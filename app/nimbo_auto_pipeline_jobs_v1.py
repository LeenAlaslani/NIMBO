"""M7a local-only asynchronous pipeline; real M1->M4 calls, no fake progress.

Not a multi-tenant sandbox, network-facing service, or production scheduler.
No quantization is promised here: objective is explicitly speed / ONNX FP32.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.nimbo_model_intake_v1 import check_model_id, validate_requested_revision, read_evaluation_file
from app.nimbo_heldout_validation_v1 import load_heldout

ROOT_FOLDER = Path('results/auto_optimizer_jobs_v1')
INPUT_FOLDER = Path('results/auto_optimizer_job_inputs_v1')
HELDOUT_FOLDER = Path('results/auto_optimizer_heldout_v1')
ID_RE = re.compile(r'^[0-9a-f]{32}$')
FILE_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,90}\.(?:csv|jsonl)$')
FINAL = frozenset({'COMPLETED_REVIEW_ONLY', 'COMPLETED_REJECTED', 'FAILED', 'INTERRUPTED'})
ACTIVE = frozenset({'QUEUED', 'RUNNING'})
MAX_EVENTS = 100


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _file(root: Path, folder: Path, name: str) -> Path:
    if not isinstance(name, str) or not FILE_RE.fullmatch(name) or name.startswith('.'):
        raise ValueError('Dataset must be a simple .csv/.jsonl filename; arbitrary paths not accepted')
    top = (root / folder).resolve()
    if not top.is_relative_to(root.resolve()):
        raise ValueError('Dataset input directory must remain inside the local project')
    target = top / name
    if target.is_symlink() or not target.is_file():
        raise ValueError('Required local dataset is missing or symlinked: '+name)
    actual = target.resolve(strict=True)
    if actual.parent != top:
        raise ValueError('Dataset must stay in the designated local inputs folder')
    return actual


def validate_request(root: Path, request: dict) -> dict:
    if not isinstance(request, dict) or set(request) != {'model_id', 'revision', 'pilot_file', 'heldout_file', 'preprocess', 'objective'}:
        raise ValueError('Invalid exact M7 request fields')
    model_id = check_model_id(request['model_id'])
    revision = validate_requested_revision(request['revision'])
    if revision is None:
        raise ValueError('M7 requires an immutable model revision SHA')
    if request['objective'] != 'speed':
        raise ValueError('M7a supports speed/ONNX FP32 only; size/balanced is not implemented')
    if request['preprocess'] not in ('identity', 'cardiff_tweet'):
        raise ValueError('Unsupported preprocessing policy')
    pilot = _file(root, INPUT_FOLDER, request['pilot_file'])
    heldout = _file(root, HELDOUT_FOLDER, request['heldout_file'])
    pilot_rows, pilot_sha = read_evaluation_file(pilot)
    heldout_rows, heldout_sha, _ = load_heldout(heldout)
    if not all(row['label'] is not None for row in pilot_rows):
        raise ValueError('Pilot dataset must be fully labeled')
    if len(heldout_rows) < 200 or len(pilot_rows) < 2:
        raise ValueError('Insufficient labeled evaluation samples')
    if pilot_sha == heldout_sha:
        raise ValueError('Heldout dataset must differ from pilot dataset')
    return {'model_id': model_id, 'revision': revision, 'pilot_file': pilot.name,
            'heldout_file': heldout.name, 'preprocess': request['preprocess'], 'objective': 'speed',
            'pilot_sha256': pilot_sha, 'heldout_sha256': heldout_sha,
            'pilot_rows': len(pilot_rows), 'heldout_rows': len(heldout_rows)}


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.parent / ('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load(path: Path) -> dict:
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def job_folder(root: Path, job_id: str) -> Path:
    if not ID_RE.fullmatch(job_id):
        raise ValueError('Invalid job ID')
    folder = (root / ROOT_FOLDER / job_id)
    if folder.is_symlink() or not folder.is_dir():
        raise FileNotFoundError('Job not found')
    return folder


def _process_alive(pid: int | None) -> bool:
    if type(pid) is not int or pid <= 0:
        return False
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_job(root: Path, job_id: str) -> dict:
    folder = job_folder(root, job_id)
    status = _load(folder/'status.json')
    if status.get('state') in ACTIVE:
        launcher = folder/'launcher.json'
        if launcher.is_file():
            info = _load(launcher)
            if not _process_alive(info.get('pid')):
                # Worker exited without a final receipt (OOM/crash/restart).
                _event(folder,'INTERRUPTED','error','Worker exited before writing a final receipt')
                status = _load(folder/'status.json')
    return status


def read_events(root: Path, job_id: str, after: int = 0) -> dict:
    if not isinstance(after, int) or after < 0:
        raise ValueError('after must be >= 0')
    folder = job_folder(root, job_id)
    events = []
    with (folder/'events.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if e['seq'] > after:
                    events.append(e)
                if len(events) >= MAX_EVENTS:
                    break
    return {'job_id': job_id, 'events': events, 'last_seq': events[-1]['seq'] if events else after}


def _event(folder: Path, state: str, stage: str, message: str, *, extra: dict | None = None) -> None:
    status = _load(folder/'status.json')
    seq = status['seq'] + 1
    event = {'seq': seq, 'at': utc(), 'state': state, 'stage': stage, 'message': message}
    if extra:
        event['evidence'] = extra
    with (folder/'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(event, ensure_ascii=False)+'\n')
        f.flush()
        os.fsync(f.fileno())
    status.update({'state': state, 'stage': stage, 'seq': seq, 'updated_at': event['at']})
    if extra:
        status['last_evidence'] = extra
    _atomic_json(folder/'status.json',status)


@contextmanager
def submission_lock(root: Path):
    jobs = root / ROOT_FOLDER
    jobs.mkdir(parents=True, exist_ok=True)
    lock = jobs/'.submission.lock'
    with lock.open('a+b') as fd:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        try:
            yield jobs
        finally:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)


def submit(root: Path, request: dict, *, spawn=subprocess.Popen) -> dict:
    """Accept at most one CPU-heavy job. Uses a separate real Python process."""
    root = root.expanduser().resolve(strict=True)
    cleaned = validate_request(root, request)
    with submission_lock(root) as jobs:
        for directory in jobs.iterdir():
            if directory.is_dir() and ID_RE.fullmatch(directory.name):
                if read_job(root, directory.name).get('state') in ACTIVE:
                    raise RuntimeError('One optimization job is already running; wait for it to finish')
        job_id = uuid.uuid4().hex
        folder = jobs/job_id
        folder.mkdir(mode=0o700)
        _atomic_json(folder/'request.json',cleaned)
        _atomic_json(folder/'status.json', {'job_id':job_id, 'state':'QUEUED', 'stage':'queued',
                   'seq':0,'created_at':utc(),'updated_at':utc(),'eligible_for_deployment':False})
        (folder/'events.jsonl').touch(mode=0o600)
        _event(folder,'QUEUED','queued','Accepted one local supported-model speed optimization job')
        argv = [sys.executable, '-m', 'tools.run_nimbo_auto_pipeline_job_v1',
                '--root', str(root), '--job-id', job_id]
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        try:
            local_log = folder/'worker.stderr.log'
            with local_log.open('ab') as err:
                os.chmod(local_log,0o600)
                process = spawn(argv, cwd=str(root), env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=err,
                                start_new_session=True)
            _atomic_json(folder/'launcher.json',{'pid':process.pid, 'started_at':utc()})
        except Exception as exc:
            _event(folder,'FAILED','error','Worker failed to start',extra={'error_type':type(exc).__name__})
            raise RuntimeError('Cannot start the optimization worker') from exc
    return {'job_id': job_id, 'state':'QUEUED', 'status_url':f'/nimbo/auto/jobs/{job_id}',
            'events_url':f'/nimbo/auto/jobs/{job_id}/events',
            'eligible_for_deployment':False}


def run_worker(root: Path, job_id: str, *, baseline=None, candidate=None, heldout=None, packager=None) -> dict:
    """All stages execute REAL original modules; injections used only by unit tests."""
    from app.nimbo_model_intake_v1 import run_real_baseline
    from app.nimbo_onnx_candidate_v1 import run_paired_candidate
    from app.nimbo_heldout_validation_v1 import validate_existing_candidate
    from app.nimbo_package_v1 import package_model
    baseline = baseline or run_real_baseline
    candidate = candidate or run_paired_candidate
    heldout = heldout or validate_existing_candidate
    packager = packager or package_model
    root = root.expanduser().resolve(strict=True)
    folder = job_folder(root, job_id)
    req = _load(folder/'request.json')
    pilot = _file(root, INPUT_FOLDER, req['pilot_file'])
    hold = _file(root, HELDOUT_FOLDER, req['heldout_file'])
    if hashlib.sha256(pilot.read_bytes()).hexdigest() != req['pilot_sha256'] or hashlib.sha256(hold.read_bytes()).hexdigest() != req['heldout_sha256']:
        _event(folder,'FAILED','error','Dataset changed after job submission')
        return _finish(folder,{},'EXECUTION_FAILED','Input dataset changed after submission')
    reports = {}
    try:
        _event(folder,'RUNNING','analyze','Starting real pinned PyTorch CPU baseline')
        m1, m1_report = baseline(req['model_id'], pilot, revision=req['revision'])
        reports['baseline_report'] = str(m1)
        _event(folder,'RUNNING','analyze','PyTorch baseline measured',extra={
            'accuracy':m1_report['metrics']['accuracy'], 'mean_forward_ms':m1_report['metrics']['mean_warm_forward_ms']})
        _event(folder,'RUNNING','optimize','Starting ONNX FP32 graph export and paired benchmark')
        m2, candidate_report = candidate(req['model_id'], pilot, revision=req['revision'], baseline_report=m1)
        reports['candidate_report'] = str(m2)
        if not candidate_report['decision']['local_gates_passed']:
            reason = 'ONNX did not pass paired development quality/performance gates'
            _event(folder,'COMPLETED_REJECTED','verify',reason,extra={'decision':candidate_report['decision']['decision']})
            return _finish(folder,reports,'SELECT_ORIGINAL_NO_ONNX_PACKAGE',reason)
        _event(folder,'RUNNING','optimize','ONNX paired benchmark passed local gates',extra={
            'decision':candidate_report['decision']['decision']})
        _event(folder,'RUNNING','verify','Starting new heldout paired evaluation using the exact ONNX graph')
        m3, heldout_report = heldout(m2, hold, root=root, preprocess=req['preprocess'])
        reports['heldout_report'] = str(m3)
        if not heldout_report['decision']['heldout_gates_passed']:
            reason = 'ONNX rejected by heldout quality/resource gates'
            _event(folder,'COMPLETED_REJECTED','verify',reason,extra={'decision':heldout_report['decision']['decision']})
            return _finish(folder,reports,'SELECT_ORIGINAL_NO_ONNX_PACKAGE',reason)
        _event(folder,'RUNNING','verify','Heldout quality and performance gates passed',extra={
            'decision':heldout_report['decision']['decision'],
            'accuracy':heldout_report['candidate']['accuracy']})
        _event(folder,'RUNNING','deliver','Creating and independently offline-verifying ONNX FP32 package')
        package, summary = packager(m3, hold, root=root)
        if summary.get('status') != 'RUNNABLE_PACKAGE_VERIFIED_REVIEW_ONLY' or summary.get('verified_offline_after_extract') is not True:
            raise RuntimeError('Packager did not certify offline reload')
        reports['package_path'] = str(package)
        reports['package_sha256'] = summary['archive_sha256']
        _event(folder,'COMPLETED_REVIEW_ONLY','deliver','Offline model package verified; review only',extra={
            'archive_sha256':summary['archive_sha256'], 'reload_accuracy':summary['reload_accuracy']})
        return _finish(folder,reports,'SELECT_ONNX_FP32_REVIEW_ONLY','Passes local M1-M4 checks; no deployment approval')
    except Exception as exc:
        # Detailed error stays inside the local 0700 job directory, never in API responses.
        local_error = folder/'worker.error.log'
        local_error.write_text(traceback.format_exc()[-6000:],encoding='utf-8')
        os.chmod(local_error,0o600)
        _event(folder,'FAILED','error','Optimization stage failed; inspect local job worker.error.log',
               extra={'error_type':type(exc).__name__})
        return _finish(folder,reports,'EXECUTION_FAILED','A real stage failed; inspect local CLI/dependency logs')


def _finish(folder: Path, reports: dict, decision: str, reason: str) -> dict:
    status = _load(folder/'status.json')
    payload = {'schema_version':'nimbo_auto_job_receipt_v1','job_id':status['job_id'],
               'state':status['state'],'decision':decision,'reason':reason,
               'evidence':reports,'eligible_for_deployment':False,
               'eligible_for_downloadable_package':status['state']=='COMPLETED_REVIEW_ONLY',
               'finished_at':utc()}
    _atomic_json(folder/'receipt.json',payload)
    status['receipt'] = str(folder/'receipt.json')
    status['decision'] = decision
    status['eligible_for_review_download'] = status['state']=='COMPLETED_REVIEW_ONLY'
    _atomic_json(folder/'status.json',status)
    return status


def package_for_download(root: Path, job_id: str) -> Path:
    from app.nimbo_onnx_candidate_v1 import digest
    status = read_job(root,job_id)
    if status.get('state') != 'COMPLETED_REVIEW_ONLY':
        raise ValueError('No verified package for this job')
    receipt = _load(job_folder(root,job_id)/'receipt.json')
    if receipt.get('decision') != 'SELECT_ONNX_FP32_REVIEW_ONLY':
        raise ValueError('Job has no eligible reviewed package')
    ref = receipt.get('evidence') or {}
    folder = (root / 'results' / 'auto_optimizer_packages_v1').resolve()
    candidate = Path(ref['package_path'])
    if candidate.is_symlink() or not candidate.is_file() or candidate.resolve().parent != folder:
        raise ValueError('Package is missing or outside allowed output folder')
    if digest(candidate) != ref['package_sha256']:
        raise ValueError('Package checksum changed; download refused')
    return candidate
