"""Single-instance, same-origin public reviewer service."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
from threading import Event, Lock, Thread
import time

import psutil
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles

from app import nimbo_auto_pipeline_jobs_v1 as engine
from app.nimbo_reviewer_catalog import CASE, DATASET, DATA_REVISION, HELDOUT, MODEL, PILOT, REQUEST, REVISION, RUNTIME

MIB = 1024**2
DAY = 86400
COOKIE = 'nimbo_review_session'
WALL_SECONDS = 360
WORKER_RSS = 1_500 * MIB
WORKSPACE_BYTES = 1_350 * MIB
RETENTION_SECONDS = 3600
START_INTERVAL = 1200
STARTS_PER_DAY = 6
DOWNLOAD_BUDGET = 3 * 1024**3
DOWNLOAD_SECONDS = 300
SOURCE = Path(__file__).resolve().parents[1]
LOG = logging.getLogger('uvicorn.error')
METRICS = ('accuracy', 'mean_warm_forward_ms', 'p95_warm_forward_ms', 'process_cpu_forward_total_ms')
GATES = ('quality_pass', 'performance_pass', 'heldout_gates_passed', 'mean_latency_reduction_fraction',
         'process_cpu_time_reduction_fraction', 'p95_regression_fraction', 'prediction_agreement',
         'max_abs_logit_error', 'repeatability_mismatches', 'criteria', 'decision')


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def numeric_fields(data):
    return {key: data[key] for key in METRICS if type(data.get(key)) in (int, float) and math.isfinite(data[key])}


def public_report(evidence, directory):
    for field, scope, candidate in [('heldout_report', 'heldout', 'candidate'),
                                     ('candidate_report', 'pilot', 'onnx_candidate')]:
        value = evidence.get(field)
        if not value:
            continue
        path = Path(value).resolve(strict=True)
        if not path.is_relative_to(directory):
            raise ValueError('Evidence escaped the job directory')
        result = load_json(path)
        verdict = result['decision']
        return {'scope': scope, 'examples': result.get('heldout_unique_examples') or result['unique_dataset_rows'],
                'rounds': result['rounds'], 'baseline': numeric_fields(result['baseline']),
                'candidate': numeric_fields(result[candidate]),
                'gates': {key: verdict[key] for key in GATES if key in verdict}}
    return None


class PackageResponse(FileResponse):
    def __init__(self, package, release):
        super().__init__(package, filename='nimbo-verified-review.zip', media_type='application/zip',
                         headers={'Accept-Ranges': 'none'})
        self.release = release

    async def __call__(self, scope, receive, send):
        try:
            await asyncio.wait_for(super().__call__(scope, receive, send), timeout=DOWNLOAD_SECONDS)
        finally:
            self.release()


@dataclass
class Job:
    root: Path
    owner: str
    process: subprocess.Popen
    started: float
    status: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    peak_rss: int = 0
    peak_disk: int = 0
    previous_cpu: float = 0
    previous_time: float = 0
    current_cpu: float | None = None
    current_rss: int = 0
    ended: float | None = None
    package: Path | None = None
    streaming: bool = False
    downloads: int = 0


class Reviewer:
    def __init__(self, state, inputs):
        self.state = state
        self.inputs = inputs
        self.lock = Lock()
        self.stop = Event()
        self.jobs = {}
        self.active = None
        self.streaming = False

    def start(self):
        self.state.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.instance_lock = (self.state / 'reviewer.lock').open('a+b')
        try:
            fcntl.flock(self.instance_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.instance_lock.close()
            raise RuntimeError('The reviewer workspace already has an active supervisor') from None
        for stale in self.state.glob('instance-*'):
            if not stale.is_symlink() and (stale / '.nimbo-reviewer').is_file():
                shutil.rmtree(stale)
        self.work = Path(tempfile.mkdtemp(prefix='instance-', dir=self.state))
        (self.work / '.nimbo-reviewer').touch(mode=0o600)
        self.ledger = sqlite3.connect(self.state / 'limits.sqlite3', check_same_thread=False)
        self.ledger.execute('CREATE TABLE IF NOT EXISTS quota (kind TEXT, at REAL, value INTEGER, owner TEXT)')
        self.thread = Thread(target=self.watch, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join()
        if self.active:
            self.kill(self.jobs[self.active])
        self.ledger.close()
        shutil.rmtree(self.work)
        self.instance_lock.close()

    def check_quota(self, owner):
        now = time.time()
        with self.ledger:
            self.ledger.execute('DELETE FROM quota WHERE at < ?', (now - 30 * DAY,))
        starts = self.ledger.execute("SELECT at, owner FROM quota WHERE kind='start' AND at > ?",
                                     (now - DAY,)).fetchall()
        if len(starts) >= STARTS_PER_DAY or any(at > now - START_INTERVAL or who == owner for at, who in starts):
            raise HTTPException(429, 'The live optimization quota is temporarily full')

    def record(self, kind, value, owner):
        with self.ledger:
            self.ledger.execute('INSERT INTO quota VALUES (?, ?, ?, ?)', (kind, time.time(), value, owner))

    def submit(self, owner):
        with self.lock:
            if self.active or self.streaming:
                raise HTTPException(409, 'Another reviewer job or package download is active')
            self.check_quota(owner)
            if len(self.jobs) >= 4 or shutil.disk_usage(self.work).free < 2 * WORKSPACE_BYTES:
                raise HTTPException(503, 'The reviewer workspace is temporarily at capacity')
            self.record('start', 1, owner)
            root = Path(tempfile.mkdtemp(prefix='job-', dir=self.work))
            process = None

            def spawn(argv, **kwargs):
                nonlocal process
                env = {'PATH': os.defpath, 'HOME': str(root), 'TMPDIR': str(root), 'PYTHONPATH': str(SOURCE),
                       'HF_HUB_CACHE': str(self.inputs / 'hub'), 'HF_HUB_OFFLINE': '1',
                       'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                       'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'NIMBO_PREPARED_MODEL': str(self.inputs / 'model.json'),
                       'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                       'TOKENIZERS_PARALLELISM': 'false', 'PYTHONDONTWRITEBYTECODE': '1'}
                process = subprocess.Popen([argv[0], '-m', 'tools.run_nimbo_reviewer_job', str(root), argv[-1]],
                                           **(kwargs | {'env': env}))
                return process

            try:
                for folder, name in [(engine.INPUT_FOLDER, 'pilot'), (engine.HELDOUT_FOLDER, 'heldout')]:
                    output = root / folder
                    output.mkdir(parents=True)
                    for suffix in ('.csv', '.source.json'):
                        file = f'{name}{suffix}'
                        shutil.copyfile(self.inputs / file, output / file)
                accepted = engine.submit(root, REQUEST, spawn=spawn)
            except Exception as exc:
                if process and process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                shutil.rmtree(root)
                LOG.error('Reviewer start failed: %s', type(exc).__name__)
                raise HTTPException(503, 'Could not start the optimization worker') from exc
            self.active = accepted['job_id']
            self.jobs[self.active] = Job(root, owner, process, time.monotonic(),
                                         {'job_id': self.active, 'state': 'QUEUED'})
            return self.jobs[self.active].status

    def kill(self, job):
        if job.process.poll() is None:
            try:
                os.killpg(job.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            job.process.wait()

    def refresh(self, job_id, job):
        prefix = job.root / engine.ROOT_FOLDER / job_id
        status = load_json(prefix / 'status.json')
        committed, _, _ = (prefix / 'events.jsonl').read_text(encoding='utf-8').rpartition('\n')
        result = [json.loads(line) for line in committed.splitlines()]
        job.events = [{key: event[key] for key in ('seq', 'stage', 'state', 'message')}
                      for event in result if event['seq'] <= status['seq']]
        return status

    def measure(self, job, now):
        try:
            root = psutil.Process(job.process.pid)
            rss = cpu = 0
            try:
                children = root.children(recursive=True)
            except (OSError, psutil.AccessDenied):
                children = []
            for worker in [root, *children]:
                try:
                    rss += worker.memory_info().rss
                    times = worker.cpu_times()
                    cpu += times.user + times.system
                except (OSError, psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            job.current_cpu = round(max(0, cpu - job.previous_cpu) * 100 / (now - job.previous_time), 1) if job.previous_time else None
            job.current_rss = rss
            job.previous_time, job.previous_cpu = now, cpu
            job.peak_rss = max(job.peak_rss, rss)
        except (OSError, psutil.NoSuchProcess, psutil.AccessDenied):
            job.current_rss = 0
            job.current_cpu = None
        size = 0
        for path in job.root.rglob('*'):
            try:
                if path.is_file():
                    size += path.stat().st_size
            except FileNotFoundError:
                continue
        job.peak_disk = max(job.peak_disk, size)

    def finish(self, job_id, job, status, error):
        comparison = None
        reason = error
        state = status['state']
        decision = status.get('decision')
        if not error:
            try:
                receipt = load_json(job.root / engine.ROOT_FOLDER / job_id / 'receipt.json')
                comparison = public_report(receipt['evidence'], job.root)
                if state == 'COMPLETED_REVIEW_ONLY':
                    if job.process.returncode != 0 or not comparison or comparison['scope'] != 'heldout':
                        raise ValueError('Missing completed worker evidence')
                    job.package = engine.package_for_download(job.root, job_id)
            except (OSError, KeyError, ValueError):
                reason = 'The worker stopped before producing a valid result'
        if reason:
            state, decision = 'FAILED', 'EXECUTION_FAILED'
            job.events = [event for event in job.events if event['state'] not in engine.FINAL]
            job.events.append({'seq': status['seq'] + 1, 'stage': 'error', 'state': state, 'message': reason})
        else:
            reason = {'COMPLETED_REVIEW_ONLY': 'Held-out gates and offline package reload passed',
                      'COMPLETED_REJECTED': 'The candidate failed a measured gate. The original model was retained.',
                      'FAILED': 'A real optimization stage failed.'}.get(state, 'The worker was interrupted.')
        if job.package:
            evidence = receipt['evidence']
            package = {'size_bytes': job.package.stat().st_size, 'sha256': evidence['package_sha256'],
                       'verified_offline_after_extract': True}
        else:
            package = None
        job.status = {'job_id': job_id, 'state': state, 'decision': decision, 'reason': reason,
                      'eligible_for_review_download': package is not None, 'package': package,
                      'comparison': comparison, 'usage': {'wall_seconds': round(time.monotonic() - job.started, 2),
                      'observed_peak_worker_rss_mib': round(job.peak_rss / MIB, 1),
                      'observed_peak_workspace_mib': round(job.peak_disk / MIB, 1)}}
        LOG.info('Reviewer job %s %s %s', job_id, state, job.status['usage'])
        job.ended = time.monotonic()
        self.active = None
        for path in job.root.iterdir():
            if job.package and path.name == 'results':
                for folder in path.iterdir():
                    if folder != job.package.parent:
                        shutil.rmtree(folder)
            elif path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    def watch(self):
        while not self.stop.wait(0.25):
            with self.lock:
                now = time.monotonic()
                for job_id, job in list(self.jobs.items()):
                    if job.ended and not job.streaming and now - job.ended > RETENTION_SECONDS:
                        shutil.rmtree(job.root)
                        del self.jobs[job_id]
                if self.active is None:
                    continue
                job_id = self.active
                job = self.jobs[job_id]
                try:
                    status = self.refresh(job_id, job)
                    self.measure(job, now)
                    limit = next((label for hit, label in [
                        (now - job.started > WALL_SECONDS, f'Worker exceeded the {WALL_SECONDS}-second wall limit'),
                        (job.peak_rss > WORKER_RSS, 'Worker exceeded the memory limit'),
                        (job.peak_disk > WORKSPACE_BYTES, 'Worker exceeded the temporary storage limit'),
                    ] if hit), None)
                    if limit or job.process.poll() is not None:
                        if limit:
                            self.kill(job)
                        self.finish(job_id, job, status, limit)
                    else:
                        job.status = {'job_id': job_id, 'state': status['state'] if status['state'] in engine.ACTIVE else 'RUNNING'}
                except Exception as exc:
                    LOG.error('Reviewer supervisor: %s', type(exc).__name__)
                    self.kill(job)
                    self.finish(job_id, job, {'state': 'FAILED', 'seq': job.events[-1]['seq'] if job.events else 0},
                                'The optimizer could not complete this job')

    def require(self, job_id, owner):
        job = self.jobs.get(job_id)
        if job is None or not secrets.compare_digest(job.owner, owner):
            raise HTTPException(404, 'Job unavailable or expired')
        return job

    def download(self, job_id, owner):
        with self.lock:
            job = self.require(job_id, owner)
            if not job.package:
                raise HTTPException(404, 'No verified review package')
            if self.active or self.streaming:
                raise HTTPException(409, 'Another reviewer job or package download is active')
            size = job.package.stat().st_size
            used = self.ledger.execute("SELECT COALESCE(SUM(value), 0) FROM quota WHERE kind='zip' AND at > ?",
                                       (time.time() - 30 * DAY,)).fetchone()[0]
            if job.downloads >= 2 or used + size > DOWNLOAD_BUDGET:
                raise HTTPException(429, 'The package download quota is full')
            self.record('zip', size, owner)
            job.downloads += 1
            self.streaming = job.streaming = True

        def release():
            with self.lock:
                self.streaming = job.streaming = False

        return PackageResponse(job.package, release)


def owner_key(request):
    token = request.cookies.get(COOKIE, '')
    if 40 <= len(token) <= 64 and all(c.isascii() and (c.isalnum() or c in '-_') for c in token):
        return hashlib.sha256(token.encode()).hexdigest()
    return ''


def create_app(*, state, inputs, web, replay, origin):
    service = Reviewer(state, inputs)

    @asynccontextmanager
    async def lifespan(_):
        for path in ('model.json', 'pilot.csv', 'pilot.source.json', 'heldout.csv', 'heldout.source.json'):
            if not (inputs / path).is_file():
                raise RuntimeError('Prepared reviewer inputs are missing')
        historical = load_json(replay)
        app.state.historical = historical
        service.start()
        try:
            yield
        finally:
            service.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.reviewer = service

    @app.middleware('http')
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        if request.url.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/healthz')
    def health():
        if not service.thread.is_alive():
            raise HTTPException(503, 'Reviewer worker is unavailable')
        return {'status': 'ready'}

    @app.get('/api/reviewer/catalog')
    def catalog():
        return {'cases': [{'id': CASE, 'model': MODEL, 'revision': REVISION, 'dataset': DATASET,
                           'dataset_revision': DATA_REVISION, 'pilot': PILOT, 'heldout': HELDOUT,
                           'threads': RUNTIME['threads'], 'max_length': RUNTIME['length']}]}

    @app.get('/api/reviewer/replay')
    def history():
        return app.state.historical

    @app.post('/api/reviewer/jobs', status_code=202)
    async def submit(request: Request, response: Response):
        if request.headers.get('origin') != origin:
            raise HTTPException(403, 'Live job submission requires the application origin')
        if request.headers.get('content-type', '').split(';', 1)[0] != 'application/json':
            raise HTTPException(415, 'Expected JSON')
        body = bytearray()
        try:
            async with asyncio.timeout(8):
                async for chunk in request.stream():
                    if len(body) + len(chunk) > 256:
                        raise HTTPException(413, 'The reviewer request is too large')
                    body.extend(chunk)
        except TimeoutError as exc:
            raise HTTPException(408, 'Reviewer request timed out') from exc
        try:
            selected = json.loads(body)
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(422, 'Invalid reviewer request') from exc
        if selected != {'case': CASE}:
            raise HTTPException(422, 'Select an available public model and dataset pair')
        token = request.cookies.get(COOKIE) if owner_key(request) else secrets.token_urlsafe(32)
        result = await asyncio.to_thread(service.submit, hashlib.sha256(token.encode()).hexdigest())
        response.set_cookie(COOKIE, token, max_age=DAY, secure=origin.startswith('https://'), httponly=True,
                            samesite='strict', path='/')
        return result

    @app.get('/api/reviewer/jobs/{job_id}')
    def status(job_id: str, request: Request):
        with service.lock:
            job = service.require(job_id, owner_key(request))
            return {**job.status, 'events': job.events}

    @app.get('/api/reviewer/jobs/{job_id}/events')
    def events(job_id: str, request: Request):
        with service.lock:
            return {'events': service.require(job_id, owner_key(request)).events}

    @app.get('/api/reviewer/jobs/{job_id}/resources')
    def resources(job_id: str, request: Request):
        with service.lock:
            job = service.require(job_id, owner_key(request))
            return {'monitor_state': 'finished' if job.ended else 'live' if job.current_rss else 'waiting',
                    'sampled_at': datetime.now(timezone.utc).isoformat(),
                    'job_cpu_percent_one_core': job.current_cpu if not job.ended else None,
                    'job_rss_mib': round(job.current_rss / MIB, 1) if job.current_rss and not job.ended else None,
                    'observed_peak_job_rss_mib': round(job.peak_rss / MIB, 1) if job.peak_rss else None}

    @app.get('/api/reviewer/jobs/{job_id}/package')
    def package(job_id: str, request: Request):
        if request.headers.get('range'):
            raise HTTPException(416, 'Range downloads are unavailable')
        return service.download(job_id, owner_key(request))

    @app.get('/')
    def index():
        return FileResponse(web / 'index.html')

    @app.get('/nimbo-brand-mark.png')
    def logo():
        return FileResponse(web / 'nimbo-brand-mark.png')

    app.mount('/assets', StaticFiles(directory=web / 'assets'), name='assets')
    return app


if __name__ == '__main__':
    import uvicorn

    port = int(os.environ.get('PORT', '10000'))
    origin = os.environ.get('NIMBO_PUBLIC_ORIGIN') or os.environ.get('RENDER_EXTERNAL_URL') or f'http://127.0.0.1:{port}'
    application = create_app(state=Path(os.environ['NIMBO_STATE']), inputs=Path(os.environ['NIMBO_INPUTS']),
                             web=Path(os.environ['NIMBO_WEB']), replay=Path(os.environ['NIMBO_REPLAY']),
                             origin=origin.rstrip('/'))
    uvicorn.run(application, host='0.0.0.0', port=port, workers=1, proxy_headers=False, access_log=False)
