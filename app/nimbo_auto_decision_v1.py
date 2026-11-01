"""M6: evidence-bound ONNX FP32 fallback when all tested INT8 variants were rejected.

Decision is LOCAL REVIEW ONLY, never deployment approval. Revalidates the exact
M2/M3/M4/M5/M5.1 lineage and reruns packaged ONNX predictions offline in an
independent Python process before creating a decision receipt.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from app.nimbo_model_intake_v1 import resolve_labels
from app.nimbo_heldout_validation_v1 import load_heldout
from app.nimbo_int8_candidate_v1 import validate_int8_gate, verify_and_extract_m4
from app.nimbo_quality_quantization_v1 import verify_m5_failure, PROFILES
from app.nimbo_package_v1 import check_review
from app.nimbo_onnx_candidate_v1 import digest

SCHEMA = 'nimbo_auto_selection_v1'


def _report_path(value: str | Path, project: Path, subfolder: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise ValueError(f'Symlinked {subfolder} report is not allowed')
    path = candidate.resolve(strict=True)
    folder = (project / 'results' / subfolder).resolve()
    if folder not in path.parents or path.name != 'report.json':
        raise ValueError(f'Report must be a report.json under results/{subfolder}')
    if path.stat().st_size > 1_000_000:
        raise ValueError('Oversized report refused')
    return path


def verify_rejected_search(report: dict, *, m5_sha: str, m4_sha: str,
                           m3_sha: str, fp32_sha: str, dev_sha: str,
                           dev_examples: int) -> list[dict]:
    """Do not trust the selected field or displayed short summary of a search."""
    if report.get('schema_version') != 'nimbo_quality_int8_search_v1' or report.get('status') != 'DEVELOPMENT_ONLY_NO_EXPORT':
        raise ValueError('Invalid M5.1 report identity')
    if any(report.get(k) != v for k, v in {
        'source_m5_report_sha256': m5_sha,
        'source_m4_archive_sha256': m4_sha,
        'source_m3_report_sha256': m3_sha,
        'source_fp32_graph_sha256': fp32_sha,
        'dev_dataset_sha256': dev_sha,
    }.items()):
        raise ValueError('M5.1 report does not match original artifact/evidence chain')
    if (report.get('decision') != 'ALL_SELECTIVE_INT8_CANDIDATES_REJECTED'
            or report.get('selected') is not None
            or report.get('eligible_for_deployment') is not False
            or report.get('eligible_for_downloadable_package') is not False):
        raise ValueError('Search must conclude with zero accepted INT8 candidates')
    if report.get('unique_dev_examples') != dev_examples or not 2 <= report.get('rounds', 0) <= 5:
        raise ValueError('M5.1 development sample size/rounds inconsistent')
    names = report.get('quantized_profiles_tested')
    records = report.get('trials')
    if (not isinstance(names, list) or not isinstance(records, list)
            or not 1 <= len(names) <= len(PROFILES)
            or len(names) != len(records) or len(set(names)) != len(names)
            or any(name not in PROFILES for name in names)):
        raise ValueError('Invalid selective candidate trial listing')
    safe = []
    for name, entry in zip(names, records):
        if not isinstance(entry, dict) or entry.get('profile') != name:
            raise ValueError('Selective profile order or record does not match')
        if entry.get('status') == 'BENCHMARKED':
            verdict = entry.get('decision') or {}
            if verdict.get('dev_gates_passed') is not False or verdict.get('decision') != 'INT8_REJECTED_DEV_GATES':
                raise ValueError('Accepted/ambiguous candidate cannot be silently ignored')
            recomputed = validate_int8_gate(entry['fp32_metrics'], entry['int8_metrics'],
                old_bytes=entry['fp32_graph_bytes'], new_bytes=entry['int8_graph_bytes'],
                sample_count=dev_examples, agreement=verdict['prediction_agreement'],
                mismatches=verdict['repeatability_mismatches'], objective=report['objective'])
            if recomputed != verdict:
                raise ValueError('Candidate gate recalculation differs from original decision')
            if not isinstance(entry.get('int8_graph_sha256'), str) or len(entry['int8_graph_sha256']) != 64:
                raise ValueError('Missing candidate graph identity')
            safe.append({'profile':name,'status':'REJECTED','reason':'FAILED_FIXED_QUALITY_SIZE_OR_PERFORMANCE_GATES',
                         'quality_pass':verdict['quality_pass'],
                         'size_pass':verdict['size_pass'],'performance_pass':verdict['performance_pass'],
                         'prediction_agreement':verdict['prediction_agreement'],
                         'graph_bytes':entry['int8_graph_bytes']})
        elif entry.get('status') == 'FAILED':
            if not isinstance(entry.get('error_type'), str):
                raise ValueError('Failed trial must contain error_type')
            safe.append({'profile':name,'status':'FAILED','reason':'CANDIDATE_NOT_EVALUABLE'})
        else:
            raise ValueError('Unknown candidate trial state')
    return safe


def verify_offline_package(package_dir: Path, dataset: Path, *, project: Path) -> dict:
    """Reload the extracted archive without network in a fresh subprocess."""
    env = dict(os.environ)
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    command = [sys.executable, '-m', 'tools.verify_nimbo_package_v1',
               '--package', str(package_dir), '--dataset', str(dataset)]
    run = subprocess.run(command, cwd=project, env=env,
                         text=True, capture_output=True, timeout=180, check=False)
    if run.returncode != 0 or 'STATUS: OFFLINE_BUNDLE_RELOAD_VERIFIED_REVIEW_ONLY' not in run.stdout:
        raise RuntimeError('Independent offline package reload rejected: '+run.stderr.strip()[-250:])
    payload = run.stdout.split('STATUS: OFFLINE_BUNDLE_RELOAD_VERIFIED_REVIEW_ONLY',1)[1].strip()
    data = json.loads(payload)
    if data.get('status') != 'OFFLINE_BUNDLE_RELOAD_VERIFIED_REVIEW_ONLY' or data.get('eligible_for_deployment') is not False:
        raise RuntimeError('Offline reload report is not in review-only state')
    return {'status':data['status'], 'samples':data['samples'], 'accuracy':data['accuracy'],
            'model_graph_sha256':data['model_graph_sha256'],
            'prediction_fingerprint':data['prediction_fingerprint']}


def select_verified_fallback(package: str | Path, package_sha: str, m3_report: str | Path,
                             m3_dataset: str | Path, m5_report: str | Path,
                             search_report: str | Path, dev_dataset: str | Path,
                             *, root: str | Path='.') -> tuple[Path, dict]:
    """Actual FP32 package reload + full evidence-gate verification; fail closed."""
    project = Path(root).expanduser().resolve(strict=True)
    if project.is_symlink():
        raise ValueError('Symlinked project root refused')
    m3_path = Path(m3_report).expanduser().resolve(strict=True)
    m5_path = _report_path(m5_report,project,'auto_optimizer_int8_v1')
    search_path = _report_path(search_report,project,'auto_optimizer_quality_int8_v1')
    test_data = Path(m3_dataset).expanduser().resolve(strict=True)
    development = Path(dev_dataset).expanduser().resolve(strict=True)
    archive = Path(package).expanduser()
    if archive.is_symlink():
        raise ValueError('Symlink package rejected')
    archive = archive.resolve(strict=True)
    if (project/'results'/'auto_optimizer_packages_v1').resolve() not in archive.parents:
        raise ValueError('M4 package must originate from original NIMBO project results')

    m3, m2, graph, _, m3_dataset_sha = check_review(m3_path,test_data,root=project)
    rows, dev_sha, source = load_heldout(development)
    if dev_sha in (m3_dataset_sha,m2['dataset_sha256']):
        raise ValueError('Development dataset must not equal prior test data')
    if source.get('source') == 'cardiffnlp/tweet_eval' and source.get('split') != 'validation':
        raise ValueError('Only TweetEval validation may be used for INT8 candidate selection')
    m5 = verify_m5_failure(m5_path,project=project,expected_m4_sha=package_sha,
           expected_dev_sha=dev_sha,expected_m3_sha=digest(m3_path),
           expected_graph_sha=m2['artifact']['sha256'])
    search = json.loads(search_path.read_text(encoding='utf-8'))
    rejected = verify_rejected_search(search,m5_sha=digest(m5_path),m4_sha=package_sha,
           m3_sha=digest(m3_path),fp32_sha=m2['artifact']['sha256'],
           dev_sha=dev_sha,dev_examples=len(rows))
    with tempfile.TemporaryDirectory(prefix='nimbo_selection_') as directory:
        saved, manifest = verify_and_extract_m4(archive,Path(directory)/'extracted',
            expected_sha256=package_sha,m3=m3,m3_report=m3_path,
            fp32_graph_sha256=m2['artifact']['sha256'])
        offline = verify_offline_package(saved,test_data,project=project)
        if offline['samples'] != m3['heldout_unique_examples']:
            raise RuntimeError('Offline package verified against an unexpected test count')
        if offline['model_graph_sha256'] != m2['artifact']['sha256']:
            raise RuntimeError('Offline reloaded ONNX graph differs from M3 graph')
        if abs(offline['accuracy'] - m3['candidate']['accuracy']) > 1e-6:
            raise RuntimeError('Package reload accuracy differs from M3 verified accuracy')
    verdict = {
        'schema_version':SCHEMA,
        'status':'VERIFIED_FALLBACK_SELECTED_REVIEW_ONLY',
        'decision':'SELECT_ONNX_FP32_REJECT_INT8',
        'selected_variant':'ONNX_RUNTIME_FP32',
        'selected_package':{'path':str(archive),'sha256':package_sha,'graph_sha256':m2['artifact']['sha256']},
        'model':{'model_id':m3['model']['model_id'],
                 'resolved_revision':m3['model']['resolved_revision']},
        'evidence':{'m3_report_sha256':digest(m3_path),'m5_report_sha256':digest(m5_path),
                    'm5_search_sha256':digest(search_path),'int8_candidates_rejected':len(rejected),
                    'm5_decision':m5['decision']['decision'],
                    'm5_1_decision':search['decision'],'offline_reload':offline,
                    'm3_baseline':m3['baseline'],'m3_fp32_candidate':m3['candidate'],
                    'm3_prediction_agreement':m3['decision']['prediction_agreement']},
        'rejections':rejected,
        'eligible_for_deployment':False,
        'eligible_for_downloadable_package':True,
        'package_status':'EXISTING_VERIFIED_FP32_REVIEW_PACKAGE_ONLY',
        'limits':['No INT8 variant passed development quality policy; all remain rejected.',
                  'Existing verified M4 FP32 bundle is a review-only downloadable artifact, not a production deployment.',
                  'M3 TweetEval test is not proven independent of public model development.',
                  'Measured speed improvement applies only to the recorded Mac CPU tests.',
                  'FP32 ONNX graph was not proven to be smaller than the original model.',
                  'No measured electrical energy, FLOPs, RAM savings or cloud billing.'],
    }
    dest_root = project/'results'/'auto_optimizer_decisions_v1'
    dest_root.mkdir(parents=True,exist_ok=True)
    dest = dest_root / ('selection_'+uuid.uuid4().hex[:16]+'.json')
    with dest.open('x',encoding='utf-8') as out:
        json.dump(verdict,out,indent=2,ensure_ascii=False)
        out.write('\n')
    return dest,verdict
