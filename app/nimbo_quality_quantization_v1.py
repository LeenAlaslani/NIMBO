"""NIMBO M5.1: bounded selective dynamic INT8 search on M5 development data.

No automatic deployment or export. All candidates derive from the *exact* M4
FP32 graph. M3 TEST data is used ONLY to verify the M4 evidence chain, never
for profile ranking or tuning. Source code alone is NOT real-model evidence.
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
import uuid
from pathlib import Path

from app.nimbo_int8_candidate_v1 import (
    verify_and_extract_m4, paired_evaluation, validate_int8_gate,
)
from app.nimbo_package_v1 import check_review
from app.nimbo_heldout_validation_v1 import load_heldout
from app.nimbo_onnx_candidate_v1 import digest

# Deterministic, small development-only search. Not an unbounded hyperparameter sweep.
PROFILES = {
    'front_50': ('front', 0.50, False),
    'back_50': ('back', 0.50, False),
    'front_75': ('front', 0.75, False),
    'front_50_per_channel': ('front', 0.50, True),
    'back_50_per_channel': ('back', 0.50, True),
}
DEFAULT_PROFILES = ('front_50', 'back_50', 'front_75', 'front_50_per_channel')
MAX_BYTES = 950 * 1024 * 1024


def verify_m5_failure(report_file: str | Path, *, project: Path, expected_m4_sha: str,
                      expected_dev_sha: str, expected_m3_sha: str, expected_graph_sha: str) -> dict:
    """Read the original rejected M5 attempt; no data or artifacts are overwritten."""
    path = Path(report_file).expanduser().resolve(strict=True)
    root = (project / 'results' / 'auto_optimizer_int8_v1').resolve()
    if root not in path.parents or path.name != 'report.json' or path.is_symlink():
        raise ValueError('M5 report must belong to a local original M5 run')
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema_version') != 'nimbo_int8_dev_trial_v1' or data.get('status') != 'INT8_DEVELOPMENT_ONLY_NO_EXPORT':
        raise ValueError('Invalid M5 report schema or status')
    expected = {
        'source_m4_archive_sha256': expected_m4_sha,
        'source_m3_report_sha256': expected_m3_sha,
        'source_fp32_graph_sha256': expected_graph_sha,
        'dev_dataset_sha256': expected_dev_sha,
    }
    if any(data.get(k) != v for k, v in expected.items()):
        raise ValueError('M5 source evidence does not match M4/M3/development dataset')
    decision = data.get('decision') or {}
    if decision.get('decision') != 'INT8_REJECTED_DEV_GATES' or decision.get('dev_gates_passed') is not False:
        raise ValueError('M5.1 requires the original rejected M5 decision, not an approved variant')
    graph = path.parent / 'model.int8.onnx'
    if graph.is_symlink() or not graph.is_file() or not 1024 <= graph.stat().st_size <= MAX_BYTES:
        raise ValueError('Original M5 INT8 graph is missing or invalid')
    if data.get('int8_graph_bytes') != graph.stat().st_size or data.get('int8_graph_sha256') != digest(graph):
        raise ValueError('M5 INT8 graph checksum mismatch')
    return data


def normalize_profiles(names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if not 1 <= len(names) <= len(PROFILES) or len(set(names)) != len(names):
        raise ValueError('Choose 1–5 distinct profiles')
    if any(p not in PROFILES for p in names):
        raise ValueError('Unknown profile; allowed: ' + ', '.join(PROFILES))
    return tuple(names)


def select_matmul_nodes(model) -> list[str]:
    """Require stable unique ONNX node names; never silently quantize unnamed nodes."""
    nodes = [n for n in model.graph.node if n.op_type == 'MatMul']
    names = [n.name for n in nodes]
    if len(nodes) < 4 or any(not x for x in names) or len(set(names)) != len(names):
        raise ValueError('Expected 4+ distinctly named ONNX MatMul nodes; selective quantization unsupported')
    return names


def profile_nodes(all_names: list[str], profile: str) -> tuple[list[str], bool]:
    if profile not in PROFILES:
        raise ValueError('Unknown profile')
    direction, fraction, per_channel = PROFILES[profile]
    if len(all_names) < 4 or len(set(all_names)) != len(all_names):
        raise ValueError('Invalid source MatMul node catalog')
    n = max(1, math.floor(len(all_names) * fraction))
    # ONNX graph node order approximates execution order. This is a heuristic,
    # not a semantic transformer-layer mapping; evidence records this limitation.
    chosen = all_names[:n] if direction == 'front' else all_names[-n:]
    return chosen, per_channel


def choose_winner(records: list[dict], objective: str) -> dict | None:
    """Require all original M5 size/quality/performance gates. Never relax them."""
    if objective not in ('size', 'balanced'):
        raise ValueError('Unknown objective')
    good = [r for r in records if r.get('decision', {}).get('dev_gates_passed') is True
            and r.get('status') == 'BENCHMARKED']
    if not good:
        return None
    if objective == 'size':
        return min(good, key=lambda r: (r['int8_graph_bytes'], r['int8_metrics']['mean_warm_forward_ms'], r['profile']))
    return min(good, key=lambda r: (r['int8_metrics']['mean_warm_forward_ms'], r['int8_graph_bytes'], r['profile']))


def quantize_profile(source: Path, target: Path, *, all_names: list[str], profile: str) -> dict:
    """Actual ORT quantize_dynamic; only selected named MatMul nodes are eligible."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnx
    chosen, per_channel = profile_nodes(all_names, profile)
    quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8,
                     per_channel=per_channel, nodes_to_quantize=chosen,
                     op_types_to_quantize=['MatMul'])
    if not target.is_file() or not 1024 <= target.stat().st_size <= MAX_BYTES:
        raise RuntimeError('No valid selective INT8 graph created')
    new = onnx.load(str(target), load_external_data=False)
    ops = {n.op_type for n in new.graph.node}
    if not ops.intersection({'MatMulInteger', 'QLinearMatMul'}):
        raise RuntimeError('Selective quantization created no INT8 MatMul operators')
    old = onnx.load(str(source), load_external_data=False)
    # Require the new graph to have fewer FP32 MatMul ops, and verify that some
    # are still present. Otherwise selection may have silently quantized all.
    before = sum(n.op_type == 'MatMul' for n in old.graph.node)
    after = sum(n.op_type == 'MatMul' for n in new.graph.node)
    if not 0 < after < before:
        raise RuntimeError('Graph does not prove selective INT8/FP32 MatMul mix')
    return {'requested_matmul_nodes': len(chosen), 'total_matmul_nodes': len(all_names),
            'per_channel': per_channel, 'matmul_remaining_fp32': after,
            'int8_operator_types': sorted(ops.intersection({'MatMulInteger','QLinearMatMul'}))}


def run_quality_search(m4_package: str | Path, package_sha: str, m5_report: str | Path,
                       m3_report: str | Path, m3_dataset: str | Path, dev_dataset: str | Path,
                       *, root: str | Path = '.', objective: str = 'size', rounds: int = 3,
                       profiles: tuple[str, ...] = DEFAULT_PROFILES) -> tuple[Path, dict]:
    if objective not in ('size', 'balanced') or not 2 <= rounds <= 5:
        raise ValueError('Invalid optimization objective or rounds')
    profiles = normalize_profiles(profiles)
    project = Path(root).expanduser().resolve(strict=True)
    m3, m2, original_graph, _, original_m3_dataset_sha = check_review(m3_report, m3_dataset, root=project)
    m3_path = Path(m3_report).expanduser().resolve(strict=True)
    rows, dev_sha, provenance = load_heldout(dev_dataset)
    if dev_sha in (original_m3_dataset_sha, m2['dataset_sha256']):
        raise ValueError('Development dataset duplicates prior M2/M3 dataset')
    if provenance.get('source') == 'cardiffnlp/tweet_eval' and provenance.get('split') != 'validation':
        raise ValueError('Do not select quantization candidates on TweetEval test split')
    source_sha = m2['artifact']['sha256']
    original_attempt = verify_m5_failure(m5_report, project=project,
         expected_m4_sha=package_sha, expected_dev_sha=dev_sha,
         expected_m3_sha=digest(m3_path), expected_graph_sha=source_sha)

    results_root = project / 'results' / 'auto_optimizer_quality_int8_v1'
    results_root.mkdir(parents=True, exist_ok=True)
    run_dir = results_root / ('search_' + uuid.uuid4().hex[:16])
    run_dir.mkdir(parents=False, exist_ok=False)
    records = []
    try:
        with tempfile.TemporaryDirectory(prefix='source_', dir=run_dir) as temp:
            pkg, manifest = verify_and_extract_m4(Path(m4_package).expanduser().resolve(strict=True),
                Path(temp)/'extracted', expected_sha256=package_sha,
                m3=m3, m3_report=m3_path, fp32_graph_sha256=source_sha)
            graph = pkg / 'model.onnx'
            import onnx
            source_model = onnx.load(str(graph), load_external_data=False)
            names = select_matmul_nodes(source_model)
            for profile in profiles:
                target = run_dir / ('model.' + profile + '.int8.onnx')
                try:
                    details = quantize_profile(graph, target, all_names=names, profile=profile)
                    base, candidate, decision = paired_evaluation(
                         graph, target, pkg, rows, manifest=manifest,
                         objective=objective, rounds=rounds)
                    # Same unchanged M5 gate is applied by paired_evaluation.
                    expected_decision = validate_int8_gate(base, candidate,
                        old_bytes=graph.stat().st_size, new_bytes=target.stat().st_size,
                        sample_count=len(rows), agreement=decision['prediction_agreement'],
                        mismatches=decision['repeatability_mismatches'], objective=objective)
                    if expected_decision != decision:
                        raise RuntimeError('Original M5 acceptance gate mismatch')
                    record = {'profile':profile,'status':'BENCHMARKED',
                              'fp32_graph_bytes':graph.stat().st_size,
                              'int8_graph_bytes':target.stat().st_size,
                              'int8_graph_sha256':digest(target),
                              'settings':details,'fp32_metrics':base,
                              'int8_metrics':candidate,'decision':decision}
                except (ValueError, RuntimeError, OSError, ImportError) as exc:
                    record = {'profile':profile,'status':'FAILED',
                              'error_type':type(exc).__name__, 'error':str(exc)[:500]}
                records.append(record)
                # Until final ranking only preserve summary evidence, not rejected models.
                if target.exists() and (record['status'] != 'BENCHMARKED' or
                                        not record['decision']['dev_gates_passed']):
                    target.unlink()
            winner = choose_winner(records, objective)
            for record in records:
                if record['status'] == 'BENCHMARKED' and (winner is None or record['profile'] != winner['profile']):
                    target = run_dir / ('model.' + record['profile'] + '.int8.onnx')
                    if target.exists(): target.unlink()
            selection = 'INT8_DEV_PASS_NEEDS_NEW_HELDOUT_AND_RELOAD' if winner else 'ALL_SELECTIVE_INT8_CANDIDATES_REJECTED'
            result = {'schema_version':'nimbo_quality_int8_search_v1',
                      'status':'DEVELOPMENT_ONLY_NO_EXPORT',
                      'decision':selection, 'objective':objective,
                      'source_m5_report_sha256':digest(Path(m5_report).expanduser().resolve(strict=True)),
                      'source_m5_decision':original_attempt['decision']['decision'],
                      'source_m4_archive_sha256':package_sha,
                      'source_fp32_graph_sha256':source_sha,
                      'source_m3_report_sha256':digest(m3_path),
                      'dev_dataset_sha256':dev_sha,'dev_dataset_provenance':provenance,
                      'unique_dev_examples':len(rows),'rounds':rounds,
                      'quantized_profiles_tested':list(profiles),'trials':records,
                      'selected':None if winner is None else {
                          'profile':winner['profile'], 'graph_sha256':winner['int8_graph_sha256'],
                          'graph_bytes':winner['int8_graph_bytes'],
                          'path':str(run_dir / ('model.' + winner['profile'] + '.int8.onnx')),
                          'decision':winner['decision']['decision']},
                      'eligible_for_downloadable_package':False,'eligible_for_deployment':False,
                      'limits':['Selection only on the development validation split; NOT a fresh holdout.',
                                'Node list selects first/last MatMul in ONNX topological order, not verified semantic transformer layers.',
                                'Per-channel and selective profiles may not run on every ONNX Runtime version/CPU.',
                                'Graph size only; not full package, RAM, electricity, FLOPs or billing.',
                                'Requires new independent validation and offline package verification before export.']}
            report = run_dir / 'report.json'
            report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            return report, result
    except Exception:
        # preserve prior M1-M5 results; incomplete M5.1 run is not a success.
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
