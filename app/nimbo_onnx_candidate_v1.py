"""NIMBO M2: real experimental ONNX FP32 candidate for a verified HF classifier.

A run can create a test-only ONNX graph and a measured report. It never
approves/deploys/exports a runnable product package; M3 handles validation.
Only immutable, safetensors-backed built-in transformer models from M1 intake.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import statistics
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from app.nimbo_model_intake_v1 import (
    inspect_public_classifier, percentile, read_evaluation_file, resolve_labels,
)

SUPPORTED_INPUTS = ('input_ids', 'attention_mask', 'token_type_ids')


def select_inputs(tokenized: dict, accepted_model_inputs: set[str]) -> tuple[str, ...]:
    """Select only tensor feeds consumed by the *original* model forward."""
    names = tuple(name for name in SUPPORTED_INPUTS if name in tokenized and name in accepted_model_inputs)
    if 'input_ids' not in names:
        raise ValueError('Model/tokenizer does not provide input_ids in an accepted forward signature')
    if any(name not in SUPPORTED_INPUTS for name in tokenized):
        raise ValueError('Tokenizer requires unsupported input fields; refusing to silently discard them')
    return names


def validate_reference_report(previous: dict, *, model_id: str, sha: str, dataset_sha: str,
                              cpu_threads: int, max_length: int) -> None:
    """Forbid comparing against an unrelated or differently configured M1 run."""
    if previous.get('schema_version') != 'nimbo_baseline_v1' or previous.get('status') != 'BASELINE_ONLY_NOT_OPTIMIZED':
        raise ValueError('Reference is not an M1 measured baseline report')
    source = previous.get('model') or {}
    if (source.get('model_id') != model_id or source.get('resolved_revision') != sha or
            previous.get('dataset_sha256') != dataset_sha):
        raise ValueError('Existing baseline refers to another model revision or dataset')
    if previous.get('cpu_threads') != cpu_threads or previous.get('max_length') != max_length:
        raise ValueError('Baseline CPU threads or max_length differ; use matching options')
    if not isinstance((previous.get('metrics') or {}).get('mean_warm_forward_ms'), (float, int)):
        raise ValueError('Reference baseline has no measured forward latency')


def summarize_measurements(wall: list[float], cpu: list[float], labels: list[int | None],
                           predictions: list[int]) -> dict:
    n = len(wall)
    if not n or len(cpu) != n or len(labels) != n or len(predictions) != n:
        raise ValueError('Missing paired measurements')
    if any(not math.isfinite(v) or v < 0 for v in wall + cpu):
        raise ValueError('Invalid benchmark timing')
    fully_labeled = all(y is not None for y in labels)
    return {
        'n_forward_calls': n,
        'accuracy': round(sum(p == y for p, y in zip(predictions, labels)) / n, 6) if fully_labeled else None,
        'mean_warm_forward_ms': round(statistics.mean(wall), 4),
        'p95_warm_forward_ms': round(percentile(wall, .95), 4),
        'process_cpu_forward_total_ms': round(sum(cpu), 4),
    }


def decide_candidate(original: dict, candidate: dict, agreement: float, *, samples: int,
                     max_abs_logit_error: float, min_speedup: float = .05,
                     min_agreement: float = 1.0, max_accuracy_drop: float = 0.0,
                     max_cpu_increase: float = 0.0, max_p95_increase: float = 0.0) -> dict:
    """Evidence gate, not permission for release. Small samples are always exploratory."""
    if not 2 <= samples <= 100 or not 0 <= agreement <= 1 or not math.isfinite(max_abs_logit_error):
        raise ValueError('Invalid comparison data')
    if not (0 <= min_speedup <= 1 and 0 <= min_agreement <= 1 and
            0 <= max_accuracy_drop <= 1 and 0 <= max_cpu_increase <= 1 and 0 <= max_p95_increase <= 1):
        raise ValueError('Invalid acceptance limits')
    for name in ('mean_warm_forward_ms', 'p95_warm_forward_ms', 'process_cpu_forward_total_ms'):
        if original[name] <= 0 or candidate[name] < 0:
            raise ValueError(f'Invalid measured {name}')
    speedup = 1 - candidate['mean_warm_forward_ms'] / original['mean_warm_forward_ms']
    cpu_gain = 1 - candidate['process_cpu_forward_total_ms'] / original['process_cpu_forward_total_ms']
    p95_regression = candidate['p95_warm_forward_ms'] / original['p95_warm_forward_ms'] - 1
    label_quality = original.get('accuracy') is not None and candidate.get('accuracy') is not None
    accuracy_drop = original['accuracy'] - candidate['accuracy'] if label_quality else None
    quality_pass = (label_quality and accuracy_drop <= max_accuracy_drop + 1e-9
                    and agreement + 1e-9 >= min_agreement and max_abs_logit_error <= .01)
    performance_pass = (speedup + 1e-9 >= min_speedup and cpu_gain + 1e-9 >= -max_cpu_increase
                        and p95_regression <= max_p95_increase + 1e-9)
    exploratory_only = samples < 50
    pass_local_gates = bool(quality_pass and performance_pass)
    return {
        'local_gates_passed': pass_local_gates,
        'quality_pass': bool(quality_pass), 'performance_pass': bool(performance_pass),
        'small_sample_exploratory': exploratory_only,
        'mean_latency_reduction_fraction': round(speedup, 6),
        'process_cpu_time_reduction_fraction': round(cpu_gain, 6),
        'p95_regression_fraction': round(p95_regression, 6),
        'accuracy_drop': round(accuracy_drop, 6) if accuracy_drop is not None else None,
        'prediction_agreement': round(agreement, 6),
        'max_abs_logit_error': round(max_abs_logit_error, 8),
        'decision': ('REJECTED_LOCAL_GATES' if not pass_local_gates else
                     'EXPLORATORY_PASS_NOT_APPROVED' if exploratory_only else
                     'LOCAL_PASS_REQUIRES_HELDOUT_AND_RELOAD'),
        'eligible_for_deployment': False,
        'eligible_for_downloadable_package': False,
        'criteria': {'min_mean_latency_reduction': min_speedup, 'min_prediction_agreement': min_agreement,
                     'max_accuracy_drop': max_accuracy_drop, 'max_process_cpu_increase': max_cpu_increase,
                     'max_p95_increase': max_p95_increase, 'max_abs_logit_error': .01},
    }


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _load_onnx_session(model, tokenizer, tokenized: dict, export_path: Path, *, threads: int):
    """Export same weights (not quantization). Return a validated CPU ORT session.

    Uses legacy ONNX exporter explicitly for predictable dynamic_axes behavior
    across the existing local PyTorch and Optimum installations.
    """
    import torch
    import onnxruntime as ort
    model_inputs = set(inspect.signature(model.forward).parameters)
    names = select_inputs(tokenized, model_inputs)
    args = tuple(tokenized[name] for name in names)

    class LogitsOnly(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, *inputs):
            named = {name: value for name, value in zip(names, inputs)}
            return self.base(**named).logits

    axes = {name: {0: 'batch', 1: 'sequence'} for name in names}
    axes['logits'] = {0: 'batch'}
    export_path.parent.mkdir(parents=True, exist_ok=False)
    torch.onnx.export(LogitsOnly(model).eval(), args, str(export_path),
                      export_params=True, opset_version=17, do_constant_folding=True,
                      input_names=list(names), output_names=['logits'], dynamic_axes=axes,
                      dynamo=False)
    if not export_path.is_file() or export_path.stat().st_size < 1024:
        raise RuntimeError('Export did not produce a valid ONNX graph file')
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(export_path), sess_options=opts, providers=['CPUExecutionProvider'])
    if session.get_providers() != ['CPUExecutionProvider']:
        raise RuntimeError('ONNX must run using CPUExecutionProvider only')
    ort_names = tuple(x.name for x in session.get_inputs())
    if 'input_ids' not in ort_names or any(x not in names for x in ort_names):
        raise RuntimeError('ONNX input signature does not match source model')
    if len(session.get_outputs()) != 1:
        raise RuntimeError('ONNX output signature must contain a single logits tensor')
    return session, ort_names


def run_paired_candidate(model_id: str, dataset: str | Path, *, revision: str | None = None,
                         baseline_report: str | Path | None = None, threads: int = 2,
                         max_length: int = 256, rounds: int = 3,
                         output_root: str | Path = 'results/auto_optimizer_candidates_v1') -> tuple[Path, dict]:
    if not 1 <= threads <= 8 or not 32 <= max_length <= 256 or not 2 <= rounds <= 5:
        raise ValueError('Threads 1–8, max_length 32–256, rounds 2–5 required')
    rows, dataset_sha = read_evaluation_file(dataset)
    spec = inspect_public_classifier(model_id, revision=revision)
    expected = resolve_labels(rows, spec.id2label)
    prior = None
    if baseline_report is not None:
        reference_path = Path(baseline_report).expanduser().resolve(strict=True)
        prior = json.loads(reference_path.read_text(encoding='utf8'))
        validate_reference_report(prior, model_id=spec.model_id, sha=spec.resolved_revision,
                                  dataset_sha=dataset_sha, cpu_threads=threads, max_length=max_length)

    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.resolved_revision,
                                               trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.model_id, revision=spec.resolved_revision, trust_remote_code=False,
        use_safetensors=True, low_cpu_mem_usage=False).eval().cpu()

    toks = [tokenizer(row['text'], max_length=max_length, truncation=True, return_tensors='pt')
            for row in rows]
    export_input = toks[0]
    run_id = 'candidate_' + uuid.uuid4().hex[:16]
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifact_dir = root / run_id
    export_path = artifact_dir / 'model.onnx'
    # No graph artifacts are placed into the Git worktree's tracked paths.
    session, names = _load_onnx_session(model, tokenizer, export_input, export_path, threads=threads)
    prepared = []
    model_forward_inputs = set(inspect.signature(model.forward).parameters)
    for tokenized, y in zip(toks, expected):
        names_for_pt = select_inputs(tokenized, model_forward_inputs)
        pt_inputs = {name: tokenized[name] for name in names_for_pt}
        ort_inputs = {name: tokenized[name].detach().cpu().numpy().astype(np.int64)
                      for name in names}
        prepared.append((pt_inputs, ort_inputs, y))

    with torch.inference_mode():
        for pt, ort, _ in prepared[:min(3, len(prepared))]:
            model(**pt)
            session.run(None, ort)

    walls = {'baseline': [], 'candidate': []}
    cpus = {'baseline': [], 'candidate': []}
    predictions = {'baseline': [], 'candidate': []}
    labels = []
    matches = 0
    mismatched_across_rounds = 0
    max_logit_error = 0.0
    first_predictions = {}
    for r in range(rounds):
        for i, (pt_inputs, ort_inputs, y) in enumerate(prepared):
            outcomes = {}
            order = ('baseline','candidate') if (r + i) % 2 == 0 else ('candidate','baseline')
            for which in order:
                t0 = time.perf_counter_ns()
                c0 = time.process_time_ns()
                if which == 'baseline':
                    with torch.inference_mode():
                        value = model(**pt_inputs).logits.detach().cpu().numpy()
                else:
                    value = np.asarray(session.run(None, ort_inputs)[0])
                cpu_ms = (time.process_time_ns() - c0) / 1e6
                wall_ms = (time.perf_counter_ns() - t0) / 1e6
                if value.ndim != 2 or value.shape != (1, spec.num_labels) or not np.isfinite(value).all():
                    raise RuntimeError(f'Unexpected {which} output shape or non-finite logits')
                walls[which].append(wall_ms)
                cpus[which].append(cpu_ms)
                pred = int(value.argmax(axis=-1)[0])
                predictions[which].append(pred)
                outcomes[which] = (pred, value)
            labels.append(y)
            matches += int(outcomes['baseline'][0] == outcomes['candidate'][0])
            err = float(np.max(np.abs(outcomes['baseline'][1] - outcomes['candidate'][1])))
            max_logit_error = max(max_logit_error, err)
            now = (outcomes['baseline'][0], outcomes['candidate'][0])
            if i in first_predictions and first_predictions[i] != now:
                mismatched_across_rounds += 1
            first_predictions[i] = now

    base = summarize_measurements(walls['baseline'], cpus['baseline'], labels, predictions['baseline'])
    candidate = summarize_measurements(walls['candidate'], cpus['candidate'], labels, predictions['candidate'])
    comparison = decide_candidate(base, candidate, matches / (rounds * len(rows)),
                                   samples=len(rows), max_abs_logit_error=max_logit_error)
    if mismatched_across_rounds:
        comparison.update({'local_gates_passed': False, 'quality_pass': False,
                           'decision': 'REJECTED_NONDETERMINISTIC_PREDICTIONS'})
    comparison['repeatability_mismatches'] = mismatched_across_rounds
    report = {
        'schema_version': 'nimbo_onnx_candidate_v1',
        'status': 'EXPERIMENT_ONLY_NOT_EXPORTED',
        'model': asdict(spec), 'dataset_sha256': dataset_sha,
        'model_revision': spec.resolved_revision,
        'source_baseline_report': str(Path(baseline_report).expanduser().resolve()) if prior else None,
        'source_baseline_metrics': prior['metrics'] if prior else None,
        'reference_comparison_note': 'M1 baseline run is provenance only; compare new paired measurements below, not separate runs.',
        'cpu_threads_each_path': threads, 'max_length': max_length,
        'rounds': rounds, 'unique_dataset_rows': len(rows),
        'forward_calls_each_path': rounds * len(rows),
        'baseline': base, 'onnx_candidate': candidate, 'decision': comparison,
        'artifact': {'path': str(export_path), 'sha256': digest(export_path),
                     'bytes': export_path.stat().st_size, 'format': 'ONNX FP32',
                     'status': 'TEST_ONLY_GRAPH_NOT_A_RUNNABLE_PACKAGE'},
        'limits': ['Not a user-downloadable runnable package; source tokenizer/config not bundled.',
                   'No independent heldout validation or fresh reload completed.',
                   'Small IMDb sample is exploratory, no resource saving claims beyond this run.',
                   'CPU forward timings exclude tokenizer, setup, model export, downloads, energy, dollars and memory.',
                   'No production deployment or automatic adoption.'],
    }
    report_path = artifact_dir / 'report.json'
    with report_path.open('x', encoding='utf8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    return report_path, report
