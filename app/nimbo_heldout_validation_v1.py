"""M3: re-open the exact M2 ONNX candidate and validate on disjoint labeled data.

Local validation; no production approval, export package or deployment.
Do not claim training-independent holdout: public model may already have used
TweetEval for training or architecture selection. Only independent of M2's IMDb
12-row development evaluation if dataset fingerprints differ.
"""
from __future__ import annotations

import csv
import hashlib
import io
import inspect
import json
import math
import statistics
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from app.nimbo_model_intake_v1 import inspect_public_classifier, resolve_labels, percentile
from app.nimbo_onnx_candidate_v1 import digest, select_inputs, summarize_measurements, decide_candidate

MIN_ROWS = 200
MAX_ROWS = 800
MAX_DATASET_BYTES = 2 * 1024 * 1024
MAX_GRAPH_BYTES = 750 * 1024 * 1024


def load_heldout(path: str | Path) -> tuple[list[dict], str, dict]:
    """Read and fingerprint a 200-800 row labeled CSV/JSONL (no guessed labels)."""
    path = Path(path).expanduser().resolve(strict=True)
    if path.suffix.lower() not in ('.csv', '.jsonl'):
        raise ValueError('Heldout evaluation file must be .csv or .jsonl')
    size = path.stat().st_size
    if not 0 < size <= MAX_DATASET_BYTES:
        raise ValueError('Heldout file must be nonempty and <=2 MiB')
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode('utf-8-sig')
    if path.suffix.lower() == '.csv':
        csv_reader = csv.DictReader(io.StringIO(text))
        fields = csv_reader.fieldnames
        if not fields or 'text' not in fields or 'label' not in fields or len(set(fields)) != len(fields):
            raise ValueError('CSV must have unique text,label columns')
        rows = list(csv_reader)
    else:
        rows = []
        for number,line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSONL row {number}') from exc
            if not isinstance(parsed, dict):
                raise ValueError(f'JSONL row {number} is not an object')
            rows.append(parsed)
    if not MIN_ROWS <= len(rows) <= MAX_ROWS:
        raise ValueError(f'Heldout needs {MIN_ROWS}-{MAX_ROWS} labeled examples')
    cleaned = []
    for i, row in enumerate(rows, 1):
        message = row.get('text')
        label = row.get('label')
        if not isinstance(message, str) or not 1 <= len(message.strip()) <= 2000:
            raise ValueError(f'Row {i} has invalid text')
        if not isinstance(label, (int, str)) or isinstance(label, bool) or not str(label).strip():
            raise ValueError(f'Row {i} missing reference label')
        cleaned.append({'text': message.strip(), 'label': str(label).strip()})
    sidecar = path.with_suffix('.source.json')
    provenance = {'source': 'user_supplied_not_independently_audited', 'dataset_sha256': sha}
    if sidecar.is_file():
        metadata = json.loads(sidecar.read_text(encoding='utf-8'))
        if not isinstance(metadata, dict) or metadata.get('csv_sha256') != sha or metadata.get('rows') != len(rows):
            raise ValueError('Dataset provenance sidecar does not match file checksum or row count')
        provenance = {key: metadata[key] for key in ('source','split','revision','seed','rows','csv_sha256') if key in metadata}
        provenance['provenance_status'] = 'metadata_claim_not_independent_audit'
    return cleaned, sha, provenance



def pilot_gate_disposition(m2: dict) -> str | None:
    """Let *small* quality-faithful pilots continue to the STRICT heldout gate.

    This does not make a rejected pilot pass: M2's original decision and
    measurements stay unchanged. Only a preliminary CPU/P95 timing regression
    can be deferred. The ≥5% mean latency requirement and ALL quality checks
    must already pass; M3 independently rechecks CPU/P95 and quality on 200+ rows.
    """
    if not isinstance(m2, dict):
        return None
    d = m2.get('decision') or {}
    if d.get('local_gates_passed') is True and d.get('decision') in (
            'EXPLORATORY_PASS_NOT_APPROVED', 'LOCAL_PASS_REQUIRES_HELDOUT_AND_RELOAD'):
        return 'PILOT_LOCAL_PASS'
    if (d.get('local_gates_passed') is not False or
            d.get('decision') != 'REJECTED_LOCAL_GATES' or
            d.get('quality_pass') is not True or d.get('performance_pass') is not False or
            d.get('small_sample_exploratory') is not True or
            d.get('repeatability_mismatches') != 0 or
            d.get('prediction_agreement') != 1.0 or
            d.get('accuracy_drop') != 0.0 or
            type(m2.get('unique_dataset_rows')) is not int or
            not 2 <= m2['unique_dataset_rows'] < 50):
        return None
    try:
        old = m2['baseline']
        new = m2['onnx_candidate']
        rows = m2['unique_dataset_rows']
        rounds = m2['rounds']
        if type(rounds) is not int or not 2 <= rounds <= 5:
            return None
        if (m2.get('forward_calls_each_path') != rows * rounds or
                old.get('n_forward_calls') != rows * rounds or
                new.get('n_forward_calls') != rows * rounds):
            return None
        expected = decide_candidate(old, new, d['prediction_agreement'],
                                    samples=rows, max_abs_logit_error=d['max_abs_logit_error'])
        # The original gate policy must be unchanged and the reported failure
        # must be explained solely by CPU or p95 (not insufficient speedup).
        if d.get('criteria') != expected['criteria']:
            return None
        for field in ('decision', 'local_gates_passed', 'quality_pass',
                      'performance_pass', 'small_sample_exploratory',
                      'mean_latency_reduction_fraction', 'process_cpu_time_reduction_fraction',
                      'p95_regression_fraction', 'accuracy_drop',
                      'prediction_agreement', 'max_abs_logit_error'):
            if d.get(field) != expected[field]:
                return None
        if (not expected['quality_pass'] or expected['performance_pass'] or
                expected['mean_latency_reduction_fraction'] < .05 or
                old['accuracy'] != new['accuracy']):
            return None
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return 'PILOT_CPU_P95_TIMING_DEFERRED_TO_HELDOUT'


def read_m2_candidate(report_path: str | Path, *, root: Path, model_id: str, sha: str,
                      heldout_sha: str, threads: int, max_length: int) -> tuple[dict, Path]:
    """Require an M2 report and matching hash-verified graph under its run folder."""
    report_file = Path(report_path).expanduser().resolve(strict=True)
    expected_root = (root / 'results' / 'auto_optimizer_candidates_v1').resolve()
    if expected_root not in report_file.parents or report_file.name != 'report.json':
        raise ValueError('M2 report must be in the original project candidate run folder')
    report = json.loads(report_file.read_text(encoding='utf-8'))
    if report.get('schema_version') != 'nimbo_onnx_candidate_v1' or report.get('status') != 'EXPERIMENT_ONLY_NOT_EXPORTED':
        raise ValueError('Unsupported or incomplete M2 candidate report')
    src = report.get('model') or {}
    if src.get('model_id') != model_id or src.get('resolved_revision') != sha:
        raise ValueError('M2 model ID/revision does not match requested model')
    if report.get('dataset_sha256') == heldout_sha:
        raise ValueError('Heldout data must differ from the M2 development dataset')
    if report.get('cpu_threads_each_path') != threads or report.get('max_length') != max_length:
        raise ValueError('CPU threads / max_length differ from M2; preserve configuration')
    decision = report.get('decision') or {}
    if pilot_gate_disposition(report) is None:
        raise ValueError('M2 pilot gates failed quality/speed checks, or its performance deferral is invalid')
    artifact = report.get('artifact') or {}
    raw_graph = report_file.parent / 'model.onnx'
    if raw_graph.is_symlink():
        raise ValueError('Symlinked ONNX graphs are not accepted')
    graph = raw_graph.resolve(strict=True)
    if artifact.get('path') != str(graph) or graph.parent != report_file.parent:
        raise ValueError('M2 artifact path mismatch')
    if not 1024 <= graph.stat().st_size <= MAX_GRAPH_BYTES or graph.stat().st_size != artifact.get('bytes'):
        raise ValueError('M2 graph size mismatch')
    if digest(graph) != artifact.get('sha256'):
        raise ValueError('M2 graph checksum mismatch; refuse execution')
    return report, graph


def preprocess_text(text: str, mode: str = 'identity') -> str:
    if mode == 'identity':
        return text
    if mode != 'cardiff_tweet':
        raise ValueError('Unknown preprocessing mode')
    return ' '.join('@user' if token.startswith('@') and len(token) > 1 else
                    'http' if token.startswith('http') else token for token in text.split(' '))


def decide_heldout(base: dict, candidate: dict, *, sample_count: int, agreement: float,
                    max_logit_error: float, repeats: int) -> dict:
    if not MIN_ROWS <= sample_count <= MAX_ROWS or repeats < 0 or not 0 <= agreement <= 1 or not math.isfinite(max_logit_error):
        raise ValueError('Invalid heldout evidence')
    if base.get('accuracy') is None or candidate.get('accuracy') is None:
        raise ValueError('Quality reference labels are mandatory')
    for name in ('mean_warm_forward_ms', 'p95_warm_forward_ms', 'process_cpu_forward_total_ms'):
        if base[name] <= 0 or candidate[name] < 0:
            raise ValueError('Invalid performance metrics')
    gain = 1 - candidate['mean_warm_forward_ms'] / base['mean_warm_forward_ms']
    cpu_gain = 1 - candidate['process_cpu_forward_total_ms'] / base['process_cpu_forward_total_ms']
    p95_delta = candidate['p95_warm_forward_ms'] / base['p95_warm_forward_ms'] - 1
    delta = base['accuracy'] - candidate['accuracy']
    quality = bool(delta <= 1e-9 and agreement >= 1 - 1e-9 and max_logit_error <= .01 and repeats == 0)
    performance = bool(gain >= .05 - 1e-9 and cpu_gain >= -1e-9 and p95_delta <= 1e-9)
    ok = quality and performance
    return {
        'heldout_gates_passed': ok,
        'quality_pass': quality,
        'performance_pass': performance,
        'decision': 'HELDOUT_LOCAL_PASS_REQUIRES_PACKAGE_RELOAD' if ok else 'REJECTED_HELDOUT_GATES',
        'mean_latency_reduction_fraction': round(gain, 6),
        'process_cpu_time_reduction_fraction': round(cpu_gain, 6),
        'p95_regression_fraction': round(p95_delta, 6),
        'accuracy_drop': round(delta, 6),
        'prediction_agreement': round(agreement, 6),
        'max_abs_logit_error': round(max_logit_error, 8),
        'repeatability_mismatches': repeats,
        'criteria': {'minimum_unique_examples': MIN_ROWS, 'minimum_mean_speedup': .05,
                     'maximum_accuracy_drop': 0.0, 'minimum_prediction_agreement': 1.0,
                     'maximum_cpu_increase': 0.0, 'maximum_p95_increase': 0.0,
                     'maximum_absolute_logit_error': .01, 'repeatability_mismatches': 0},
        'eligible_for_deployment': False,
        'eligible_for_downloadable_package': False,
    }


def validate_existing_candidate(m2_report: str | Path, heldout_path: str | Path, *, root: str | Path='.',
                                rounds: int = 3, preprocess: str = 'identity',
                                output_root: str | Path = 'results/auto_optimizer_heldout_v1') -> tuple[Path,dict]:
    """Execute a *new* process-local ONNX Runtime reload, paired with pinned PyTorch."""
    if not 2 <= rounds <= 5 or preprocess not in ('identity','cardiff_tweet'):
        raise ValueError('Rounds 2-5 and supported preprocessing required')
    project = Path(root).expanduser().resolve(strict=True)
    rows, sha, provenance = load_heldout(heldout_path)
    # Read provenance before executing an untrusted graph.
    report_path = Path(m2_report).expanduser().resolve(strict=True)
    candidate_report = json.loads(report_path.read_text(encoding='utf-8'))
    old_model = candidate_report.get('model') or {}
    model_id, revision = old_model.get('model_id'), old_model.get('resolved_revision')
    if not isinstance(model_id,str) or not isinstance(revision,str):
        raise ValueError('Missing M2 model identity')
    threads = candidate_report.get('cpu_threads_each_path')
    max_length = candidate_report.get('max_length')
    if type(threads) is not int or not 1 <= threads <= 8 or type(max_length) is not int or not 32 <= max_length <= 256:
        raise ValueError('Invalid M2 CPU/thread policy')
    spec = inspect_public_classifier(model_id,revision=revision)
    m2, graph = read_m2_candidate(report_path, root=project, model_id=spec.model_id,
                                   sha=spec.resolved_revision, heldout_sha=sha, threads=threads,max_length=max_length)
    expected = resolve_labels(rows, spec.id2label)
    if any(x is None for x in expected):
        raise ValueError('All heldout rows must be labeled')
    import numpy as np
    import torch
    import onnxruntime as ort
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    baseline = AutoModelForSequenceClassification.from_pretrained(
        model_id,revision=revision,trust_remote_code=False,use_safetensors=True,low_cpu_mem_usage=False).eval().cpu()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(graph),sess_options=opts,providers=['CPUExecutionProvider'])
    if session.get_providers()!=['CPUExecutionProvider']:
        raise RuntimeError('Candidate failed to initialize CPU-only ONNX Runtime')
    names = tuple(x.name for x in session.get_inputs())
    if 'input_ids' not in names or any(x not in ('input_ids','attention_mask','token_type_ids') for x in names):
        raise RuntimeError('Unrecognized ONNX graph input fields')
    if len(session.get_outputs()) != 1:
        raise RuntimeError('Expected exactly one ONNX output')
    accepted = set(inspect.signature(baseline.forward).parameters)
    prepared = []
    for row,label in zip(rows,expected):
        tokens = tokenizer(preprocess_text(row['text'],preprocess),max_length=max_length,
                           truncation=True,return_tensors='pt')
        pt_names = select_inputs(tokens, accepted)
        if any(key not in pt_names for key in names):
            raise ValueError('ONNX model needs an input not accepted by PyTorch')
        pt = {key:tokens[key] for key in pt_names}
        ort_feed = {key:tokens[key].detach().cpu().numpy().astype(np.int64) for key in names}
        prepared.append((pt,ort_feed,label))
    with torch.inference_mode():
        for pt,ort_feed,_ in prepared[:3]:
            baseline(**pt)
            session.run(None,ort_feed)
    wall = {'baseline':[],'candidate':[]}
    cpu = {'baseline':[],'candidate':[]}
    preds = {'baseline':[],'candidate':[]}
    repeated = {}
    mismatch_count = 0
    agrees = 0
    max_error = 0.0
    for r in range(rounds):
        for i,(pt,ort_feed,y) in enumerate(prepared):
            observations = {}
            for name in (('baseline','candidate') if (r+i)%2==0 else ('candidate','baseline')):
                start_wall = time.perf_counter_ns()
                start_cpu = time.process_time_ns()
                if name == 'baseline':
                    with torch.inference_mode():
                        logits = baseline(**pt).logits.detach().cpu().numpy()
                else:
                    logits = np.asarray(session.run(None,ort_feed)[0])
                cpu[name].append((time.process_time_ns()-start_cpu)/1e6)
                wall[name].append((time.perf_counter_ns()-start_wall)/1e6)
                if logits.shape != (1,spec.num_labels) or not np.isfinite(logits).all():
                    raise RuntimeError(f'Invalid logits for {name}')
                predicted = int(logits.argmax(axis=-1)[0])
                preds[name].append(predicted)
                observations[name] = (predicted,logits)
            agrees += observations['baseline'][0]==observations['candidate'][0]
            max_error = max(max_error,float(np.max(np.abs(observations['baseline'][1]-observations['candidate'][1]))))
            pair=(observations['baseline'][0], observations['candidate'][0])
            if i in repeated and pair!=repeated[i]:
                mismatch_count+=1
            repeated[i]=pair
    labels = expected * rounds
    base_metrics = summarize_measurements(wall['baseline'],cpu['baseline'],labels,preds['baseline'])
    candidate_metrics = summarize_measurements(wall['candidate'],cpu['candidate'],labels,preds['candidate'])
    decision = decide_heldout(base_metrics,candidate_metrics,sample_count=len(rows),
                              agreement=agrees/(rounds*len(rows)),max_logit_error=max_error,
                              repeats=mismatch_count)
    result = {
        'schema_version':'nimbo_heldout_validation_v1',
        'status':'HELDOUT_VALIDATION_COMPLETED_REVIEW_ONLY',
        'model':asdict(spec),
        'source_candidate_report':str(report_path),
        'source_graph_sha256':m2['artifact']['sha256'],
        'source_m2_dataset_sha256':m2['dataset_sha256'],
        'pilot_gate_disposition':pilot_gate_disposition(m2),
        'heldout_dataset_sha256':sha,
        'heldout_dataset_provenance':provenance,
        'heldout_unique_examples':len(rows),
        'rounds':rounds,'forward_calls_per_path':rounds*len(rows),
        'cpu_threads_each_path':threads,'max_length':max_length,
        'preprocessing':preprocess,
        'baseline':base_metrics,'candidate':candidate_metrics,
        'decision':decision,
        'limitations': ['Disjoint from M2 IMDb development data by checksum; not proof of separation from source model pretraining/fine-tuning.',
                        'One machine CPU local evaluation, warm forward only (excludes tokenizer/setup).',
                        'No production deployment approval; no downloadable runnable model package.',
                        'No FLOPs, cloud billing, electrical power or energy measurement.',
                        'M2 artifact reloaded in a new CLI process, but tokenizer/config still sourced from pinned Hugging Face revision.'],
    }
    dst_root = Path(output_root).expanduser().resolve()
    dst_root.mkdir(parents=True,exist_ok=True)
    target = dst_root / ('heldout_'+uuid.uuid4().hex[:16]+'.json')
    with target.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,indent=2,ensure_ascii=False)
        handle.write('\n')
    return target,result
