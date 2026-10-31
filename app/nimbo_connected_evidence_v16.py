"""Unified evidence endpoint on the main Mission API.
Includes paired comparison metadata while keeping untrusted labels separate from audited quality.
"""
from __future__ import annotations
import json
from pathlib import Path
from fastapi import APIRouter

router=APIRouter(tags=['evidence'])

@router.get('/nimbo/evidence/unified')
def unified_evidence(limit:int=100):
    limit=max(1,min(limit,500))
    try:
        from nimbo.unified_evidence import collect
        result=collect(limit=limit)
    except (ImportError, OSError, ValueError):
        result={'schema_version':'1.0','runtime_summary':{'request_count':0,'by_workload':{},'invalid_lines':0,'no_quality_rate_claim':True},'records':[], 'limitations':['Historical evidence collector is unavailable; pair logs are still read below.']}
    path=Path('results/baseline_comparisons_v15.jsonl')
    invalid=0
    if path.exists():
        for index,line in enumerate(path.read_text(encoding='utf-8').splitlines()[-limit:]):
            if not line.strip():continue
            try:
                r=json.loads(line)
                if not isinstance(r,dict):raise ValueError('Expected record')
                result['records'].append({
                    'id':'paired:'+str(index)+':'+str(r.get('timestamp_utc','')),
                    'source':'baseline_comparisons_v15.jsonl',
                    'evidence_kind':'local_paired_cpu_benchmark',
                    'workload':r.get('workload','classification'),
                    'route':'paired_baseline_candidate',
                    'timestamp_utc':r.get('timestamp_utc'),
                    'baseline_model':r.get('baseline_model'),
                    'candidate_model':r.get('candidate_model'),
                    'sample_count':r.get('sample_count'),
                    'baseline_mean_inference_ms':r.get('baseline_mean_inference_ms'),
                    'candidate_mean_inference_ms':r.get('candidate_mean_inference_ms'),
                    'mean_latency_reduction_fraction':r.get('candidate_latency_reduction_fraction'),
                    'baseline_accuracy_supplied_labels':r.get('baseline_accuracy_supplied_labels'),
                    'candidate_accuracy_supplied_labels':r.get('candidate_accuracy_supplied_labels'),
                    'all_labels_present':r.get('quality_labels_supplied') is True,
                    'quality_certified':False,'financial_cost':None,
                    'latency_ms':None,
                    'criterion':'warm CPU inference, not full request latency, FLOPs or energy',
                    'success_label':None,
                })
            except (ValueError,TypeError):invalid+=1
    result['paired_summary']={'count':len([x for x in result['records'] if x.get('evidence_kind')=='local_paired_cpu_benchmark']), 'invalid_lines':invalid}
    result.setdefault('limitations',[]).append('Paired CPU comparison is not a live runtime request count and supplied labels are not independently audited.')
    # NIMBO_V20_SAME_MODEL_EVIDENCE: saved experiment metadata only; no input text.
    from pathlib import Path as _P20
    _v20_file = _P20('results/same_model_trials_v20.jsonl')
    if _v20_file.is_file():
        for _i20, _line20 in enumerate(_v20_file.read_text(encoding='utf8').splitlines()[-limit:]):
            try:
                _r20 = json.loads(_line20)
                _c20 = _r20['comparison']
                result['records'].append({
                    'id': _r20['run_id'], 'source': 'same_model_trials_v20.jsonl',
                    'evidence_kind': 'local_paired_cpu_benchmark',
                    'optimization_kind': 'dynamic_int8_same_model',
                    'experiment_phase': _r20.get('phase'),
                    'workload': 'classification', 'route': 'same_model_dynamic_int8',
                    'timestamp_utc': _r20.get('timestamp_utc'),
                    'baseline_model': _c20.get('baseline_model'),
                    'candidate_model': _c20.get('candidate_model'),
                    'baseline_mean_inference_ms': _c20['baseline']['mean_inference_ms'],
                    'candidate_mean_inference_ms': _c20['candidate']['mean_inference_ms'],
                    'mean_latency_reduction_fraction': _c20.get('mean_latency_reduction_fraction'),
                    'sample_count': _c20.get('sample_count'),
                    'all_labels_present': True, 'quality_certified': False,
                    'baseline_accuracy_supplied_labels': _c20['baseline']['accuracy_on_supplied_labels'],
                    'candidate_accuracy_supplied_labels': _c20['candidate']['accuracy_on_supplied_labels'],
                    'quality_gate': _r20['optimization']['gate'],
                    'baseline_weight_bytes': _c20.get('baseline_state_dict_bytes'),
                    'candidate_weight_bytes': _c20.get('candidate_state_dict_bytes'),
                    'financial_cost': None, 'latency_ms': None,
                    'criterion': 'Warm CPU forward latency; same pretrained model quantized to INT8',
                    'success_label': None
                })
            except (ValueError, TypeError, KeyError):
                continue
    # NIMBO_V20_2_ONNX_TRIAL: same-model ONNX evidence, independent of live requests.
    from pathlib import Path as _ONNX_PATH
    _onnx_log = _ONNX_PATH('results/same_model_onnx_trials_v20.jsonl')
    if _onnx_log.is_file():
        for _line in _onnx_log.read_text(encoding='utf8').splitlines()[-limit:]:
            try:
                _r = json.loads(_line)
                _c = _r['comparison']
                result['records'].append({
                    'id':_r['run_id'], 'source':'same_model_onnx_trials_v20.jsonl',
                    'evidence_kind':'local_paired_cpu_benchmark',
                    'optimization_kind':'onnx_runtime_fp32_same_model',
                    'experiment_phase':_r.get('phase'), 'workload':'classification',
                    'route':'same_model_onnx_fp32','timestamp_utc':_r.get('timestamp_utc'),
                    'baseline_model':_c.get('baseline_model'),
                    'candidate_model':_c.get('candidate_model'),
                    'sample_count':_c.get('sample_count'),
                    'baseline_mean_inference_ms':_c['baseline']['mean_inference_ms'],
                    'candidate_mean_inference_ms':_c['candidate']['mean_inference_ms'],
                    'mean_latency_reduction_fraction':_c.get('mean_latency_reduction_fraction'),
                    'baseline_accuracy_supplied_labels':_c['baseline']['accuracy_on_supplied_labels'],
                    'candidate_accuracy_supplied_labels':_c['candidate']['accuracy_on_supplied_labels'],
                    'baseline_p95_ms':_c['baseline']['p95_ms'],
                    'candidate_p95_ms':_c['candidate']['p95_ms'],
                    'prediction_agreement_fraction':_c.get('prediction_agreement_fraction'),
                    'quality_gate':_r['optimization']['gate'],
                    'quality_certified':False, 'all_labels_present':True,
                    'financial_cost':None, 'latency_ms':None,
                    'criterion':'warm CPU inference on identical token IDs; PYTORCH FP32 vs ONNX FP32',
                    'success_label':None
                })
            except (ValueError,TypeError,KeyError):
                continue
    # NIMBO_V21_INDEPENDENT_VALIDATION: separate from development and live-request counts.
    _v21_log = Path('results/independent_validation_v21.jsonl')
    if _v21_log.is_file():
        for _line in _v21_log.read_text(encoding='utf8').splitlines()[-limit:]:
            try:
                _r=json.loads(_line)
                _c=_r['comparison']
                _d=_r['optimization']['gate']
                result['records'].append({
                    'id':_r['run_id'],'source':'independent_validation_v21.jsonl',
                    'evidence_kind':'heldout_repeated_cpu_validation',
                    'workload':'classification','route':'same_model_onnx_fp32_heldout',
                    'experiment_phase':'heldout', 'timestamp_utc':_r.get('timestamp_utc'),
                    'baseline_model':_c.get('baseline_model'),
                    'candidate_model':_c.get('candidate_model'),
                    'sample_count':_c['sample_count'],'rounds':_c.get('rounds'),
                    'baseline_mean_inference_ms':_c['baseline']['mean_inference_ms'],
                    'candidate_mean_inference_ms':_c['candidate']['mean_inference_ms'],
                    'baseline_p95_ms':_c['baseline']['p95_ms'],
                    'candidate_p95_ms':_c['candidate']['p95_ms'],
                    'local_pipeline_speedup_fraction':_c.get('local_pipeline_speedup_fraction'),
                    'mean_latency_reduction_fraction':_c['mean_latency_reduction_fraction'],
                    'baseline_accuracy_supplied_labels':_c['baseline']['accuracy_on_reference_labels'],
                    'candidate_accuracy_supplied_labels':_c['candidate']['accuracy_on_reference_labels'],
                    'prediction_agreement_fraction':_c.get('prediction_agreement_fraction'),
                    'quality_gate':_d,'decision':_r['optimization']['approval'],
                    'artifact':_r['optimization'].get('export'),
                    'all_labels_present':True,'quality_certified':False,
                    'financial_cost':None,'latency_ms':None,
                    'criterion':'Disjoint IMDb holdout, repeated same-model CPU forward + local tokenizer pipeline',
                    'success_label':None,
                })
            except (ValueError,KeyError,TypeError):
                continue
    result.setdefault('limitations',[]).append('v21 holdout is disjoint from NIMBO development samples, not a third-party audit; accepted variant is not deployed.')
    # NIMBO_V21_1_CONTROLLED_ADOPTION: local canary, separate evidence class.
    _adopt_log = Path('results/adoption_requests_v21_1.jsonl')
    if _adopt_log.is_file():
        for _line in _adopt_log.read_text(encoding='utf8').splitlines()[-limit:]:
            try:
                _r=json.loads(_line)
                result['records'].append({
                    'id':'local_adoption:'+str(_r.get('timestamp_utc')),
                    'source':'adoption_requests_v21_1.jsonl',
                    'evidence_kind':'local_candidate_canary',
                    'workload':'classification','route':_r.get('route_served'),
                    'timestamp_utc':_r.get('timestamp_utc'),
                    'local_mode':_r.get('mode'),
                    'baseline_pipeline_ms':_r.get('baseline_pipeline_ms'),
                    'candidate_pipeline_ms':_r.get('candidate_pipeline_ms'),
                    'latency_ms':_r.get('request_wall_ms'),
                    'quality_certified':False,'success_label':None,
                    'financial_cost':None,'energy_joules':None,
                    'criterion':'Local opt-in canary or shadow; no production routing changed'
                })
            except (ValueError,KeyError,TypeError):continue
    result.setdefault('limitations',[]).append('v21.1 local adoption tests do not modify production routing; shadow executes both paths and saves no compute.')
    # NIMBO_IMAGE_EVIDENCE_V22: separate local GPU measurements; no automatic quality certification.
    _image_file=Path('results/image_gateway_runs_v22.jsonl')
    _review_file=Path('results/image_gateway_human_reviews_v22.jsonl')
    _human={}
    if _review_file.is_file():
        for _line in _review_file.read_text(encoding='utf8').splitlines():
            try:
                _entry=json.loads(_line)
                if isinstance(_entry,dict):_human[_entry.get('run_id')]=_entry
            except (ValueError,TypeError):continue
    if _image_file.is_file():
        for _line in _image_file.read_text(encoding='utf8').splitlines()[-limit:]:
            try:
                _r=json.loads(_line)
                if not isinstance(_r,dict) or not _r.get('run_id'):continue
                _label=_human.get(_r['run_id'],{})
                result['records'].append({
                    'id':_r['run_id'],'source':'image_gateway_runs_v22.jsonl',
                    'evidence_kind':'local_image_gpu_execution','workload':'image',
                    'route':_r.get('mode'),'timestamp_utc':_r.get('at_utc'),
                    'device':_r.get('device'),'seed':_r.get('seed'),
                    'fast_inference_ms':_r.get('fast_inference_ms'),
                    'standard_inference_ms':_r.get('standard_inference_ms'),
                    'end_to_end_ms':_r.get('end_to_end_ms'),
                    'human_review_fast':_label.get('fast_success'),
                    'human_review_standard':_label.get('standard_success'),
                    'human_review_available':bool(_label),
                    'quality_certified':False,'success_label':None,
                    'latency_ms':None,'criterion':'GPU inference time only; human critical-object label is unaudited',
                    'cost_usd':None,'energy_joules':None,
                })
            except (ValueError,TypeError,KeyError):continue
    result.setdefault('limitations',[]).append('Image gateway records reflect local GPU inference; human object-presence assessments are not independent certification.')
    return result
