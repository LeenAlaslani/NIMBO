"""Same-model PyTorch FP32 versus ONNX Runtime FP32 benchmark.

Only the pinned-to-allowlist IMDb model/dataset; no arbitrary Python or model URLs.
Same evaluation samples/token ids, alternating execution order, warm forwards.
No data or inference outputs are written to evidence.
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.nimbo_optimize_v20 import MODEL_ID, DATASET_ID, TrialRequest, evaluation_set, p95, weight_bytes

RESULTS = Path('results/same_model_onnx_trials_v20.jsonl')
CACHE = Path('results/model_cache_v20/onnx_fp32')
EXPORTS = Path('results/optimized_models_v20_2')
router = APIRouter(prefix='/nimbo/optimize/onnx', tags=['same-model-onnx'])


class OnnxRequest(TrialRequest):
    min_mean_speedup: float = Field(default=0.05, ge=0.0, le=0.5)
    max_p95_regression: float = Field(default=0.0, ge=0.0, le=0.25)
    min_prediction_agreement: float = Field(default=0.995, ge=0.9, le=1.0)
    threads: int = Field(default=4, ge=1, le=8)


def validate(req: OnnxRequest):
    if req.model_id != MODEL_ID:
        raise ValueError('Only the allowlisted external IMDb classifier is available for this trial.')
    if req.phase not in ('development', 'heldout'):
        raise ValueError('phase must be development or heldout')
    if req.examples is None and req.phase == 'development' and req.sample_count > 200:
        raise ValueError('Development partition has at most 200 disjoint samples')
    if req.examples is None and req.seed != 20261031:
        raise ValueError('IMDb partition seed must stay fixed for disjoint development/heldout sets')
    if req.examples is not None:
        if not 10 <= len(req.examples) <= req.sample_count:
            raise ValueError('A supplied evaluation set requires 10..sample_count labeled examples')
        if any(e.expected_label.strip().upper() not in ('POSITIVE', 'NEGATIVE') for e in req.examples):
            raise ValueError('Every label must be POSITIVE or NEGATIVE')
    return req


def decision_for(baseline, candidate, agreement, req):
    accuracy_drop = baseline['accuracy_on_supplied_labels'] - candidate['accuracy_on_supplied_labels']
    mean_reduction = (baseline['mean_inference_ms'] - candidate['mean_inference_ms']) / baseline['mean_inference_ms']
    p95_regression = (candidate['p95_ms'] - baseline['p95_ms']) / baseline['p95_ms']
    quality_pass = accuracy_drop <= req.quality_max_drop + 1e-12 and agreement >= req.min_prediction_agreement
    latency_pass = mean_reduction >= req.min_mean_speedup and p95_regression <= req.max_p95_regression
    eligible = quality_pass and latency_pass
    if not quality_pass:
        reason = 'REJECTED: accuracy or FP32 prediction agreement missed its guardrail'
    elif not latency_pass:
        reason = 'REJECTED: mean speedup or P95 latency requirement not met'
    else:
        reason = 'PASSED local quality/latency gates; not yet approved for deployment'
    return {'eligible': eligible, 'quality_pass': quality_pass, 'latency_pass': latency_pass,
            'prediction_agreement': round(agreement, 5), 'mean_speedup_fraction': round(mean_reduction, 5),
            'p95_regression_fraction': round(p95_regression, 5), 'reason': reason,
            'constraints': {'max_accuracy_drop': req.quality_max_drop,
                            'min_prediction_agreement': req.min_prediction_agreement,
                            'min_mean_speedup': req.min_mean_speedup,
                            'max_p95_regression': req.max_p95_regression}}


def summarize(times, correct, n):
    return {'mean_inference_ms': round(statistics.mean(times), 4),
            'p50_ms': round(statistics.median(times), 4), 'p95_ms': round(p95(times), 4),
            'accuracy_on_supplied_labels': round(correct / n, 5), 'successes': correct,
            'inference_seconds_per_labeled_success': round(sum(times)/(1000*correct), 6) if correct else None}


def _load_onnx_model(model_id, cache_dir, threads, revision):
    """Only save a complete export in cache; don't use a partial export after failure."""
    import onnxruntime as ort
    import shutil
    from optimum.onnxruntime import ORTModelForSequenceClassification
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    manifest = cache_dir / 'nimbo_cache_manifest.json'
    files = sorted(cache_dir.glob('*.onnx')) if cache_dir.exists() else []
    trusted_cache = False
    if files and manifest.is_file():
        try:
            trusted_cache = json.loads(manifest.read_text()) == {'model_id':model_id,'format':'onnx_fp32','revision':revision}
        except (ValueError, OSError):
            pass
    if not trusted_cache:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        # Hugging Face Optimum supported export path for DistilBERT classification.
        candidate = ORTModelForSequenceClassification.from_pretrained(
            model_id, revision=revision, export=True, provider='CPUExecutionProvider',
            session_options=opts, trust_remote_code=False)
        candidate.save_pretrained(str(cache_dir))
        files = sorted(cache_dir.glob('*.onnx'))
        if len(files) != 1:
            raise RuntimeError(f'Expected exactly one ONNX file, found {len(files)}')
        manifest.write_text(json.dumps({'model_id':model_id,'format':'onnx_fp32','revision':revision}), encoding='utf8')
    file = files[0]
    session = ort.InferenceSession(str(file), sess_options=opts, providers=['CPUExecutionProvider'])
    if 'CPUExecutionProvider' not in session.get_providers():
        raise RuntimeError('No CPUExecutionProvider was initialized')
    return session, file


def run_onnx_trial(req: OnnxRequest, *, data_adapter=None, model_adapter=None, clock=None):
    validate(req)
    if clock is None:
        clock = time.perf_counter
    start = clock()
    run_id = 'onnx_' + uuid.uuid4().hex[:16]
    def event(stage, message, **extra):
        return {'type':'event', 'stage':stage, 'message':message,
                'at_ms':round((clock()-start)*1000, 2), 'run_id':run_id, **extra}
    yield event('analyze', 'Preparing original IMDb FP32 versus its ONNX Runtime FP32 execution path')
    rows, source = data_adapter(req) if data_adapter else evaluation_set(req)
    if not rows or len(rows) > req.sample_count:
        raise ValueError('Invalid evaluation dataset size')
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(req.threads)
    yield event('route', 'Loading the exact same pretrained model for PyTorch and ONNX CPU inference', model=MODEL_ID)
    warm_start = clock()
    if model_adapter:
        tokenizer, original, session, onnx_file = model_adapter(req)
    else:
        original = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, trust_remote_code=False).eval().cpu()
        revision = getattr(original.config, '_commit_hash', None)
        if not revision:
            raise RuntimeError('Could not pin the source model revision; refusing a non-reproducible comparison')
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision, trust_remote_code=False)
        yield event('execute', 'Exporting/reusing the FP32 ONNX graph; this setup time is excluded from warm inference')
        session, onnx_file = _load_onnx_model(MODEL_ID, CACHE, req.threads, revision)
    revision = getattr(getattr(original,'config',None),'_commit_hash',None) or 'test_injected'
    input_names = [x.name for x in session.get_inputs()]
    if not input_names or 'input_ids' not in input_names:
        raise RuntimeError('ONNX session does not accept input_ids')
    prepared=[]
    for txt, label in rows:
        tok = tokenizer(txt, max_length=req.max_length, truncation=True, return_tensors='np')
        ort_inputs = {key:np.asarray(tok[key],dtype=np.int64) for key in input_names}
        pt_inputs = {key:torch.from_numpy(value) for key,value in ort_inputs.items()}
        prepared.append((pt_inputs,ort_inputs,int(label)))
    yield event('execute', 'Both paths share identical token IDs and attention masks; warming three samples')
    with torch.inference_mode():
        for inp_t,inp_n,_ in prepared[:min(3,len(prepared))]:
            original(**inp_t)
            session.run(None,inp_n)
    setup_ms=round((clock()-warm_start)*1000,2)
    times={'baseline':[],'candidate':[]}
    successes={'baseline':0,'candidate':0}
    different_predictions=0
    yield event('execute','Comparing paired warm CPU inference; input tokenization excluded',total=len(prepared),completed=0)
    for i,(pt_inputs,ort_inputs,actual) in enumerate(prepared):
        order=('baseline','candidate') if i%2==0 else ('candidate','baseline')
        predictions={}
        for which in order:
            if which=='baseline':
                with torch.inference_mode():
                    t=clock();logits=original(**pt_inputs).logits
                    # Synchronization not needed: inference strictly local CPU.
                    elapsed=(clock()-t)*1000
                predicted=int(torch.argmax(logits,dim=-1).item())
            else:
                t=clock();logits=session.run(None,ort_inputs)[0];elapsed=(clock()-t)*1000
                predicted=int(np.asarray(logits).argmax(axis=-1).item())
            times[which].append(elapsed)
            predictions[which]=predicted
            successes[which]+=int(predicted==actual)
        different_predictions+=int(predictions['baseline']!=predictions['candidate'])
        if (i+1)%max(1,len(prepared)//20)==0 or i+1==len(prepared):
            yield event('execute',f'Evaluated input {i+1}/{len(prepared)}',completed=i+1,total=len(prepared))
    baseline=summarize(times['baseline'],successes['baseline'],len(prepared))
    candidate=summarize(times['candidate'],successes['candidate'],len(prepared))
    agreement=(len(prepared)-different_predictions)/len(prepared)
    gate=decision_for(baseline,candidate,agreement,req)
    yield event('verify', 'Compared accuracy, prediction agreement, mean latency and P95 against explicit acceptance gates')
    original_bytes=weight_bytes(original)
    file_bytes=Path(onnx_file).stat().st_size
    artifact=None
    approval='not_approved'
    if req.export_variant and req.phase=='heldout' and req.examples is None and gate['eligible']:
        import shutil
        EXPORTS.mkdir(parents=True,exist_ok=True)
        target=EXPORTS/(run_id+'_model.onnx')
        shutil.copy2(onnx_file,target)
        dest_manifest=target.with_suffix('.manifest.json')
        dest_manifest.write_text(json.dumps({'model_id':MODEL_ID,'model_revision':revision,
            'artifact_type':'onnx_fp32','source_run_id':run_id,'threads':req.threads,
            'max_length':req.max_length,'provider':'CPUExecutionProvider',
            'tokenizer':'Use the source model tokenizer',
            'quality_and_latency':'Local benchmark acceptance; human deployment review required'},indent=2),encoding='utf8')
        artifact={'onnx_model':str(target),'manifest':str(dest_manifest)}
        approval='eligible_for_review'
    comparison={'baseline_model':MODEL_ID+' · PyTorch FP32',
        'candidate_model':MODEL_ID+' · ONNX Runtime FP32', 'sample_count':len(prepared),
        'baseline':baseline,'candidate':candidate,
        'mean_latency_reduction_fraction':gate['mean_speedup_fraction'],
        'prediction_agreement_fraction':round(agreement,5),
        'all_labels_present':True, 'labels_origin':source, 'quality_audited':False,
        'baseline_state_dict_bytes':original_bytes,'onnx_artifact_bytes':file_bytes,
        'artifact_size_caveat':'PyTorch state_dict and ONNX graph file are not equivalent weight-only formats.',
        'criterion':'Paired warm CPU forward inference; tokenizer excluded; CPUExecutionProvider',
        'model_revision':revision}
    report={'workload':'classification','route':'same_model_onnx_fp32',
        'result':{'label':f'{len(prepared)} inputs evaluated'},
        'metrics':{'end_to_end_ms':round((clock()-start)*1000,2),'model_setup_and_warmup_ms':setup_ms},
        'comparison':comparison,
        'optimization':{'kind':'onnx_runtime_fp32', 'original_model_id':MODEL_ID,
            'gate':gate,'approval':approval,'export':artifact},
        'guardrail':{'action':'HELDOUT_LOCAL_EVALUATION_UNAUDITED' if req.phase=='heldout' else 'DEVELOPMENT_LABELS_UNAUDITED',
            'reason':'IMDb reference labels, not independently audited certification'},
        'measurement_note':'Same FP32 model and exact token IDs. Mean and P95 from warm CPU inference. Cached export/setup excluded; no FLOPs, power, or dollars measured.'}
    record={'run_id':run_id,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'evidence_kind':'local_same_model_onnx_optimization_trial','phase':req.phase,
        'model_id':MODEL_ID,'sample_count':len(rows), 'dataset':DATASET_ID if req.examples is None else 'user_supplied_local',
        'dataset_partition':source,'seed':req.seed,'max_length':req.max_length,
        'python_version':sys.version.split()[0],'pytorch_version':torch.__version__,
        'onnxruntime_version':__import__('onnxruntime').__version__,
        'hardware':{'machine':platform.machine(),'processor':platform.processor(),'system':platform.system()},
        'thread_count':req.threads,'quality_certified':False,'comparison':comparison,
        'optimization':report['optimization'],'workload':'classification','financial_cost':None}
    RESULTS.parent.mkdir(parents=True,exist_ok=True)
    with RESULTS.open('a',encoding='utf8') as f:
        f.write(json.dumps(record,ensure_ascii=False)+'\n')
    yield event('deliver', 'ONNX variant evaluated and evidence saved; eligibility is not deployment approval', decision=gate['reason'])
    yield {'type':'result','report':report}


@router.get('/catalog')
def catalog():
    return {'model':MODEL_ID,'variants':['pytorch_fp32','onnx_runtime_fp32'],
            'provider':'CPUExecutionProvider','ready':'install_optimum_onnx_on_local_cpu',
            'phases':['development','heldout'],'candidate_kind':'same_weights_different_execution_runtime'}


@router.get('/recent')
def recent(limit:int=20):
    rows=[]
    if RESULTS.is_file():
        for line in RESULTS.read_text(encoding='utf8').splitlines()[-min(50,max(1,limit)):]:
            try:rows.append(json.loads(line))
            except (ValueError,TypeError):pass
    return {'records':rows,'evidence_kind':'local_same_model_onnx_optimization_trial'}


@router.post('/run')
def run(req:OnnxRequest):
    try:validate(req)
    except ValueError as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc
    def stream():
        try:
            for row in run_onnx_trial(req):
                yield json.dumps(row,ensure_ascii=False)+'\n'
        except Exception as exc:
            yield json.dumps({'type':'error','message':f'{type(exc).__name__}: {exc}'})+'\n'
    return StreamingResponse(stream(),media_type='application/x-ndjson',headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='NIMBO FP32 same-model PyTorch vs ONNX Runtime trial')
    parser.add_argument('--phase',choices=['development','heldout'],default='development')
    parser.add_argument('--samples',type=int,default=200)
    parser.add_argument('--export-variant',action='store_true')
    args=parser.parse_args()
    req=OnnxRequest(phase=args.phase,sample_count=args.samples,export_variant=args.export_variant)
    for packet in run_onnx_trial(req):
        if packet['type']=='event':
            if packet['stage']!='execute' or packet.get('completed') in (None,50,100,150,200,400,600,800):
                print(f"[{packet['stage']:7s}] {packet['message']}",flush=True)
        else:
            print(json.dumps(packet['report'],indent=2))
