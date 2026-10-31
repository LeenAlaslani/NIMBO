"""NIMBO v20: REAL same-model FP32 -> INT8 local CPU evaluation.

No arbitrary model URL, no remote code, no fabricated savings. Model & dataset
IDs are currently allowlisted. Does not execute during import.
"""
from __future__ import annotations
import json, io, os, platform, random, statistics, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

MODEL_ID = 'lvwerra/distilbert-imdb'
DATASET_ID = 'stanfordnlp/imdb'
RESULTS = Path('results/same_model_trials_v20.jsonl')
router = APIRouter(prefix='/nimbo/optimize/v20', tags=['same-model-optimization'])

class Example(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    expected_label: str

class TrialRequest(BaseModel):
    model_id: str = MODEL_ID
    phase: str = 'development'
    sample_count: int = Field(default=200, ge=10, le=1000)
    seed: int = 20261031
    max_length: int = Field(default=256, ge=32, le=512)
    quality_max_drop: float = Field(default=0.01, ge=0.0, le=0.1)
    examples: Optional[list[Example]] = Field(default=None, max_length=1000)
    export_variant: bool = False


def validate_request(req: TrialRequest):
    if req.model_id != MODEL_ID:
        raise ValueError('Only the allowlisted IMDb classifier is supported in this first model onboarding trial.')
    if req.phase not in ('development', 'heldout'):
        raise ValueError('phase must be development or heldout.')
    if req.examples is None and req.phase == 'development' and req.sample_count > 200:
        raise ValueError('Development partition has at most 200 disjoint IMDb test samples; heldout has up to 1000.')
    if req.seed != 20261031 and req.examples is None:
        raise ValueError('IMDb data split seed is fixed to keep development and heldout records disjoint.')
    if req.examples is not None:
        if len(req.examples) < 10 or len(req.examples) > req.sample_count:
            raise ValueError('A custom evaluation set must contain 10 to sample_count examples.')
        if any(x.expected_label.strip().upper() not in {'POSITIVE', 'NEGATIVE'} for x in req.examples):
            raise ValueError('All examples need POSITIVE or NEGATIVE labels.')
    return req


def gate_decision(baseline_accuracy, candidate_accuracy, baseline_ms, candidate_ms, maximum_drop):
    if baseline_accuracy is None or candidate_accuracy is None:
        return {'eligible': False, 'reason': 'Missing ground-truth labels', 'quality_pass': False, 'latency_pass': False}
    quality_pass = candidate_accuracy >= baseline_accuracy - maximum_drop - 1e-12
    latency_pass = candidate_ms < baseline_ms
    return {'eligible': bool(quality_pass and latency_pass), 'quality_pass': bool(quality_pass),
            'latency_pass': bool(latency_pass),
            'reason': ('Candidate passed this local benchmark' if quality_pass and latency_pass else
                       'Candidate rejected: quality drop exceeded policy' if not quality_pass else
                       'Candidate rejected: no measured latency improvement')}


def p95(values):
    a = sorted(values)
    return a[min(len(a)-1, max(0, int(.95 * len(a) + .999999) - 1))]


def calc_summary(times, correct, count):
    return {'mean_inference_ms': round(statistics.mean(times), 4),
            'p50_ms': round(statistics.median(times), 4), 'p95_ms': round(p95(times), 4),
            'accuracy_on_supplied_labels': round(correct / count, 5), 'successes': correct,
            'inference_seconds_per_labeled_success': round(sum(times)/(1000*correct), 6) if correct else None}


def evaluation_set(req):
    if req.examples is not None:
        return [(x.text, 1 if x.expected_label.strip().upper()=='POSITIVE' else 0) for x in req.examples], 'user_supplied_labels'
    from datasets import load_dataset
    # Two disjoint partitions of official IMDb test split. This is NOT an independent
    # external audit and may overlap a published model's previous evaluation domain.
    dataset = load_dataset(DATASET_ID, split='test')
    if len(dataset) < 1200: raise RuntimeError('IMDb test split unexpectedly small')
    ids = list(range(len(dataset)))
    random.Random(req.seed).shuffle(ids)
    ids = ids[:200] if req.phase == 'development' else ids[200:1200]
    ids = ids[:req.sample_count]
    return [(str(dataset[i]['text']), int(dataset[i]['label'])) for i in ids], 'imdb_test_disjoint_dev_heldout'


def weight_bytes(model):
    # These are serialized weights, NOT process RAM and not GPU memory.
    buffer = io.BytesIO()
    import torch
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def run_trial(req: TrialRequest, model_loader=None, dataset_loader=None, perf=None):
    """Yield genuine progress events followed by a result. Inject adapters in tests."""
    validate_request(req)
    if perf is None: perf = time.perf_counter
    begin = perf()
    run_id = 'trial_' + uuid.uuid4().hex[:16]
    def event(stage, message, **details):
        return {'type':'event', 'stage':stage, 'message':message,
                'at_ms':round(1000*(perf()-begin),2), 'run_id':run_id, **details}
    yield event('analyze', 'Preparing same-model original FP32 versus dynamic INT8 evaluation')
    rows, source = (dataset_loader(req) if dataset_loader else evaluation_set(req))
    if not rows or len(rows) > req.sample_count:
        raise ValueError('No valid samples or too many samples supplied')
    import torch
    from torch import nn
    if hasattr(torch,'set_num_threads'):torch.set_num_threads(min(4,max(1,os.cpu_count() or 1)))
    supported=list(torch.backends.quantized.supported_engines)
    prefer = 'qnnpack' if platform.machine().lower() in ('arm64','aarch64') else 'x86'
    engine = prefer if prefer in supported else ('fbgemm' if 'fbgemm' in supported else (supported[0] if supported else None))
    if engine is None or engine == 'none': raise RuntimeError('No supported CPU INT8 quantization engine detected')
    torch.backends.quantized.engine=engine
    yield event('route', 'One allowlisted external model; comparing its original weights and an INT8-transformed variant',model=MODEL_ID)
    load_start=perf()
    if model_loader:
        tokenizer, original = model_loader(req)
        revision='test_mock'
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        # No trust_remote_code; this is a fixed allowlist, not arbitrary uploaded Python.
        tokenizer=AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
        original=AutoModelForSequenceClassification.from_pretrained(MODEL_ID,trust_remote_code=False).cpu().eval()
        revision=getattr(original.config,'_commit_hash',None) or 'not_pinned'
    baseline_bytes=weight_bytes(original)
    yield event('execute','Model loaded; transforming Linear layers to INT8 dynamic quantization')
    from torch.ao.quantization import quantize_dynamic
    optimized=quantize_dynamic(original,{nn.Linear},dtype=torch.qint8,inplace=False).cpu().eval()
    changed=[m for m in optimized.modules() if 'quantized.dynamic' in type(m).__module__ and type(m).__name__=='Linear']
    if not changed:raise RuntimeError('Quantization created zero INT8 Linear layers; refusing an optimization claim')
    quantized_bytes=weight_bytes(optimized)
    weight_ratio=(1-quantized_bytes/baseline_bytes) if baseline_bytes else None
    load_warmup_s=perf()-load_start
    prepared=[]
    for txt,label in rows:
        x=tokenizer(txt,max_length=req.max_length,truncation=True,return_tensors='pt')
        prepared.append((x,label))
    # Warm up both variants, excluded from warm inference timing.
    with torch.inference_mode():
        for mdl in (original,optimized):
            for x,_ in prepared[:min(3,len(prepared))]:mdl(**x)
    warmed_ms=(perf()-load_start)*1000
    yield event('execute','Both variants warmed; evaluating paired CPU forward passes',total=len(prepared),completed=0)
    times={'baseline':[], 'candidate':[]};correct={'baseline':0,'candidate':0};mismatches=0
    for i,(inputs,actual) in enumerate(prepared):
        order=[('baseline',original),('candidate',optimized)] if i%2==0 else [('candidate',optimized),('baseline',original)]
        preds={}
        with torch.inference_mode():
            for kind,mdl in order:
                t=perf();out=mdl(**inputs);elapsed=(perf()-t)*1000
                predicted=int(torch.argmax(out.logits,dim=-1).item())
                preds[kind]=predicted;correct[kind]+=int(predicted==actual);times[kind].append(elapsed)
        mismatches+=int(preds['baseline']!=preds['candidate'])
        if (i+1)%max(1,len(prepared)//20)==0 or i==len(prepared)-1:
            yield event('execute',f'Evaluated input {i+1}/{len(prepared)}',completed=i+1,total=len(prepared))
    baseline=calc_summary(times['baseline'],correct['baseline'],len(prepared))
    candidate=calc_summary(times['candidate'],correct['candidate'],len(prepared))
    reduction=(baseline['mean_inference_ms']-candidate['mean_inference_ms'])/baseline['mean_inference_ms'] if baseline['mean_inference_ms'] else None
    decision=gate_decision(baseline['accuracy_on_supplied_labels'],candidate['accuracy_on_supplied_labels'],baseline['mean_inference_ms'],candidate['mean_inference_ms'],req.quality_max_drop)
    yield event('verify','Compared both model variants against reference labels; applying latency and quality gates')
    artifact=None
    if req.export_variant and req.phase == 'heldout' and req.examples is None and decision['eligible']:
        dest=Path('results/optimized_models_v20');dest.mkdir(parents=True,exist_ok=True)
        weights=dest/(run_id+'_dynamic_int8_state_dict.pt')
        torch.save(optimized.state_dict(),weights)
        manifest=dest/(run_id+'_manifest.json')
        manifest.write_text(json.dumps({'model_id':MODEL_ID,'model_revision':revision,'variant':'dynamic_int8','source_run_id':run_id,'quantized_linear_layers':len(changed),'quantization_engine':engine,'max_length':req.max_length,'reloading':'Load the original model architecture, quantize its Linear layers and then load this state_dict. The original tokenizer remains required.'},indent=2)+'\n',encoding='utf8')
        artifact={'state_dict':str(weights),'manifest':str(manifest),'note':'Quantized state_dict, not a self-contained model package.'}
    comparison={'baseline_model':MODEL_ID+' · FP32','candidate_model':MODEL_ID+' · INT8 dynamic',
        'sample_count':len(rows),'baseline':baseline,'candidate':candidate,
        'mean_latency_reduction_fraction':round(reduction,5) if reduction is not None else None,
        'all_labels_present':True,'labels_origin':source,'model_revision':revision,
        'criterion':'warm CPU forward time, tokenizer excluded; serialized state_dict size separately measured',
        'matched_prediction_count':len(rows)-mismatches,'quality_audited':False,
        'baseline_state_dict_bytes':baseline_bytes,'candidate_state_dict_bytes':quantized_bytes,
        'serialized_weight_reduction_fraction':round(weight_ratio,5) if weight_ratio is not None else None}
    report={'workload':'classification','route':'same_model_dynamic_int8','result':{'label':f'{len(rows)} inputs evaluated'},
        'metrics':{'end_to_end_ms':round((perf()-begin)*1000,2),'load_and_quantize_ms':round(load_warmup_s*1000,2),'prep_and_warmup_ms':round(warmed_ms-load_warmup_s*1000,2)},
        'comparison':comparison,'optimization':{'kind':'post_training_dynamic_int8','original_model_id':MODEL_ID,
             'quality_floor_max_drop':req.quality_max_drop,'quantized_linear_layers':len(changed),
             'quantization_engine':engine,'gate':decision,'export':artifact,
             'approval':'eligible_for_review' if req.phase=='heldout' and req.examples is None and decision['eligible'] else 'not_approved'},
        'guardrail':{'action':'HELDOUT_LOCAL_EVALUATION_UNAUDITED' if req.phase=='heldout' else 'DEVELOPMENT_LABELS_UNAUDITED',
                     'reason':'IMDb labeled evaluation, not independent certification.'},
        'measurement_note':'Same pretrained model and same tokenizer; FP32 vs quantized INT8 Linear layers. CPU warm forward only; not energy, dollars, FLOPs or whole-request throughput.'}
    record={'run_id':run_id,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'evidence_kind':'local_same_model_optimization_trial','phase':req.phase,'model_id':MODEL_ID,
        'sample_count':len(rows),'dataset':DATASET_ID if req.examples is None else 'user_supplied_local',
        'dataset_partition':source,'seed':req.seed,'max_length':req.max_length,
        'torch_version':torch.__version__,'python_version':sys.version.split()[0],
        'hardware':{'machine':platform.machine(),'processor':platform.processor(),'system':platform.system()},
        'quality_certified':False,'comparison':comparison,'optimization':report['optimization'],
        'result_quality':'locally_evaluated_not_independently_audited',
        'workload':'classification','financial_cost':None}
    RESULTS.parent.mkdir(parents=True,exist_ok=True)
    with RESULTS.open('a',encoding='utf8') as fh:fh.write(json.dumps(record,ensure_ascii=False)+'\n')
    yield event('deliver','Variant measured and recorded; adoption requires held-out check and passing quality/latency gates')
    yield {'type':'result','report':report}

@router.get('/catalog')
def catalog():
    return {'models':[{'id':MODEL_ID,'workload':'classification','source':'Hugging Face / allowlisted',
                      'variants':['fp32','int8_dynamic'],'device':'local CPU','health':'not_loaded_until_requested'}],
            'dataset':DATASET_ID,'supported_phases':['development','heldout'],
            'source':'No arbitrary user-supplied code or remote URLs accepted.'}

@router.get('/recent')
def recent(limit:int=15):
    limit=max(1,min(50,limit));rows=[]
    if RESULTS.is_file():
        for line in RESULTS.read_text(encoding='utf8').splitlines()[-limit:]:
            try:rows.append(json.loads(line))
            except (ValueError,TypeError):pass
    return {'records':rows,'evidence_kind':'local_same_model_optimization_trial'}

@router.post('/run')
def optimize(req:TrialRequest):
    try:validate_request(req)
    except ValueError as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc
    def stream():
        try:
            for packet in run_trial(req):yield json.dumps(packet,ensure_ascii=False)+'\n'
        except Exception as exc:
            yield json.dumps({'type':'error','message':f'{type(exc).__name__}: {exc}'},ensure_ascii=False)+'\n'
    return StreamingResponse(stream(),media_type='application/x-ndjson',headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Optimize the same external IMDb classifier, FP32 versus dynamic INT8')
    parser.add_argument('--phase',choices=['development','heldout'],default='development')
    parser.add_argument('--samples',type=int,default=200)
    parser.add_argument('--max-length',type=int,default=256)
    parser.add_argument('--export-variant',action='store_true')
    args=parser.parse_args()
    request=TrialRequest(phase=args.phase,sample_count=args.samples,max_length=args.max_length,export_variant=args.export_variant)
    for packet in run_trial(request):
        if packet['type']=='result':
            print(json.dumps(packet['report'],indent=2,ensure_ascii=False))
        else:print(packet['stage'].upper(),packet['message'],flush=True)
