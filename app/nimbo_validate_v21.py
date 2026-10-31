"""NIMBO v21.0: independent-of-development holdout validation for same IMDb model.

Data: fixed-seed disjoint IMDb test indices 200:1200, never the v20.2 dev 0:200.
Compare PyTorch and ONNX CPU on identical texts, in alternating order across rounds.
Keep model selection, artifact staging and deployment authorization separate.
This is *not* independent third-party auditing, full API latency, FLOPs, or energy.
"""
from __future__ import annotations

import hashlib
import json
import platform
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.nimbo_optimize_v20 import MODEL_ID, DATASET_ID, evaluation_set, p95, weight_bytes
from app.nimbo_onnx_v20 import _load_onnx_model, CACHE

LOG = Path('results/independent_validation_v21.jsonl')
STAGED = Path('results/onnx_review_candidates_v21')
router = APIRouter(prefix='/nimbo/validate/v21', tags=['onnx-heldout-validation'])


class ValidationRequest(BaseModel):
    model_id: str = MODEL_ID
    phase: str = 'heldout'
    sample_count: int = Field(default=800, ge=500, le=1000)
    rounds: int = Field(default=3, ge=2, le=5)
    seed: int = 20261031
    max_length: int = Field(default=256, ge=32, le=512)
    threads: int = Field(default=4, ge=1, le=8)
    quality_max_drop: float = Field(default=0.01, ge=0.0, le=0.05)
    min_prediction_agreement: float = Field(default=0.995, ge=0.99, le=1)
    min_mean_speedup: float = Field(default=0.05, ge=0.01, le=0.5)
    min_local_pipeline_speedup: float = Field(default=0.05, ge=0.0, le=0.5)
    max_p95_regression: float = Field(default=0, ge=0, le=0.1)
    export_variant: bool = True
    examples: Optional[list] = None


def validate_request(req: ValidationRequest):
    if req.model_id != MODEL_ID:
        raise ValueError('Only the fixed, allowlisted IMDb model is available.')
    if req.phase != 'heldout' or req.seed != 20261031 or req.examples is not None:
        raise ValueError('v21 requires the fixed held-out IMDb test partition; custom/development rows are not accepted.')
    return req


def _summarize(times):
    return {'mean_ms':round(statistics.mean(times),4), 'p50_ms':round(statistics.median(times),4),
            'p95_ms':round(p95(times),4), 'count':len(times)}


def gate_v21(baseline, candidate, agreement, rounds, req):
    baseline_forward=baseline['warm_forward']; candidate_forward=candidate['warm_forward']
    baseline_local=baseline['tokenizer_plus_forward']; candidate_local=candidate['tokenizer_plus_forward']
    def reduction(a,b):
        return (a-b)/a if a>0 else float('-inf')
    forward_gain=reduction(baseline_forward['mean_ms'],candidate_forward['mean_ms'])
    local_gain=reduction(baseline_local['mean_ms'],candidate_local['mean_ms'])
    accuracy_drop=baseline['accuracy_on_reference_labels']-candidate['accuracy_on_reference_labels']
    quality=accuracy_drop<=req.quality_max_drop+1e-12 and agreement>=req.min_prediction_agreement
    p95= candidate_forward['p95_ms'] <= baseline_forward['p95_ms']*(1+req.max_p95_regression)+1e-12
    repeatable=all(x['candidate_forward_mean_ms'] < x['baseline_forward_mean_ms'] for x in rounds)
    latency=(forward_gain >= req.min_mean_speedup and local_gain >= req.min_local_pipeline_speedup
             and p95 and repeatable)
    eligible=bool(quality and latency)
    return {
        'eligible':eligible,'quality_pass':quality,'latency_pass':latency,'p95_pass':p95,
        'repeatability_pass':repeatable,'prediction_agreement':round(agreement,5),
        'mean_speedup_fraction':round(forward_gain,5), 'local_pipeline_speedup_fraction':round(local_gain,5),
        'accuracy_drop_fraction':round(accuracy_drop,5),
        'reason':('ELIGIBLE FOR MANUAL REVIEW; no variant has been deployed' if eligible else
                  'REJECTED: held-out accuracy/agreement condition failed' if not quality else
                  'REJECTED: latency, P95 or repeatability condition failed'),
        'constraints':{'max_accuracy_drop':req.quality_max_drop, 'min_prediction_agreement':req.min_prediction_agreement,
                       'min_mean_speedup':req.min_mean_speedup,'min_local_pipeline_speedup':req.min_local_pipeline_speedup,
                       'max_p95_regression':req.max_p95_regression,'require_faster_each_round':True}}


def _sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def _stage_candidate(session_file, tokenizer, revision, req, run_id, source_digest):
    """A review-only artifact: safe local copies, revision/digest manifest. No deployment."""
    import shutil
    folder=STAGED/run_id
    folder.mkdir(parents=True,exist_ok=False)
    try:
        file=folder/'model.onnx'
        shutil.copy2(session_file,file)
        tokenizer.save_pretrained(str(folder))
        manifest={
            'schema_version':'1.0','status':'eligible_for_manual_review_only',
            'deployment_approved':False,'model_id':MODEL_ID,'model_revision':revision,
            'source_run_id':run_id,'heldout_partition':source_digest,'sample_count':req.sample_count,
            'rounds':req.rounds,'cpu_threads':req.threads,'max_length':req.max_length,
            'runtime':'onnxruntime','provider':'CPUExecutionProvider',
            'model_sha256':_sha256_file(file),
            'tokenizer_files':sorted(p.name for p in folder.iterdir() if p.name not in ('model.onnx','manifest.json')),
            'claim_limitations':['No deployment performed','Not an independent third-party audit',
                                 'Warm CPU and local tokenization timing, not full HTTP request or energy']}
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
        return {'directory':str(folder),'onnx_model':str(file),'manifest':str(folder/'manifest.json'),
                'sha256':manifest['model_sha256']}
    except Exception:
        shutil.rmtree(folder,ignore_errors=True)
        raise


def validate_onnx(req:ValidationRequest, *, data_adapter=None, model_adapter=None, clock=None, stage_candidate=None):
    """Generator streams measured stages and returns final evidence. Dependency hooks for tests."""
    validate_request(req)
    if clock is None: clock=time.perf_counter
    start=clock();run_id='heldout_'+uuid.uuid4().hex[:16]
    def event(stage,message,**fields):
        return {'type':'event','stage':stage,'message':message,'run_id':run_id,
                'at_ms':round((clock()-start)*1000,2),**fields}
    yield event('analyze','Loading the disjoint IMDb test holdout (never the 200 development samples)')
    if data_adapter:
        rows, source=data_adapter(req)
    else:
        from app.nimbo_optimize_v20 import TrialRequest
        rows, source=evaluation_set(TrialRequest(phase='heldout',sample_count=req.sample_count,seed=req.seed))
    if len(rows)!=req.sample_count or any(not isinstance(t,str) or int(label) not in (0,1) for t,label in rows):
        raise RuntimeError('Heldout dataset must contain exactly the requested number of binary-labeled items')
    sample_digest=hashlib.sha256(('\n'.join(hashlib.sha256(t.encode('utf8')).hexdigest() for t,_ in rows)).encode()).hexdigest()
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    torch.set_num_threads(req.threads)
    yield event('route',f'Loading one pinned IMDb model; {req.rounds} paired CPU rounds; {req.sample_count} distinct rows')
    setup_start=clock()
    if model_adapter:
        tokenizer, pt_model, session, graph_path, revision=model_adapter(req)
    else:
        pt_model=AutoModelForSequenceClassification.from_pretrained(MODEL_ID,trust_remote_code=False).eval().cpu()
        revision=getattr(pt_model.config,'_commit_hash',None)
        if not revision: raise RuntimeError('Source model has no pinned revision; validation stopped')
        tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=revision,trust_remote_code=False)
        yield event('execute','Loading or exporting ONNX graph; setup is excluded from per-item timings')
        session,graph_path=_load_onnx_model(MODEL_ID,CACHE,req.threads,revision)
    names=[item.name for item in session.get_inputs()]
    if not names or 'input_ids' not in names:raise RuntimeError('ONNX input schema missing input_ids')
    def model_call(text,which):
        begun=clock()
        x=tokenizer(text,max_length=req.max_length,truncation=True,return_tensors='np')
        inputs={k:np.asarray(x[k],dtype=np.int64) for k in names}
        if which=='baseline':
            pt={k:torch.from_numpy(v) for k,v in inputs.items()}
        forward_start=clock()
        if which=='baseline':
            with torch.inference_mode():output=pt_model(**pt).logits
            pred=int(torch.argmax(output,dim=-1).item())
        else:
            logits=session.run(None,inputs)[0]
            pred=int(np.asarray(logits).argmax(axis=-1).item())
        forward_end=clock()
        return pred,(forward_end-forward_start)*1000,(forward_end-begun)*1000
    # Warm up outside all measurements.
    for text,_ in rows[:3]:
        model_call(text,'baseline');model_call(text,'candidate')
    setup_ms=round((clock()-setup_start)*1000,2)
    yield event('execute','Both variants warmed. Measuring tokenizer + model and warm model forward separately',completed=0,total=req.rounds*len(rows))
    measures={which:{'warm_forward':[], 'tokenizer_plus_forward':[]} for which in ('baseline','candidate')}
    predictions_first={}
    per_round=[]
    for repeat in range(req.rounds):
        forward_round={'baseline':[],'candidate':[]}; pipeline_round={'baseline':[],'candidate':[]}
        for idx,(txt,label) in enumerate(rows):
            # Reverse ordering across samples and rounds to reduce order bias.
            order=('baseline','candidate') if (idx+repeat)%2==0 else ('candidate','baseline')
            preds={}
            for which in order:
                pred,inf,local=model_call(txt,which)
                preds[which]=pred
                measures[which]['warm_forward'].append(inf)
                measures[which]['tokenizer_plus_forward'].append(local)
                forward_round[which].append(inf)
                pipeline_round[which].append(local)
            if repeat==0:
                predictions_first[idx]=(preds['baseline'],preds['candidate'],int(label))
            elif (preds['baseline'],preds['candidate']) != predictions_first[idx][:2]:
                raise RuntimeError('Prediction instability between repeated rounds; comparison refused')
            done=repeat*len(rows)+idx+1
            if done%max(1,len(rows)//10)==0 or done==req.rounds*len(rows):
                yield event('execute',f'Paired measurement {done}/{req.rounds*len(rows)}',completed=done,total=req.rounds*len(rows),round=repeat+1)
        per_round.append({'round':repeat+1,
            'baseline_forward_mean_ms':round(statistics.mean(forward_round['baseline']),4),
            'candidate_forward_mean_ms':round(statistics.mean(forward_round['candidate']),4),
            'baseline_local_pipeline_mean_ms':round(statistics.mean(pipeline_round['baseline']),4),
            'candidate_local_pipeline_mean_ms':round(statistics.mean(pipeline_round['candidate']),4)})
    baseline_correct=sum(int(a==y) for a,b,y in predictions_first.values())
    candidate_correct=sum(int(b==y) for a,b,y in predictions_first.values())
    agreement=sum(int(a==b) for a,b,y in predictions_first.values())/len(rows)
    summaries={}
    for which,successes in [('baseline',baseline_correct),('candidate',candidate_correct)]:
        summaries[which]={'warm_forward':_summarize(measures[which]['warm_forward']),
            'tokenizer_plus_forward':_summarize(measures[which]['tokenizer_plus_forward']),
            'accuracy_on_reference_labels':round(successes/len(rows),5),
            'accuracy_on_supplied_labels':round(successes/len(rows),5), # backward compatible display
            'mean_inference_ms':_summarize(measures[which]['warm_forward'])['mean_ms'],
            'p95_ms':_summarize(measures[which]['warm_forward'])['p95_ms'],
            'correct_unique':successes}
    baseline=summaries['baseline'];candidate=summaries['candidate']
    decision=gate_v21(baseline,candidate,agreement,per_round,req)
    yield event('verify','Checking disjoint heldout accuracy, agreement, P95, local pipeline and round-by-round speed')
    artifact=None
    status='rejected'
    if decision['eligible']:
        status='eligible_for_review'
        if req.export_variant:
            yield event('verify','Staging ONNX graph and tokenizer for manual review; not deploying the model')
            staging=stage_candidate or _stage_candidate
            artifact=staging(graph_path,tokenizer,revision,req,run_id,sample_digest)
    comparison={
        'baseline_model':MODEL_ID+' · PyTorch FP32',
        'candidate_model':MODEL_ID+' · ONNX Runtime FP32',
        'sample_count':len(rows),'measurement_count':len(rows)*req.rounds,'rounds':req.rounds,
        'baseline':baseline,'candidate':candidate,'per_round':per_round,
        'prediction_agreement_fraction':round(agreement,5),
        'mean_latency_reduction_fraction':decision['mean_speedup_fraction'],
        'local_pipeline_speedup_fraction':decision['local_pipeline_speedup_fraction'],
        'all_labels_present':True,'quality_audited':False,
        'criterion':'Same model and IMDb test holdout, alternating paired warm CPU passes; local tokenization+forward separately measured',
        'partition':'imdb_test_fixed_seed_disjoint_from_dev','partition_digest':sample_digest,
        'model_revision':revision,'baseline_state_dict_bytes':weight_bytes(pt_model),
        'onnx_artifact_bytes':Path(graph_path).stat().st_size,
        'baseline_mean_inference_ms':baseline['mean_inference_ms'],
        'candidate_mean_inference_ms':candidate['mean_inference_ms']}
    report={'workload':'classification','route':'same_model_onnx_fp32_heldout',
        'result':{'label':f'{len(rows)} heldout inputs · {req.rounds} repetitions'},
        'metrics':{'end_to_end_ms':round((clock()-start)*1000,2), 'model_setup_and_warmup_ms':setup_ms},
        'comparison':comparison,
        'optimization':{'kind':'onnx_runtime_fp32','validation':'heldout_repeated_v21',
                        'approval':status,'gate':decision,'export':artifact},
        'guardrail':{'action':'HELDOUT_REFERENCE_LABELS_UNAUDITED',
                     'reason':'Official IMDb reference labels on disjoint local evaluation set; not external certification'},
        'measurement_note':'Model and tokenizer same pinned revision; 800 separate from 200 development samples; mean and P95 are repeated CPU forward. Local pipeline includes tokenizer+forward (not HTTP), independent auditing and hardware cost not measured.'}
    record={'schema_version':'v21.0','run_id':run_id,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'workload':'classification','phase':'heldout','model_id':MODEL_ID,'model_revision':revision,
        'dataset':DATASET_ID,'dataset_partition':source,'dataset_fingerprint':sample_digest,
        'sample_count':len(rows),'rounds':req.rounds,'seed':req.seed,'max_length':req.max_length,
        'thread_count':req.threads,'quality_certified':False,
        'hardware':{'machine':platform.machine(),'processor':platform.processor(),'system':platform.system()},
        'python_version':sys.version.split()[0], 'pytorch_version':torch.__version__,
        'onnxruntime_version':__import__('onnxruntime').__version__,
        'optimization':report['optimization'],'comparison':comparison,'financial_cost':None}
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open('a',encoding='utf8') as f:
        f.write(json.dumps(record,ensure_ascii=False)+'\n')
    yield event('deliver',f'Validation {status.upper()}. Decision and evidence saved; deployment unchanged',decision=decision['reason'])
    yield {'type':'result','report':report}


@router.get('/catalog')
def catalog():
    return {'model':MODEL_ID,'phase':'heldout','default_sample_count':800,'rounds':3,
            'dataset':DATASET_ID,'deployment_policy':'never_auto_deploy',
            'candidate':'onnx_runtime_fp32','gate':'quality+mean+P95+tokenizer_plus_forward+repeatability'}


@router.get('/recent')
def recent(limit:int=20):
    limit=max(1,min(int(limit),100))
    records=[]
    if LOG.exists():
        for line in LOG.read_text(encoding='utf8').splitlines()[-limit:]:
            try:records.append(json.loads(line))
            except (TypeError,ValueError):continue
    return {'records':records,'evidence_kind':'local_repeated_heldout_validation'}


@router.post('/run')
def run(req:ValidationRequest):
    try:validate_request(req)
    except ValueError as e:raise HTTPException(status_code=422,detail=str(e)) from e
    def stream():
        try:
            for packet in validate_onnx(req):
                yield json.dumps(packet,ensure_ascii=False)+'\n'
        except Exception as e:
            yield json.dumps({'type':'error','message':f'{type(e).__name__}: {e}'})+'\n'
    return StreamingResponse(stream(),media_type='application/x-ndjson',headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--samples',type=int,default=800)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--no-export',action='store_true')
    args=parser.parse_args()
    req=ValidationRequest(sample_count=args.samples,rounds=args.rounds,export_variant=not args.no_export)
    for row in validate_onnx(req):
        if row['type']=='event':print(f"{row['stage']:7s} | {row['message']}",flush=True)
        else:print(json.dumps(row['report'],indent=2,ensure_ascii=False))
