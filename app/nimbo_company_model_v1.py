"""Local-only, *real* evaluation of a user-uploaded supported ONNX model.

Security boundary: only the SHA-256-pinned ONNX review artifact of the pinned
public DistilBERT-IMDb classifier can be staged. The actual staged bytes are
loaded with ONNX Runtime and evaluated on user-provided labeled examples.
This does not deploy or register arbitrary company models, nor re-run official
held-out judging results. No raw customer texts are written to disk.
"""
from __future__ import annotations
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from app.nimbo_demo_artifact_v1 import EXPECTED_SHA256, MAX_BYTES, MIN_BYTES, _local_only

router = APIRouter(prefix='/nimbo/demo/company-model', tags=['company-model-pilot'])
MODEL_ID='lvwerra/distilbert-imdb'
MODEL_REVISION='0fc02cd68445b599a9cb2da2368050e7fb31d29a'
STAGING=Path('results/company_model_pilot_v1')
CANDIDATE=STAGING/'supported_company_candidate.onnx'
MAX_ROWS=30

class Example(BaseModel):
    text: str = Field(min_length=1,max_length=2000)
    expected_label: str

class PilotRequest(BaseModel):
    examples: list[Example] = Field(min_length=2,max_length=MAX_ROWS)

def validate_rows(raw):
    rows=[]
    for x in raw:
        val=x.expected_label.strip().upper()
        if val not in ('POSITIVE','NEGATIVE') or not x.text.strip():
            raise ValueError('Every example must have a nonempty text and POSITIVE/NEGATIVE label')
        rows.append((x.text, 1 if val=='POSITIVE' else 0))
    return rows

def is_staged_and_verified(path=None):
    path=CANDIDATE if path is None else path
    try:
        if path.is_symlink() or not path.is_file() or not MIN_BYTES<=path.stat().st_size<=MAX_BYTES:
            return False
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
        return digest.hexdigest()==EXPECTED_SHA256
    except OSError:
        return False

@router.get('/status')
def get_status(request:Request):
    _local_only(request)
    return {'imported':is_staged_and_verified(), 'supported_model':MODEL_ID,
            'revision':MODEL_REVISION,'format':'ONNX FP32',
            'supported_only':True,'deployed':False,
            'note':'Imported model is loaded for local pilot inference only; it is not activated in production.'}

@router.post('/import')
async def import_model(request:Request):
    _local_only(request)
    if request.headers.get('content-type','').split(';')[0].strip().lower() != 'application/octet-stream':
        raise HTTPException(415,'Upload the raw .onnx file (application/octet-stream)')
    STAGING.mkdir(parents=True,exist_ok=True)
    tmp=STAGING/('incoming_'+str(os.getpid())+'_'+str(time.time_ns())+'.tmp')
    h=hashlib.sha256();size=0
    try:
        with tmp.open('xb') as f:
            async for chunk in request.stream():
                size+=len(chunk)
                if size>MAX_BYTES:raise HTTPException(413,'ONNX model exceeds 420 MB')
                f.write(chunk);h.update(chunk)
        if size<MIN_BYTES or h.hexdigest()!=EXPECTED_SHA256:
            raise HTTPException(422,'Model is real but not the supported pinned IMDb ONNX example; no arbitrary ONNX execution is permitted.')
        os.chmod(tmp,0o600)
        os.replace(tmp,CANDIDATE)
        return {'imported':True,'sha256':h.hexdigest(),'bytes':size,
                'model':MODEL_ID,'deployed':False,'executed':False,
                'note':'Supported ONNX file staged. Run the labeled pilot evaluation to execute these uploaded bytes.'}
    finally:
        tmp.unlink(missing_ok=True)

def _mean(values):return round(statistics.mean(values),4)

def real_pilot_inference(rows,model_path=CANDIDATE):
    """Actually executes staged candidate file; compute times are exploratory."""
    import numpy as np
    import torch
    import onnxruntime as ort
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    old_threads=torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION,trust_remote_code=False)
        baseline=AutoModelForSequenceClassification.from_pretrained(MODEL_ID,revision=MODEL_REVISION,trust_remote_code=False).cpu().eval()
        options=ort.SessionOptions()
        options.intra_op_num_threads=2
        options.inter_op_num_threads=1
        session=ort.InferenceSession(str(model_path),sess_options=options,providers=['CPUExecutionProvider'])
        if session.get_providers()!=['CPUExecutionProvider']:
            raise RuntimeError('Only CPUExecutionProvider is supported')
        inputs={x.name for x in session.get_inputs()}
        if not {'input_ids','attention_mask'}.issubset(inputs) or not inputs.issubset({'input_ids','attention_mask','token_type_ids'}):
            raise RuntimeError('Unexpected ONNX model input signature')
        timings={'baseline':[],'candidate':[]}
        counts={'baseline':0,'candidate':0};agreement=0
        prepared=[]
        for text,label in rows:
            token=tokenizer(text,truncation=True,max_length=128,return_tensors='np')
            feeds={k:np.asarray(token[k],dtype=np.int64) for k in inputs}
            prepared.append((feeds,label))
        with torch.inference_mode():
            for feeds,_ in prepared[:2]:
                baseline(**{k:torch.from_numpy(v) for k,v in feeds.items()})
                session.run(None,feeds)
            for idx,(feeds,label) in enumerate(prepared):
                preds={}
                for mode in (('baseline','candidate') if idx%2==0 else ('candidate','baseline')):
                    start=time.perf_counter_ns()
                    if mode=='baseline':
                        arr=baseline(**{k:torch.from_numpy(v) for k,v in feeds.items()}).logits
                        pred=int(torch.argmax(arr,dim=-1).item())
                    else:
                        arr=session.run(None,feeds)[0]
                        pred=int(np.asarray(arr).argmax(axis=-1).item())
                    duration=(time.perf_counter_ns()-start)/1e6
                    timings[mode].append(duration)
                    counts[mode]+=int(pred==label)
                    preds[mode]=pred
                agreement+=int(preds['baseline']==preds['candidate'])
        n=len(rows)
        baseline_ms=_mean(timings['baseline']);candidate_ms=_mean(timings['candidate'])
        return {'sample_count':n,'baseline':{'accuracy':round(counts['baseline']/n,4),'mean_warm_ms':baseline_ms},
                'candidate':{'accuracy':round(counts['candidate']/n,4),'mean_warm_ms':candidate_ms},
                'prediction_agreement':round(agreement/n,4),
                'warm_latency_reduction_fraction':round((baseline_ms-candidate_ms)/baseline_ms,5) if baseline_ms else None}
    finally:
        torch.set_num_threads(old_threads)

@router.post('/evaluate')
def evaluate(request:Request,payload:PilotRequest):
    _local_only(request)
    if not is_staged_and_verified():
        raise HTTPException(409,'Import a supported real ONNX file before evaluating it')
    try:rows=validate_rows(payload.examples)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    start=time.perf_counter()
    try:result=real_pilot_inference(rows)
    except (RuntimeError,ValueError,ImportError,OSError) as exc:
        raise HTTPException(503,'Real model evaluation failed: '+str(exc)) from exc
    report={'status':'COMPLETED_LOCAL_PILOT','scope':'actual uploaded ONNX file vs original pinned PyTorch model',
            'model':MODEL_ID,'model_revision':MODEL_REVISION,'source_artifact_sha256':EXPECTED_SHA256,
            'evaluation':result,'total_elapsed_ms':round((time.perf_counter()-start)*1000,2),
            'assessment':'EXPLORATORY_ONLY_NOT_HELDOUT_CERTIFICATION','deployed':False,
            'note':'Real inference on uploaded, allowlisted ONNX bytes; labeled small dataset provided by user. No change to model registry or live route. Warm timing excludes setup; no electricity, billing or CPU-saving claim.'}
    # Aggregate receipt only; no user text or predictions saved to disk.
    STAGING.mkdir(exist_ok=True,parents=True)
    (STAGING/'latest_pilot_summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return report
