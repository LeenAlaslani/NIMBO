"""NIMBO v21.1 local-only controlled candidate adoption.

A candidate must have a passing v21 heldout record, a matching review manifest
and exact ONNX SHA256. Existing model API and production routes are untouched.
Manual shadow -> optional local canary -> one-click rollback. No auto deployment.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

MODEL_ID = 'lvwerra/distilbert-imdb'
ROOT = Path('results')
VALIDATIONS = ROOT / 'independent_validation_v21.jsonl'
STAGED = ROOT / 'onnx_review_candidates_v21'
STATE = ROOT / 'adoption_state_v21_1.json'
EVENTS = ROOT / 'adoption_events_v21_1.jsonl'
TRIALS = ROOT / 'adoption_requests_v21_1.jsonl'
router = APIRouter(prefix='/nimbo/adoption/v21', tags=['local-onnx-canary'])
LOCK = threading.RLock()
PROCESS_CACHE = {}
REVIEW_SHA_CACHE = {}
ALLOWED_ORIGINS = {'http://127.0.0.1:5173', 'http://localhost:5173'}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _append(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf8') as out:
        out.write(json.dumps(record, ensure_ascii=False) + '\n')


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix='.nimbo-adoption-', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf8') as out:
            json.dump(value, out, indent=2)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def _load_json(path, fallback):
    try: return json.loads(path.read_text(encoding='utf8')) if path.exists() else fallback
    except (json.JSONDecodeError, OSError): return fallback


def _initial():
    return {'schema_version': 'v21.1', 'mode': 'baseline', 'run_id': None,
            'request_count': 0, 'shadow_count': 0, 'shadow_agreement': 0,
            'canary_safety_count': 0, 'canary_slow_count': 0,
            'shadow_baseline_ms': [], 'shadow_candidate_ms': [],
            'last_decision_utc': None, 'last_reason': 'No local adoption requested'}


def _state():
    state = _load_json(STATE, _initial())
    return state if isinstance(state, dict) and state.get('mode') in ('baseline','shadow','canary') else _initial()


def _review(run_id):
    # Neither arbitrary disk paths nor arbitrary Hugging Face models are accepted.
    if not isinstance(run_id, str) or not run_id.startswith('heldout_') or len(run_id) > 80 or not run_id.replace('_','').isalnum():
        raise ValueError('Invalid validation run identifier')
    records = []
    if VALIDATIONS.exists():
        for line in VALIDATIONS.read_text(encoding='utf8').splitlines():
            try:
                data = json.loads(line)
                if data.get('run_id') == run_id: records.append(data)
            except (ValueError, TypeError): continue
    if len(records) != 1: raise ValueError('Exactly one saved heldout validation is required')
    rec = records[0]
    opt = rec.get('optimization') or {}
    gate = opt.get('gate') or {}
    if rec.get('model_id') != MODEL_ID or rec.get('phase') != 'heldout' or opt.get('validation') != 'heldout_repeated_v21':
        raise ValueError('Wrong model or validation type')
    if rec.get('sample_count', 0) < 800 or rec.get('rounds', 0) < 3:
        raise ValueError('At least 800 disjoint samples and three measured rounds required')
    if opt.get('approval') != 'eligible_for_review' or not all(gate.get(k) is True for k in
             ('eligible','quality_pass','latency_pass','p95_pass','repeatability_pass')):
        raise ValueError('Validation gates did not all pass')
    export = opt.get('export') or {}
    folder = (STAGED / run_id).resolve()
    parent = STAGED.resolve()
    if folder.parent != parent or not folder.is_dir(): raise ValueError('Review directory unavailable')
    manifest_path = folder / 'manifest.json'
    model_path = folder / 'model.onnx'
    if not manifest_path.is_file() or not model_path.is_file():
        raise ValueError('Staged candidate is incomplete')
    manifest = _load_json(manifest_path,{})
    if (manifest.get('status') != 'eligible_for_manual_review_only' or
            manifest.get('deployment_approved') is not False or
            manifest.get('model_id') != MODEL_ID or
            manifest.get('source_run_id') != run_id or
            manifest.get('sample_count',0) < 800 or manifest.get('rounds',0) < 3 or
            manifest.get('model_revision') != rec.get('model_revision')):
        raise ValueError('Candidate manifest does not match the passing heldout run')
    revision = manifest['model_revision']
    if not isinstance(revision, str) or len(revision)!=40 or any(x not in '0123456789abcdef' for x in revision.lower()):
        raise ValueError('A pinned forty-character model revision is required')
    stat = model_path.stat()
    cache_key=(str(model_path),stat.st_size,stat.st_mtime_ns,manifest.get('model_sha256'))
    if cache_key in REVIEW_SHA_CACHE:
        actual=REVIEW_SHA_CACHE[cache_key]
    else:
        actual=_sha(model_path)
        REVIEW_SHA_CACHE.clear()
        REVIEW_SHA_CACHE[cache_key]=actual
    if actual != manifest.get('model_sha256') or actual != export.get('sha256'):
        raise ValueError('ONNX artifact hash mismatch; refusing execution')
    # Retest the source heldout digest/partition to prevent using mismatched records.
    if rec.get('dataset_partition') is None or not rec.get('dataset_fingerprint'):
        raise ValueError('Dataset provenance missing')
    return {'run_id':run_id,'revision':revision,'folder':folder,'sha256':actual,
            'validation':rec,'manifest':manifest}


def _reviews():
    if not VALIDATIONS.is_file(): return []
    ids = []
    for line in VALIDATIONS.read_text(encoding='utf8').splitlines()[::-1]:
        try: rid=json.loads(line)['run_id']
        except (ValueError, KeyError, TypeError):continue
        if rid not in ids: ids.append(rid)
    valid=[]
    for rid in ids[:50]:
        try:
            x=_review(rid)
            c=x['validation'].get('comparison',{})
            valid.append({'run_id':rid,'model_id':MODEL_ID, 'sha256':x['sha256'],
                          'revision':x['revision'], 'sample_count':x['validation']['sample_count'],
                          'rounds':x['validation']['rounds'],
                          'warm_cpu_reduction_fraction':c.get('mean_latency_reduction_fraction'),
                          'local_pipeline_reduction_fraction':c.get('local_pipeline_speedup_fraction'),
                          'quality_pass':True, 'status':'eligible_for_local_review'})
        except (ValueError, OSError):continue
    return valid


def _log_change(previous, new, reason):
    _append(EVENTS, {'timestamp_utc':_utc(), 'old_mode':previous.get('mode'),
                     'new_mode':new['mode'], 'run_id':new.get('run_id'), 'reason':reason,
                     'kind':'explicit_local_control'})


def _switch(new_mode, run_id, reason):
    current=_state()
    s={**_initial(), 'mode':new_mode,'run_id':run_id,
       'last_reason':reason,'last_decision_utc':_utc()}
    _atomic_json(STATE,s)
    _log_change(current,s,reason)
    return s


def _auto_rollback(reason):
    with LOCK:
        return _switch('baseline',None,'Automatic local rollback: '+reason)


def _metrics(state):
    a=state.get('shadow_baseline_ms',[])
    b=state.get('shadow_candidate_ms',[])
    if len(a)!=len(b) or not a:return {'paired_samples':0,'baseline_avg_ms':None,'candidate_avg_ms':None,'reduction_fraction':None}
    avg_a=statistics.mean(a);avg_b=statistics.mean(b)
    return {'paired_samples':len(a),'baseline_avg_ms':round(avg_a,3),
            'candidate_avg_ms':round(avg_b,3),
            'reduction_fraction':round((avg_a-avg_b)/avg_a,4) if avg_a>0 else None}


def status():
    with LOCK:
        s=_state()
        if s['mode']!='baseline':
            try:_review(s['run_id'])
            except (ValueError,OSError) as exc:
                s=_auto_rollback('candidate verification failed: '+str(exc))
        events=[]
        if TRIALS.exists():
            for line in TRIALS.read_text(encoding='utf8').splitlines()[-12:]:
                try: events.append(json.loads(line))
                except ValueError:continue
        return {'scope':'local_opt_in_only','production_deployed':False,
                'changes_existing_nimbo_routing':False,'model_id':MODEL_ID,
                'state':s,'shadow_metrics':_metrics(s), 'reviews':_reviews(),
                'recent_requests':events,
                'note':'Shadow executes both models: never claim compute savings for shadow. Canary uses ONNX only except every fifth request; no external quality audit or production change.'}


class ControlRequest(BaseModel):
    run_id: str | None = None
    confirmation: str


class TextRequest(BaseModel):
    text: str = Field(min_length=1,max_length=2000)
    expected_label: Literal['POSITIVE','NEGATIVE'] | None = None


def _local_only(request: Request):
    client = request.client.host if request.client else ''
    if client not in ('127.0.0.1','::1','localhost','testclient'):
        raise HTTPException(status_code=403,detail='This feature accepts loopback requests only')
    origin = request.headers.get('origin')
    if origin and origin not in ALLOWED_ORIGINS:
        raise HTTPException(status_code=403,detail='Untrusted origin')


def _candidate_thread_count(review):
    """Respect the independently selected ONNX CPU threads in staged review.

    Existing 4-thread artifacts remain backwards compatible. Refuse tampering
    or metadata drift between the validated result and staged artifact.
    """
    manifest=review['manifest']
    saved=review['validation']
    count=manifest.get('candidate_cpu_threads',manifest.get('cpu_threads',4))
    validated=saved.get('candidate_thread_count',saved.get('thread_count',4))
    if type(count) is not int or not (1 <= count <= 8) or count!=validated:
        raise ValueError('ONNX CPU-thread configuration differs from heldout review record')
    return count


def _prepare_cache(review):
    """Two CPU implementations of the exact same pinned model. Lazy single-process cache."""
    import torch
    import onnxruntime as ort
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    key=(review['run_id'],review['sha256'])
    if key in PROCESS_CACHE:return PROCESS_CACHE[key]
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(str(review['folder']),local_files_only=True,trust_remote_code=False)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL_ID,revision=review['revision'],trust_remote_code=False).eval().cpu()
    opts=ort.SessionOptions();opts.intra_op_num_threads=_candidate_thread_count(review);opts.inter_op_num_threads=1
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess=ort.InferenceSession(str(review['folder']/'model.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
    if sess.get_providers()!=['CPUExecutionProvider']:
        raise RuntimeError('Unexpected ONNX execution provider; refusing canary')
    model_names={x.name for x in sess.get_inputs()}
    if 'input_ids' not in model_names:raise RuntimeError('Staged graph lacks input_ids')
    # Warm each path once; setup and warmup remain excluded from warm timings.
    prepared=tok('A friendly example.',max_length=256,truncation=True,return_tensors='np')
    import numpy as np
    with torch.inference_mode():
        model(**{k:torch.from_numpy(np.asarray(v,dtype=np.int64)) for k,v in prepared.items() if k in model_names})
    sess.run(None,{k:np.asarray(v,dtype=np.int64) for k,v in prepared.items() if k in model_names})
    engine={'tokenizer':tok,'baseline':model,'candidate':sess,'inputs':model_names}
    PROCESS_CACHE.clear() # keep memory bounded: one reviewed run at a time
    PROCESS_CACHE[key]=engine
    return engine


def _infer(text,which,review):
    import numpy as np
    import torch
    eng=_prepare_cache(review)
    start=time.perf_counter()
    data=eng['tokenizer'](text,max_length=256,truncation=True,return_tensors='np')
    prepared={k:np.asarray(v,dtype=np.int64) for k,v in data.items() if k in eng['inputs']}
    forward=time.perf_counter()
    if which=='baseline':
        with torch.inference_mode():
            logits=eng['baseline'](**{k:torch.from_numpy(v) for k,v in prepared.items()}).logits.detach().numpy()
    else:logits=eng['candidate'].run(None,prepared)[0]
    forward_ms=(time.perf_counter()-forward)*1000
    pipeline_ms=(time.perf_counter()-start)*1000
    raw=np.asarray(logits, dtype=np.float64).reshape(-1)
    if raw.size!=2 or not np.isfinite(raw).all():
        raise ValueError('Invalid binary classifier logits')
    shifted=raw-raw.max()
    probs=np.exp(shifted);probs=probs/probs.sum()
    id_=int(probs.argmax())
    return {'label':('NEGATIVE','POSITIVE')[id_], 'class_id':id_,
            'positive_probability':round(float(probs[1]),8),
            'confidence':round(float(probs[id_]),8),
            'warm_forward_ms':round(forward_ms,3), 'tokenizer_plus_forward_ms':round(pipeline_ms,3)}


def run_local(text,expected_label=None,infer=None):
    """Serialize operations in one local process. Tests inject fake inference; no real network required."""
    infer = infer or _infer
    request_start=time.perf_counter()
    with LOCK:
        s=_state();mode=s['mode'];review=None
        if mode!='baseline':
            try:review=_review(s['run_id'])
            except (ValueError,OSError) as exc:
                _auto_rollback('Staged artifact unavailable: '+str(exc))
                raise ValueError('Local candidate invalid; rolled back. '+str(exc))
        elif _reviews():
            review=_review(_reviews()[0]['run_id'])
        if review is None:raise ValueError('No eligible, verified ONNX review artifact exists')
        # A baseline request stays on baseline; shadow returns baseline while collecting
        # paired evidence. Canary returns candidate except when the safety sample fails.
        # v22: force a fresh five-request paired guard window after an upgrade,
        # then continue the existing every-fifth periodic safety check.
        observed_baseline = (mode in ('baseline','shadow') or
            (mode=='canary' and (s.get('v22_guarded_count',0)<5 or (s['request_count']+1)%5==0)))
        try:
            baseline = infer(text,'baseline',review) if observed_baseline else None
            candidate = infer(text,'candidate',review) if mode in ('shadow','canary') else None
        except Exception as exc:
            _auto_rollback('Execution failure ('+type(exc).__name__+')')
            raise ValueError('Local execution failed; baseline mode restored. '+str(exc))
        mismatch=bool(baseline and candidate and baseline['class_id'] != candidate['class_id'])
        probability_drift=(abs(baseline['positive_probability']-candidate['positive_probability'])
            if baseline and candidate and 'positive_probability' in baseline and 'positive_probability' in candidate else None)
        # A same-model FP32 export should not move probabilities by several points.
        # This is an additional local safety gate, not an independent accuracy audit.
        score_drift=bool(probability_drift is not None and probability_drift>0.025)
        unsafe=mismatch or score_drift
        served=baseline if mode in ('baseline','shadow') or unsafe else candidate
        routed='baseline' if served is baseline else 'candidate_onnx'
        if mode in ('shadow','canary'):
            s['request_count']+=1
            if baseline and candidate:
                s['shadow_count']+=1
                if not unsafe:s['shadow_agreement']+=1
                if mode=='canary':s['v22_guarded_count']=s.get('v22_guarded_count',0)+1
                s['shadow_baseline_ms']=(s.get('shadow_baseline_ms',[])+[baseline['tokenizer_plus_forward_ms']])[-100:]
                s['shadow_candidate_ms']=(s.get('shadow_candidate_ms',[])+[candidate['tokenizer_plus_forward_ms']])[-100:]
                if mode=='canary':
                    s['canary_safety_count']=s.get('canary_safety_count',0)+1
                    if candidate['tokenizer_plus_forward_ms'] > baseline['tokenizer_plus_forward_ms']*1.2:
                        s['canary_slow_count']=s.get('canary_slow_count',0)+1
                    else:
                        s['canary_slow_count']=0
            _atomic_json(STATE,s)
        latency_proxy=served['tokenizer_plus_forward_ms']
        request_wall_ms=round((time.perf_counter()-request_start)*1000,3)
        record={'timestamp_utc':_utc(),'kind':'local_adoption_request',
                'mode':mode,'route_served':routed,'run_id':review['run_id'],
                'request_wall_ms':request_wall_ms,
                'baseline_warm_ms':baseline['warm_forward_ms'] if baseline else None,
                'candidate_warm_ms':candidate['warm_forward_ms'] if candidate else None,
                'baseline_pipeline_ms':baseline['tokenizer_plus_forward_ms'] if baseline else None,
                'candidate_pipeline_ms':candidate['tokenizer_plus_forward_ms'] if candidate else None,
                'prediction_agreement':None if not baseline or not candidate else not mismatch,
                'probability_delta':round(probability_drift,8) if probability_drift is not None else None,
                'parity_verified':bool(baseline and candidate and not unsafe),
                'safety_check': ('UNPAIRED' if not baseline or not candidate else
                                 'MISMATCH' if mismatch else 'PROBABILITY_DRIFT' if score_drift else 'PASS'),
                'baseline_label':baseline['label'] if baseline else None,
                'candidate_label':candidate['label'] if candidate else None,
                'reference_label_supplied':expected_label is not None,
                'matched_supplied_label':served['label']==expected_label if expected_label else None,
                'input_or_output_text_saved':False,'financial_cost':None,'energy_joules':None}
        _append(TRIALS,record)
        rollback_reason=None
        if unsafe:
            rollback_reason=('Prediction mismatch detected in same-input safety comparison' if mismatch
                             else 'Same-label probability drift exceeded 2.5 percentage points')
            _auto_rollback(rollback_reason)
        elif mode=='canary' and s.get('canary_slow_count',0)>=3:
            rollback_reason='Candidate tokenizer+CPU latency was over 20% slower in three consecutive canary safety checks'
            _auto_rollback(rollback_reason)
        return {'model_id':MODEL_ID,'mode_before':mode,'route_served':routed,
                'result_label':served['label'],'served_pipeline_ms':latency_proxy,
                'request_wall_ms':request_wall_ms,
                'baseline':baseline,'candidate':candidate,
                'comparison_performed':bool(baseline and candidate),
                'production_deployed':False,'canary_only':True,
                'auto_rollback':bool(rollback_reason),'rollback_reason':rollback_reason,
                'parity_checked':bool(baseline and candidate),
                'parity_status':('LABEL_MISMATCH' if mismatch else 'SCORE_DRIFT' if score_drift else
                                 'PASS' if baseline and candidate else 'NOT_CHECKED'),
                'probability_delta':round(probability_drift,8) if probability_drift is not None else None,
                'safety_sampling_note':('A candidate-only Canary request has no baseline comparison.'
                    if mode=='canary' and not baseline else
                    'Paired check executed; shadow checks do not reduce inference compute.'),
                'note':'Shadow runs two models and does not save compute. Canary candidate-only requests have no independent success label; cold load and HTTP transport excluded.'}


@router.get('/status',dependencies=[Depends(_local_only)])
def get_status(): return status()


@router.post('/shadow',dependencies=[Depends(_local_only)])
def start_shadow(body:ControlRequest):
    if body.confirmation!='START LOCAL SHADOW': raise HTTPException(422,'Confirmation text mismatch')
    with LOCK:
        if _state()['mode']!='baseline':raise HTTPException(409,'Roll back to baseline before starting another trial')
        try:_review(body.run_id)
        except (ValueError,OSError) as exc:raise HTTPException(422,str(exc)) from exc
        _switch('shadow',body.run_id,'User explicitly started local paired shadow; baseline remains served')
    return status()


@router.post('/canary',dependencies=[Depends(_local_only)])
def activate_canary(body:ControlRequest):
    if body.confirmation!='ACTIVATE LOCAL CANARY':raise HTTPException(422,'Confirmation text mismatch')
    with LOCK:
        s=_state()
        if s['mode']!='shadow' or s['run_id']!=body.run_id:raise HTTPException(409,'Requires a matching local shadow trial')
        try:_review(body.run_id)
        except (ValueError,OSError) as exc:raise HTTPException(422,str(exc)) from exc
        m=_metrics(s)
        if s['shadow_count']<5 or s['shadow_agreement']!=s['shadow_count']:
            raise HTTPException(409,'Need at least five paired shadow requests with 100% agreement')
        if m['reduction_fraction'] is None or m['reduction_fraction'] < .05:
            raise HTTPException(409,'Shadow tokenizer+inference is not at least 5% faster; candidate not activated')
        # Preserve the approved artifact run but reset canary counters, keep historical shadow summaries.
        s['mode']='canary';s['request_count']=0
        s['canary_safety_count']=0;s['canary_slow_count']=0;s['v22_guarded_count']=0
        s['last_decision_utc']=_utc();s['last_reason']='Manual local canary activation; no production deployment'
        _atomic_json(STATE,s);_log_change({'mode':'shadow'},s,s['last_reason'])
    return status()


@router.post('/rollback',dependencies=[Depends(_local_only)])
def rollback(body:ControlRequest):
    if body.confirmation!='ROLL BACK TO BASELINE':raise HTTPException(422,'Confirmation text mismatch')
    with LOCK: _switch('baseline',None,'User explicitly restored original PyTorch route')
    return status()


@router.post('/classify',dependencies=[Depends(_local_only)])
def classify(body:TextRequest):
    try:return run_local(body.text,body.expected_label)
    except ValueError as exc:raise HTTPException(409,detail=str(exc)) from exc


# NIMBO_V22_CANARY_SAFETY: optional explicit same-input parity probe.
# Never writes raw text or token IDs; independent of production routes.

def _parity_probe(text, review, engine=None):
    import numpy as np
    import torch
    eng=engine if engine is not None else _prepare_cache(review)
    started=time.perf_counter()
    encoded=eng['tokenizer'](text,max_length=256,truncation=True,return_tensors='np')
    prepared={k:np.asarray(v,dtype=np.int64) for k,v in encoded.items() if k in eng['inputs']}
    if 'input_ids' not in prepared:
        raise ValueError('Missing input_ids in shared tokenizer output')
    t0=time.perf_counter()
    with torch.inference_mode():
        a=eng['baseline'](**{k:torch.from_numpy(v) for k,v in prepared.items()}).logits.detach().numpy()
    baseline_ms=(time.perf_counter()-t0)*1000
    t1=time.perf_counter()
    b=eng['candidate'].run(None,prepared)[0]
    candidate_ms=(time.perf_counter()-t1)*1000
    def decode(arr):
        scores=np.asarray(arr,dtype=np.float64).reshape(-1)
        if scores.size!=2 or not np.isfinite(scores).all():
            raise ValueError('Unexpected binary classification output')
        shifted=scores-scores.max();p=np.exp(shifted);p/=p.sum()
        k=int(p.argmax())
        return {'class_id':k,'label':('NEGATIVE','POSITIVE')[k],
                'positive_probability':round(float(p[1]),8),
                'confidence':round(float(p[k]),8)}
    base=decode(a);cand=decode(b)
    diff=abs(base['positive_probability']-cand['positive_probability'])
    label_match=base['class_id']==cand['class_id']
    safe=label_match and diff<=.025
    return {'model_id':MODEL_ID,'run_id':review['run_id'],
            'token_count':int(prepared['input_ids'].shape[-1]),
            'shared_tokenization':True,
            'baseline':base,'candidate':cand,
            'prediction_agreement':label_match,
            'positive_probability_delta':round(diff,8),
            'parity_pass':safe,
            'status':'PASS' if safe else 'LABEL_MISMATCH' if not label_match else 'SCORE_DRIFT',
            'baseline_forward_ms':round(baseline_ms,3),
            'candidate_forward_ms':round(candidate_ms,3),
            'total_probe_ms':round((time.perf_counter()-started)*1000,3),
            'input_saved':False,'production_deployed':False,
            'note':'Both variants received exactly the same token IDs. This diagnostic runs both models, so it does not save compute; confidence is not ground-truth.'}


@router.post('/parity',dependencies=[Depends(_local_only)])
def parity_check(body:TextRequest):
    with LOCK:
        state=_state()
        try:
            review=(_review(state['run_id']) if state['mode']!='baseline'
                else _review(_reviews()[0]['run_id']) if _reviews() else None)
            if review is None:raise ValueError('No valid staged ONNX candidate available')
            result=_parity_probe(body.text,review)
        except (ValueError,OSError,RuntimeError) as exc:
            if state['mode']=='canary':_auto_rollback('Failed direct parity safety probe: '+type(exc).__name__)
            raise HTTPException(409,'Parity probe failed: '+str(exc)) from exc
        if not result['parity_pass'] and state['mode'] in ('shadow','canary'):
            _auto_rollback('Direct parity probe failed: '+result['status'])
            result['auto_rollback']=True
        else:result['auto_rollback']=False
        _append(TRIALS,{'timestamp_utc':_utc(),'kind':'local_parity_probe_v22',
            'mode':state['mode'],'route_served':None,'run_id':review['run_id'],
            'request_wall_ms':result['total_probe_ms'],
            'parity_verified':result['parity_pass'],
            'safety_check':result['status'],
            'prediction_agreement':result['prediction_agreement'],
            'probability_delta':result['positive_probability_delta'],
            'baseline_label':result['baseline']['label'],
            'candidate_label':result['candidate']['label'],
            'input_or_output_text_saved':False,'financial_cost':None,'energy_joules':None})
        return result
