"""VERA automatic image triage added to the actual NIMBO v24 route.

No image/prompt content saved; no model inference fabricated. The v24 human
manual decisions remain a fallback, and automated acceptance is NOT audited.
"""
from __future__ import annotations
import base64
import hashlib
import importlib.util
import math
import threading
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app import nimbo_adaptive_image_v24 as adaptive
from app import nimbo_vera_v25 as engine

router=APIRouter(prefix='/nimbo/image/vera/v25',tags=['VERA local image verification'])

class VerifyRequest(BaseModel):
    run_id:str=Field(min_length=40,max_length=45)
    prompt:str=Field(min_length=3,max_length=500)
    image_base64:str=Field(min_length=50,max_length=engine.MAX_IMAGE_B64)


def _check(rec,req,kind):
    if not rec:raise HTTPException(404,'Adaptive run does not exist')
    expected='awaiting_human' if kind=='fast' else 'fallback_generated'
    if rec.get('phase')!=expected:
        raise HTTPException(409,f'Run is not in {expected} state; start a new mission if already reviewed')
    if hashlib.sha256(req.prompt.encode('utf8')).hexdigest()!=rec.get('prompt_sha256'):
        raise HTTPException(409,'Request prompt does not match the original generation')
    try:
        _,raw=engine._base64_image(req.image_base64)
    except Exception as exc:
        raise HTTPException(422,'Invalid generated PNG supplied to VERA') from exc
    digest=hashlib.sha256(raw).hexdigest()
    key='fast_image_sha256' if kind=='fast' else 'standard_image_sha256'
    if not rec.get(key):
        raise HTTPException(409,'Image digest is not available for this older mission. Start a new adaptive image mission')
    if digest!=rec[key]:raise HTTPException(409,'Image differs from the image generated for this exact run')


def _wrap(rec,result,payload=None):
    response=adaptive._payload(rec,
            image_result=(payload or {}).get('result') if payload else None,
            trace=(payload or {}).get('trace') if payload else None)
    response['vera']=result
    response['adaptive'].update({'vera_status':result['verdict'],'vera_route_advice':result['route_advice'],
                                'verification_source':'owlv2_local_development',
                                'independently_audited':False})
    response['metrics'].update({'vera_fast_wall_ms':rec.get('vera_fast',{}).get('verification_ms'),
                                'vera_standard_wall_ms':rec.get('vera_standard',{}).get('verification_ms'),
                                'request_wall_fast_ms':rec.get('fast_request_wall_ms'),
                                'request_wall_standard_ms':rec.get('standard_request_wall_ms')})
    fw=rec.get('fast_request_wall_ms');sw=rec.get('standard_request_wall_ms')
    if isinstance(fw,(int,float)):
        response['metrics']['executed_generation_request_wall_ms']=round(fw+(sw or 0),3)
    return response


@router.get('/health')
def health():
    return {'detector':engine.MODEL_ID,'installed':importlib.util.find_spec('transformers') is not None,
            'mode':'real_local_zero_shot_detection','device_preference':engine.os.getenv('NIMBO_VERA_DEVICE','cpu'),
            'loaded':engine._load.cache_info().currsize>0,
            'conservative_manual_review_for_uncertain':True,
            'independently_audited':False}


@router.post('/fast')
def verify_fast(req:VerifyRequest):
    with adaptive.LOCK:
        previous=adaptive._last(req.run_id)
        _check(previous,req,'fast')
    result=engine.verify(req.image_base64,req.prompt)
    with adaptive.LOCK:
        current=adaptive._last(req.run_id)
        _check(current,req,'fast')
        if result['route_advice'] in ('accept_fast','fallback'):
            decision='accept_fast' if result['route_advice']=='accept_fast' else 'fallback'
            response=adaptive.decide(adaptive.Decide(run_id=req.run_id,prompt=req.prompt,
                             decision=decision,fast_object_present=True if decision=='accept_fast' else None))
            latest=adaptive._last(req.run_id)
            latest={**latest,'vera_fast':result,'verification_source':'automatic_owlv2_unverified',
                    'human_label_source':None}
            adaptive._append(latest)
            body=_wrap(latest,result,response)
            body['result']=(response or {}).get('result',{})
            return body
        latest={**current,'vera_fast':result,'verification_source':'automatic_owlv2_unverified',
               'at_utc':adaptive._utc()}
        adaptive._append(latest)
        return _wrap(latest,result)


@router.post('/standard')
def verify_standard(req:VerifyRequest):
    with adaptive.LOCK:
        previous=adaptive._last(req.run_id)
        _check(previous,req,'standard')
    result=engine.verify(req.image_base64,req.prompt)
    with adaptive.LOCK:
        current=adaptive._last(req.run_id)
        _check(current,req,'standard')
        if result['verdict'] in ('PASS','FAIL'):
            adaptive.review_standard(adaptive.FinalReview(run_id=req.run_id,
                                              standard_object_present=result['verdict']=='PASS'))
            latest=adaptive._last(req.run_id)
            latest={**latest,'vera_standard':result,'verification_source':'automatic_owlv2_unverified',
                    'human_label_source':None}
            adaptive._append(latest)
        else:
            latest={**current,'vera_standard':result,'verification_source':'automatic_owlv2_unverified',
                    'at_utc':adaptive._utc()}
            adaptive._append(latest)
        return _wrap(latest,result)

@router.get('/evidence')
def evidence():
    """Measured run counts, never inferred dollar, energy, or quality savings."""
    states={}
    for rec in adaptive._read():
        states[rec['run_id']]=rec
    auto=[r for r in states.values() if r.get('vera_fast') or r.get('vera_standard')]
    finished=[r for r in auto if r.get('phase') in ('accepted_fast','fallback_reviewed')]
    verifier_total_ms=sum((r.get('vera_fast') or {}).get('verification_ms',0)+
                          (r.get('vera_standard') or {}).get('verification_ms',0) for r in auto)
    return {'automatic_checks_runs':len(auto), 'automatically_triaged_complete':len(finished),
            'automatically_accepted_fast':sum(r.get('phase')=='accepted_fast' and
                 (r.get('vera_fast') or {}).get('verdict')=='PASS' for r in auto),
            'automatic_fast_fallbacks':sum((r.get('vera_fast') or {}).get('route_advice')=='fallback' for r in auto),
            'fast_uncertain':sum((r.get('vera_fast') or {}).get('verdict')=='UNCERTAIN' for r in auto),
            'standard_failed':sum((r.get('vera_standard') or {}).get('verdict')=='FAIL' for r in auto),
            'verifier_wall_time_total_ms':round(verifier_total_ms,2),
            'independent_success_labels':0,'independently_audited':False,
            'note':'Heuristic local detection results. No certified task success, electricity, FLOPs, billing or cross-device comparison.'}
