"""NIMBO v24: real sequential, human-gated adaptive image execution.

Fast (LCM) is generated first. A human decision explicitly accepts the object
or requests a standard SD1.5 fallback; the baseline is never precomputed.
This is intentionally NOT an automatic quality detector, an independent audit,
or a claim of energy/financial savings. Prompts and PNGs are never persisted.
"""
from __future__ import annotations

import hashlib
import base64
# NIMBO_VERA_V25
import json
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import nimbo_image_connector_v22 as gateway

router = APIRouter(prefix='/nimbo/image/adaptive/v24', tags=['NIMBO adaptive image'])
FILE = Path('results/image_adaptive_runs_v24.jsonl')
LOCK = threading.RLock()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _append(entry):
    FILE.parent.mkdir(parents=True, exist_ok=True)
    with FILE.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + '\n')


def _read():
    if not FILE.exists():
        return []
    rows = []
    for line in FILE.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
            if isinstance(row, dict) and row.get('run_id'):
                rows.append(row)
        except (ValueError, TypeError):
            pass
    return rows


def _last(run_id):
    return next((r for r in reversed(_read()) if r.get('run_id') == run_id), None)


class Start(BaseModel):
    prompt: str = Field(min_length=3, max_length=500)
    seed: int = Field(default=42, ge=0, le=2**31-1)
    mode: str = 'adaptive'


class Decide(BaseModel):
    run_id: str = Field(min_length=38, max_length=48)
    prompt: str = Field(min_length=3, max_length=500)
    decision: str = Field(description='accept_fast or fallback')
    fast_object_present: bool | None = None


class FinalReview(BaseModel):
    run_id: str = Field(min_length=38, max_length=48)
    standard_object_present: bool


def _as_ms(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise HTTPException(502, 'Image gateway did not return a valid measured GPU time')
    return round(float(value), 3)


def _gateway_call(prompt, seed, mode):
    # Reuses the existing tested v22 gateway and real local worker. No fake result.
    result = gateway.generate(gateway.ImageRequest(prompt=prompt, mode=mode, seed=seed))
    if not isinstance(result, dict) or not result.get('run_id'):
        raise HTTPException(502, 'Image gateway returned no valid run')
    return result


def _payload(rec, image_result=None, trace=None):
    fast = rec['fast_inference_ms']
    standard = rec.get('standard_inference_ms')
    finished = rec['phase'] != 'awaiting_human'
    served = 'fast' if rec['phase'] == 'accepted_fast' else 'standard' if rec['phase'] in ('fallback_generated', 'fallback_reviewed') else None
    state = {'phase':rec['phase'], 'run_id':rec['run_id'], 'served_by':served,
             'fallback_used':standard is not None, 'fast_object_present':rec.get('fast_object_present'),
             'standard_object_present':rec.get('standard_object_present'),
             'review_source':rec.get('verification_source','human_self_report'), 'independently_audited':False,
             'human_review_complete':rec['phase'] in ('accepted_fast', 'fallback_reviewed') and rec.get('human_label_source') is not None,
             'quality_verified':False, 'finished':finished}
    return {
        'workload':'image', 'route':'sd15_adaptive_human_gated_gpu', 'run_id':rec['run_id'],
        'adaptive':state, 'result':image_result or {},
        'metrics':{'end_to_end_ms':round(fast+(standard or 0),3), 'fast_inference_ms':fast,
                   'standard_inference_ms':standard, 'gpu_inference_total_ms':round(fast+(standard or 0),3),
                   'device':rec['device'], 'fast_request_wall_ms':rec.get('fast_request_wall_ms'),
                   'standard_request_wall_ms':rec.get('standard_request_wall_ms')},
        'image_comparison':{'mode':'adaptive','adaptive_phase':rec['phase'],'seed':rec['seed'],'fast_steps':4,'standard_steps':25,
                            'fast_inference_ms':fast,'standard_inference_ms':standard,
                            'measured_fast_latency_reduction_fraction':None,
                            'quality_verified':False},
        'trace':trace or [],
        'measurement_note':'Sequential GPU inference time measured for executed paths only, excluding human review time. No independent quality, FLOPs, energy or dollars claim.'
    }


@router.post('/start')
def start(request:Start):
    _request_start=time.perf_counter()
    if request.mode != 'adaptive':
        raise HTTPException(422, 'This endpoint supports adaptive mode only')
    generated = _gateway_call(request.prompt, request.seed, 'fast')
    fast_ms = _as_ms(generated.get('metrics', {}).get('fast_inference_ms'))
    image = generated.get('result', {}).get('fast_image_base64')
    if not image:
        raise HTTPException(502, 'Fast GPU image was not returned')
    run_id = 'adaptive_' + uuid.uuid4().hex
    fast_wall=round((time.perf_counter()-_request_start)*1000,3)
    rec = {'run_id':run_id, 'at_utc':_utc(), 'phase':'awaiting_human',
           'prompt_sha256':_digest(request.prompt), 'seed':request.seed,
           'fast_image_sha256':hashlib.sha256(base64.b64decode(image)).hexdigest(),
           'fast_request_wall_ms':fast_wall, 'verification_scope':'full_prompt_v25',
           'device':generated['metrics'].get('device'),
           'fast_gateway_run_id':generated['run_id'], 'standard_gateway_run_id':None,
           'fast_inference_ms':fast_ms, 'standard_inference_ms':None,
           'fast_object_present':None, 'standard_object_present':None,
           'human_label_source':'user_manual_review', 'independently_audited':False}
    with LOCK:
        _append(rec)
    output = _payload(rec, {'fast_image_base64':image, 'image_base64':image},
                      [{'stage':'execute','message':'Fast 4-step LCM image generated on real GPU','at_ms':fast_ms},
                       {'stage':'verify','message':'Waiting for human critical-object decision; standard model has NOT run','at_ms':fast_ms}])
    return output


@router.post('/decide')
def decide(request:Decide):
    _request_start=time.perf_counter()
    if not request.run_id.startswith('adaptive_') or len(request.run_id) != 41:
        raise HTTPException(422, 'Invalid adaptive run id')
    if request.decision not in ('accept_fast', 'fallback'):
        raise HTTPException(422, 'Choose accept_fast or fallback')
    if request.decision == 'accept_fast' and request.fast_object_present is not True:
        raise HTTPException(422, 'Fast acceptance requires explicit critical-object presence review')
    if request.decision == 'fallback' and request.fast_object_present is True:
        raise HTTPException(422, 'A passing fast image should not trigger fallback')
    with LOCK:
        rec = _last(request.run_id)
        if rec is None:
            raise HTTPException(404, 'Adaptive run not found')
        if _digest(request.prompt) != rec['prompt_sha256']:
            raise HTTPException(409, 'Prompt changed since fast image generation; restart the mission')
        if rec['phase'] != 'awaiting_human':
            raise HTTPException(409, 'This run already has a decision; do not re-execute fallback')
        updated = {**rec, 'at_utc':_utc(), 'fast_object_present':request.fast_object_present,
                   'verification_source':'human_full_prompt_review','human_label_source':'user_manual_review'}
        if request.decision == 'accept_fast':
            updated['phase']='accepted_fast'
            _append(updated)
            return _payload(updated, {}, [{'stage':'verify','message':'User accepted Fast critical-object presence (not independently audited)','at_ms':rec['fast_inference_ms']},
                                           {'stage':'deliver','message':'Fast result served; standard diffusion never executed','at_ms':rec['fast_inference_ms']}])
        # Keep lock until standard inference finishes to prevent duplicate fallback.
        # Failure leaves the run pending and retryable; not a false success.
        standard = _gateway_call(request.prompt, rec['seed'], 'standard')
        standard_ms = _as_ms(standard.get('metrics', {}).get('standard_inference_ms'))
        image = standard.get('result', {}).get('standard_image_base64')
        if not image:
            raise HTTPException(502, 'Standard fallback did not return an image')
        updated.update(phase='fallback_generated',standard_inference_ms=standard_ms,
                       standard_gateway_run_id=standard['run_id'],
                       standard_image_sha256=hashlib.sha256(base64.b64decode(image)).hexdigest(),
                       standard_request_wall_ms=round((time.perf_counter()-_request_start)*1000,3))
        _append(updated)
        return _payload(updated, {'standard_image_base64':image,'image_base64':image},
                        [{'stage':'verify','message':'Fast was rejected or uncertain by user; standard fallback requested','at_ms':rec['fast_inference_ms']},
                         {'stage':'execute','message':'Standard 25-step SD 1.5 generated on real GPU','at_ms':round(rec['fast_inference_ms']+standard_ms,3)},
                         {'stage':'deliver','message':'Fallback delivered; standard object presence still needs human review','at_ms':round(rec['fast_inference_ms']+standard_ms,3)}])


@router.post('/review-standard')
def review_standard(request:FinalReview):
    with LOCK:
        rec = _last(request.run_id)
        if not rec:
            raise HTTPException(404, 'Adaptive run not found')
        if rec['phase'] != 'fallback_generated':
            raise HTTPException(409, 'No pending standard fallback review')
        updated = {**rec,'phase':'fallback_reviewed','at_utc':_utc(),
                   'standard_object_present':request.standard_object_present,
                   'verification_source':'human_full_prompt_review','human_label_source':'user_manual_review'}
        _append(updated)
    return {'saved':True,'phase':updated['phase'],'standard_object_present':request.standard_object_present,
            'independently_audited':False}


@router.get('/evidence')
def evidence():
    states = {}
    for row in _read():
        states[row['run_id']] = row
    finished = [r for r in states.values() if r['phase']!='awaiting_human']
    reviewed = [r for r in finished if r['phase'] in ('accepted_fast','fallback_reviewed') and r.get('human_label_source') is not None]
    successful = sum(1 for r in reviewed if (r['phase']=='accepted_fast' and r['fast_object_present'] is True)
                     or (r['phase']=='fallback_reviewed' and r['standard_object_present'] is True))
    gpu_seconds = sum((r['fast_inference_ms'] + (r.get('standard_inference_ms') or 0))/1000 for r in reviewed)
    return {'started':len(states),'finished':len(finished),'complete_human_reviews':len(reviewed),
            'successful_critical_objects_human_labeled':successful,
            'fallback_count':sum(r['standard_inference_ms'] is not None for r in finished),
            'fast_accepted_count':sum(r['phase']=='accepted_fast' for r in finished),
            'seconds_per_human_labeled_success':round(gpu_seconds/successful,4) if successful else None,
            'independently_audited':False,'final_records':[{k:v for k,v in r.items() if k!='prompt_sha256'} for r in list(states.values())[-30:]],
            'note':'Actual GPU inference time on sequential executed paths only. Human labels are not independently audited; user waiting time, energy and billing are not included.'}
