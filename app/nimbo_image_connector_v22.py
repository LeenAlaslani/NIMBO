"""NIMBO Image v22 model gateway and manual quality evidence.

Never accepts external URLs from clients, never copies prompt text or PNGs into evidence.
The worker runs on 127.0.0.1:8011; this API runs on 127.0.0.1:8000.
"""
from __future__ import annotations
import base64
import binascii
import json
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request as URLRequest, urlopen

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

PREFIX='/nimbo/image/v22'
WORKER='http://127.0.0.1:8011'
FILE=Path('results/image_gateway_runs_v22.jsonl')
REVIEWS=Path('results/image_gateway_human_reviews_v22.jsonl')
router=APIRouter(prefix=PREFIX,tags=['NIMBO local image gateway'])
LOG_LOCK=threading.Lock()

class ImageRequest(BaseModel):
    prompt: str = Field(min_length=3,max_length=500)
    mode: str = 'paired'
    seed: int = Field(default=42,ge=0,le=2**31-1)

class ReviewRequest(BaseModel):
    run_id: str = Field(min_length=12,max_length=80)
    fast_success: bool | None = None
    standard_success: bool | None = None
    rubric: str = Field(default='critical_object_present',max_length=90)

def _http_json(url, payload=None, timeout=3):
    headers={'Content-Type':'application/json','Accept':'application/json'}
    data=None if payload is None else json.dumps(payload).encode('utf8')
    request=URLRequest(url,data=data,headers=headers,method='GET' if data is None else 'POST')
    with urlopen(request,timeout=timeout) as response:
        return json.loads(response.read(15_000_000))

def worker_status():
    try:
        result=_http_json(WORKER+'/health',timeout=1.4)
        if isinstance(result,dict) and result.get('ok') is True:
            return {'connected':True,'device':result.get('device'), 'models_loaded':bool(result.get('models_loaded')),
                    'quality_certified':False,'model':result.get('model')}
        return {'connected':False,'reason':(result or {}).get('reason','GPU unavailable')}
    except (HTTPError,URLError,TimeoutError,ValueError,OSError):
        return {'connected':False,'reason':'Local image GPU worker is not running on port 8011'}

def probe_image_gateway_v22():
    return worker_status()['connected']

def _validate_png(value):
    if not isinstance(value,str) or len(value)>12_000_000:
        raise HTTPException(502,'Worker image missing or oversized')
    try: blob=base64.b64decode(value,validate=True)
    except (binascii.Error,ValueError):raise HTTPException(502,'Worker returned invalid Base64')
    if len(blob)>8_000_000 or not blob.startswith(b'\x89PNG\r\n\x1a\n'):
        raise HTTPException(502,'Worker returned invalid PNG bytes')
    return value

def _finite_ms(val):
    if isinstance(val,bool) or not isinstance(val,(int,float)) or not math.isfinite(val) or val<0:
        raise HTTPException(502,'GPU worker did not provide measured inference duration')
    return float(val)

def _utc():return datetime.now(timezone.utc).isoformat()

def _write(path,rec):
    with LOG_LOCK:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a',encoding='utf8') as out:out.write(json.dumps(rec,ensure_ascii=False)+'\n')

def _recent(path,limit=1000):
    if not path.is_file():return []
    output=[]
    with path.open(encoding='utf8') as stream:
        for line in stream:
            try:output.append(json.loads(line))
            except (ValueError,TypeError):continue
    return output[-limit:]

@router.get('/status')
def status():return worker_status()

@router.post('/generate')
def generate(request:ImageRequest):
    if request.mode not in ('fast','standard','paired'):
        raise HTTPException(422,'Choose fast, standard or paired')
    status=worker_status()
    if not status['connected']:
        raise HTTPException(503, f'GPU worker offline: {status.get("reason")}. Start the worker first.')
    started=time.perf_counter()
    try:
        raw=_http_json(WORKER+'/generate',request.model_dump() if hasattr(request,'model_dump') else request.dict(),timeout=600)
    except HTTPError as exc:
        detail={}
        try:detail=json.loads(exc.read(10000))
        except (ValueError,OSError):pass
        raise HTTPException(502,f'GPU worker HTTP {exc.code}: {detail.get("detail","inference error")}') from exc
    except (URLError,TimeoutError,OSError,ValueError) as exc:
        raise HTTPException(502,f'GPU worker unavailable during generation ({type(exc).__name__})') from exc
    results=raw.get('results') if isinstance(raw,dict) else None
    if not isinstance(results,dict):raise HTTPException(502,'GPU worker response missing results')
    required=['fast','standard'] if request.mode=='paired' else [request.mode]
    for mode in required:
        if mode not in results or not isinstance(results[mode],dict):raise HTTPException(502,'Required image mode missing')
        _validate_png(results[mode].get('image_base64'))
        _finite_ms(results[mode].get('inference_ms'))
    if request.mode=='paired':
        fast_ms=_finite_ms(results['fast']['inference_ms'])
        std_ms=_finite_ms(results['standard']['inference_ms'])
    else:
        fast_ms=_finite_ms(results['fast']['inference_ms']) if 'fast' in results else None
        std_ms=_finite_ms(results['standard']['inference_ms']) if 'standard' in results else None
    chosen='fast' if 'fast' in results else 'standard'
    run_id='image_'+uuid.uuid4().hex
    elapsed=(time.perf_counter()-started)*1000
    record={'run_id':run_id,'at_utc':_utc(),'evidence_kind':'local_image_gpu_execution',
            'model':raw.get('model'),'lora':raw.get('lora'), 'device':status['device'],
            'mode':request.mode,'seed':request.seed,'width':512,'height':512,
            'fast_inference_ms':fast_ms,'standard_inference_ms':std_ms,
            'end_to_end_ms':round(elapsed,3), 'quality_certified':False,
            'manual_quality_label_available':False, 'image_saved_to_evidence':False,
            'prompt_saved_to_evidence':False}
    _write(FILE,record)
    trace=[e for e in (raw.get('trace') or []) if isinstance(e,dict)][:30]
    trace.append({'stage':'deliver','message':'Generated images delivered. Human quality review still required.','at_ms':round(elapsed,3)})
    return {'workload':'image','run_id':run_id,'route':f'sd15_{request.mode}_local_gpu',
            'result':{'image_base64':results[chosen]['image_base64'],
                      'fast_image_base64':results.get('fast',{}).get('image_base64'),
                      'standard_image_base64':results.get('standard',{}).get('image_base64')},
            'metrics':{'end_to_end_ms':round(elapsed,3),'fast_inference_ms':fast_ms,
                       'standard_inference_ms':std_ms,'device':status['device']},
            'image_comparison':{'mode':request.mode,'seed':request.seed,'fast_steps':4,
                                'standard_steps':25,'fast_inference_ms':fast_ms,
                                'standard_inference_ms':std_ms,
                                'measured_fast_latency_reduction_fraction':(1-fast_ms/std_ms) if fast_ms is not None and std_ms and std_ms>0 else None,
                                'quality_verified':False},
            'trace':trace,'measurement_note':
              'Actual local GPU diffusion times; the two images may differ semantically. Time is not energy, FLOPs, dollars or cost per successful image.'}

@router.post('/review')
def review(req:ReviewRequest):
    if not req.run_id.startswith('image_') or not req.run_id[6:].isalnum():
        raise HTTPException(422,'Invalid image run ID')
    if req.fast_success is None and req.standard_success is None:
        raise HTTPException(422,'At least one image needs a human review label')
    if req.rubric!='critical_object_present':
        raise HTTPException(422,'This gateway only supports the critical-object rubric')
    run=next((x for x in reversed(_recent(FILE)) if x.get('run_id')==req.run_id),None)
    if not run:raise HTTPException(404,'Run not found')
    if req.fast_success is not None and run['fast_inference_ms'] is None:raise HTTPException(422,'Fast image not generated')
    if req.standard_success is not None and run['standard_inference_ms'] is None:raise HTTPException(422,'Standard image not generated')
    rec={'at_utc':_utc(),'run_id':req.run_id,'rubric':req.rubric,
         'fast_success':req.fast_success,'standard_success':req.standard_success,
         'label_source':'user_manual_review', 'independently_audited':False}
    _write(REVIEWS,rec)
    return {'saved':True,'run_id':req.run_id,'note':'Human review saved; not an independent quality audit.'}

@router.get('/evidence')
def evidence():
    runs=_recent(FILE); labels={}
    for rec in _recent(REVIEWS):labels[rec.get('run_id')]=rec
    paired=[r for r in runs if r.get('mode')=='paired']
    paired_rated=[r for r in paired if r['run_id'] in labels and
                  labels[r['run_id']].get('fast_success') is not None and
                  labels[r['run_id']].get('standard_success') is not None]
    def stats(mode):
        count=sum(bool(labels[r['run_id']][mode+'_success']) for r in paired_rated)
        elapsed=sum(float(r[mode+'_inference_ms']) for r in paired_rated)/1000
        return {'successes':count,'total_seconds':round(elapsed,4),
                'seconds_per_success':round(elapsed/count,4) if count else None,
                'success_rate':round(count/len(paired_rated),4) if paired_rated else None}
    return {'runs':len(runs),'paired_runs':len(paired),
            'complete_human_paired_reviews':len(paired_rated),'labels_independently_audited':False,
            'fast':stats('fast'),'standard':stats('standard'),
            'last_runs':[{**r,'review':labels.get(r['run_id'])} for r in runs[-20:][::-1]],
            'metric_note':'Wall-clock GPU inference seconds per human-labeled successful critical object; excludes loading, HTTP, electricity and service billing.'}
