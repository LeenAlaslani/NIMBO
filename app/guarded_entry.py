"""Wrap the existing NIMBO FastAPI app without modifying mission_api.py.

REAL HTTP requests are traced. No response is called quality-verified by this
middleware; verification requires a separate trusted, validated verifier.
"""
from __future__ import annotations
import importlib, json, os, threading, time, uuid
from collections import deque
from dataclasses import asdict
from pathlib import Path
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from nimbo.guardrails_core import Policy, Observation, Evidence, decide

app = importlib.import_module('app.mission_api').app
log_path = Path(os.getenv('NIMBO_EVENTS_PATH', 'results/unified_events_v4.jsonl'))
_events = deque(maxlen=100)
_lock = threading.Lock()

def record(event):
    # Never log prompts, responses, auth headers or request bodies.
    sanitized = {k: v for k, v in event.items() if k in (
        'event_id','timestamp','kind','path','method','status_code','latency_ms',
        'task_type','action','reason','remaining_s','quality_certified')}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(sanitized, ensure_ascii=False) + '\n')
        _events.append(sanitized)
    return sanitized

@app.middleware('http')
async def nimbo_trace(request: Request, call_next):
    if request.url.path.startswith('/nimbo/guardrails/') or request.url.path.startswith('/nimbo/evidence/'):
        return await call_next(request)
    start = time.perf_counter()
    error = False
    try:
        response = await call_next(request)
        code = response.status_code
    except Exception:
        code = 500
        error = True
        raise
    finally:
        elapsed = round((time.perf_counter() - start) * 1000, 3)
        record(dict(event_id=uuid.uuid4().hex,timestamp=time.time(),kind='http_observation',
                    path=request.url.path,method=request.method,status_code=code,latency_ms=elapsed,
                    quality_certified=False))
    response.headers['X-NIMBO-Quality'] = 'NOT_VERIFIED'
    return response

@app.post('/nimbo/guardrails/decision')
async def guardrails_decision(request: Request):
    try:
        body = await request.json()
        policy = Policy(**body['policy'])
        raw = dict(body['observation'])
        raw['evidence'] = Evidence(raw['evidence'])
        observation = Observation(**raw)
        # API clients cannot self-certify their verifiers. A real trusted registry
        # must validate verifier identities before enabling certified ACCEPT.
        observation = Observation(**{**asdict(observation),
            'evidence': observation.evidence, 'verifier_validated': False})
        result = decide(policy, observation)
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422,detail=str(exc))
    event=record(dict(event_id=uuid.uuid4().hex,timestamp=time.time(),kind='guardrail_decision',
                      task_type=policy.task_type,action=result.action.value,reason=result.reason,
                      remaining_s=result.remaining_s,quality_certified=result.quality_certified))
    return {'decision':{**asdict(result),'action':result.action.value},'event_id':event['event_id'],
            'note':'No external verifier is trusted from caller input; this endpoint cannot certify quality.'}

@app.get('/nimbo/evidence/recent')
async def recent_evidence(limit: int = 25):
    if not (1 <= limit <= 100): raise HTTPException(status_code=422,detail='limit must be 1-100')
    return {'events':list(_events)[-limit:], 'source':'in_process_and_jsonl',
            'warning':'HTTP traces do not establish task success or verified quality.'}
