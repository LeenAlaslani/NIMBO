# NIMBO_V27_COST_AWARE_ENGINE: surgical revision of V26 (same API path)
"""VERA v26: autonomous but *conservative* local image quality routing.

Starts from a real v24 fast GPU image, runs a real v25 OWLv2 verification,
launches exactly one Standard fallback only on an actionable local FAIL.
For uncheckable complex scenes, it avoids costly checks and blind fallback. No prompt/PNG is written to evidence.

Important: OWLv2 and a pixel-color heuristic are NOT a full scene verifier.
A non-PASS means the system can finish without claiming task success. The
optional human review is separate and never presented as independent auditing.
"""
from __future__ import annotations

import base64
import hashlib
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import nimbo_adaptive_image_v24 as adaptive
from app import nimbo_vera_v25 as verifier

router = APIRouter(prefix='/nimbo/image/vera/v26', tags=['VERA autonomous image optimizer'])
_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix='nimbo-vera-v26')
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.RLock()
_FINISHED_MAX = 12


class Launch(BaseModel):
    run_id: str = Field(min_length=41, max_length=41)
    prompt: str = Field(min_length=3, max_length=500)
    fast_image_base64: str = Field(min_length=50, max_length=verifier.MAX_IMAGE_B64)


def _ms():
    return time.perf_counter() * 1000


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _validate_initial(req):
    if not req.run_id.startswith('adaptive_') or len(req.run_id) != 41:
        raise HTTPException(422, 'Invalid image trial identifier')
    with adaptive.LOCK:
        rec = adaptive._last(req.run_id)
        if not rec:
            raise HTTPException(404, 'Image mission does not exist')
        if adaptive._digest(req.prompt) != rec.get('prompt_sha256'):
            raise HTTPException(409, 'Prompt does not match the existing image mission')
        try:
            _, raw = verifier._base64_image(req.fast_image_base64)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, 'Invalid Fast PNG') from exc
        if hashlib.sha256(raw).hexdigest() != rec.get('fast_image_sha256'):
            raise HTTPException(409, 'Fast PNG digest does not match the generated image')
        return rec


def _slim(job):
    return {'run_id': job['run_id'], 'state': job['state'], 'stage': job['stage'],
            'events': list(job['events']), 'elapsed_agent_ms': round(_ms() - job['started_ms'], 2)
            if job['state'] not in ('complete', 'error') else job.get('elapsed_agent_ms'),
            'decision': job.get('decision'), 'failure': job.get('failure'),
            'report': job.get('report') if job['state'] == 'complete' else None}


def _change(run_id, state, stage, message):
    with _JOBS_LOCK:
        job = _JOBS[run_id]
        job['state'] = state
        job['stage'] = stage
        job['events'].append({'stage': stage, 'message': message,
                              'at_ms': round(_ms() - job['started_ms'], 2)})
        job['updated_ms'] = _ms()


def _append_record(run_id, **fields):
    with adaptive.LOCK:
        rec = adaptive._last(run_id)
        if not rec:
            raise RuntimeError('Run disappeared from local evidence')
        if rec.get('phase') != 'auto_verifying':
            raise RuntimeError('Image mission is no longer owned by VERA v26')
        next_rec = {**rec, **fields, 'at_utc': _utc()}
        adaptive._append(next_rec)
        return next_rec


def _successful_local_verification(outcome):
    return outcome.get('verdict') == 'PASS' and outcome.get('route_advice') == 'accept_fast' and not outcome.get('spec', {}).get('scene_unverified')


def _check_image(encoded, prompt):
    # NIMBO_V27_COST_AWARE: Do not spend 6-18s on OWLv2 when v25's
    # acceptance rule structurally forbids a PASS for these instructions.
    # This returns UNKNOWN, NEVER a fabricated quality PASS or FAIL.
    t0=_ms()
    spec=verifier.analyze_prompt(prompt)
    if not spec.get('supported') or spec.get('scene_unverified'):
        return {'verdict':'UNCERTAIN', 'route_advice':'no_blind_fallback',
                'reason':('Full scene/count/relationship requirements cannot be certified by '
                          'this object-only verifier. Expensive detection cannot establish '
                          'full-prompt success; no unproven extra GPU pass is justified.'),
                'spec':spec, 'boxes_plausible':None, 'boxes_reliable':None,
                'verification_ms':round(_ms()-t0,3), 'model_inference_ms':0,
                'model':'reasoned_capability_guard_v27', 'quality_certified':False,
                'independently_audited':False,
                'verification_skipped_reason':'structurally_unverifiable_scene'}
    # For simple supported tasks, use the actual image model, with honest timing.
    return verifier.verify(encoded,prompt)


def _execute(run_id, prompt, fast_b64):
    """Worker thread, never runs on the HTTP request thread. Unit-testable."""
    begun = _ms()
    try:
        _change(run_id, 'checking_fast', 'verify', 'VERA is evaluating the real Fast image')
        fast_check = _check_image(fast_b64, prompt)
        _append_record(run_id, vera_fast=fast_check, verification_source='automatic_vera_v27_cost_aware_unverified',
                       human_label_source=None)

        if _successful_local_verification(fast_check):
            rec = _append_record(run_id, phase='accepted_fast', fast_object_present=True,
                                 auto_outcome='fast_local_pass', auto_selected='fast',
                                 auto_decision_reason=fast_check.get('reason', 'Local object checks passed'),
                                 auto_complete=True, auto_finished_utc=_utc(),
                                 auto_agent_wall_ms=round(_ms() - begun, 3))
            decision = 'FAST_ACCEPTED_LOCAL_CHECK'
            image_result = {'fast_image_base64': fast_b64, 'image_base64': fast_b64}
            std_check = None
        elif fast_check.get('verdict') == 'UNCERTAIN':
            # V27: A second model generation cannot resolve an uncheckable
            # semantic specification; automatic blind fallback spends resources
            # without evidence it produces a better result.
            rec = _append_record(run_id, phase='auto_unverified',
                                 auto_outcome='fast_unverified_no_blind_fallback',
                                 auto_selected='none', fast_object_present=None,
                                 auto_decision_reason=fast_check.get('reason'),
                                 auto_complete=True, auto_finished_utc=_utc(),
                                 auto_agent_wall_ms=round(_ms()-begun,3))
            decision = 'FAST_RETURNED_UNVERIFIED'
            image_result = {'fast_image_base64':fast_b64, 'image_base64':fast_b64}
            std_check = None
        else:
            # Explicit local FAIL justifies one fallback. Failure is not a
            # certified absence judgment; Standard must be checked separately.
            _change(run_id, 'generating_standard', 'execute',
                    'Local checks found a failure; launching Standard as a rescue attempt')
            with adaptive.LOCK:
                current = adaptive._last(run_id)
                if not current or current.get('phase') != 'auto_verifying':
                    raise RuntimeError('The mission changed before fallback generation')
                seed = current['seed']
            standard_start = _ms()
            generated = adaptive._gateway_call(prompt, seed, 'standard')
            standard_wall = round(_ms() - standard_start, 3)
            standard_ms = adaptive._as_ms(generated.get('metrics', {}).get('standard_inference_ms'))
            standard_b64 = generated.get('result', {}).get('standard_image_base64')
            if not standard_b64:
                raise RuntimeError('Standard gateway returned no PNG')
            _, standard_raw = verifier._base64_image(standard_b64)
            _append_record(run_id, standard_inference_ms=standard_ms,
                           standard_request_wall_ms=standard_wall,
                           standard_gateway_run_id=generated.get('run_id'),
                           standard_image_sha256=hashlib.sha256(standard_raw).hexdigest())
            _change(run_id, 'checking_standard', 'verify', 'VERA is checking the Standard fallback')
            std_check = _check_image(standard_b64, prompt)
            local_pass = _successful_local_verification(std_check)
            if local_pass:
                outcome, decision, presence = 'standard_local_pass', 'STANDARD_ACCEPTED_LOCAL_CHECK', True
            elif std_check.get('verdict') == 'FAIL':
                outcome, decision, presence = 'standard_local_fail', 'NO_VERIFIED_RESULT', False
            else:
                outcome, decision, presence = 'standard_unverified', 'STANDARD_DELIVERED_UNVERIFIED', None
            rec = _append_record(run_id, phase='fallback_reviewed', vera_standard=std_check,
                                 fast_object_present=False if fast_check.get('verdict') == 'FAIL' else None,
                                 standard_object_present=presence,
                                 auto_outcome=outcome, auto_selected='standard' if local_pass else 'none',
                                 auto_decision_reason=std_check.get('reason', 'Standard quality could not be established'),
                                 auto_complete=True, auto_finished_utc=_utc(),
                                 auto_agent_wall_ms=round(_ms() - begun, 3))
            image_result = {'fast_image_base64': fast_b64,
                            'standard_image_base64': standard_b64,
                            'image_base64': standard_b64}

        # The measured wall fields remain explicit: this API begins AFTER the Fast image
        # has already been generated by the existing v24 Mission Floor route.
        payload = adaptive._payload(rec, image_result=image_result)
        total_verify = (fast_check.get('verification_ms') or 0) + ((std_check or {}).get('verification_ms') or 0)
        overall_elapsed = round(_ms() - begun, 3)
        payload['route'] = 'sd15_adaptive_autonomous_vera_v26'
        payload['vera_fast'] = fast_check
        payload['vera_standard'] = std_check
        payload['adaptive'].update({'served_by': 'fast' if decision == 'FAST_RETURNED_UNVERIFIED' else payload['adaptive'].get('served_by'), 'auto_outcome': rec.get('auto_outcome'), 'auto_selected': rec.get('auto_selected'),
                                    'auto_complete': True, 'verification_source': 'automatic_vera_v27_cost_aware_unverified',
                                    'human_review_complete': False, 'quality_verified': False,
                                    'quality_certified': False, 'decision': decision,
                                    'verdict_reason': rec.get('auto_decision_reason')})
        payload['metrics'].update({'verifier_total_wall_ms': round(total_verify, 3),
                                   'agent_after_fast_wall_ms': overall_elapsed,
                                   'fast_request_wall_ms': rec.get('fast_request_wall_ms'),
                                   'standard_request_wall_ms': rec.get('standard_request_wall_ms'),
                                   'executed_generation_request_wall_ms': round((rec.get('fast_request_wall_ms') or 0) + (rec.get('standard_request_wall_ms') or 0), 3),
                                   'measured_components_wall_ms': round((rec.get('fast_request_wall_ms') or 0) + overall_elapsed, 3)})
        # Mirror only real server transitions into the mission telemetry replay.
        # All times are measured, relative to VERA start (not fabricated progress).
        with _JOBS_LOCK:
            actual_events=list(_JOBS[run_id]['events'])
        payload['trace']=[{'stage':e['stage'], 'message':'VERA · '+e['message'],
                           'at_ms':round((rec.get('fast_request_wall_ms') or 0)+e['at_ms'],2)}
                          for e in actual_events]
        payload['trace'].append({'stage':'deliver','message':'VERA final decision recorded: '+decision,
                                 'at_ms':round((rec.get('fast_request_wall_ms') or 0)+overall_elapsed,2)})
        payload['measurement_note'] = ('Actual Fast and optional Standard inference; OWLv2 is skipped when it cannot certify the stated scene. '
            'Components wall includes Fast request + post-Fast agent; excludes frontend handoff time. '
            'No independently verified full-prompt success, energy, FLOPs or monetary savings.')
        with _JOBS_LOCK:
            job = _JOBS[run_id]
            job['report'] = payload
            job['decision'] = decision
            job['elapsed_agent_ms'] = overall_elapsed
            job['updated_ms'] = _ms()
            # Avoid retaining the input prompt or Fast PNG after completion.
            job.pop('prompt', None)
            job.pop('fast_image_base64', None)
        _change(run_id, 'complete', 'deliver', 'Agent finished; local decision and real measurements recorded')
    except Exception as exc:
        # Fail closed. Evidence records errors, never successful quality.
        try:
            _append_record(run_id, phase='auto_error', auto_complete=False,
                           auto_error_type=type(exc).__name__,
                           auto_decision_reason=str(exc)[:180])
        except Exception:
            pass
        with _JOBS_LOCK:
            job = _JOBS[run_id]
            job['failure'] = f'{type(exc).__name__}: {str(exc)[:170]}'
            job['elapsed_agent_ms'] = round(_ms() - begun, 3)
            job.pop('prompt', None)
            job.pop('fast_image_base64', None)
        _change(run_id, 'error', 'error', 'VERA or GPU worker failed; no quality success was recorded')


def _prune_jobs():
    # Keep a small bounded collection; no files or prompts/images retained in the audit log.
    finished = sorted((j for j in _JOBS.values() if j['state'] in ('complete','error')),
                      key=lambda j:j.get('updated_ms', 0))
    for job in finished[:-_FINISHED_MAX]:
        _JOBS.pop(job['run_id'], None)


@router.get('/health')
def health():
    return {'ok': True, 'version': '27.0-cost-aware', 'policy': 'cost_aware_no_blind_fallback_v27',
            'detector': verifier.MODEL_ID, 'worker_limit': 1,
            'model_loaded': verifier._load.cache_info().currsize > 0,
            'quality_certified': False, 'independently_audited': False,
            'human_decision_required_for_execution': False, 'quality_floor_guaranteed': False}


@router.post('/launch', status_code=202)
def launch(req: Launch):
    rec = _validate_initial(req)
    with _JOBS_LOCK:
        existing = _JOBS.get(req.run_id)
        if existing:
            return _slim(existing)
        if rec['phase'] not in ('awaiting_human', 'auto_verifying'):
            raise HTTPException(409, 'Image run was already finalized. Start a new mission')
        if rec['phase'] == 'auto_verifying' and rec.get('standard_inference_ms') is not None:
            raise HTTPException(409, 'Previous Standard was already generated; cannot safely replay it after an API restart. Start a new image mission')
        if sum(j['state'] not in ('complete','error') for j in _JOBS.values()) >= 2:
            raise HTTPException(429, 'VERA local worker queue is full (2 concurrent/queued missions)')
        # If previous API process restarted after a crash, there can be an orphaned
        # auto_verifying record. The original PNG digest is checked before resuming.
        with adaptive.LOCK:
            current = adaptive._last(req.run_id)
            if current['phase'] not in ('awaiting_human', 'auto_verifying'):
                raise HTTPException(409, 'Mission state changed during launch')
            adaptive._append({**current, 'phase': 'auto_verifying', 'at_utc': _utc(),
                              'verification_source': 'automatic_vera_v27_cost_aware_unverified',
                              'human_label_source': None, 'auto_complete': False})
        begun = _ms()
        _JOBS[req.run_id] = {'run_id': req.run_id, 'state': 'queued', 'stage': 'queue',
                             'started_ms': begun, 'updated_ms': begun, 'events': [],
                             'report': None, 'decision': None}
        _prune_jobs()
        _POOL.submit(_execute, req.run_id, req.prompt, req.fast_image_base64)
        return _slim(_JOBS[req.run_id])


@router.get('/status/{run_id}')
def status(run_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(run_id)
        if not job:
            raise HTTPException(404, 'No active VERA v26 job in this API process')
        return _slim(job)


@router.get('/evidence')
def evidence():
    states = {}
    for r in adaptive._read():
        states[r['run_id']] = r
    runs = [r for r in states.values() if r.get('verification_source') == 'automatic_vera_v27_cost_aware_unverified']
    complete = [r for r in runs if r.get('auto_complete') is True]
    return {'total_runs': len(runs), 'complete_runs': len(complete),
            'fast_accepted_local': sum(r.get('auto_outcome') == 'fast_local_pass' for r in complete),
            'standard_accepted_local': sum(r.get('auto_outcome') == 'standard_local_pass' for r in complete),
            'standard_unverified': sum(r.get('auto_outcome') == 'standard_unverified' for r in complete),
            'standard_local_fail': sum(r.get('auto_outcome') == 'standard_local_fail' for r in complete),
            'error_count': sum(r.get('phase') == 'auto_error' for r in runs),
            'total_gpu_inference_ms': round(sum((r.get('fast_inference_ms') or 0) + (r.get('standard_inference_ms') or 0) for r in complete), 3),
            'total_verifier_wall_ms': round(sum((r.get('vera_fast') or {}).get('verification_ms', 0) + (r.get('vera_standard') or {}).get('verification_ms', 0) for r in complete), 3),
            'quality_certified': False, 'independently_audited': False,
            'fast_unverified_no_blind_fallback': sum(r.get('auto_outcome')=='fast_unverified_no_blind_fallback' for r in complete),
            'note': 'V27 refuses blind fallback for uncheckable scenes. No independent quality guarantee, measured dollars, or energy.'}
