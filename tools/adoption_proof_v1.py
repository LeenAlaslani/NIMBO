#!/usr/bin/env python3
"""Real local HTTP Shadow -> Canary -> Rollback probe through NIMBO's adoption router.

Self-contained temporary service with isolated adoption state/log files.
Never binds the real port 8000, changes production routing or publishes artifacts.
A legitimately rejected Canary gate is not silently converted into a pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from tools.http_probe_v1 import choose_local_port, request_json
from tools.concurrency_probe_v1 import SOURCE_SELECTION, validate_lineage
from tools.staged_candidate_smoke_v1 import latest_exported_run

ROOT = Path('results/adoption_proof_v1')
PROMPTS = (
    'The direction was superb and the performances were convincing.',
    'The ending was awful and the plot was a mess.',
    'A charming film with excellent actors.',
    'I hated the predictable dialogue and terrible pacing.',
    'The soundtrack was lovely but the story was slow.',
    'A surprisingly entertaining and well-made movie.',
    'Unfocused, boring and hardly worth finishing.',
    'An enjoyable performance, even with a few weak scenes.',
)


def validate_response(result, *, mode: str, served: str, comparison: bool | None):
    if result.get('mode_before') != mode or result.get('route_served') != served:
        raise ValueError(f'Wrong serving route for {mode}: {result.get("route_served")}')
    if result.get('production_deployed') is not False:
        raise ValueError('Unexpected production deployment flag')
    if comparison is not None and result.get('comparison_performed') is not comparison:
        raise ValueError('Expected comparison flag missing')
    if result.get('auto_rollback') is True:
        raise RuntimeError('Automatic rollback detected: ' + str(result.get('rollback_reason')))
    if result.get('result_label') not in ('POSITIVE', 'NEGATIVE'):
        raise ValueError('Invalid classification output')
    return result


def audit_rolled_back(state):
    s=state.get('state') or {}
    if s.get('mode') != 'baseline' or s.get('run_id') is not None:
        raise RuntimeError('Rollback did not return to baseline')
    if state.get('production_deployed') is not False or state.get('changes_existing_nimbo_routing') is not False:
        raise RuntimeError('Production routing flag changed')
    return True


def choose_verified_run():
    import json
    from app.nimbo_adoption_v21_1 import _review
    run_id=latest_exported_run()
    review=_review(run_id)
    selection=json.loads(SOURCE_SELECTION.read_text(encoding='utf8'))
    threads=validate_lineage(selection, review['validation'],review)
    return run_id,review,threads


def run_flow(base, run_id, request=request_json):
    """One-user opt-in, actual REST endpoints. No original adoption state writes."""
    prefix='/nimbo/adoption/v21'
    def get(path): return request(base+prefix+path)
    def post(path,body): return request(base+prefix+path,body)

    initial=get('/status')
    audit_rolled_back(initial)
    baseline=post('/classify',{'text':PROMPTS[0]})
    validate_response(baseline,mode='baseline',served='baseline',comparison=False)
    print('BASELINE: PASS (served PyTorch)',flush=True)
    start=post('/shadow',{'run_id':run_id,'confirmation':'START LOCAL SHADOW'})
    if start['state']['mode']!='shadow':raise RuntimeError('Shadow did not activate')
    samples=[]
    for txt in PROMPTS[:6]:
        row=post('/classify',{'text':txt})
        validate_response(row,mode='shadow',served='baseline',comparison=True)
        if row.get('parity_status')!='PASS':raise RuntimeError('Shadow mismatch/drift')
        samples.append(row)
    shadow=get('/status')
    stats=shadow.get('shadow_metrics',{})
    if shadow['state']['mode']!='shadow' or stats.get('paired_samples',0)<5 or shadow['state'].get('shadow_count',0)<5:
        raise RuntimeError('Insufficient paired Shadow samples')
    print(f"SHADOW: PASS ({len(samples)} matched pairs; baseline served, no compute saving)",flush=True)

    from urllib.error import HTTPError
    try:
        switched=post('/canary',{'run_id':run_id,'confirmation':'ACTIVATE LOCAL CANARY'})
    except HTTPError as exc:
        # Do not bypass a legitimate gate. A 409 is an actual NIMBO refusal.
        if exc.code != 409:
            raise
        print('CANARY: GATE_BLOCKED (local shadow guard refused promotion)',flush=True)
        return {'outcome':'GATE_BLOCKED','shadow_count':len(samples),
                'shadow_metrics':stats,'canary_pairs':0,'canary_candidate_only':0,
                'reason':f'HTTP {exc.code} Conflict: promotion refused'}
    if switched['state']['mode']!='canary':raise RuntimeError('Canary did not activate')
    paired=0
    alone=0
    for txt in PROMPTS[:7]:
        row=post('/classify',{'text':txt})
        if row.get('route_served')!='candidate_onnx':
            raise RuntimeError('Canary did not serve ONNX on this diagnostic')
        validate_response(row,mode='canary',served='candidate_onnx',comparison=None)
        if row.get('comparison_performed') is True:
            paired+=1
            if row.get('parity_status')!='PASS':raise RuntimeError('Canary parity failure')
        elif row.get('comparison_performed') is False:
            alone+=1
        else:raise RuntimeError('Canary comparison flag missing')
    if paired<5 or alone<1:
        raise RuntimeError('Canary did not complete both guarded and candidate-only requests')
    print(f'CANARY: PASS ({paired} paired safety checks; {alone} candidate-only requests)',flush=True)
    return {'outcome':'CANARY_PASS','shadow_count':len(samples),'shadow_metrics':stats,
            'canary_pairs':paired,'canary_candidate_only':alone}


def cleanup_state(base, request=request_json):
    prefix='/nimbo/adoption/v21'
    status=request(base+prefix+'/rollback',{'confirmation':'ROLL BACK TO BASELINE'})
    audit_rolled_back(status)
    check=request(base+prefix+'/classify',{'text':PROMPTS[1]})
    validate_response(check,mode='baseline',served='baseline',comparison=False)
    print('ROLLBACK: PASS (PyTorch restored and served)',flush=True)
    return True


def run_probe():
    run_id,review,threads=choose_verified_run()
    now=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    session=(ROOT/f'session_{now}').resolve()
    session.mkdir(parents=True,exist_ok=False)
    port=choose_local_port()
    base=f'http://127.0.0.1:{port}'
    log=session/'server.log'
    env=os.environ.copy()
    env.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                'NIMBO_ADOPTION_PROOF_DIR':str(session),
                'NIMBO_ADOPTION_PROOF_RUN_ID':run_id})
    flow=None;issues=[];rollback_ok=False;health=None
    with log.open('w',encoding='utf8') as fh:
        proc=subprocess.Popen([sys.executable,'-m','uvicorn','tools.adoption_proof_service_v1:app',
            '--host','127.0.0.1','--port',str(port),'--workers','1','--no-access-log'],
            cwd=Path.cwd(),env=env,stdout=fh,stderr=subprocess.STDOUT)
        try:
            # Wait for the actual adoption router (not the Step16 wrapper).
            import time
            for _ in range(400):
                if proc.poll() is not None:raise RuntimeError('Service exited at startup; inspect '+str(log))
                try:
                    health=request_json(base+'/proof/health',timeout=0.9)
                    if health.get('ready') is True:break
                except Exception:pass
                time.sleep(.3)
            else:raise TimeoutError('Adoption proof service not ready; see '+str(log))
            if health.get('service_kind')!='isolated_actual_adoption_router_proof' or health.get('run_id')!=run_id or health.get('candidate_threads')!=threads:
                raise RuntimeError('Unexpected proof service or candidate lineage')
            flow=run_flow(base,run_id)
        except Exception as exc:
            issues.append(f'{type(exc).__name__}: {exc}')
        finally:
            if health is not None:
                try: rollback_ok=cleanup_state(base)
                except Exception as exc:issues.append('Rollback failed: '+str(exc))
            proc.terminate()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)

    complete=bool(flow and flow.get('outcome')=='CANARY_PASS' and rollback_ok and not issues)
    gated=bool(flow and flow.get('outcome')=='GATE_BLOCKED' and rollback_ok and not issues)
    report={
        'schema_version':'real_local_adoption_http_proof_v1',
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'status':'PASS' if complete else 'GATE_BLOCKED' if gated else 'FAIL',
        'source_run_id':run_id,
        'model_revision':review['revision'],
        'artifact_sha256':review['sha256'],
        'baseline_threads':review['validation']['thread_count'],
        'candidate_threads':threads,
        'test_service_kind':'isolated_actual_adoption_router_proof',
        'production_deployed':False,
        'original_adoption_state_modified':False,
        'test_text_saved':False,
        'scope':'Real NIMBO adoption router over temporary local HTTP; isolated state/logs; not primary mission_api, UI, or production deployment. Small unlabeled diagnostic texts; not energy or cost evidence.',
        'shadow_and_canary':flow,'rollback_verified':rollback_ok,
        'errors':issues,
        'synthetic_input_digest':hashlib.sha256('\n'.join(PROMPTS).encode('utf8')).hexdigest(),
        'server_log_file':str(log),
    }
    path=session/'report.json'
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('=== NIMBO ISOLATED ADOPTION HTTP PROOF ===')
    print('STATUS:',report['status'])
    print('RUN:',run_id)
    print('ROLLBACK VERIFIED:',rollback_ok)
    print('ORIGINAL ADOPTION STATE: UNTOUCHED BY PROOF SERVICE')
    print('EVIDENCE:',path)
    if issues:
        for e in issues:print('ERROR:',e[:300])
    if not complete:
        raise RuntimeError('Adoption proof did not complete all gated stages; inspect report')
    return report


def main():
    p=argparse.ArgumentParser()
    p.parse_args()
    run_probe()

if __name__=='__main__':main()
