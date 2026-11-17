"""M11 local-only auditable multi-agent optimizer wrapper around the real M7 executor.

Five real independently prompted LLM specialists may reason; only audited Python tools
can act. The existing model loader, ONNX generator, heldout verifier and offline
packager are reused unchanged. No benchmark is fabricated by this module.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Callable

from app.nimbo_agentic_core_v1 import (
    OllamaRoleModel, available_agent_config, gate_for_action, extract_transient,
    AgentDecisionError, AgentConfigurationError, JOB_ID, stamp, summarize_measured_evidence,
    check_reason_consistency, grounded_decision_summary,
)
from app.nimbo_auto_pipeline_jobs_v1 import (
    validate_request as validate_m7, read_job as read_m7_job,
    read_events as read_m7_events, submit as submit_m7,
    package_for_download as m7_verified_package,
)

JOB_ROOT = Path('results/agentic_jobs_v1')
FINAL = frozenset({'COMPLETED_REVIEW_ONLY','COMPLETED_REJECTED','FAILED','INTERRUPTED'})


def _atomic_json(path: Path, data: dict):
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('x',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,indent=2)
            f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def folder(root: Path, job_id: str) -> Path:
    if not JOB_ID.fullmatch(job_id):raise ValueError('Invalid job ID')
    directory=root/JOB_ROOT/job_id
    if directory.is_symlink() or not directory.is_dir():raise FileNotFoundError(job_id)
    return directory


def _event(directory: Path, stage: str, message: str, *, state='RUNNING', details: dict | None = None):
    status=_read(directory/'status.json')
    sequence=status['seq']+1
    evt={'seq':sequence,'at':stamp(),'stage':stage,'state':state,'message':str(message)[:600]}
    if details:evt['evidence']=details
    with (directory/'events.jsonl').open('a',encoding='utf-8') as out:
        out.write(json.dumps(evt,ensure_ascii=False)+'\n');out.flush();os.fsync(out.fileno())
    status.update({'state':state,'stage':stage,'seq':sequence,'updated_at':evt['at']})
    if state in FINAL:status['finished_at']=stamp()
    _atomic_json(directory/'status.json',status)


def _decision(directory, entry, stage, facts):
    # The event headline is deterministic. Qwen's reason is retained ONLY as an
    # audited model explanation, not as a verified resource/quality conclusion.
    issues=check_reason_consistency(entry.role,entry,facts)
    if issues:raise AgentDecisionError(f'Unreviewed {entry.role} reasoning reached event logger: {issues}')
    detail={'agent_role':entry.role,'model':entry.model,'tool_choice':entry.action,
            'reason_check':'bounded_consistency_pass','llm_reason_not_evidence':entry.reason}
    _event(directory,stage, f'{entry.role} decision: {entry.action} — '+grounded_decision_summary(entry.role,entry,facts),
           details=detail)



def _reviewed_decision(directory, model, role, stage, facts):
    # M11_2_BOUNDED_DECISION_RECOVERY
    # At most one factual correction and one policy-response repair.
    # Never reinvokes the M7 optimization engine.
    context = dict(facts)
    factual_repair_used = False
    policy_repair_used = False

    for _ in range(3):
        try:
            entry = model.decide(role, context)

        except AgentDecisionError as exc:
            policy_error = (
                str(exc) ==
                f'{role} returned an out-of-policy decision'
            )

            if not policy_error:
                raise

            if policy_repair_used:
                _event(
                    directory, stage,
                    f'{role} response remains out of policy; '
                    'stopping agent approval',
                    details={
                        'agent_role': role,
                        'reason_check': 'failed',
                        'error_type': 'out_of_policy'
                    }
                )
                raise

            policy_repair_used = True

            _event(
                directory, stage,
                f'{role} returned an invalid action or reason; '
                'requesting one bounded response repair',
                details={
                    'agent_role': role,
                    'reason_check': 'retry',
                    'error_type': 'out_of_policy'
                }
            )

            context = {
                **facts,
                'reasoning_feedback': {
                    'issues': ['response_out_of_policy'],
                    'instruction': (
                        'Return exactly JSON with action and reason. '
                        'Choose action ONLY from the supplied allowed_actions. '
                        'The reason must be 2-220 characters. '
                        'Use the recorded backend facts; do not invent '
                        'measurements or reverse the gate verdict. '
                        'Review-only is not production approval.'
                    )
                }
            }
            continue

        issues = check_reason_consistency(role, entry, facts)

        if not issues:
            _decision(directory, entry, stage, facts)
            return entry

        if factual_repair_used:
            _event(
                directory, stage,
                f'{role} explanation still contradicted '
                'recorded evidence; stopping agent approval',
                details={
                    'agent_role': role,
                    'reason_check': 'failed',
                    'issues': issues
                }
            )
            raise AgentDecisionError(
                f'{role} failed factual consistency after correction'
            )

        factual_repair_used = True

        _event(
            directory, stage,
            f'{role} explanation failed evidence-consistency '
            'checks; requesting one correction',
            details={
                'agent_role': role,
                'reason_check': 'failed',
                'issues': issues
            }
        )

        context = {
            **facts,
            'reasoning_feedback': {
                'issues': issues,
                'instruction': (
                    'Return JSON action and reason consistent with '
                    'the recorded backend state. Use only an allowed '
                    'action. Do not claim rejection if the verified '
                    'backend gates passed. Do not impersonate another agent.'
                )
            }
        }

    raise AgentDecisionError(
        f'{role} exhausted bounded reasoning corrections'
    )


def _finish(directory, state, decision, reason, *, engine_job_id=None, package_verified=False):
    status=_read(directory/'status.json')
    status.update({'decision':decision,'reason':reason,'engine_job_id':engine_job_id,
                   'eligible_for_review_download':bool(state=='COMPLETED_REVIEW_ONLY' and package_verified),
                   'eligible_for_deployment':False})
    _atomic_json(directory/'status.json',status)
    _event(directory,'deliver' if state=='COMPLETED_REVIEW_ONLY' else 'error',reason,state=state)
    receipt={'schema_version':'nimbo_agentic_review_v1','job_id':status['job_id'],'engine_job_id':engine_job_id,
             'state':state,'decision':decision,'reason':reason,'eligible_for_review_download':bool(state=='COMPLETED_REVIEW_ONLY' and package_verified),
             'eligible_for_deployment':False,'written_at':stamp()}
    _atomic_json(directory/'receipt.json',receipt)
    return _read(directory/'status.json')


def read_job(root: Path, job_id: str):
    directory=folder(root,job_id)
    status=_read(directory/'status.json')
    if status['state'] in ('QUEUED','RUNNING') and (directory/'launcher.json').exists():
        pid=_read(directory/'launcher.json').get('pid')
        try:
            if type(pid) is not int or pid <= 0: raise ProcessLookupError('invalid worker PID')
            try:
                ended, _ = os.waitpid(pid,os.WNOHANG)
                if ended == pid: raise ProcessLookupError('worker already exited')
            except ChildProcessError:pass
            os.kill(pid,0)
        except (ProcessLookupError,TypeError,ValueError):
            return _finish(directory,'INTERRUPTED','WORKER_INTERRUPTED','Agentic worker exited before final receipt')
        except PermissionError:pass
    return status


def read_events(root: Path, job_id: str,after: int=0):
    directory=folder(root,job_id)
    with (directory/'events.jsonl').open(encoding='utf-8') as f:
        result=[json.loads(line) for line in f if line.strip()]
    return {'job_id':job_id,'events':[e for e in result if e['seq']>after]}


def package_for_download(root: Path, job_id: str):
    status=read_job(root,job_id)
    if status['state']!='COMPLETED_REVIEW_ONLY' or status.get('eligible_for_review_download') is not True:
        raise ValueError('Agentic job has no approved review package')
    receipt=_read(folder(root,job_id)/'receipt.json')
    if receipt.get('engine_job_id')!=status.get('engine_job_id') or receipt.get('eligible_for_review_download') is not True:
        raise ValueError('Agentic receipt mismatch')
    return m7_verified_package(root,status['engine_job_id']) # recalculates artifact checksum at download


def submit(root: Path, request: dict, *, model_config=None, spawn=subprocess.Popen):
    root=root.resolve(strict=True)
    cfg=model_config or available_agent_config()
    OllamaRoleModel(cfg) # validates loopback URL; does not claim a response yet
    cleaned=validate_m7(root,request) # immutable SHA, safe paths, labeled datasets
    top=root/JOB_ROOT
    top.mkdir(parents=True,exist_ok=True)
    # Serializes agent job creation; M7 also protects its heavyweight worker.
    with (top/'.submission.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        try:
            for prior in top.iterdir():
                if prior.is_dir() and JOB_ID.fullmatch(prior.name) and read_job(root,prior.name)['state'] in ('QUEUED','RUNNING'):
                    raise RuntimeError('An agentic job is already active')
            job_id=uuid.uuid4().hex;directory=top/job_id;directory.mkdir(mode=0o700)
            _atomic_json(directory/'request.json',cleaned)
            _atomic_json(directory/'config.json',cfg)
            _atomic_json(directory/'status.json',{'job_id':job_id,'state':'QUEUED','seq':0,'stage':'queued',
                                                 'created_at':stamp(),'updated_at':stamp(),
                                                 'eligible_for_review_download':False,'eligible_for_deployment':False})
            (directory/'events.jsonl').touch(mode=0o600)
            _event(directory,'queued','CAPTAIN accepted a real model optimization mission',state='QUEUED')
            argv=[sys.executable,'-m','app.nimbo_agentic_jobs_v1','--root',str(root),'--job-id',job_id]
            try:
                with (directory/'agent.stderr.log').open('ab') as log:
                    process=spawn(argv,cwd=str(root),env={**os.environ,'PYTHONUNBUFFERED':'1'},stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL,stderr=log,start_new_session=True)
                _atomic_json(directory/'launcher.json',{'pid':process.pid,'at':stamp()})
            except Exception:
                _finish(directory,'FAILED','AGENT_START_FAILED','Could not launch local agent worker')
                raise RuntimeError('Could not start agent worker')
        finally:
            fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
    return {'job_id':job_id,'state':'QUEUED','status_url':f'/nimbo/agentic/jobs/{job_id}',
            'events_url':f'/nimbo/agentic/jobs/{job_id}/events','eligible_for_deployment':False}



def _engine_request(cleaned: dict):
    # M7's public exact field contract only. Never pass arbitrary LLM-generated text to tools.
    keys=('model_id','revision','pilot_file','heldout_file','preprocess','objective')
    return {key:cleaned[key] for key in keys}


def run_worker(root: Path, job_id: str, *, agent_model=None, engine_submit:Callable=submit_m7,
               engine_read:Callable=read_m7_job, engine_events:Callable=read_m7_events,
               engine_package:Callable=m7_verified_package, sleep:Callable=time.sleep, max_polls=3600):
    directory=folder(root,job_id);cfg=_read(directory/'config.json')
    role_model=agent_model or OllamaRoleModel(cfg)
    req=_read(directory/'request.json')
    active_id=None
    try:
        _event(directory,'analyze','CAPTAIN is evaluating the supported model and requested objective')
        planning={'model_id':req['model_id'],'revision':req['revision'],'objective':req['objective'],
                  'pilot_rows':req['pilot_rows'],'heldout_rows':req['heldout_rows'],
                  'available_tool':'run_onnx_pipeline','unsupported_methods':['arbitrary LoRA training','generic LLM quantization']}
        captain=_reviewed_decision(directory,role_model,'CAPTAIN','analyze',planning)
        if not gate_for_action('CAPTAIN',captain,planning):
            return _finish(directory,'COMPLETED_REJECTED','PLANNER_REJECTED','CAPTAIN refused unsupported optimization')
        forge=_reviewed_decision(directory,role_model,'FORGE','optimize',planning)
        if not gate_for_action('FORGE',forge,planning):
            return _finish(directory,'COMPLETED_REJECTED','FORGE_REJECTED','FORGE refused unsupported tool')
        for attempt in range(2):
            _event(directory,'optimize',f'FORGE invoked the existing real M7 CPU ONNX pipeline (attempt {attempt+1})')
            launched=engine_submit(root,_engine_request(req))
            active_id=launched['job_id']
            if not JOB_ID.fullmatch(active_id):raise RuntimeError('M7 returned invalid child job ID')
            checkpoint=0;final=None;observed=[]
            for _ in range(max_polls):
                events=engine_events(root,active_id)['events']
                for ev in events:
                    if ev.get('seq',0)>checkpoint:
                        checkpoint=ev['seq'];observed.append(ev)
                        _event(directory,ev.get('stage','optimize'), 'M7 engine: '+str(ev.get('message','recorded event'))[:440],
                               details={'engine_job_id':active_id,'engine_event_seq':checkpoint})
                final=engine_read(root,active_id)
                if final.get('state') in FINAL:break
                sleep(1)
            else:
                return _finish(directory,'FAILED','AGENT_JOB_TIMEOUT','Real optimizer did not finish within bounded polling',engine_job_id=active_id)
            error_type=extract_transient(observed)
            if final['state'] in ('FAILED','INTERRUPTED') and attempt==0 and error_type:
                diagnosis={'state':final['state'],'error_type':error_type,'retry_count':attempt,'objective':req['objective']}
                doctor=_reviewed_decision(directory,role_model,'DOCTOR','verify',diagnosis)
                if gate_for_action('DOCTOR',doctor,diagnosis):
                    _event(directory,'analyze','DOCTOR authorized ONE retry for a measured transient infrastructure failure')
                    continue
            break
        facts={'state':final['state'],'decision':final.get('decision'),
               'engine_job_id':active_id,'package_verified':False,
               'engine_reason':str(final.get('reason',''))[:200],
               'measured_metrics':summarize_measured_evidence(root,active_id)}
        if final['state']=='COMPLETED_REVIEW_ONLY':
            try:
                path=engine_package(root,active_id)
                facts['package_verified']=path.is_file() and path.stat().st_size>0
            except (ValueError,OSError,KeyError,FileNotFoundError):
                facts['package_verified']=False
        _event(directory,'verify','VERA is checking actual heldout and offline package evidence')
        vera=_reviewed_decision(directory,role_model,'VERA','verify',facts)
        if not gate_for_action('VERA',vera,facts):
            if final['state']=='COMPLETED_REJECTED':
                return _finish(directory,'COMPLETED_REJECTED','QUALITY_GATES_REJECTED','M7 rejected the candidate; baseline preserved',engine_job_id=active_id)
            return _finish(directory,'COMPLETED_REJECTED' if final['state']=='COMPLETED_REVIEW_ONLY' else 'FAILED',
                           'VERIFICATION_REJECTED','No approved package: M7 or VERA verification did not pass',engine_job_id=active_id)
        _event(directory,'verify','METRI is evaluating recorded evidence and its limitations')
        metri=_reviewed_decision(directory,role_model,'METRI','verify',facts)
        if not gate_for_action('METRI',metri,facts):
            return _finish(directory,'COMPLETED_REJECTED','EVIDENCE_REJECTED','METRI did not accept sufficient measured evidence',engine_job_id=active_id)
        # Verify AGAIN. Model decisions can veto, but never override package checksum checks.
        engine_package(root,active_id)
        return _finish(directory,'COMPLETED_REVIEW_ONLY','SELECT_ONNX_FP32_REVIEW_ONLY',
                       'M7 gates passed, VERA and METRI accepted evidence, checksum-verified package for review only',
                       engine_job_id=active_id,package_verified=True)
    except Exception as error:
        local=directory/'agent.error.log'
        import traceback
        local.write_text(traceback.format_exc()[-6000:],encoding='utf-8')
        os.chmod(local,0o600)
        return _finish(directory,'FAILED','AGENT_EXECUTION_FAILED',
                       'Agent execution failed; inspect protected local log ('+type(error).__name__+')',
                       engine_job_id=active_id)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--job-id',required=True)
    args=parser.parse_args();run_worker(Path(args.root).resolve(),args.job_id)

if __name__=='__main__':main()
