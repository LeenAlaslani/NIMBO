"""M11 tests. Mocked engine tests prove orchestration and gates, NOT model execution."""
from __future__ import annotations
import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

SRC=Path(__file__).resolve().parents[1]/'app'
sys.path.insert(0,str(SRC))
from nimbo_agentic_core_v1 import (
    OllamaRoleModel, AgentDecision, AgentDecisionError, AgentConfigurationError,
    available_agent_config, validate_local_gateway,gate_for_action,extract_transient,
)


def d(role,action,reason='Verified evidence supports this choice'):
    return AgentDecision(role,action,reason,'local-test-model')

@pytest.mark.parametrize('url',[
    'https://api.example.com/api/chat','http://evil.local:11434/api/chat',
    'file:///tmp/model','http://127.0.0.1:11434/not-chat',
    'http://user:password@127.0.0.1:11434/api/chat',
])
def test_only_explicit_loopback_local_model(url):
    with pytest.raises(AgentConfigurationError):validate_local_gateway(url)


def test_requires_real_agent_model_configuration():
    with pytest.raises(AgentConfigurationError):available_agent_config({})
    result=available_agent_config({'NIMBO_AGENTIC_MODEL':'qwen2.5:7b'})
    assert result['model']=='qwen2.5:7b'


def test_quality_gate_never_overridden_by_llm():
    claimed=d('VERA','approve_for_review')
    assert not gate_for_action('VERA',claimed,{'state':'COMPLETED_REVIEW_ONLY','decision':'SELECT_ONNX_FP32_REVIEW_ONLY','package_verified':False})
    assert not gate_for_action('VERA',claimed,{'state':'COMPLETED_REJECTED','decision':'SELECT_ONNX_FP32_REVIEW_ONLY','package_verified':True})
    assert gate_for_action('VERA',claimed,{'state':'COMPLETED_REVIEW_ONLY','decision':'SELECT_ONNX_FP32_REVIEW_ONLY','package_verified':True})


def test_doctor_cannot_retry_quality_gate_or_unknown_errors():
    decision=d('DOCTOR','retry_same_inputs')
    assert not gate_for_action('DOCTOR',decision,{'state':'COMPLETED_REJECTED','error_type':'TimeoutError','retry_count':0})
    assert not gate_for_action('DOCTOR',decision,{'state':'FAILED','error_type':'ValueError','retry_count':0})
    assert not gate_for_action('DOCTOR',decision,{'state':'FAILED','error_type':'TimeoutError','retry_count':1})
    assert gate_for_action('DOCTOR',decision,{'state':'FAILED','error_type':'TimeoutError','retry_count':0})


def test_only_verified_error_type_can_trigger_retry():
    assert extract_transient([{'state':'FAILED','evidence':{'error_type':'TimeoutError'}}])=='TimeoutError'
    assert extract_transient([{'state':'FAILED','evidence':{'error_type':'ValueError'}}]) is None
    assert extract_transient([{'state':'COMPLETED_REJECTED','evidence':{'error_type':'TimeoutError'}}]) is None


def test_json_response_enforces_action_allowlist_and_valid_reason():
    class Response:
        def __init__(self,p):self.p=p
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def read(self,*_):return json.dumps({'message':{'content':json.dumps(self.p)}}).encode()
    gateway=OllamaRoleModel({'url':'http://127.0.0.1:11434/api/chat','model':'qwen2.5:7b'},
                           transport=lambda request,timeout:Response({'action':'run_onnx_pipeline','reason':'Use measured speed objective'}))
    assert gateway.decide('CAPTAIN',{'objective':'speed'}).action=='run_onnx_pipeline'
    gateway.transport=lambda request,timeout:Response({'action':'rm -rf /','reason':'unsafe operation'})
    with pytest.raises(AgentDecisionError):gateway.decide('CAPTAIN',{})


def import_worker_with_fake_engine(monkeypatch):
    pkg=types.ModuleType('app');pkg.__path__=[str(SRC)]
    legacy=types.ModuleType('app.nimbo_auto_pipeline_jobs_v1')
    legacy.validate_request=lambda root,request:{**request,'pilot_rows':20,'heldout_rows':200}
    legacy.submit=lambda root,request:{'job_id':'a'*32}
    legacy.read_job=lambda root,job_id:{'state':'COMPLETED_REVIEW_ONLY','decision':'SELECT_ONNX_FP32_REVIEW_ONLY'}
    legacy.read_events=lambda root,job_id:{'events':[]}
    legacy.package_for_download=lambda root,job_id:root/'verified.zip'
    monkeypatch.setitem(sys.modules,'app',pkg)
    monkeypatch.setitem(sys.modules,'app.nimbo_agentic_core_v1',sys.modules['nimbo_agentic_core_v1'])
    monkeypatch.setitem(sys.modules,'app.nimbo_auto_pipeline_jobs_v1',legacy)
    spec=importlib.util.spec_from_file_location('app.nimbo_agentic_jobs_v1',SRC/'nimbo_agentic_jobs_v1.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module,legacy


def make_job(mod,tmp_path):
    top=tmp_path/'results'/'agentic_jobs_v1'/('f'*32);top.mkdir(parents=True)
    req={'model_id':'org/model','revision':'a'*40,'pilot_file':'pilot.csv','heldout_file':'test.csv',
         'preprocess':'identity','objective':'speed','pilot_rows':20,'heldout_rows':200}
    mod._atomic_json(top/'request.json',req)
    mod._atomic_json(top/'config.json',{'model':'local-test','url':'http://127.0.0.1:11434/api/chat'})
    mod._atomic_json(top/'status.json',{'job_id':'f'*32,'state':'QUEUED','seq':0,'stage':'queued'})
    (top/'events.jsonl').touch()
    return top

class FakeRoleModel:
    def __init__(self,modify=None):self.roles=[];self.modify=modify or {}
    def decide(self,role,facts):
        self.roles.append(role)
        normal={'CAPTAIN':'run_onnx_pipeline','FORGE':'run_onnx_pipeline','VERA':'approve_for_review',
                'METRI':'record_evidence','DOCTOR':'retry_same_inputs'}
        return d(role,self.modify.get(role,normal[role]))


def test_real_engine_tool_invoked_and_verification_required(monkeypatch,tmp_path):
    mod,legacy=import_worker_with_fake_engine(monkeypatch)
    make_job(mod,tmp_path)
    package=tmp_path/'verified.zip';package.write_bytes(b'engine-only-package-not-for-production')
    calls=[]; model=FakeRoleModel()
    result=mod.run_worker(tmp_path,'f'*32,agent_model=model,
        engine_submit=lambda root,request:(calls.append(request),{'job_id':'a'*32})[1],
        engine_read=lambda root,j:{'state':'COMPLETED_REVIEW_ONLY','decision':'SELECT_ONNX_FP32_REVIEW_ONLY'},
        engine_events=lambda root,j:{'events':[{'seq':1,'stage':'analyze','state':'RUNNING','message':'Measured baseline'},
                                                {'seq':2,'stage':'deliver','state':'COMPLETED_REVIEW_ONLY','message':'Verified offline package'}]},
        engine_package=lambda root,j:package,sleep=lambda _:None,max_polls=2)
    assert result['state']=='COMPLETED_REVIEW_ONLY'
    assert result['eligible_for_review_download'] is True
    assert model.roles==['CAPTAIN','FORGE','VERA','METRI']
    assert calls[0]['objective']=='speed'
    assert mod.read_events(tmp_path,'f'*32)['events'][-1]['state']=='COMPLETED_REVIEW_ONLY'


def test_rejection_does_not_trigger_retry_or_approve(monkeypatch,tmp_path):
    mod,_=import_worker_with_fake_engine(monkeypatch);make_job(mod,tmp_path)
    calls=[];model=FakeRoleModel()
    result=mod.run_worker(tmp_path,'f'*32,agent_model=model,
        engine_submit=lambda root,request:(calls.append(1),{'job_id':'a'*32})[1],
        engine_read=lambda root,j:{'state':'COMPLETED_REJECTED','decision':'SELECT_ORIGINAL_NO_ONNX_PACKAGE'},
        engine_events=lambda root,j:{'events':[{'seq':1,'stage':'verify','state':'COMPLETED_REJECTED','message':'Accuracy gate failed'}]},
        engine_package=lambda *args:pytest.fail('should not fetch rejected package'),sleep=lambda _:None,max_polls=2)
    assert result['state']=='COMPLETED_REJECTED'
    assert result['eligible_for_review_download'] is False
    assert len(calls)==1
    assert 'DOCTOR' not in model.roles


def test_doctor_retries_once_only_transient_infrastructure(monkeypatch,tmp_path):
    mod,_=import_worker_with_fake_engine(monkeypatch);make_job(mod,tmp_path)
    package=tmp_path/'verified.zip';package.write_bytes(b'checked')
    calls=[];model=FakeRoleModel()
    def submit(root,req):
        calls.append(len(calls)+1)
        return {'job_id':str(calls[-1])*32}
    def status(root,j):
        if j.startswith('1'):return {'state':'FAILED','decision':'EXECUTION_FAILED'}
        return {'state':'COMPLETED_REVIEW_ONLY','decision':'SELECT_ONNX_FP32_REVIEW_ONLY'}
    def ev(root,j):
        if j.startswith('1'):return {'events':[{'seq':1,'stage':'error','state':'FAILED','message':'Connection interrupted','evidence':{'error_type':'TimeoutError'}}]}
        return {'events':[{'seq':1,'stage':'deliver','state':'COMPLETED_REVIEW_ONLY','message':'Reload verified'}]}
    result=mod.run_worker(tmp_path,'f'*32,agent_model=model,engine_submit=submit,engine_read=status,engine_events=ev,
                          engine_package=lambda root,j:package,sleep=lambda _:None,max_polls=2)
    assert len(calls)==2
    assert 'DOCTOR' in model.roles
    assert result['state']=='COMPLETED_REVIEW_ONLY'


def test_veto_prevents_download_even_when_engine_passes(monkeypatch,tmp_path):
    mod,_=import_worker_with_fake_engine(monkeypatch);make_job(mod,tmp_path)
    package=tmp_path/'verified.zip';package.write_bytes(b'checked')
    model=FakeRoleModel({'VERA':'reject'})
    result=mod.run_worker(tmp_path,'f'*32,agent_model=model,
        engine_submit=lambda root,req:{'job_id':'a'*32},
        engine_read=lambda root,j:{'state':'COMPLETED_REVIEW_ONLY','decision':'SELECT_ONNX_FP32_REVIEW_ONLY'},
        engine_events=lambda root,j:{'events':[]}, engine_package=lambda root,j:package,sleep=lambda _:None,max_polls=2)
    assert result['state']=='COMPLETED_REJECTED'
    assert result['eligible_for_review_download'] is False
    assert 'METRI' not in model.roles


def test_metrics_read_from_real_on_disk_reports_only(monkeypatch,tmp_path):
    from nimbo_agentic_core_v1 import summarize_measured_evidence
    job_id='a'*32
    path=tmp_path/'results'/'auto_optimizer_jobs_v1'/job_id
    path.mkdir(parents=True)
    metrics=tmp_path/'results'/'auto_optimizer_heldout_v1'/'heldout_fake.json'
    metrics.parent.mkdir(parents=True)
    metrics.write_text(json.dumps({
        'baseline':{'accuracy':0.71,'mean_warm_forward_ms':21.0},
        'candidate':{'accuracy':0.71,'mean_warm_forward_ms':8.4},
        'decision':{'heldout_gates_passed':True,'decision':'LOCAL_PASS'},
        'raw_text':'SENSITIVE DATA MUST NEVER LEAK'}))
    (path/'receipt.json').write_text(json.dumps({'job_id':job_id,'evidence':{'heldout_report':str(metrics)}}))
    value=summarize_measured_evidence(tmp_path,job_id)
    assert value['heldout_report']['baseline']['accuracy']==0.71
    assert value['heldout_report']['candidate']['mean_warm_forward_ms']==8.4
    assert 'raw_text' not in json.dumps(value)


def test_metrics_refuse_to_follow_symlink_out_of_results(tmp_path):
    from nimbo_agentic_core_v1 import summarize_measured_evidence
    job_id='b'*32
    path=tmp_path/'results'/'auto_optimizer_jobs_v1'/job_id
    path.mkdir(parents=True)
    secret=tmp_path/'secret.json';secret.write_text(json.dumps({'baseline':{'accuracy':1}}))
    linked=tmp_path/'results'/'outside.json';linked.symlink_to(secret)
    (path/'receipt.json').write_text(json.dumps({'job_id':job_id,'evidence':{'heldout_report':str(linked)}}))
    assert summarize_measured_evidence(tmp_path,job_id)=={}


def test_llm_probe_requires_installed_named_model():
    from nimbo_agentic_core_v1 import probe_local_model
    class R:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,*args):return json.dumps({'models':[{'name':'small:latest'}]}).encode()
    cfg={'url':'http://127.0.0.1:11434/api/chat','model':'qwen2.5:7b'}
    assert probe_local_model(cfg,transport=lambda req,timeout:R()) is False
    cfg['model']='small:latest'
    assert probe_local_model(cfg,transport=lambda req,timeout:R()) is True


def test_optimizer_modes_share_one_mission_floor():
    src = SRC / 'web' / 'src'
    floor = (src / 'GameMissionFloor.jsx').read_text(encoding='utf-8')
    mission = (src / 'UnifiedAutoMission.jsx').read_text(encoding='utf-8')

    assert floor.count('<UnifiedAutoMission ') == 1
    assert 'MULTI-AGENT MODEL FORGE' in mission
    assert "preflight.ready?'/nimbo/agentic/jobs':'/nimbo/auto/jobs'" in mission
    assert mission.count('<NimboGardenWorld ') == 1
    assert 'selectedRecordSeq={selectedSeq}' in mission

# M11.1 - audited role explanations: regression for the 2026-11-14 live Qwen receipt.
def accepted_facts():
    return {'state':'COMPLETED_REVIEW_ONLY',
            'decision':'SELECT_ONNX_FP32_REVIEW_ONLY',
            'package_verified':True,
            'measured_metrics':{'heldout_report':{'decision':{'heldout_gates_passed':True}}}}


def test_real_metri_contradiction_is_flagged_before_publishing():
    from nimbo_agentic_core_v1 import check_reason_consistency
    wrong=d('METRI','record_evidence',
            'Evidence supports rejection based on local gates and heldout quality gates passing, '
            'with significant latency reduction and no process CPU time increase.')
    assert 'claims_rejection_despite_verified_backend_pass' in check_reason_consistency('METRI',wrong,accepted_facts())


def test_vera_cannot_impersonate_metri():
    from nimbo_agentic_core_v1 import check_reason_consistency
    wrong=d('VERA','approve_for_review',
            'METRI: Heldout quality gates passed, evidence supports local gates passing. '
            'QUALITY_GATES: No hard constraints violated.')
    assert 'other_agent_impersonation' in check_reason_consistency('VERA',wrong,accepted_facts())


def test_rejected_engine_cannot_be_described_as_approved_and_no_fake_percentages():
    from nimbo_agentic_core_v1 import check_reason_consistency
    rejected={'state':'COMPLETED_REJECTED','decision':'SELECT_ORIGINAL_NO_ONNX_PACKAGE','package_verified':False}
    wrong=d('METRI','reject','Heldout quality gates passed and the candidate was approved for review.')
    assert 'claims_approval_despite_backend_rejection' in check_reason_consistency('METRI',wrong,rejected)
    invented=d('METRI','record_evidence','Measured a 55% reduction in cloud cost; production ready.')
    issues=check_reason_consistency('METRI',invented,accepted_facts())
    assert 'unverified_numerical_or_cost_claim' in issues
    assert 'unverified_production_claim' in issues


def test_grounded_event_headline_does_not_publish_llm_claim(monkeypatch,tmp_path):
    mod,_=import_worker_with_fake_engine(monkeypatch);top=make_job(mod,tmp_path)
    from nimbo_agentic_core_v1 import AgentDecision
    facts=accepted_facts()
    decision=AgentDecision('METRI','record_evidence','The gates passed but I like the purple robot.','test-llm')
    mod._decision(top,decision,'verify',facts)
    entry=mod.read_events(tmp_path,'f'*32)['events'][-1]
    assert 'purple robot' not in entry['message']
    assert 'review-only gates' in entry['message']
    assert entry['evidence']['llm_reason_not_evidence']==decision.reason
    assert entry['evidence']['reason_check']=='bounded_consistency_pass'


def test_reviewed_metri_reprompts_once_without_rerunning_m7(monkeypatch,tmp_path):
    mod,_=import_worker_with_fake_engine(monkeypatch);make_job(mod,tmp_path)
    package=tmp_path/'verified.zip';package.write_bytes(b'checked')
    class Correcting(FakeRoleModel):
        def __init__(self):super().__init__();self.metri_attempts=0
        def decide(self,role,facts):
            self.roles.append(role)
            if role=='METRI':
                self.metri_attempts+=1
                if self.metri_attempts==1:
                    assert 'reasoning_feedback' not in facts
                    return d('METRI','record_evidence','Evidence supports rejection based on gates passing.')
                assert facts['reasoning_feedback']['issues']==['claims_rejection_despite_verified_backend_pass']
                return d('METRI','record_evidence','Heldout quality gates passed; evidence supports review-only approval.')
            return d(role,{'CAPTAIN':'run_onnx_pipeline','FORGE':'run_onnx_pipeline',
                           'VERA':'approve_for_review'}[role])
    calls=[];model=Correcting()
    result=mod.run_worker(tmp_path,'f'*32,agent_model=model,
        engine_submit=lambda root,req:(calls.append(req),{'job_id':'a'*32})[1],
        engine_read=lambda root,j:{'state':'COMPLETED_REVIEW_ONLY','decision':'SELECT_ONNX_FP32_REVIEW_ONLY'},
        engine_events=lambda root,j:{'events':[]},engine_package=lambda root,j:package,
        sleep=lambda _:None,max_polls=2)
    assert result['state']=='COMPLETED_REVIEW_ONLY'
    assert result['eligible_for_review_download'] is True
    assert len(calls)==1 # Explanation correction must never rerun the expensive model.
    assert model.metri_attempts==2
    events=mod.read_events(tmp_path,'f'*32)['events']
    assert any('requesting one correction' in e['message'] for e in events)
    assert not any('supports rejection' in e['message'] for e in events)


def test_repeated_contradiction_fails_closed_and_preserves_m7(monkeypatch,tmp_path):
    mod,_=import_worker_with_fake_engine(monkeypatch);make_job(mod,tmp_path)
    package=tmp_path/'verified.zip';package.write_bytes(b'checked')
    class Persistent(FakeRoleModel):
        def decide(self,role,facts):
            self.roles.append(role)
            if role=='METRI':
                return d('METRI','record_evidence','Evidence supports rejection based on passed heldout gates.')
            return d(role,{'CAPTAIN':'run_onnx_pipeline','FORGE':'run_onnx_pipeline',
                           'VERA':'approve_for_review'}[role])
    calls=[];model=Persistent()
    result=mod.run_worker(tmp_path,'f'*32,agent_model=model,
        engine_submit=lambda root,req:(calls.append(req),{'job_id':'a'*32})[1],
        engine_read=lambda root,j:{'state':'COMPLETED_REVIEW_ONLY','decision':'SELECT_ONNX_FP32_REVIEW_ONLY'},
        engine_events=lambda root,j:{'events':[]},engine_package=lambda root,j:package,
        sleep=lambda _:None,max_polls=2)
    assert result['state']=='FAILED'
    assert result['eligible_for_review_download'] is False
    assert result['engine_job_id']=='a'*32
    assert len(calls)==1
    assert model.roles.count('METRI')==2
    with pytest.raises(ValueError):mod.package_for_download(tmp_path,'f'*32)
    events=mod.read_events(tmp_path,'f'*32)['events']
    assert any('still contradicted' in e['message'] for e in events)
    assert not any('supports rejection' in e['message'] for e in events)


def test_vera_role_label_correction_works(monkeypatch,tmp_path):
    mod,_=import_worker_with_fake_engine(monkeypatch);top=make_job(mod,tmp_path)
    class CorrectingVera:
        def __init__(self):self.attempt=0
        def decide(self,role,facts):
            assert role=='VERA'
            self.attempt+=1
            if self.attempt==1:return d(role,'approve_for_review','METRI: Heldout gates passed; approve for review.')
            assert 'other_agent_impersonation' in facts['reasoning_feedback']['issues']
            return d(role,'approve_for_review','M7 heldout quality gates passed for review only.')
    role=CorrectingVera()
    entry=mod._reviewed_decision(top,role,'VERA','verify',accepted_facts())
    assert entry.action=='approve_for_review' and role.attempt==2
    events=mod.read_events(tmp_path,'f'*32)['events']
    assert any('requesting one correction' in e['message'] for e in events)
    assert all('METRI:' not in e['message'] for e in events)


def test_vera_can_veto_verified_candidate_with_actual_concerns():
    # LLM veto is allowed; it cannot override the M7 approval gate in the other direction.
    from nimbo_agentic_core_v1 import check_reason_consistency
    veto=d('VERA','reject','Insufficient evidence for this specific external deployment request.')
    assert check_reason_consistency('VERA',veto,accepted_facts())==[]
    assert gate_for_action('VERA',veto,accepted_facts()) is False


def test_metri_does_not_record_wrong_backend_decision():
    facts=accepted_facts();facts['decision']='EXECUTION_FAILED'
    assert gate_for_action('METRI',d('METRI','record_evidence'),facts) is False


def test_failed_engine_must_not_claim_approved_package():
    from nimbo_agentic_core_v1 import check_reason_consistency
    failed={'state':'FAILED','decision':'EXECUTION_FAILED','package_verified':False}
    claim=d('VERA','reject','The quality gates passed; this package is approved for review.')
    assert 'claims_approval_despite_backend_rejection' in check_reason_consistency('VERA',claim,failed)


# M11_2_BOUNDED_DECISION_RECOVERY

def test_m112_responds_to_out_of_policy_during_vera_correction(tmp_path):
    import json
    from app.nimbo_agentic_core_v1 import AgentDecision, AgentDecisionError
    from app.nimbo_agentic_jobs_v1 import _reviewed_decision

    (tmp_path / "status.json").write_text(
        json.dumps({"seq": 0, "state": "RUNNING"})
    )
    (tmp_path / "events.jsonl").touch()

    facts = {
        "state": "COMPLETED_REVIEW_ONLY",
        "decision": "SELECT_ONNX_FP32_REVIEW_ONLY",
        "package_verified": True,
    }

    class RecoveryModel:
        calls = 0

        def decide(self, role, context):
            self.calls += 1
            if self.calls == 1:
                return AgentDecision(
                    role, "approve_for_review",
                    "Evidence supports rejection despite passed heldout gates.",
                    "test-model"
                )
            if self.calls == 2:
                raise AgentDecisionError(
                    "VERA returned an out-of-policy decision"
                )
            return AgentDecision(
                role, "approve_for_review",
                "Heldout gates passed and package verified for review only.",
                "test-model"
            )

    model = RecoveryModel()
    decision = _reviewed_decision(
        tmp_path, model, "VERA", "verify", facts
    )

    assert model.calls == 3
    assert decision.action == "approve_for_review"


def test_m112_repeated_out_of_policy_fails_closed(tmp_path):
    import json
    from app.nimbo_agentic_core_v1 import AgentDecisionError
    from app.nimbo_agentic_jobs_v1 import _reviewed_decision

    (tmp_path / "status.json").write_text(
        json.dumps({"seq": 0, "state": "RUNNING"})
    )
    (tmp_path / "events.jsonl").touch()

    facts = {
        "state": "COMPLETED_REVIEW_ONLY",
        "decision": "SELECT_ONNX_FP32_REVIEW_ONLY",
        "package_verified": True,
    }

    class InvalidModel:
        calls = 0

        def decide(self, role, context):
            self.calls += 1
            raise AgentDecisionError(
                "VERA returned an out-of-policy decision"
            )

    model = InvalidModel()

    try:
        _reviewed_decision(
            tmp_path, model, "VERA", "verify", facts
        )
    except AgentDecisionError:
        pass
    else:
        raise AssertionError("Out-of-policy decisions were accepted")

    assert model.calls == 2

    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert not any(
        "VERA decision:" in e["message"] for e in events
    )
