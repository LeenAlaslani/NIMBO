import json
from types import SimpleNamespace
import pytest

from tools.adoption_proof_v1 import validate_response, audit_rolled_back, run_flow, cleanup_state
from tools.adoption_proof_service_v1 import check_proof_directory


def response(mode,route,paired,**kwargs):
    return {'mode_before':mode,'route_served':route,'comparison_performed':paired,
            'parity_status':'PASS','result_label':'POSITIVE','production_deployed':False,
            'auto_rollback':False,**kwargs}


def test_validate_serving_routes_and_fail_closed():
    assert validate_response(response('shadow','baseline',True),mode='shadow',served='baseline',comparison=True)
    with pytest.raises(ValueError):validate_response(response('shadow','candidate_onnx',True),mode='shadow',served='baseline',comparison=True)
    with pytest.raises(ValueError):validate_response(response('shadow','baseline',False),mode='shadow',served='baseline',comparison=True)
    with pytest.raises(RuntimeError):validate_response(response('shadow','baseline',True,rollback_reason='mismatch',auto_rollback=True),mode='shadow',served='baseline',comparison=True)


def test_rollback_must_be_baseline_and_no_production():
    assert audit_rolled_back({'state':{'mode':'baseline','run_id':None},'production_deployed':False,'changes_existing_nimbo_routing':False})
    with pytest.raises(RuntimeError):audit_rolled_back({'state':{'mode':'canary','run_id':'x'},'production_deployed':False,'changes_existing_nimbo_routing':False})
    with pytest.raises(RuntimeError):audit_rolled_back({'state':{'mode':'baseline','run_id':None},'production_deployed':True,'changes_existing_nimbo_routing':False})


def test_service_isolation_requires_session_path(tmp_path, monkeypatch):
    from tools import adoption_proof_service_v1 as service
    root=tmp_path/'results'/'adoption_proof_v1'
    root.mkdir(parents=True)
    session=root/'session_1';session.mkdir()
    monkeypatch.setattr(service,'ROOT',root.resolve())
    assert check_proof_directory(str(session))==session.resolve()
    with pytest.raises(ValueError):check_proof_directory(str(tmp_path))
    with pytest.raises(ValueError):check_proof_directory(str(root/'something'))
    with pytest.raises(ValueError):check_proof_directory(None)


def fake_state(mode, count=0):
    return {'state':{'mode':mode,'run_id':None if mode=='baseline' else 'heldout_example',
                      'shadow_count':count},'production_deployed':False,
            'changes_existing_nimbo_routing':False,
            'shadow_metrics':{'paired_samples':count,'reduction_fraction':.12}}


def test_real_routes_flow_with_stubbed_transport():
    counts={'shadow':0,'canary':0}
    def transport(url,payload=None,timeout=10.0):
        if url.endswith('/status'):return fake_state('shadow',counts['shadow'])
        if url.endswith('/rollback'):return fake_state('baseline')
        if url.endswith('/shadow'):return fake_state('shadow')
        if url.endswith('/canary'):return fake_state('canary')
        if url.endswith('/classify'):
            if counts['shadow']<6:
                if counts['shadow']==0:return response('baseline','baseline',False)
                counts['shadow']+=1
                return response('shadow','baseline',True)
        raise AssertionError(url)
    # Drive a normal flow without touching disk or model downloads.
    counter={'calls':0,'canary':0}
    def simulated(url,payload=None,timeout=10.0):
        if url.endswith('/status'):
            return fake_state('baseline') if counter['calls']==0 else fake_state('shadow',6)
        if url.endswith('/shadow'):return fake_state('shadow')
        if url.endswith('/canary'):return fake_state('canary')
        if url.endswith('/classify'):
            counter['calls']+=1
            if counter['calls']==1:return response('baseline','baseline',False)
            if counter['calls']<=7:return response('shadow','baseline',True)
            counter['canary']+=1
            return response('canary','candidate_onnx',counter['canary']<=5)
        raise AssertionError(url)
    flow=run_flow('http://127.0.0.1:9','heldout_example',request=simulated)
    assert flow['outcome']=='CANARY_PASS' and flow['canary_pairs']==5 and flow['canary_candidate_only']==2


def test_legitimate_gate_denial_is_not_faked():
    from urllib.error import HTTPError
    counter={'calls':0}
    def denied(url,payload=None,timeout=10.0):
        if url.endswith('/status'):return fake_state('baseline' if counter['calls']==0 else 'shadow',6)
        if url.endswith('/shadow'):return fake_state('shadow')
        if url.endswith('/canary'):raise HTTPError(url,409,'Conflict',{},None)
        if url.endswith('/classify'):
            counter['calls']+=1
            return response('baseline' if counter['calls']==1 else 'shadow','baseline',counter['calls']>1)
        raise AssertionError(url)
    outcome=run_flow('http://127.0.0.1:9','heldout_example',request=denied)
    assert outcome['outcome']=='GATE_BLOCKED'


def test_manual_rollback_routes_back_to_baseline():
    def sample(url,payload=None,timeout=10.0):
        if url.endswith('/rollback'):return fake_state('baseline')
        if url.endswith('/classify'):return response('baseline','baseline',False)
        raise AssertionError(url)
    assert cleanup_state('http://127.0.0.1:9', request=sample)


def test_actual_adoption_http_controls_with_stubbed_model(monkeypatch, tmp_path):
    """HTTP route wiring with simulated inference; no model download in CI."""
    from fastapi.testclient import TestClient
    from tools import adoption_proof_service_v1 as service
    from app import nimbo_adoption_v21_1 as a
    monkeypatch.setattr(service, 'prepare_proof', lambda: {'ready':True,'service_kind':service.SERVICE_KIND})
    monkeypatch.setattr(a,'STATE',tmp_path/'state.json')
    monkeypatch.setattr(a,'EVENTS',tmp_path/'events.jsonl')
    monkeypatch.setattr(a,'TRIALS',tmp_path/'requests.jsonl')
    monkeypatch.setattr(a,'_review', lambda run_id: {'run_id':run_id})
    monkeypatch.setattr(a,'_reviews', lambda: [{'run_id':'heldout_fake'}])
    monkeypatch.setattr(a,'_infer',lambda text,which,review: {
        'class_id':1,'label':'POSITIVE','positive_probability':.95,
        'confidence':.95,'warm_forward_ms':12 if which=='baseline' else 7,
        'tokenizer_plus_forward_ms':12 if which=='baseline' else 7})
    with TestClient(service.app) as client:
        prefix='/nimbo/adoption/v21'
        assert client.get('/proof/health').status_code == 200
        assert client.get(prefix+'/status').json()['state']['mode']=='baseline'
        assert client.post(prefix+'/canary',json={'run_id':'heldout_fake','confirmation':'ACTIVATE LOCAL CANARY'}).status_code == 409
        assert client.post(prefix+'/shadow',json={'run_id':'heldout_fake','confirmation':'wrong'}).status_code == 422
        assert client.post(prefix+'/shadow',json={'run_id':'heldout_fake','confirmation':'START LOCAL SHADOW'}).status_code == 200
        for _ in range(6):
            x=client.post(prefix+'/classify',json={'text':'good movie'})
            assert x.status_code==200 and x.json()['route_served']=='baseline' and x.json()['comparison_performed']
        x=client.post(prefix+'/canary',json={'run_id':'heldout_fake','confirmation':'ACTIVATE LOCAL CANARY'})
        assert x.status_code==200
        paired=0; unpaired=0
        for _ in range(7):
            x=client.post(prefix+'/classify',json={'text':'good movie'})
            assert x.status_code==200 and x.json()['route_served']=='candidate_onnx'
            paired+=int(x.json()['comparison_performed'])
            unpaired+=int(not x.json()['comparison_performed'])
        assert paired==5 and unpaired==2
        x=client.post(prefix+'/rollback',json={'confirmation':'ROLL BACK TO BASELINE'})
        assert x.status_code==200 and x.json()['state']['mode']=='baseline'
        assert client.post(prefix+'/classify',json={'text':'good movie'}).json()['route_served']=='baseline'


def test_actual_adoption_automatic_rollback_on_disagreement(monkeypatch,tmp_path):
    from fastapi.testclient import TestClient
    from tools import adoption_proof_service_v1 as service
    from app import nimbo_adoption_v21_1 as a
    monkeypatch.setattr(service,'prepare_proof',lambda:{'ready':True,'service_kind':service.SERVICE_KIND})
    monkeypatch.setattr(a,'STATE',tmp_path/'state.json')
    monkeypatch.setattr(a,'EVENTS',tmp_path/'events.jsonl')
    monkeypatch.setattr(a,'TRIALS',tmp_path/'requests.jsonl')
    monkeypatch.setattr(a,'_review',lambda rid:{'run_id':rid})
    monkeypatch.setattr(a,'_reviews',lambda:[{'run_id':'heldout_fake'}])
    monkeypatch.setattr(a,'_infer',lambda text,which,review: {
        'class_id':int(which=='baseline'),'label':'POSITIVE' if which=='baseline' else 'NEGATIVE',
        'positive_probability':.99 if which=='baseline' else .01,
        'confidence':.99,'warm_forward_ms':10,'tokenizer_plus_forward_ms':10})
    with TestClient(service.app) as client:
        prefix='/nimbo/adoption/v21'
        assert client.post(prefix+'/shadow',json={'run_id':'heldout_fake','confirmation':'START LOCAL SHADOW'}).status_code==200
        res=client.post(prefix+'/classify',json={'text':'test'})
        assert res.status_code==200
        assert res.json()['route_served']=='baseline'
        assert res.json()['auto_rollback'] is True
        assert client.get(prefix+'/status').json()['state']['mode']=='baseline'
