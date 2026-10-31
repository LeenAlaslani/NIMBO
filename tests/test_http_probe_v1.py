import pytest
from tools.http_probe_v1 import run_http_batch, summarize
from tools.http_probe_service_v1 import inference, local_only, InferRequest


def test_invalid_batch_fails_closed():
    with pytest.raises(ValueError):run_http_batch('http://127.0.0.1','bad',['x']*8,1)
    with pytest.raises(ValueError):run_http_batch('http://127.0.0.1','baseline',['x']*3,1)
    with pytest.raises(ValueError):run_http_batch('http://127.0.0.1','baseline',['x']*8,10)


def test_http_transport_and_predictions():
    def mock_transport(url,payload,timeout):
        return {'path':payload['path'], 'class_id':int('great' in payload['text']),
                'server_processing_ms':1.0,'production_deployed':False}
    stats,preds=run_http_batch('http://127.0.0.1:1','candidate',['great' if i%2 else 'bad' for i in range(8)],2,request_fn=mock_transport)
    assert stats['successes']==8 and stats['errors']==0
    assert sorted(preds)==list(range(8))
    assert stats['throughput_successes_per_sec']>0


def test_http_errors_not_fake_success():
    def fail(url,payload,timeout):
        if payload['text']=='bad':raise TimeoutError('test timeout')
        return {'path':payload['path'],'class_id':0,'server_processing_ms':1,'production_deployed':False}
    stats,preds=run_http_batch('http://127.0.0.1:1','candidate',['bad']+['good']*7,4,request_fn=fail)
    assert stats['errors']==1 and stats['successes']==7
    assert 0 not in preds


def test_http_rejects_wrong_path_or_malformed_label():
    def bad(url,payload,timeout):
        return {'path':'baseline','class_id':None,'server_processing_ms':1,'production_deployed':False}
    stats,_=run_http_batch('http://127.0.0.1:1','candidate',['good']*8,2,request_fn=bad)
    assert stats['errors']==8


def test_service_rejects_nonloopback_and_origin():
    from fastapi import HTTPException
    from types import SimpleNamespace
    with pytest.raises(HTTPException) as exc:
        local_only(SimpleNamespace(client=SimpleNamespace(host='10.0.0.1'),headers={}))
    assert exc.value.status_code==403
    with pytest.raises(HTTPException):
        local_only(SimpleNamespace(client=SimpleNamespace(host='127.0.0.1'),headers={'origin':'https://example.com'}))


def test_http_payload_scope():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):InferRequest(path='third_party',text='hello')
    with pytest.raises(ValidationError):InferRequest(path='baseline',text='')


def test_actual_fastapi_routes_with_stubbed_engine(monkeypatch):
    """Route wiring and request schema; no real model download in CI."""
    from fastapi.testclient import TestClient
    from tools import http_probe_service_v1 as service

    monkeypatch.setattr(service, 'initialize_engine', lambda: (
        {'sentinel': True}, {'run_id': 'heldout_test', 'model_revision': 'rev',
                            'artifact_sha256': 'hash', 'baseline_threads': 4, 'candidate_threads': 2}))
    monkeypatch.setattr(service, 'inference', lambda eng, which, txt: (
        {'class_id': int('good' in txt), 'engine_forward_ms': 1., 'engine_pipeline_ms': 2.}))
    with TestClient(service.app) as client:
        result = client.get('/health')
        assert result.status_code == 200 and result.json()['ready']
        reply = client.post('/infer', json={'path':'candidate','text':'good movie'})
        assert reply.status_code == 200
        assert reply.json()['label'] == 'POSITIVE' and reply.json()['production_deployed'] is False
        assert client.post('/infer', json={'path':'other','text':'movie'}).status_code == 422
        assert client.post('/infer', json={'path':'baseline','text':''}).status_code == 422
        assert client.post('/infer', json={'path':'baseline','text':'movie'}, headers={'Origin':'https://outside.example'}).status_code == 403
