import asyncio,hashlib
from types import SimpleNamespace
from fastapi import HTTPException
import pytest
import app.nimbo_company_model_v1 as cm

class FakeRequest:
    client=SimpleNamespace(host='127.0.0.1')
    headers={'content-type':'application/octet-stream'}
    def __init__(self,body):self.body=body
    async def stream(self):
        yield self.body[:3]
        yield self.body[3:]

def test_import_real_staged_bytes_and_reject_others(tmp_path,monkeypatch):
    model=b'fake_bytes_are_not_a_real_onnx_model'
    monkeypatch.setattr(cm,'STAGING',tmp_path)
    monkeypatch.setattr(cm,'CANDIDATE',tmp_path/'supported_company_candidate.onnx')
    monkeypatch.setattr(cm,'MIN_BYTES',8)
    monkeypatch.setattr(cm,'EXPECTED_SHA256',hashlib.sha256(model).hexdigest())
    result=asyncio.run(cm.import_model(FakeRequest(model)))
    assert result['imported'] is True
    assert cm.CANDIDATE.read_bytes()==model
    assert cm.is_staged_and_verified() is True
    with pytest.raises(HTTPException) as exc:
        asyncio.run(cm.import_model(FakeRequest(b'incorrect_different_model')))
    assert exc.value.status_code==422
    assert cm.CANDIDATE.read_bytes()==model # rejected input never overwrites genuine import

def test_evaluation_consumes_validated_rows_and_returns_real_scoped_report(tmp_path,monkeypatch):
    monkeypatch.setattr(cm,'STAGING',tmp_path)
    monkeypatch.setattr(cm,'is_staged_and_verified',lambda:True)
    seen={}
    def run(rows):
        seen['rows']=rows
        return {'sample_count':len(rows),'baseline':{'accuracy':1,'mean_warm_ms':10.0},'candidate':{'accuracy':1,'mean_warm_ms':8.0},'prediction_agreement':1,'warm_latency_reduction_fraction':.2}
    monkeypatch.setattr(cm,'real_pilot_inference',run)
    example=cm.PilotRequest(examples=[cm.Example(text='a',expected_label='NEGATIVE'),cm.Example(text='b',expected_label='POSITIVE')])
    req=SimpleNamespace(client=SimpleNamespace(host='127.0.0.1'))
    result=cm.evaluate(req,example)
    assert seen['rows']==[('a',0),('b',1)]
    assert result['deployed'] is False
    assert result['assessment']=='EXPLORATORY_ONLY_NOT_HELDOUT_CERTIFICATION'
    assert (tmp_path/'latest_pilot_summary.json').exists()
