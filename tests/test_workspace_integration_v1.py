import pytest
from tools.workspace_integration_v1 import (BASELINE,CANDIDATE,loopback_url,parse_stream,find_receipt)
import json


def pack(t,**kw):return json.dumps({'type':t,**kw})


def fixture_stream(*,guardrail='UNVERIFIED',recorded=True):
    events=[pack('event',stage=x,message='actual event') for x in ['analyze','route','execute','verify','deliver']]
    report={'comparison':{'sample_count':2,'all_labels_present':False,'baseline_model':BASELINE,
        'candidate_model':CANDIDATE,'evidence_recorded':recorded,'baseline':{'mean_inference_ms':8.5},'candidate':{'mean_inference_ms':6.4}},'guardrail':{'action':guardrail}}
    return '\n'.join(events+[pack('result',report=report)])+'\n'


def test_valid_stream_accepts_unverified_but_measured():
    data=parse_stream(fixture_stream())
    assert data['evidence_recorded'] is True
    assert data['guardrail']=='UNVERIFIED'
    assert 'deliver' in data['event_stages']


def test_refuses_claim_of_quality_for_unlabeled_inputs():
    with pytest.raises(ValueError,match='incorrectly represented'):
        parse_stream(fixture_stream(guardrail='VERIFIED'))


def test_refuses_missing_evidence_write():
    with pytest.raises(ValueError,match='acknowledge'):
        parse_stream(fixture_stream(recorded=False))


def test_refuses_backend_error_in_stream():
    with pytest.raises(ValueError,match='Backend emitted error'):
        parse_stream(fixture_stream()+pack('error',message='GPU not available'))


def test_refuses_absent_evidence_receipt():
    data={'records':[{'source':'baseline_comparisons_v15.jsonl','timestamp_utc':'2020','sample_count':2,'all_labels_present':False,'quality_certified':False}], 'runtime_summary':{}}
    assert find_receipt(data,timestamp='2020') is True
    assert find_receipt(data,timestamp='2021') is False
    data['records'][0]['quality_certified']=True
    assert find_receipt(data,timestamp='2020') is False


def test_loopback_only():
    assert loopback_url('http://localhost:5173/')=='http://localhost:5173'
    assert loopback_url('http://127.0.0.1:8000')=='http://127.0.0.1:8000'
    for url in ('https://localhost:5173','http://example.com:8000','http://127.0.0.1:8000@badhost:80',
                'http://127.0.0.1:8000/admin','http://localhost:5173?key=secret'):
        with pytest.raises(ValueError):loopback_url(url)


def test_requires_actual_result():
    with pytest.raises(ValueError,match='Expected exactly one result'):
        parse_stream(pack('event',stage='analyze')+'\n')
