import json
import pytest
from nimbo.policy_runtime import load_text_policy, run_text_policy


def report(tmp_path, kind='adaptive', threshold=.99):
    path=tmp_path/'report.json'
    path.write_text(json.dumps({'selected_policy':{'type':kind,'threshold':threshold},
                                'models':{'small':'checkpoint-small','large':'checkpoint-large'},
                                'test_strategies': {'nimbo': {'accuracy': .1}}}))
    return path


def test_calibrated_policy_routes_low_confidence(tmp_path):
    policy=load_text_policy(report(tmp_path))
    calls=[]
    def small(text):
        calls.append('small');return {'label':'NEGATIVE','confidence': .96}
    def large(text):
        calls.append('large');return {'label':'POSITIVE','confidence': .99}
    result=run_text_policy('hello',policy,small,large)
    assert calls == ['small','large']
    assert result.decision['route']=='small-then-large'
    assert result.result['label']=='POSITIVE'
    assert result.events[0]['state']=='policy_loaded'


def test_adaptive_high_confidence_stays_small(tmp_path):
    policy=load_text_policy(report(tmp_path))
    calls=[]
    def small(text): calls.append('small');return {'label':'POSITIVE','confidence':.995}
    def large(text): calls.append('large');return {'label':'NEGATIVE','confidence': .99}
    trace=run_text_policy('hello',policy,small,large)
    assert calls==['small']
    assert trace.decision['route']=='small-first'


def test_small_only_never_runs_large(tmp_path):
    policy=load_text_policy(report(tmp_path,'small_only',None))
    trace=run_text_policy('hello',policy,lambda _: {'label':'POSITIVE','confidence':.51},
                          lambda _: (_ for _ in ()).throw(AssertionError('large called')))
    assert trace.decision['route']=='small-only'


def test_reject_invalid_policy(tmp_path):
    with pytest.raises(ValueError):
        load_text_policy(report(tmp_path,'adaptive',1.2))
    with pytest.raises(ValueError):
        load_text_policy(report(tmp_path,'arbitrary',.9))


def test_empty_input(tmp_path):
    policy=load_text_policy(report(tmp_path))
    with pytest.raises(ValueError):
        run_text_policy(' ',policy,lambda _:None,lambda _:None)
