from experiments.cost_per_success import evaluate


def rec(i, route, fast, std):
    fallback = route == 'escalated_to_standard'
    return dict(prompt_id=str(i), route=route, fast_s='1', standard_s='3', checker_s='.2',
                nimbo_s=str(1.2+(3 if fallback else 0)),
                human_fast_object_present=fast, human_standard_object_present=std)


def test_unlabeled_refuses_quality():
    result=evaluate([rec(0,'fast_accepted','','')])
    assert result['quality_validated'] is False
    assert result['strategies']['nimbo']['latency_seconds_per_success'] is None


def test_quality_and_cost():
    result=evaluate([rec(0,'fast_accepted','yes','yes'),rec(1,'escalated_to_standard','no','yes')])
    assert result['quality_validated']
    assert result['strategies']['nimbo']['successful_tasks']==2
    assert result['strategies']['nimbo']['latency_seconds_per_success']==2.7
    assert result['strategies']['always_fast']['latency_seconds_per_success']==2
    assert result['false_accept_count']==0


def test_false_accept_and_unneeded_fallback():
    result=evaluate([rec(0,'fast_accepted','no','yes'),rec(1,'escalated_to_standard','yes','yes')])
    assert result['false_accept_count']==1
    assert result['unnecessary_fallback_count']==1


def test_detects_bad_timing():
    import pytest
    row=rec(0,'fast_accepted','yes','yes')
    row['nimbo_s']='.1'
    with pytest.raises(ValueError): evaluate([row])
