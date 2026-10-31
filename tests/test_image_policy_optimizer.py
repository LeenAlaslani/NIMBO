from experiments.image_policy_optimizer import metrics,diagnostic,select_policy,label

def rows():
    return [
     dict(id='0',score=.9,fast=True,standard=True,fast_s=.7,checker_s=.04,standard_s=4.0,category='simple'),
     dict(id='1',score=.2,fast=False,standard=True,fast_s=.7,checker_s=.04,standard_s=4.0,category='scene'),
     dict(id='2',score=.8,fast=False,standard=True,fast_s=.7,checker_s=.04,standard_s=4.0,category='complex'),
    ]

def test_detect_false_accept_and_useful_escalation():
    m=metrics(rows(),.65)
    assert m['false_accepts']==1 and m['useful_escalations']==1
    assert m['successes']==2

def test_quality_floor_leads_more_escalation():
    s=select_policy(rows(),quality_floor=1.,max_fallback=.7)
    assert s['selection']['successes']==3
    assert s['selection']['fallback_count']==2

def test_infeasible_policy_is_explicit():
    assert select_policy(rows(),quality_floor=1.,max_fallback=0)['status']=='no_feasible_adaptive_policy'

def test_binary_parse_rejects_missing():
    assert label('YES') is True
    try:label('')
    except ValueError:pass
    else:raise AssertionError('missing must fail')

def test_do_not_mark_test_as_validated():
    assert diagnostic(rows())['scope']=='development_diagnostics_only'
