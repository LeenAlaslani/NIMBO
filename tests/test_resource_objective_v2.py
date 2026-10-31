"""CPU resource-objective gate regression checks; no pretrained models required."""
import pytest
from app.nimbo_validate_v21 import ValidationRequest, gate_v21
from nimbo.resource_policy_v2 import assess_record

def m(acc=.9175,wall=30,local=31,p95=40,cpu=None):
    d={'accuracy_on_reference_labels':acc,
       'warm_forward':{'mean_ms':wall,'p95_ms':p95},
       'tokenizer_plus_forward':{'mean_ms':local}}
    if cpu is not None:d['process_cpu_forward']={'mean_ms':cpu}
    return d

def rs():
    return [{'baseline_forward_mean_ms':30,'candidate_forward_mean_ms':20} for _ in range(3)]

def test_cpu_budget_rejects_measured_regression_but_latency_allows_review():
    b,c=m(cpu=50),m(wall=20,local=21,p95=30,cpu=60)
    latency=gate_v21(b,c,1.0,rs(),ValidationRequest(optimization_objective='latency'))
    resource=gate_v21(b,c,1.0,rs(),ValidationRequest(optimization_objective='cpu_time'))
    assert latency['eligible'] and not resource['eligible']
    assert latency['cpu_pass'] is False and resource['cpu_gate_required'] is True
    assert resource['eligible_for_latency_review'] and not resource['eligible_for_cpu_time_review']
    assert resource['cpu_time_reduction_fraction']==-.2
    assert 'CPU time' in resource['reason']

def test_cpu_objective_rejects_missing_cpu_measurements():
    r=gate_v21(m(),m(wall=20,local=21,p95=30),1,rs(),ValidationRequest(optimization_objective='cpu_time'))
    assert not r['eligible'] and not r['cpu_time_measured']
    assert 'not measured' in r['reason']

def test_cpu_budget_accepts_measured_nonregression():
    r=gate_v21(m(cpu=50),m(wall=20,local=21,p95=30,cpu=45),1,rs(),ValidationRequest(optimization_objective='cpu_time'))
    assert r['eligible'] and r['cpu_pass']
    assert r['cpu_time_reduction_fraction']==.1

def test_quality_cannot_be_bypassed_by_lower_cpu():
    r=gate_v21(m(cpu=50),m(acc=.7,wall=20,local=21,p95=30,cpu=30),.7,rs(),ValidationRequest(optimization_objective='cpu_time'))
    assert not r['eligible'] and not r['quality_pass']

def test_p95_failure_cannot_be_bypassed_by_lower_cpu():
    r=gate_v21(m(cpu=50),m(wall=20,local=21,p95=60,cpu=30),1,rs(),ValidationRequest(optimization_objective='cpu_time'))
    assert not r['eligible'] and not r['p95_pass']

def test_nan_and_zero_cpu_cannot_claim_cpu_reduction():
    for bad in (0,float('nan'),float('inf')):
        r=gate_v21(m(cpu=bad),m(wall=20,local=21,p95=30,cpu=30),1,rs(),ValidationRequest(optimization_objective='cpu_time'))
        assert not r['eligible'] and not r['cpu_time_measured']

def test_assess_record_is_derived_only_no_deployment():
    b,c=m(cpu=50),m(wall=20,local=21,p95=30,cpu=60)
    record={'run_id':'heldout_sample','comparison':{'sample_count':800,'rounds':3,'all_labels_present':True,
        'baseline':b,'candidate':c,'prediction_agreement_fraction':1,'per_round':rs()},
        'optimization':{'approval':'eligible_for_review'}}
    out=assess_record(record)
    assert out['latency_review_pass'] and not out['cpu_time_review_pass']
    assert out['recommended_scope']=='LATENCY_ONLY' and not out['deployment_approved']
    assert out['not_new_experiment'] is True

def test_assess_record_refuses_incomplete_holdout():
    with pytest.raises(ValueError,match='Expected real labeled'):
        assess_record({'run_id':'heldout_x','comparison':{'sample_count':3,'rounds':3}})
