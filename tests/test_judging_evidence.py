"""Fail-closed checks for public judging reports."""
import copy
import pytest
from tools.judging_evidence import summarize


def record():
    return {'run_id':'heldout_test','optimization':{'validation':'heldout_repeated_v21', 'approval':'eligible_for_review',
             'gate':{'eligible':True,'quality_pass':True,'latency_pass':True,'p95_pass':True,'repeatability_pass':True}},
            'comparison':{'sample_count':800,'rounds':3,'all_labels_present':True,
              'baseline':{'accuracy_on_reference_labels':.9175,'mean_inference_ms':28.5501},
              'candidate':{'accuracy_on_reference_labels':.9175,'mean_inference_ms':17.7637},
              'mean_latency_reduction_fraction':.37781,'local_pipeline_speedup_fraction':.37163,
              'prediction_agreement_fraction':1.0,'measurement_count':2400,'per_round':[]}}


def test_accept_realistic_heldout_structure():
    summary=summarize(record())
    assert summary['approval']=='eligible_for_review'
    assert 'less electrical energy' in summary['claims_not_supported']


def test_refuse_missing_labels():
    r=record();r['comparison']['all_labels_present']=False
    with pytest.raises(ValueError):summarize(r)


def test_refuse_inconsistent_speedup():
    r=record();r['comparison']['mean_latency_reduction_fraction']=.65
    with pytest.raises(ValueError):summarize(r)


def test_refuse_false_eligible_gate():
    r=record();r['optimization']['gate']['quality_pass']=False
    with pytest.raises(ValueError):summarize(r)
