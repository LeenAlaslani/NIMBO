"""No downloads: pure selection tests, threading parameter and fail-closed checks."""
import copy
import pytest
from app.nimbo_validate_v21 import ValidationRequest, validate_request
from tools.cpu_thread_tuner_v3 import (score_candidate,choose_candidate,
    eligible_heldout_config, digest_texts, DEV_ROUNDS, CANDIDATE_COUNTS)


def measure(wall=30,cpu=60,p95=38):
    return {'wall':{'mean_ms':wall,'p95_ms':p95},'cpu':{'mean_ms':cpu,'p95_ms':cpu+10}}


def rounds():
    return [{'baseline_wall_ms':30,'candidate_wall_ms':20,
             'baseline_cpu_ms':60,'candidate_cpu_ms':45} for _ in range(DEV_ROUNDS)]


def score(b=None,c=None,agreement=1, bacc=.9,cacc=.9,rr=None):
    return score_candidate(b or measure(),c or measure(20,45,30),
                           agreement=agreement,baseline_accuracy=bacc,
                           candidate_accuracy=cacc,per_round=rr or rounds())


def record(threads,gain,eligible):
    return {'candidate_threads':threads,
            'gate':{'eligible_on_development':eligible,
                    'cpu_time_improvement_fraction':gain,
                    'wall_improvement_fraction':.1}}


def sample_saved():
    return {'status':'SELECTED_DEV_ONLY','heldout_evaluated_for_this_selection':False,
        'model_id':'lvwerra/distilbert-imdb','model_revision':'a'*40,
        'source_review_run_id':'heldout_abc','source_model_sha256':'z'*64,
        'development_samples':200,'development_rounds':2,
        'selected_candidate_threads':2,
        'results':[record(1,.05,True),record(2,.10,True),record(4,-.2,False)]}


def sample_review():
    return {'revision':'a'*40,'run_id':'heldout_abc','sha256':'z'*64}


def test_selector_passes_dual_objective():
    s=score()
    assert s['eligible_on_development']
    assert s['wall_improvement_fraction']>0
    assert s['cpu_time_improvement_fraction']>0


def test_selector_rejects_cpu_regression_despite_wall_gain():
    s=score(c=measure(20,72,30))
    assert not s['eligible_on_development']
    assert any('CPU' in x for x in s['rejection_reasons'])


def test_selector_rejects_quality_regression_and_p95():
    assert not score(cacc=.75)['eligible_on_development']
    assert not score(c=measure(20,45,42))['eligible_on_development']
    assert not score(agreement=.9)['eligible_on_development']


def test_selector_rejects_one_round_cpu_regression():
    rr=rounds()
    rr[1]={**rr[1],'candidate_cpu_ms':75}
    assert not score(rr=rr)['eligible_on_development']


def test_selector_requires_full_two_rounds():
    assert not score(rr=rounds()[:1])['eligible_on_development']


def test_selector_rejects_invalid_numbers():
    with pytest.raises(ValueError,match='Invalid'):
        score(c=measure(20,float('nan'),30))


def test_no_eligible_candidate_fails_closed():
    assert choose_candidate([record(1,.9,False),record(2,-.2,False)]) is None


def test_choose_highest_cpu_gain_not_fastest_only():
    r=[record(1,.05,True),record(2,.13,True),record(4,.9,False)]
    assert choose_candidate(r)['candidate_threads']==2


def test_heldout_rejects_unselected_and_modified_policy():
    data=sample_saved();review=sample_review()
    assert eligible_heldout_config(data,review)==2
    bad=copy.deepcopy(data);bad['selected_candidate_threads']=1
    with pytest.raises(ValueError,match='deterministic'):
        eligible_heldout_config(bad,review)
    bad=copy.deepcopy(data);bad['status']='NO_ELIGIBLE_CANDIDATE'
    with pytest.raises(ValueError,match='No passing'):
        eligible_heldout_config(bad,review)
    bad=copy.deepcopy(data);bad['heldout_evaluated_for_this_selection']=True
    with pytest.raises(ValueError,match='already'):
        eligible_heldout_config(bad,review)


def test_heldout_rejects_different_model_revision_or_artifact():
    data=sample_saved();review=sample_review()
    bad=copy.deepcopy(data);bad['model_revision']='b'*40
    with pytest.raises(ValueError,match='revision'):
        eligible_heldout_config(bad,review)
    bad=copy.deepcopy(data);bad['source_model_sha256']='different'
    with pytest.raises(ValueError,match='lineage'):
        eligible_heldout_config(bad,review)


def test_validation_accepts_separate_onnx_threads_and_rejects_out_of_range():
    req=ValidationRequest(threads=4,candidate_threads=2,optimization_objective='cpu_time')
    assert validate_request(req).candidate_threads==2
    with pytest.raises(Exception):
        ValidationRequest(threads=4,candidate_threads=9)


def test_dev_digest_is_deterministic_sensitive_to_text():
    a=[('abc',1),('def',0)]
    assert digest_texts(a)==digest_texts(a)
    assert digest_texts(a)!=digest_texts([('changed',1),('def',0)])
    assert len(digest_texts(a))==64


def test_staged_adoption_respects_selected_threads_and_checks_lineage():
    from app.nimbo_adoption_v21_1 import _candidate_thread_count
    review={'manifest':{'candidate_cpu_threads':2,'cpu_threads':4},
            'validation':{'candidate_thread_count':2,'thread_count':4}}
    assert _candidate_thread_count(review)==2
    legacy={'manifest':{'cpu_threads':4},'validation':{'thread_count':4}}
    assert _candidate_thread_count(legacy)==4
    review['validation']['candidate_thread_count']=4
    with pytest.raises(ValueError,match='differs'):
        _candidate_thread_count(review)
    review['manifest']['candidate_cpu_threads']=9
    with pytest.raises(ValueError,match='differs'):
        _candidate_thread_count(review)
