import pytest
from tools.concurrency_probe_v1 import (
    make_prompts, percentile_ms, summarize_batch, compare_predictions,
    run_batch, validate_lineage,
)


def test_synthetic_inputs_deterministic_and_bounded():
    a=make_prompts(24)
    assert a==make_prompts(24) and len(a)==24 and all(isinstance(t,str) for t in a)
    assert len(set(a))>=16
    with pytest.raises(ValueError):make_prompts(0)
    with pytest.raises(ValueError):make_prompts(1000)


def test_percentile_and_empty_fail_closed():
    assert percentile_ms([1,2,3,4,5])==4.8
    with pytest.raises(ValueError):percentile_ms([])


def test_parallel_worker_and_parity():
    def worker(kind, text): return int('great' in text)
    prompts=['great'+str(i) if i%2 else 'bad'+str(i) for i in range(8)]
    base,ba=run_batch('baseline',worker,prompts,2)
    cand,ca=run_batch('candidate',worker,prompts,2)
    assert base['successes']==8 and cand['errors']==0
    assert compare_predictions(ba,ca,8)['all_agree']
    assert base['batch_process_cpu_ms']>=0


def test_worker_failure_does_not_create_fake_success():
    def fail(kind,text):
        if '3' in text:raise RuntimeError('intentional error')
        return 1
    stats,preds=run_batch('baseline',fail,[str(i) for i in range(8)],4)
    assert stats['errors']==1 and stats['successes']==7
    assert not compare_predictions(preds,{i:1 for i in range(8)},8)['all_agree']


def test_malformed_worker_prediction_rejected():
    stats, preds=run_batch('candidate',lambda kind,text:None,['x']*8,1)
    assert stats['errors']==8 and preds=={}


def test_incomplete_results_rejected():
    with pytest.raises(ValueError):
        summarize_batch([],request_count=8,elapsed_ms=10,cpu_ms=8,concurrency=1,path='baseline')


def test_tampered_or_unvalidated_review_rejected():
    selection={'status':'SELECTED_DEV_ONLY','heldout_evaluated_for_this_selection':True,
               'selected_candidate_threads':2,'model_revision':'abc'}
    record={'candidate_thread_count':2,'thread_count':4,'optimization_objective':'cpu_time',
            'model_revision':'abc','run_id':'heldout_foo',
            'optimization':{'approval':'eligible_for_review','gate':{
                k:True for k in ('eligible','quality_pass','latency_pass','cpu_pass','p95_pass','repeatability_pass')}}}
    review={'revision':'abc','sha256':'digest','validation':record}
    assert validate_lineage(selection,record,review)==2
    bad={**record,'candidate_thread_count':4}
    with pytest.raises(ValueError):validate_lineage(selection,bad,review)
    bad={**record,'optimization_objective':'latency'}
    with pytest.raises(ValueError):validate_lineage(selection,bad,review)
    bad={**record,'optimization':{'approval':'eligible_for_review','gate':{}}}
    with pytest.raises(ValueError):validate_lineage(selection,bad,review)
