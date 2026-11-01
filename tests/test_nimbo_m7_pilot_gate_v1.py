"""M7c safety tests: pilot timing is provisional, M3 strict gates are unchanged.

Metrics are synthetic replicas of user-supplied aggregates. These tests do NOT
run real RoBERTa, ONNX Runtime, or Mac CPU benchmarks.
"""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.nimbo_heldout_validation_v1 import pilot_gate_disposition, read_m2_candidate
from app.nimbo_onnx_candidate_v1 import decide_candidate
from app.nimbo_auto_pipeline_jobs_v1 import run_worker, submit, read_events


def pilot(*, cpu=2110.483, acc=0.833333, agreement=1.0, p95=45.3236,
          mean=30.3989, rows=12, repeat=0):
    old = {'n_forward_calls':rows*3, 'accuracy':0.833333,
           'mean_warm_forward_ms':43.7664, 'p95_warm_forward_ms':57.1232,
           'process_cpu_forward_total_ms':2035.969}
    candidate = {'n_forward_calls':rows*3, 'accuracy':acc,
           'mean_warm_forward_ms':mean, 'p95_warm_forward_ms':p95,
           'process_cpu_forward_total_ms':cpu}
    decision = decide_candidate(old,candidate,agreement,
                                samples=rows,max_abs_logit_error=1.025e-5)
    decision['repeatability_mismatches'] = repeat
    return {'schema_version':'nimbo_onnx_candidate_v1',
            'status':'EXPERIMENT_ONLY_NOT_EXPORTED',
            'model':{'model_id':'team/roberta','resolved_revision':'a'*40},
            'dataset_sha256':'f'*64,'cpu_threads_each_path':2,'max_length':256,
            'rounds':3,'unique_dataset_rows':rows,'forward_calls_each_path':rows*3,
            'baseline':old,'onnx_candidate':candidate,'decision':decision}


def test_exact_real_m7_aggregates_defer_cpu_only():
    report=pilot()
    assert report['decision']['decision']=='REJECTED_LOCAL_GATES'
    assert report['decision']['process_cpu_time_reduction_fraction']==-0.036599
    assert pilot_gate_disposition(report)=='PILOT_CPU_P95_TIMING_DEFERRED_TO_HELDOUT'
    assert report['decision']['local_gates_passed'] is False  # no false approval


def test_m2_pass_has_original_semantics():
    r=pilot(cpu=1900)
    assert r['decision']['local_gates_passed']
    assert pilot_gate_disposition(r)=='PILOT_LOCAL_PASS'


@pytest.mark.parametrize('alter',[
    lambda x: x['decision'].update(prediction_agreement=.99),
    lambda x: x['decision'].update(quality_pass=False),
    lambda x: x['decision'].update(accuracy_drop=.01),
    lambda x: x['decision'].update(repeatability_mismatches=1),
    lambda x: x['decision']['criteria'].update(max_process_cpu_increase=.05),
    lambda x: x['decision'].update(mean_latency_reduction_fraction=.99),
    lambda x: x.update(forward_calls_each_path=1),
    lambda x: x.update(unique_dataset_rows=51),
    lambda x: x['decision'].update(max_abs_logit_error=.02),
])
def test_pilot_defer_fail_closed_for_inconsistent_or_unsafe_evidence(alter):
    r=pilot(); alter(r)
    assert pilot_gate_disposition(r) is None


def test_no_defer_when_accuracy_or_speed_is_lost():
    assert pilot_gate_disposition(pilot(acc=.75)) is None
    assert pilot_gate_disposition(pilot(mean=43.0)) is None
    assert pilot_gate_disposition(pilot(agreement=.98)) is None


def test_graph_and_evidence_m3_accept_deferral_without_editing_m2(tmp_path):
    p=tmp_path/'results'/'auto_optimizer_candidates_v1'/'candidate_x'
    p.mkdir(parents=True)
    graph=p/'model.onnx';graph.write_bytes(b'x'*2048)
    r=pilot()
    r['artifact']={'path':str(graph.resolve()),'sha256':hashlib.sha256(graph.read_bytes()).hexdigest(),
                   'bytes':graph.stat().st_size}
    filename=p/'report.json';raw=json.dumps(r);filename.write_text(raw)
    loaded,loaded_graph=read_m2_candidate(filename,root=tmp_path,model_id='team/roberta',
        sha='a'*40,heldout_sha='e'*64,threads=2,max_length=256)
    assert loaded_graph == graph
    assert loaded['decision']['local_gates_passed'] is False
    assert filename.read_text() == raw


def test_m3_still_rejects_unsafe_deferred_candidate(tmp_path):
    p=tmp_path/'results'/'auto_optimizer_candidates_v1'/'candidate_x'
    p.mkdir(parents=True)
    graph=p/'model.onnx';graph.write_bytes(b'x'*2048)
    r=pilot(acc=.7)
    r['artifact']={'path':str(graph.resolve()),'sha256':hashlib.sha256(graph.read_bytes()).hexdigest(),
                   'bytes':graph.stat().st_size}
    filename=p/'report.json';filename.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='gates'):
        read_m2_candidate(filename,root=tmp_path,model_id='team/roberta',
            sha='a'*40,heldout_sha='e'*64,threads=2,max_length=256)


def setup_job(root):
    pilot_dir=root/'results'/'auto_optimizer_job_inputs_v1';pilot_dir.mkdir(parents=True)
    pilot_dir.joinpath('pilot.csv').write_text('text,label\nhello,POSITIVE\nawful,NEGATIVE\n')
    heldout_dir=root/'results'/'auto_optimizer_heldout_v1';heldout_dir.mkdir(parents=True)
    heldout_dir.joinpath('test.csv').write_text('text,label\n'+''.join(f'text {i},{i%2}\n' for i in range(200)))
    req={'model_id':'team/roberta','revision':'a'*40,'pilot_file':'pilot.csv',
         'heldout_file':'test.csv','preprocess':'identity','objective':'speed'}
    class Spawn:
        pid=777777
    return submit(root,req,spawn=lambda *a,**k:Spawn())


def test_m7_proceeds_to_heldout_and_package_when_pilot_cpu_fluctuates(tmp_path):
    job=setup_job(tmp_path)
    def base(*a,**k):return tmp_path/'m1.json',{'metrics':{'accuracy':.833333,'mean_warm_forward_ms':40}}
    def candidate(*a,**k):return tmp_path/'m2.json',pilot()
    def heldout(*a,**kw):return tmp_path/'m3.json',{
        'decision':{'heldout_gates_passed':True,'decision':'HELDOUT_LOCAL_PASS_REQUIRES_PACKAGE_RELOAD'},
        'candidate':{'accuracy':.71}}
    package=tmp_path/'results'/'auto_optimizer_packages_v1'/'package.zip'
    package.parent.mkdir(parents=True);package.write_bytes(b'zip_placeholder')
    def packaged(*a,**k):return package,{'status':'RUNNABLE_PACKAGE_VERIFIED_REVIEW_ONLY',
        'verified_offline_after_extract':True,'archive_sha256':hashlib.sha256(package.read_bytes()).hexdigest(),
        'reload_accuracy':.71}
    result=run_worker(tmp_path,job['job_id'],baseline=base,candidate=candidate,heldout=heldout,packager=packaged)
    assert result['state']=='COMPLETED_REVIEW_ONLY'
    events=read_events(tmp_path,job['job_id'])['events']
    evidence=[e for e in events if e.get('evidence',{}).get('pilot_gate_disposition')]
    assert len(evidence)==1
    assert evidence[0]['evidence']['pilot_decision']=='REJECTED_LOCAL_GATES'
    assert any('heldout' in e['message'].lower() for e in events)


def test_m7_heldout_cpu_failure_still_rejects_no_package(tmp_path):
    job=setup_job(tmp_path)
    def base(*a,**k):return tmp_path/'m1.json',{'metrics':{'accuracy':.833333,'mean_warm_forward_ms':40}}
    def candidate(*a,**k):return tmp_path/'m2.json',pilot()
    def heldout(*a,**k):return tmp_path/'m3.json',{'decision':{'heldout_gates_passed':False,'decision':'REJECTED_HELDOUT_GATES'}}
    def never_package(*a,**k):raise AssertionError('Unsafe to package M3 rejected candidate')
    result=run_worker(tmp_path,job['job_id'],baseline=base,candidate=candidate,heldout=heldout,packager=never_package)
    assert result['state']=='COMPLETED_REJECTED'
    assert result['eligible_for_review_download'] is False
