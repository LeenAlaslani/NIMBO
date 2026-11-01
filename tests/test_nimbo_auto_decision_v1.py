"""M6 decision tests, with a mocked offline-runner orchestration (not live RoBERTa inference)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from app.nimbo_auto_decision_v1 import verify_rejected_search, select_verified_fallback
from app.nimbo_int8_candidate_v1 import validate_int8_gate


def metrics(accuracy=.71, latency=8., p95=11., cpu=9000.):
    return {'accuracy':accuracy,'mean_warm_forward_ms':latency,
            'p95_warm_forward_ms':p95,'process_cpu_forward_total_ms':cpu}


def search_data():
    original=metrics()
    candidate=metrics(.70,6.,9.,7500.)
    gate=validate_int8_gate(original,candidate,old_bytes=498831348,new_bytes=250000000,
                           sample_count=200,agreement=.945,mismatches=0,objective='size')
    assert gate['dev_gates_passed'] is False
    return {
        'schema_version':'nimbo_quality_int8_search_v1',
        'status':'DEVELOPMENT_ONLY_NO_EXPORT','decision':'ALL_SELECTIVE_INT8_CANDIDATES_REJECTED',
        'source_m5_report_sha256':'a'*64,'source_m4_archive_sha256':'b'*64,
        'source_m3_report_sha256':'c'*64,'source_fp32_graph_sha256':'d'*64,
        'dev_dataset_sha256':'e'*64,'objective':'size',
        'unique_dev_examples':200,'rounds':3,
        'selected':None,'eligible_for_deployment':False,
        'eligible_for_downloadable_package':False,
        'quantized_profiles_tested':['front_50','front_75'],
        'trials':[{'profile':'front_50','status':'BENCHMARKED',
                   'fp32_graph_bytes':498831348,'int8_graph_bytes':250000000,
                   'int8_graph_sha256':'f'*64,'fp32_metrics':original,
                   'int8_metrics':candidate,'decision':gate},
                  {'profile':'front_75','status':'FAILED','error_type':'RuntimeError','error':'ORT unsupported'}],
    }


def checked(r):
    return verify_rejected_search(r,m5_sha='a'*64,m4_sha='b'*64,
                                  m3_sha='c'*64,fp32_sha='d'*64,dev_sha='e'*64,
                                  dev_examples=200)


def test_only_rejected_candidates_are_reported():
    results=checked(search_data())
    assert len(results)==2 and results[0]['status']=='REJECTED'
    assert results[1]['status']=='FAILED'
    assert results[0]['prediction_agreement']==.945


def test_reject_tampered_agreement_even_if_decision_stays_rejected():
    r=search_data()
    r['trials'][0]['decision']['prediction_agreement']=.99
    with pytest.raises(ValueError,match='recalculation'):
        checked(r)


def test_reject_candidate_appearing_approved():
    r=search_data()
    r['trials'][0]['decision']['dev_gates_passed']=True
    with pytest.raises(ValueError,match='Accepted'):
        checked(r)


@pytest.mark.parametrize('alter',[
    lambda r:r.update(decision='INT8_DEV_PASS_NEEDS_NEW_HELDOUT_AND_RELOAD'),
    lambda r:r.update(selected={'profile':'front_50'}),
    lambda r:r.update(eligible_for_deployment=True),
    lambda r:r.update(source_m4_archive_sha256='f'*64),
    lambda r:r.update(quantized_profiles_tested=['front_50','front_50']),
    lambda r:r.update(unique_dev_examples=199),
])
def test_reject_forged_search_metadata(alter):
    r=search_data(); alter(r)
    with pytest.raises(ValueError):checked(r)


def test_fail_closed_wrong_trial_order():
    r=search_data();r['trials'].reverse()
    with pytest.raises(ValueError):checked(r)


def test_selection_e2e_mocked_dependency_boundaries(tmp_path, monkeypatch):
    """Mock orchestration. Actual Mac invocation must perform independent offline reload."""
    import app.nimbo_auto_decision_v1 as m6
    project=tmp_path
    (project/'results'/'auto_optimizer_packages_v1').mkdir(parents=True)
    (project/'results'/'auto_optimizer_int8_v1'/'int8_a').mkdir(parents=True)
    (project/'results'/'auto_optimizer_quality_int8_v1'/'search_a').mkdir(parents=True)
    graph=project/'graph.onnx';graph.write_bytes(b'graph data')
    archive=project/'results'/'auto_optimizer_packages_v1'/'pack.zip';archive.write_bytes(b'archive')
    m3=project/'results'/'m3.json';m3.write_bytes(b'm3')
    m3data=project/'test.csv';m3data.write_text('test')
    devdata=project/'dev.csv';devdata.write_text('dev')
    m5=project/'results'/'auto_optimizer_int8_v1'/'int8_a'/'report.json';m5.write_text('{}')
    m51=project/'results'/'auto_optimizer_quality_int8_v1'/'search_a'/'report.json';m51.write_text('{}')
    m2={'dataset_sha256':'f'*64,'artifact':{'sha256':'d'*64}}
    m3_object={'heldout_unique_examples':200,'candidate':{'accuracy':.71},
               'baseline':metrics(.71,21.,28.,15000.),'decision':{'prediction_agreement':1.0},
               'model':{'model_id':'team/test','resolved_revision':'f'*40}}
    monkeypatch.setattr(m6,'check_review',lambda *a,**kw:(m3_object,m2,graph,[], 'c'*64))
    monkeypatch.setattr(m6,'load_heldout',lambda *a,**kw:([{}]*200,'e'*64,{'source':'cardiffnlp/tweet_eval','split':'validation'}))
    monkeypatch.setattr(m6,'verify_m5_failure',lambda *a,**kw:{'decision':{'decision':'INT8_REJECTED_DEV_GATES'}})
    obj=search_data()
    obj['source_m5_report_sha256']=hashlib.sha256(m5.read_bytes()).hexdigest()
    obj['source_m3_report_sha256']=hashlib.sha256(m3.read_bytes()).hexdigest()
    obj['source_m4_archive_sha256']='b'*64
    m51.write_text(json.dumps(obj))
    class Saved:
        def __init__(self,path):self.path=path
    def fake_extract(source,target,**kw):
        dest=target/'nimbo_onnx_review_a';dest.mkdir(parents=True)
        return dest,{'schema_version':'nimbo_optimized_package_v1'}
    monkeypatch.setattr(m6,'verify_and_extract_m4',fake_extract)
    monkeypatch.setattr(m6,'verify_offline_package',lambda *a,**kw:{
        'status':'OFFLINE_BUNDLE_RELOAD_VERIFIED_REVIEW_ONLY','samples':200,
        'accuracy':.71,'model_graph_sha256':'d'*64,'prediction_fingerprint':'abc'})
    path,receipt=m6.select_verified_fallback(archive,'b'*64,m3,m3data,m5,m51,devdata,root=project)
    assert path.is_file() and receipt['decision']=='SELECT_ONNX_FP32_REJECT_INT8'
    assert receipt['eligible_for_deployment'] is False
    assert receipt['evidence']['int8_candidates_rejected']==2
    assert receipt['selected_package']['path']==str(archive)
    assert receipt['selected_variant']=='ONNX_RUNTIME_FP32'


def test_offline_reload_rejects_failed_subprocess(tmp_path,monkeypatch):
    import app.nimbo_auto_decision_v1 as m6
    def fake_run(args, **kwargs):
        assert kwargs['env']['HF_HUB_OFFLINE']=='1'
        assert kwargs['env']['TRANSFORMERS_OFFLINE']=='1'
        return SimpleNamespace(returncode=1,stdout='',stderr='Bad package checksum')
    monkeypatch.setattr(m6.subprocess,'run',fake_run)
    with pytest.raises(RuntimeError,match='Bad package checksum'):
        m6.verify_offline_package(tmp_path,tmp_path,project=tmp_path)


def test_offline_reload_accepts_only_review_receipt(tmp_path,monkeypatch):
    import app.nimbo_auto_decision_v1 as m6
    payload={'status':'OFFLINE_BUNDLE_RELOAD_VERIFIED_REVIEW_ONLY','eligible_for_deployment':False,
             'samples':200,'accuracy':.71,'model_graph_sha256':'a'*64,'prediction_fingerprint':'b'*64}
    stdout='STATUS: OFFLINE_BUNDLE_RELOAD_VERIFIED_REVIEW_ONLY\n'+json.dumps(payload)
    monkeypatch.setattr(m6.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0,stdout=stdout,stderr=''))
    assert m6.verify_offline_package(tmp_path,tmp_path,project=tmp_path)['accuracy']==.71
    payload['eligible_for_deployment']=True
    stdout='STATUS: OFFLINE_BUNDLE_RELOAD_VERIFIED_REVIEW_ONLY\n'+json.dumps(payload)
    with pytest.raises(RuntimeError,match='review-only'):
        m6.verify_offline_package(tmp_path,tmp_path,project=tmp_path)
