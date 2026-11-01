"""M5.1 policy + mocked pipeline checks. These are NOT live INT8 hardware benchmarks."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.nimbo_quality_quantization_v1 import (
    PROFILES, DEFAULT_PROFILES, choose_winner, normalize_profiles, profile_nodes,
    select_matmul_nodes, verify_m5_failure, run_quality_search,
)


def test_fixed_profiles_nonempty_unique():
    assert len(PROFILES) == 5 and len(set(DEFAULT_PROFILES)) == len(DEFAULT_PROFILES)
    assert normalize_profiles(('front_50','front_75')) == ('front_50','front_75')


@pytest.mark.parametrize('names', [(), ('unknown',), ('front_50','front_50'), tuple(PROFILES)*2])
def test_reject_invalid_search_space(names):
    with pytest.raises(ValueError):normalize_profiles(names)


def test_selects_deterministic_segments_without_overlap_claim():
    names=[f'matmul_{i}' for i in range(12)]
    front,pc=profile_nodes(names,'front_50')
    back,_=profile_nodes(names,'back_50')
    front75,_=profile_nodes(names,'front_75')
    _,per_channel=profile_nodes(names,'front_50_per_channel')
    assert front==names[:6] and back==names[6:] and front75==names[:9]
    assert pc is False and per_channel is True


def test_refuse_unnamed_ambiguous_graph():
    class N:
        def __init__(self,name):self.name=name;self.op_type='MatMul'
    assert select_matmul_nodes(SimpleNamespace(graph=SimpleNamespace(node=[N(str(x)) for x in range(4)])))==['0','1','2','3']
    with pytest.raises(ValueError):select_matmul_nodes(SimpleNamespace(graph=SimpleNamespace(node=[N('x') for _ in range(4)])))
    with pytest.raises(ValueError):select_matmul_nodes(SimpleNamespace(graph=SimpleNamespace(node=[N(str(x)) for x in range(3)])))


def rec(profile, size, latency, passed=True):
    return {'profile':profile,'status':'BENCHMARKED','int8_graph_bytes':size,
            'int8_metrics':{'mean_warm_forward_ms':latency},
            'decision':{'dev_gates_passed':passed}}


def test_selection_only_passing_and_by_objective():
    data=[rec('front_50',300,7),rec('back_50',260,8),rec('front_75',200,5,False)]
    assert choose_winner(data,'size')['profile']=='back_50'
    assert choose_winner(data,'balanced')['profile']=='front_50'
    assert choose_winner([rec('x',100,2,False)],'size') is None


def test_no_downgrade_of_original_m5_quality_requirement():
    from app.nimbo_int8_candidate_v1 import validate_int8_gate
    base={'accuracy':.71,'mean_warm_forward_ms':8.8,'p95_warm_forward_ms':11.9,
          'process_cpu_forward_total_ms':12000}
    cand={'accuracy':.70,'mean_warm_forward_ms':6.3,'p95_warm_forward_ms':9.2,
          'process_cpu_forward_total_ms':9600}
    verdict=validate_int8_gate(base,cand,old_bytes=498_831_348,new_bytes=242_447_471,
                               sample_count=200,agreement=.925,mismatches=0)
    assert verdict['decision']=='INT8_REJECTED_DEV_GATES'
    assert verdict['size_pass'] and verdict['performance_pass'] and not verdict['quality_pass']


def _fake_m5(tmp_path):
    results=tmp_path/'results'/'auto_optimizer_int8_v1'/'int8_1234'
    results.mkdir(parents=True)
    graph=results/'model.int8.onnx'
    graph.write_bytes(b'x'*3000)
    sha=hashlib.sha256(graph.read_bytes()).hexdigest()
    payload={'schema_version':'nimbo_int8_dev_trial_v1','status':'INT8_DEVELOPMENT_ONLY_NO_EXPORT',
             'source_m4_archive_sha256':'a'*64,'source_m3_report_sha256':'b'*64,
             'source_fp32_graph_sha256':'c'*64,'dev_dataset_sha256':'d'*64,
             'int8_graph_bytes':3000,'int8_graph_sha256':sha,
             'decision':{'decision':'INT8_REJECTED_DEV_GATES','dev_gates_passed':False}}
    report=results/'report.json';report.write_text(json.dumps(payload))
    return report,payload,graph


def _verify(report,root):
    return verify_m5_failure(report,project=root,expected_m4_sha='a'*64,
                              expected_dev_sha='d'*64,expected_m3_sha='b'*64,
                              expected_graph_sha='c'*64)


def test_m5_report_requires_real_rejected_evidence_chain(tmp_path):
    report,data,graph=_fake_m5(tmp_path)
    assert _verify(report,tmp_path)['int8_graph_bytes']==3000
    graph.write_bytes(b'altered')
    with pytest.raises(ValueError,match='checksum|size|missing or invalid'): _verify(report,tmp_path)


def test_m5_must_be_rejected_not_allowed_to_forge_success(tmp_path):
    report,data,graph=_fake_m5(tmp_path)
    data['decision']={'decision':'INT8_DEV_PASS_NEEDS_NEW_HELDOUT_AND_RELOAD','dev_gates_passed':True}
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='rejected'): _verify(report,tmp_path)


def test_m5_must_reference_same_development_data(tmp_path):
    report,_,_=_fake_m5(tmp_path)
    with pytest.raises(ValueError,match='does not match'):
        verify_m5_failure(report,project=tmp_path,expected_m4_sha='a'*64,
                          expected_dev_sha='f'*64,expected_m3_sha='b'*64,
                          expected_graph_sha='c'*64)


def test_injected_end_to_end_selects_only_passed_profile(tmp_path, monkeypatch):
    """Pure orchestration fake. No onnxruntime/real quantization is performed."""
    import sys
    import app.nimbo_quality_quantization_v1 as mod
    import app.nimbo_int8_candidate_v1 as m5
    project=tmp_path
    (project/'results').mkdir()
    m3=project/'m3.json';m3.write_bytes(b'm3')
    m3_dataset=project/'test.csv';m3_dataset.write_text('test data')
    dev_dataset=project/'dev.csv';dev_dataset.write_text('dev data')
    archive=project/'package.zip';archive.write_bytes(b'archive')
    fake_m5=project/'old_report.json';fake_m5.write_text('m5')
    m2={'dataset_sha256':'f'*64, 'artifact':{'sha256':'c'*64}}
    monkeypatch.setattr(mod,'check_review',lambda *a,**kw:({},m2,None,None,'e'*64))
    monkeypatch.setattr(mod,'load_heldout',lambda *a,**kw:([{'text':'x','label':'0'}]*200,'d'*64,{'source':'cardiffnlp/tweet_eval','split':'validation'}))
    monkeypatch.setattr(mod,'verify_m5_failure',lambda *a,**kw:{'decision':{'decision':'INT8_REJECTED_DEV_GATES'}})
    def extract(*a,**kw):
        pkg=Path(a[1])/'pack';pkg.mkdir(parents=True)
        (pkg/'model.onnx').write_bytes(b'O'*6000)
        return pkg,{'preprocessing':'identity'}
    monkeypatch.setattr(mod,'verify_and_extract_m4',extract)
    nodes=[SimpleNamespace(op_type='MatMul',name=f'n{i}') for i in range(8)]
    monkeypatch.setitem(sys.modules,'onnx',SimpleNamespace(load=lambda *a,**kw:SimpleNamespace(graph=SimpleNamespace(node=nodes))))
    def fake_quantize(source,target,*,all_names,profile):
        target.write_bytes(b'X'* (2000 if profile=='front_50' else 3000))
        return {'requested_matmul_nodes':4,'total_matmul_nodes':8,'per_channel':False}
    monkeypatch.setattr(mod,'quantize_profile',fake_quantize)
    from app.nimbo_int8_candidate_v1 import validate_int8_gate
    base={'accuracy':.71,'mean_warm_forward_ms':10.,'p95_warm_forward_ms':15.,
          'process_cpu_forward_total_ms':2000.}
    def fake_eval(graph,target,pkg,rows,*,manifest,objective,rounds):
        match=target.name=='model.back_50.int8.onnx'
        cand={'accuracy':.71,'mean_warm_forward_ms':8.,'p95_warm_forward_ms':13.,
              'process_cpu_forward_total_ms':1700.}
        verdict=validate_int8_gate(base,cand,old_bytes=graph.stat().st_size,new_bytes=target.stat().st_size,
                                   sample_count=200,agreement=1.0 if match else .90,mismatches=0,objective=objective)
        return base,cand,verdict
    monkeypatch.setattr(mod,'paired_evaluation',fake_eval)
    monkeypatch.setattr(mod,'digest',lambda p:'a'*64 if Path(p)==archive else 'b'*64)
    report,result=run_quality_search(archive,'a'*64,fake_m5,m3,m3_dataset,dev_dataset,
                                    root=project,profiles=('front_50','back_50'))
    assert report.is_file()
    assert result['selected']['profile']=='back_50'
    assert not result['eligible_for_downloadable_package']
    assert not result['eligible_for_deployment']
    assert not (report.parent/'model.front_50.int8.onnx').exists()
    assert (report.parent/'model.back_50.int8.onnx').is_file()
