"""M5 tests exercise real decision logic and mocked orchestration.

They DO NOT establish any INT8 speedup/accuracy without Mac model execution.
"""
from __future__ import annotations
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
import pytest

from app.nimbo_int8_candidate_v1 import (validate_int8_gate,verify_and_extract_m4,run_int8_trial)
from tools.prepare_nimbo_int8_dev_v1 import prepare_dev


def m(latency=25.,p95=38.,cpu=6000.,accuracy=.71):
    return {'mean_warm_forward_ms':latency,'p95_warm_forward_ms':p95,
            'process_cpu_forward_total_ms':cpu,'accuracy':accuracy}


def gate(**kw):
    return validate_int8_gate(m(),m(27.,40.,6200.,.705),old_bytes=4000,
                             new_bytes=1800,sample_count=200,agreement=.99,mismatches=0,**kw)


def test_int8_size_objective_pass_can_be_slower_than_fp32():
    result=gate(objective='size')
    assert result['dev_gates_passed']
    assert result['decision']=='INT8_DEV_PASS_NEEDS_NEW_HELDOUT_AND_RELOAD'
    assert result['graph_size_reduction_fraction']==.55
    assert not result['eligible_for_deployment']
    assert not result['eligible_for_downloadable_package']


def test_balanced_objective_rejects_slow_int8():
    result=gate(objective='balanced')
    assert result['size_pass'] and result['quality_pass']
    assert not result['performance_pass']


@pytest.mark.parametrize('candidate,new_bytes,agreement,mismatches,reason',[
    (m(27.,40.,6200.,.705),3500,.99,0,'size_pass'),
    (m(27.,40.,6200.,.68),1800,.99,0,'quality_pass'),
    (m(27.,40.,6200.,.705),1800,.90,0,'quality_pass'),
    (m(27.,40.,6200.,.705),1800,.99,1,'quality_pass'),
    (m(40.,40.,6200.,.705),1800,.99,0,'performance_pass'),
    (m(27.,60.,6200.,.705),1800,.99,0,'performance_pass'),
    (m(27.,40.,8000.,.705),1800,.99,0,'performance_pass'),
])
def test_quality_size_and_performance_rejection(candidate,new_bytes,agreement,mismatches,reason):
    verdict=validate_int8_gate(m(),candidate,old_bytes=4000,new_bytes=new_bytes,
                                sample_count=200,agreement=agreement,mismatches=mismatches)
    assert verdict['decision']=='INT8_REJECTED_DEV_GATES'
    assert verdict[reason] is False


@pytest.mark.parametrize('kw',[
    {'sample_count':199},{'agreement':1.1},{'mismatches':-1},
    {'objective':'speed'},{'old_bytes':0},{'new_bytes':0}
])
def test_gate_fails_closed_invalid_params(kw):
    base=dict(old_bytes=4000,new_bytes=1800,sample_count=200,agreement=.99,mismatches=0,objective='size')
    base.update(kw)
    with pytest.raises(ValueError):validate_int8_gate(m(),m(),**base)


def test_gate_rejects_nan_and_missing_labels():
    a=m();a['accuracy']=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):validate_int8_gate(a,m(),old_bytes=4000,new_bytes=1800,sample_count=200,agreement=1,mismatches=0)
    b=m();b['accuracy']=None
    with pytest.raises(ValueError,match='Missing numeric'):validate_int8_gate(m(),b,old_bytes=4000,new_bytes=1800,sample_count=200,agreement=1,mismatches=0)


def test_real_dev_preparation_provenance(tmp_path):
    fake=[{'text':f'dataset tweet #{i}','label':i%3} for i in range(280)]
    dest=tmp_path/'dev.csv'
    p,info=prepare_dev(fake,revision='a'*40,output=dest,samples=200)
    assert p.is_file() and info['source']=='cardiffnlp/tweet_eval'
    assert info['split']=='validation' and info['rows']==200
    assert info['csv_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
    assert len(set(info['source_indices']))==200
    assert json.loads(dest.with_suffix('.source.json').read_text())['split']=='validation'
    with pytest.raises(FileExistsError):prepare_dev(fake,revision='a'*40,output=dest,samples=200)


def test_source_revision_validation(tmp_path):
    with pytest.raises(ValueError):prepare_dev([{'text':'x','label':0}]*200,
                                               revision='main',output=tmp_path/'rows.csv')


def make_archive(tmp_path, *, tamper=False, extra=False, symlink=False):
    dirname='nimbo_onnx_review_aaa'
    files={
        'model.onnx':b'ONNX'+b'x'*2048,
        'config.json':b'{}',
        'tokenizer_config.json':b'{}',
        'run_model.py':b'print(1)',
        'README.md':b'REVIEW',
        'evidence.json':b'{}',
    }
    m3=tmp_path/'m3.json';m3.write_text('M3 review receipt')
    graph_sha=hashlib.sha256(files['model.onnx']).hexdigest()
    manifest={'schema_version':'nimbo_optimized_package_v1',
              'review_status':'LOCAL_REVIEW_ONLY_NOT_DEPLOYED',
              'source_m3_report_sha256':hashlib.sha256(m3.read_bytes()).hexdigest(),
              'source_m2_graph_sha256':graph_sha,'heldout_dataset_sha256':'a'*64,
              'model':{'model_id':'team/model','resolved_revision':'b'*40},
              'preprocessing':'identity',
              'files':{k:hashlib.sha256(v).hexdigest() for k,v in files.items()}}
    archive=tmp_path/'old.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for k,v in files.items():
            if tamper and k=='model.onnx':v+=b'bad'
            z.writestr(f'{dirname}/{k}',v)
        z.writestr(f'{dirname}/manifest.json',json.dumps(manifest))
        if extra:z.writestr(f'{dirname}/other.bin',b'BAD')
        if symlink:
            info=zipfile.ZipInfo(f'{dirname}/link')
            info.create_system=3
            info.external_attr=(0o120777 << 16)
            z.writestr(info,b'../evil')
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    old={'heldout_dataset_sha256':'a'*64,'model':{'model_id':'team/model','resolved_revision':'b'*40},
         'preprocessing':'identity'}
    return archive,sha,m3,graph_sha,old


def extract(tmp_path,**kwargs):
    archive,sha,receipt,graph,old=make_archive(tmp_path,**kwargs)
    return verify_and_extract_m4(archive,tmp_path/'fresh',expected_sha256=sha,
                                  m3=old,m3_report=receipt,fp32_graph_sha256=graph)


def test_reject_zip_modified_files(tmp_path):
    with pytest.raises(ValueError,match='checksums'):
        extract(tmp_path,tamper=True)


def test_reject_unexpected_zip_members(tmp_path):
    with pytest.raises(ValueError,match='Unexpected ZIP'):
        extract(tmp_path,extra=True)


def test_reject_symlink_zip_entries(tmp_path):
    with pytest.raises(ValueError):extract(tmp_path,symlink=True)


def test_reject_wrong_zip_hash(tmp_path):
    archive,sha,receipt,graph,old=make_archive(tmp_path)
    with pytest.raises(ValueError,match='hash differs'):
        verify_and_extract_m4(archive,tmp_path/'fresh',expected_sha256='f'*64,
                              m3=old,m3_report=receipt,fp32_graph_sha256=graph)


def test_m4_archive_verification_good_path(tmp_path):
    folder,manifest=extract(tmp_path)
    assert folder.joinpath('model.onnx').is_file()
    assert manifest['schema_version']=='nimbo_optimized_package_v1'


def test_full_int8_run_injected_fake_quantization_not_real_model(tmp_path,monkeypatch):
    """Injection proves orchestration and receipt, not any actual INT8 inference."""
    import app.nimbo_int8_candidate_v1 as mod
    project=tmp_path
    (project/'results').mkdir()
    (project/'results'/'auto_optimizer_int8_v1').mkdir()
    m3=project/'m3.json';m3.write_text('local dev receipt')
    archive=project/'m4.zip';archive.write_bytes(b'fake zip')
    dataset=project/'dev.csv';dataset.write_text('fake development rows')
    m3data=project/'test.csv';m3data.write_text('fake test data')
    original=project/'source.onnx';original.write_bytes(b'X'*5000)
    m3_report={'source_graph_sha256':'0'*64,'heldout_dataset_sha256':'a'*64}
    m2={'dataset_sha256':'b'*64,'artifact':{'sha256':hashlib.sha256(original.read_bytes()).hexdigest()}}
    monkeypatch.setattr(mod,'check_review',lambda *a,**kw:(m3_report,m2,original,[], 'a'*64))
    rows=[{'text':'test '+str(i),'label':str(i%3)} for i in range(200)]
    monkeypatch.setattr(mod,'load_heldout',lambda *a,**kw:(rows,'c'*64,{'source':'cardiffnlp/tweet_eval','split':'validation'}))
    def fake_extract(_archive,target,**kw):
        pkg=target/'nimbo_onnx_review_test';pkg.mkdir(parents=True)
        (pkg/'model.onnx').write_bytes(original.read_bytes())
        return pkg,{'model':{'model_id':'team/model'},'preprocessing':'identity'}
    monkeypatch.setattr(mod,'verify_and_extract_m4',fake_extract)
    class Q:
        QInt8=0
    def quantize(source,output,**kw):
        assert kw['weight_type']==0 and kw['per_channel'] is False
        assert kw['op_types_to_quantize']==['MatMul']
        Path(output).write_bytes(b'INT8'*300)
    monkeypatch.setitem(sys.modules,'onnxruntime',SimpleNamespace(quantization=SimpleNamespace(quantize_dynamic=quantize,QuantType=Q)))
    monkeypatch.setitem(sys.modules,'onnxruntime.quantization',SimpleNamespace(quantize_dynamic=quantize,QuantType=Q))
    monkeypatch.setitem(sys.modules,'onnx',SimpleNamespace(load=lambda *a,**kw:SimpleNamespace(graph=SimpleNamespace(node=[SimpleNamespace(op_type='MatMulInteger')]))))
    monkeypatch.setattr(mod,'paired_evaluation',lambda *a,**kw:(m(),m(24.,36.,5900.,.71),
             validate_int8_gate(m(),m(24.,36.,5900.,.71),old_bytes=5000,new_bytes=1200,sample_count=200,agreement=1,mismatches=0)))
    result_file,result=run_int8_trial(archive,'1'*64,m3,m3data,dataset,root=project)
    assert result_file.is_file()
    assert result['status']=='INT8_DEVELOPMENT_ONLY_NO_EXPORT'
    assert result['int8_graph_bytes']==1200
    assert result['decision']['eligible_for_downloadable_package'] is False
    assert not result['decision']['eligible_for_deployment']
    assert not any('test 1' in v for v in json.dumps(result).splitlines())


def test_reject_dev_reuse_of_m3_set(tmp_path,monkeypatch):
    import app.nimbo_int8_candidate_v1 as mod
    m3=tmp_path/'m3.json';m3.write_text('{}')
    f=tmp_path/'x.csv';f.write_text('data')
    (tmp_path/'results').mkdir()
    monkeypatch.setattr(mod,'check_review',lambda *a,**kw:({}, {'dataset_sha256':'b'*64},None,[],'a'*64))
    monkeypatch.setattr(mod,'load_heldout',lambda *a,**kw:([{}]*200,'a'*64,{}))
    with pytest.raises(ValueError,match='must differ'):
        run_int8_trial(f,'f'*64,m3,f,f,root=tmp_path)
