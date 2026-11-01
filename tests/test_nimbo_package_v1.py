"""M4 dependency-light tests. Mocking reload is NOT a real external model benchmark."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.nimbo_package_v1 as m4


def _save(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2),encoding='utf8')
    return path


@pytest.fixture
def demo(tmp_path):
    project=tmp_path/'nimbo'
    project.mkdir()
    candidates=project/'results'/'auto_optimizer_candidates_v1'/'candidate_test'
    candidates.mkdir(parents=True)
    graph=candidates/'model.onnx'
    graph.write_bytes(b'X'*2048)
    model={'model_id':'public/roberta','resolved_revision':'a'*40,
           'num_labels':2,'architecture':'roberta','id2label':{'0':'NEGATIVE','1':'POSITIVE'}}
    dataset=project/'results'/'auto_optimizer_heldout_v1'/'heldout.csv'
    dataset.parent.mkdir(parents=True)
    dataset.write_text('text,label\n'+('good,POSITIVE\nbad,NEGATIVE\n'*100),encoding='utf8')
    sha=hashlib.sha256(dataset.read_bytes()).hexdigest()
    meta={'schema_version':'nimbo_onnx_candidate_v1','status':'EXPERIMENT_ONLY_NOT_EXPORTED',
          'model':model,'dataset_sha256':'1'*64,'cpu_threads_each_path':2,'max_length':256,
          'decision':{'local_gates_passed':True,'decision':'EXPLORATORY_PASS_NOT_APPROVED'},
          'artifact':{'path':str(graph),'sha256':m4.digest(graph),'bytes':2048}}
    m2=_save(candidates/'report.json',meta)
    d={'heldout_gates_passed':True,'quality_pass':True,'performance_pass':True,
       'decision':'HELDOUT_LOCAL_PASS_REQUIRES_PACKAGE_RELOAD',
       'repeatability_mismatches':0,'prediction_agreement':1.0,'accuracy_drop':0.0}
    m3=_save(project/'results'/'auto_optimizer_heldout_v1'/'heldout_report.json',{
        'schema_version':'nimbo_heldout_validation_v1','status':'HELDOUT_VALIDATION_COMPLETED_REVIEW_ONLY',
        'decision':d,'model':model,'source_candidate_report':str(m2),
        'source_graph_sha256':meta['artifact']['sha256'],'source_m2_dataset_sha256':'1'*64,
        'heldout_dataset_sha256':sha,'heldout_unique_examples':200,'rounds':3,
        'cpu_threads_each_path':2,'max_length':256,'preprocessing':'identity',
        'heldout_dataset_provenance':{'source':'user_supplied_not_independently_audited','dataset_sha256':sha},
        'baseline':{'accuracy':.5,'n_forward_calls':600},
        'candidate':{'accuracy':.5,'n_forward_calls':600}})
    return SimpleNamespace(root=project,m2=m2,m3=m3,graph=graph,data=dataset,model=model)


def test_valid_gated_review_binds_original_graph(demo):
    m3,m2,graph,rows,sha=m4.check_review(demo.m3,demo.data,root=demo.root)
    assert graph==demo.graph and len(rows)==200
    assert m3['decision']['heldout_gates_passed'] and sha==m3['heldout_dataset_sha256']
    assert m2['artifact']['sha256']==m4.digest(graph)


@pytest.mark.parametrize('field,value',[('decision',{'heldout_gates_passed':False}),
    ('heldout_unique_examples',12),('rounds',2),('source_graph_sha256','a'*64),
    ('preprocessing','unsafe')])
def test_invalid_review_refused(demo,field,value):
    data=json.loads(demo.m3.read_text())
    data[field]=value
    demo.m3.write_text(json.dumps(data))
    with pytest.raises((ValueError,KeyError)):
        m4.check_review(demo.m3,demo.data,root=demo.root)


def test_modified_graph_sha_refused(demo):
    demo.graph.write_bytes(b'Q'*2048)
    with pytest.raises(ValueError,match='checksum'):
        m4.check_review(demo.m3,demo.data,root=demo.root)


def test_modified_dataset_refused(demo):
    demo.data.write_text(demo.data.read_text().replace('good','changed',1))
    with pytest.raises(ValueError,match='dataset identity'):
        m4.check_review(demo.m3,demo.data,root=demo.root)


def test_new_file_manifest_mismatch_rejected(tmp_path):
    path=tmp_path/'bundle';path.mkdir()
    (path/'model.onnx').write_bytes(b'x'*2048)
    hashes=m4._hashes(path)
    (path/'unknown.py').write_text('unexpected')
    with pytest.raises(ValueError,match='checksums'):
        m4.verify_bundle_files(path,{'files':hashes})


def test_symlinked_package_file_refused(tmp_path):
    path=tmp_path/'bundle';path.mkdir()
    outside=tmp_path/'outside';outside.write_text('outside')
    (path/'tokenizer.json').symlink_to(outside)
    with pytest.raises(ValueError,match='symbolic link'):
        m4._hashes(path)


def test_prediction_hash_stable():
    assert m4._sha_predictions([0,1,2]) == hashlib.sha256(b'0,1,2').hexdigest()
    assert m4._sha_predictions([0,1]) != m4._sha_predictions([1,0])


def test_m4_staged_export_and_zip_subprocess_invocations_mocked(demo,monkeypatch):
    import sys,zipfile
    from types import ModuleType
    class FakeTokenizer:
        @staticmethod
        def from_pretrained(*args,**kwargs):
            assert kwargs.get('trust_remote_code') is False
            return FakeTokenizer()
        def save_pretrained(self,folder):
            Path(folder,'tokenizer_config.json').write_text('{}')
            Path(folder,'tokenizer.json').write_text('{}')
    class FakeConfig:
        model_type='roberta'
        num_labels=2
        @staticmethod
        def from_pretrained(*args,**kwargs):return FakeConfig()
        def save_pretrained(self,folder):Path(folder,'config.json').write_text('{}')
    module=ModuleType('transformers')
    module.AutoTokenizer=FakeTokenizer
    module.AutoConfig=FakeConfig
    monkeypatch.setitem(sys.modules,'transformers',module)
    monkeypatch.setattr(m4,'inspect_public_classifier',lambda *_a,**_kw:SimpleNamespace(
        model_id=demo.model['model_id'],resolved_revision=demo.model['resolved_revision'],
        num_labels=2,id2label={0:'NEGATIVE',1:'POSITIVE'},architecture='roberta'))
    monkeypatch.setattr(m4,'_predict_bundle',lambda _folder,rows,**kw:[0]*len(rows))
    calls=[]
    def fake_run(args,**kw):
        calls.append((args,kw))
        return SimpleNamespace(returncode=0,stdout='OFFLINE_BUNDLE_RELOAD_VERIFIED_REVIEW_ONLY\n{"review_only": true}',stderr='')
    monkeypatch.setattr(m4.subprocess,'run',fake_run)
    archive,report=m4.package_model(demo.m3,demo.data,root=demo.root,
        output_root=demo.root/'results'/'auto_optimizer_packages_v1')
    assert report['verified_offline_after_extract'] and not report['eligible_for_deployment']
    assert len(calls)==2
    assert calls[0][1]['env']['HF_HUB_OFFLINE']=='1'
    assert calls[1][0][-2]=='--text'
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        assert any(x.endswith('/manifest.json') for x in names)
        assert any(x.endswith('/run_model.py') for x in names)
        assert any(x.endswith('/config.json') for x in names)
        assert any(x.endswith('/tokenizer.json') for x in names)
        assert any(x.endswith('/model.onnx') for x in names)


def test_no_package_when_fresh_reloaded_accuracy_regresses(demo,monkeypatch):
    import sys
    from types import ModuleType
    class Fake:
        @staticmethod
        def from_pretrained(*a,**kw):return Fake()
        model_type='roberta'
        num_labels=2
        def save_pretrained(self,p):
            Path(p,'config.json').write_text('{}')
            Path(p,'tokenizer_config.json').write_text('{}')
    mod=ModuleType('transformers');mod.AutoTokenizer=Fake;mod.AutoConfig=Fake
    monkeypatch.setitem(sys.modules,'transformers',mod)
    monkeypatch.setattr(m4,'inspect_public_classifier',lambda *_a,**kw:SimpleNamespace(
        model_id='public/roberta',resolved_revision='a'*40,num_labels=2,
        id2label={0:'NEGATIVE',1:'POSITIVE'},architecture='roberta'))
    monkeypatch.setattr(m4,'_predict_bundle',lambda *_a,**kw:[1]+[0]*199)
    root=demo.root/'results'/'auto_optimizer_packages_v1'
    with pytest.raises(ValueError,match='accuracy'):
        m4.package_model(demo.m3,demo.data,root=demo.root,output_root=root)
    assert list(root.glob('*.zip'))==[]
