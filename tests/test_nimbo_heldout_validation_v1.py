"""M3 dependency-light tests, do not imply external model measurements."""
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from app.nimbo_heldout_validation_v1 import (
    load_heldout, read_m2_candidate, decide_heldout, preprocess_text,
    validate_existing_candidate,
)
from tools.prepare_nimbo_tweeteval_holdout_v1 import prepare


def write_set(path, n=200, labels=('0','1','2')):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f)
        w.writerow(['text','label'])
        for i in range(n):
            w.writerow([f'Tweet number {i}', labels[i%len(labels)]])
    return path


def metrics(*,accuracy=.80, mean=30.,p95=39.,cpu=1000.):
    return {'accuracy':accuracy,'mean_warm_forward_ms':mean,
            'p95_warm_forward_ms':p95,'process_cpu_forward_total_ms':cpu}


def write_m2(root, graph_data=b'X'*4096):
    folder=root/'results'/'auto_optimizer_candidates_v1'/'candidate_test'
    folder.mkdir(parents=True)
    graph=folder/'model.onnx'
    graph.write_bytes(graph_data)
    sha=hashlib.sha256(graph_data).hexdigest()
    report={
      'schema_version':'nimbo_onnx_candidate_v1',
      'status':'EXPERIMENT_ONLY_NOT_EXPORTED',
      'model':{'model_id':'team/unseen-model','resolved_revision':'a'*40},
      'dataset_sha256':'d'*64,
      'cpu_threads_each_path':2,'max_length':256,
      'decision':{'local_gates_passed':True,'decision':'EXPLORATORY_PASS_NOT_APPROVED'},
      'artifact':{'path':str(graph),'sha256':sha,'bytes':4096},
    }
    p=folder/'report.json'
    p.write_text(json.dumps(report),encoding='utf8')
    return p,graph,report


def load_source(root, report, heldout='f'*64, threads=2):
    return read_m2_candidate(report,root=root,model_id='team/unseen-model',sha='a'*40,
                             heldout_sha=heldout,threads=threads,max_length=256)


def test_valid_200_labeled_rows(tmp_path):
    path=write_set(tmp_path/'data.csv')
    rows, sha, provenance=load_heldout(path)
    assert len(rows)==200 and len(sha)==64
    assert rows[0]['label']=='0'
    assert provenance['source']=='user_supplied_not_independently_audited'


def test_reject_small_evaluation_dataset(tmp_path):
    path=write_set(tmp_path/'data.csv',n=12)
    with pytest.raises(ValueError,match='200-800'):
        load_heldout(path)


def test_reject_missing_labels(tmp_path):
    path=write_set(tmp_path/'data.csv',n=200)
    data=path.read_text().replace('Tweet number 1,1','Tweet number 1,')
    path.write_text(data)
    with pytest.raises(ValueError,match='label'):
        load_heldout(path)


def test_reject_invalid_provenance(tmp_path):
    path=write_set(tmp_path/'data.csv')
    path.with_suffix('.source.json').write_text(json.dumps({'rows':200,'csv_sha256':'x'*64}))
    with pytest.raises(ValueError,match='provenance'):
        load_heldout(path)


def test_preparer_uses_actual_rows_and_source_indices(tmp_path):
    source=[{'text':f'original test tweet {i}', 'label':i%3} for i in range(260)]
    path,meta=prepare(source, revision='f'*40, output=tmp_path/'genuine.csv',samples=200)
    out,sha,prov=load_heldout(path)
    assert len(out)==200
    assert meta['csv_sha256']==sha
    assert out[0]['text'].startswith('original test tweet ')
    assert prov['source']=='cardiffnlp/tweet_eval'
    assert meta['source_indices']==sorted(meta['source_indices'])


def test_preparer_refuses_overwrite(tmp_path):
    source=[{'text':f't{i}', 'label':i%3} for i in range(240)]
    path=tmp_path/'set.csv'
    prepare(source,revision='f'*40,output=path,samples=200)
    with pytest.raises(FileExistsError):
        prepare(source,revision='f'*40,output=path,samples=200)


def test_m2_candidate_hash_verified(tmp_path):
    report,graph,_=write_m2(tmp_path)
    obj,path=load_source(tmp_path,report)
    assert path==graph and obj['artifact']['bytes']==4096


def test_m2_candidate_reject_tampered_graph(tmp_path):
    report,graph,_=write_m2(tmp_path)
    graph.write_bytes(b'Z'*4096)
    with pytest.raises(ValueError,match='checksum'):
        load_source(tmp_path,report)


def test_m2_candidate_reject_mismatched_dataset(tmp_path):
    report,_,_=write_m2(tmp_path)
    with pytest.raises(ValueError,match='differ'):
        load_source(tmp_path,report,heldout='d'*64)


def test_m2_candidate_reject_changed_threads(tmp_path):
    report,_,_=write_m2(tmp_path)
    with pytest.raises(ValueError,match='CPU threads'):
        load_source(tmp_path,report,threads=4)


def test_m2_candidate_reject_failed_local_gates(tmp_path):
    report,_,doc=write_m2(tmp_path)
    doc['decision']['local_gates_passed']=False
    report.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='gates'):
        load_source(tmp_path,report)


def test_good_holdout_review_only():
    result=decide_heldout(metrics(),metrics(mean=20,p95=30,cpu=950),sample_count=200,agreement=1.,max_logit_error=.001,repeats=0)
    assert result['heldout_gates_passed']
    assert result['decision']=='HELDOUT_LOCAL_PASS_REQUIRES_PACKAGE_RELOAD'
    assert result['eligible_for_deployment'] is False
    assert result['eligible_for_downloadable_package'] is False


@pytest.mark.parametrize('bad,agreement,err,repeats', [
    (metrics(mean=29,p95=39,cpu=950),1.,0.,0),
    (metrics(mean=20,p95=45,cpu=950),1.,0.,0),
    (metrics(mean=20,p95=30,cpu=1150),1.,0.,0),
    (metrics(accuracy=.79,mean=20,p95=30,cpu=950),1.,0.,0),
    (metrics(mean=20,p95=30,cpu=950),.99,0.,0),
    (metrics(mean=20,p95=30,cpu=950),1.,.02,0),
    (metrics(mean=20,p95=30,cpu=950),1.,0.,1),
])
def test_reject_failed_metrics(bad,agreement,err,repeats):
    result=decide_heldout(metrics(),bad,sample_count=200,agreement=agreement,
                          max_logit_error=err,repeats=repeats)
    assert not result['heldout_gates_passed']
    assert result['decision']=='REJECTED_HELDOUT_GATES'


def test_cardiff_preprocess():
    assert preprocess_text('hi @Leen see https://x', 'cardiff_tweet')=='hi @user see http'
    assert preprocess_text('a @Leen', 'identity')=='a @Leen'
    with pytest.raises(ValueError):preprocess_text('hi','new-mode')


def test_toy_full_revalidation_with_new_session(tmp_path,monkeypatch):
    """An injected toy baseline/session checks workflow; NOT external ONNX inference."""
    import sys
    import numpy as np
    import torch
    import app.nimbo_heldout_validation_v1 as m3
    from app.nimbo_model_intake_v1 import ModelIntake
    dataset=write_set(tmp_path/'holdout.csv')
    report,graph,_=write_m2(tmp_path)
    spec=ModelIntake('team/unseen-model','a'*40,'bert',3,{0:'negative',1:'neutral',2:'positive'},4096,('model.safetensors',),'apache-2.0')
    monkeypatch.setattr(m3,'inspect_public_classifier',lambda *args,**kwargs:spec)

    class Tokenizer:
        def __call__(self,text,**kwargs):
            index=int(text.split()[-1])%3
            return {'input_ids':torch.tensor([[index,0]],dtype=torch.long),
                    'attention_mask':torch.tensor([[1,1]],dtype=torch.long)}

    class Baseline(torch.nn.Module):
        def forward(self,input_ids,attention_mask):
            x=input_ids[:,0].long()
            logits=torch.nn.functional.one_hot(x,num_classes=3).float()
            return SimpleNamespace(logits=logits)

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*a,**kwargs):return Tokenizer()
    class AutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(*a,**kwargs):return Baseline()
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoTokenizer=AutoTokenizer,AutoModelForSequenceClassification=AutoModelForSequenceClassification))

    class SessionOptions:pass
    class FakeSession:
        def __init__(self,path,sess_options,providers):
            assert path==str(graph)
        def get_providers(self):return ['CPUExecutionProvider']
        def get_inputs(self):return [SimpleNamespace(name='input_ids'),SimpleNamespace(name='attention_mask')]
        def get_outputs(self):return [SimpleNamespace(name='logits')]
        def run(self,unused,feeds):
            idx=feeds['input_ids'][:,0].astype(int)
            return [np.eye(3,dtype=np.float32)[idx]]
    monkeypatch.setitem(sys.modules,'onnxruntime',SimpleNamespace(
        SessionOptions=SessionOptions,InferenceSession=FakeSession,
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1)))
    output,result=validate_existing_candidate(report,dataset,root=tmp_path,rounds=2,
                           output_root=tmp_path/'reports')
    assert output.is_file()
    assert result['status']=='HELDOUT_VALIDATION_COMPLETED_REVIEW_ONLY'
    assert result['baseline']['accuracy']==1.
    assert result['candidate']['accuracy']==1.
    assert result['decision']['prediction_agreement']==1.
    assert result['decision']['eligible_for_deployment'] is False
    assert result['source_graph_sha256']==hashlib.sha256(graph.read_bytes()).hexdigest()
    assert 'Tweet number' not in output.read_text()
