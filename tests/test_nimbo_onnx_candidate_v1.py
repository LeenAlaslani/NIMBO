"""Dependency-light M2 tests. Do not claim real model inference from these."""
from copy import deepcopy

import pytest

from app.nimbo_onnx_candidate_v1 import (
    select_inputs, summarize_measurements, decide_candidate, validate_reference_report,
)


def measurements(*, latency=20.0, p95=30.0, cpu=300.0, accuracy=0.833333):
    return {'mean_warm_forward_ms':latency, 'p95_warm_forward_ms':p95,
            'process_cpu_forward_total_ms':cpu, 'accuracy':accuracy}


def reference():
    return {'schema_version':'nimbo_baseline_v1', 'status':'BASELINE_ONLY_NOT_OPTIMIZED',
            'model':{'model_id':'team/roberta-model','resolved_revision':'a'*40},
            'dataset_sha256':'f'*64,'cpu_threads':2, 'max_length':256,
            'metrics':{'mean_warm_forward_ms':39.472}}


def check_reference(p):
    validate_reference_report(p, model_id='team/roberta-model', sha='a'*40,
                              dataset_sha='f'*64, cpu_threads=2, max_length=256)


def test_reference_exact_match():
    check_reference(reference())


@pytest.mark.parametrize('field,value', [('schema_version','other'),('status','fake'),('dataset_sha256','d'*64),
   ('cpu_threads',4),('max_length',128)])
def test_refuse_incompatible_baseline(field,value):
    doc=reference();doc[field]=value
    with pytest.raises(ValueError):check_reference(doc)


def test_refuse_different_model_sha():
    doc=reference();doc['model']['resolved_revision']='b'*40
    with pytest.raises(ValueError):check_reference(doc)


def test_refuse_absent_original_metrics():
    doc=reference();doc['metrics']={}
    with pytest.raises(ValueError):check_reference(doc)


def test_selects_roberta_inputs_only():
    assert select_inputs({'input_ids':1,'attention_mask':2},{'input_ids','attention_mask'})==('input_ids','attention_mask')


def test_selects_bert_token_type_ids():
    assert select_inputs({'attention_mask':2,'input_ids':1,'token_type_ids':3},
                         {'input_ids','attention_mask','token_type_ids'}) == ('input_ids','attention_mask','token_type_ids')


def test_does_not_pass_unaccepted_token_type_ids():
    assert select_inputs({'input_ids':1,'token_type_ids':3},{'input_ids'}) == ('input_ids',)


def test_refuses_unknown_required_tokens():
    with pytest.raises(ValueError,match='unsupported'):
        select_inputs({'input_ids':1,'custom_mask':2},{'input_ids','custom_mask'})


def test_refuses_missing_input_ids():
    with pytest.raises(ValueError,match='input_ids'):
        select_inputs({'attention_mask':1},{'attention_mask'})


def test_summary_uses_supplied_labels_not_guessing():
    out=summarize_measurements([10.,20.,30.],[12.,22.,32.],[1,0,1],[1,1,1])
    assert out['accuracy']==round(2/3,6)
    assert out['mean_warm_forward_ms']==20.
    assert out['process_cpu_forward_total_ms']==66.
    assert out['p95_warm_forward_ms']==29.


def test_summary_missing_labels_reports_none():
    assert summarize_measurements([10.,10.],[1.,1.],[None,None],[0,0])['accuracy'] is None


def test_summary_rejects_non_finite_timings():
    with pytest.raises(ValueError):summarize_measurements([float('nan')],[1.],[None],[0])


def test_speed_and_cpu_gates_pass_but_12_samples_never_approved():
    result=decide_candidate(measurements(), measurements(latency=14,p95=23,cpu=220),1.,samples=12,max_abs_logit_error=.001)
    assert result['local_gates_passed']
    assert result['decision']=='EXPLORATORY_PASS_NOT_APPROVED'
    assert result['eligible_for_deployment'] is False
    assert result['eligible_for_downloadable_package'] is False


def test_80_samples_only_local_pass_with_no_deployment():
    result=decide_candidate(measurements(), measurements(latency=12,p95=25,cpu=250),1.,samples=80,max_abs_logit_error=.001)
    assert result['decision']=='LOCAL_PASS_REQUIRES_HELDOUT_AND_RELOAD'
    assert result['eligible_for_deployment'] is False


@pytest.mark.parametrize('candidate,agreement,err', [
    (measurements(latency=22),1.,.001),
    (measurements(latency=15,p95=31),1.,.001),
    (measurements(latency=15,p95=25,cpu=320),1.,.001),
    (measurements(latency=15,p95=25,cpu=250,accuracy=.75),1.,.001),
    (measurements(latency=15,p95=25,cpu=250),.9,.001),
    (measurements(latency=15,p95=25,cpu=250),1.,.03),
])
def test_rejection_gates(candidate,agreement,err):
    decision=decide_candidate(measurements(),candidate,agreement,samples=12,max_abs_logit_error=err)
    assert decision['decision']=='REJECTED_LOCAL_GATES'
    assert decision['local_gates_passed'] is False


def test_unlabeled_set_never_quality_verified():
    base=measurements(accuracy=None)
    cand=measurements(latency=10,p95=25,cpu=250,accuracy=None)
    result=decide_candidate(base,cand,1.,samples=12,max_abs_logit_error=.001)
    assert result['quality_pass'] is False and result['accuracy_drop'] is None


def test_invalid_thresholds_fail_closed():
    with pytest.raises(ValueError):decide_candidate(measurements(),measurements(),1.,samples=12,
                                                   max_abs_logit_error=0, min_speedup=1.5)


def test_invalid_sample_count_fail_closed():
    with pytest.raises(ValueError):decide_candidate(measurements(),measurements(),1.,samples=1,max_abs_logit_error=0)


def test_exporter_preserves_dynamic_axes_cpu_provider_and_input_names(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import torch
    from app.nimbo_onnx_candidate_v1 import _load_onnx_session

    seen = {}

    class Model(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=torch.stack((input_ids.sum(-1), attention_mask.sum(-1)), dim=1).float())

    def fake_export(model, args, target, **kwargs):
        seen.update(kwargs)
        assert len(args) == 2 and args[0].shape == (1, 3)
        from pathlib import Path
        Path(target).write_bytes(b'x'*2048)

    class SessionOptions:
        pass

    class Session:
        def __init__(self, path, sess_options, providers):
            seen['providers'] = providers
        def get_providers(self):
            return ['CPUExecutionProvider']
        def get_inputs(self):
            return [SimpleNamespace(name='input_ids'), SimpleNamespace(name='attention_mask')]
        def get_outputs(self):
            return [SimpleNamespace(name='logits')]

    ort = SimpleNamespace(SessionOptions=SessionOptions, InferenceSession=Session,
                          GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1))
    monkeypatch.setitem(sys.modules, 'onnxruntime', ort)
    monkeypatch.setattr(torch.onnx, 'export', fake_export)
    tensor=torch.tensor([[1,2,3]], dtype=torch.long)
    sess, names=_load_onnx_session(Model(), None, {'input_ids':tensor, 'attention_mask':tensor},
                                  tmp_path/'export'/'model.onnx', threads=2)
    assert names == ('input_ids','attention_mask')
    assert seen['providers']==['CPUExecutionProvider']
    assert seen['dynamo'] is False
    assert seen['input_names']==['input_ids','attention_mask']
    assert seen['dynamic_axes']['input_ids'][1]=='sequence'
    assert seen['opset_version']==17


def test_full_report_flow_with_injected_toy_model_not_real_benchmark(tmp_path, monkeypatch):
    import sys
    import json
    import numpy as np
    import torch
    from types import SimpleNamespace
    from app.nimbo_model_intake_v1 import ModelIntake
    import app.nimbo_onnx_candidate_v1 as m2

    dataset = tmp_path/'texts.csv'
    dataset.write_text('text,label\n"excellent",POSITIVE\n"awful",NEGATIVE\n', encoding='utf8')
    import hashlib
    sha=hashlib.sha256(dataset.read_bytes()).hexdigest()
    spec = ModelIntake('team/toy-model','a'*40,'bert',2,{0:'NEGATIVE',1:'POSITIVE'},1000,
                       ('model.safetensors',),'apache-2.0')
    monkeypatch.setattr(m2,'inspect_public_classifier',lambda *_args,**_kw:spec)

    class Tokenizer:
        def __call__(self, text, **kwargs):
            value=1 if 'excellent' in text else 0
            return {'input_ids':torch.tensor([[value, 0]],dtype=torch.long),
                    'attention_mask':torch.tensor([[1,1]],dtype=torch.long)}

    class TinyModel(torch.nn.Module):
        def forward(self,input_ids,attention_mask):
            x=input_ids.float()[:,0]
            return SimpleNamespace(logits=torch.stack((1-x,x),dim=1))

    class Session:
        def run(self, unused, ort_inputs):
            x=ort_inputs['input_ids'][:,0]
            return [np.stack((1-x,x),axis=1).astype(np.float32)]

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*a,**kw):return Tokenizer()

    class AutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(*a,**kw):return TinyModel()

    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoTokenizer=AutoTokenizer, AutoModelForSequenceClassification=AutoModelForSequenceClassification))

    def fake_export(model,tokenizer,tokens,graph_path,*,threads):
        graph_path.parent.mkdir(parents=True,exist_ok=False)
        graph_path.write_bytes(b'x'*2048)
        return Session(),('input_ids','attention_mask')

    monkeypatch.setattr(m2,'_load_onnx_session',fake_export)

    previous = reference()
    previous['model']['model_id']='team/toy-model'
    previous['dataset_sha256']=sha
    old=tmp_path/'baseline.json'
    old.write_text(json.dumps(previous),encoding='utf8')

    location, report=m2.run_paired_candidate('team/toy-model',dataset,revision='a'*40,
                                            baseline_report=old,rounds=2,output_root=tmp_path/'candidates')
    assert location.is_file()
    assert report['status']=='EXPERIMENT_ONLY_NOT_EXPORTED'
    assert report['unique_dataset_rows']==2
    assert report['forward_calls_each_path']==4
    assert report['baseline']['accuracy']==1
    assert report['onnx_candidate']['accuracy']==1
    assert report['decision']['prediction_agreement']==1
    assert report['decision']['eligible_for_downloadable_package'] is False
    assert report['artifact']['bytes']==2048
    assert 'excellent' not in location.read_text(encoding='utf8')
    assert 'awful' not in location.read_text(encoding='utf8')
