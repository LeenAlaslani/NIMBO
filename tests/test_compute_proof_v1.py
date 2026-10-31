"""Regressions: evidence must be genuine; failed quality/speed cannot stage."""
import copy
import pytest
from app.nimbo_validate_v21 import ValidationRequest, gate_v21, _process_peak_rss_mb
from tools.compute_proof_v1 import summarize
from tools.staged_candidate_smoke_v1 import inspect_results, latest_exported_run

REQ=ValidationRequest()

def measures(acc=.9175, mean=30.0, local=31.0, p95=40.0):
    return {'accuracy_on_reference_labels':acc,
            'warm_forward':{'mean_ms':mean,'p95_ms':p95},
            'tokenizer_plus_forward':{'mean_ms':local}}

def rounds(b=30,c=20):
    return [{'baseline_forward_mean_ms':b,'candidate_forward_mean_ms':c} for _ in range(3)]

def test_accepts_faster_equal_accuracy():
    a=gate_v21(measures(),measures(mean=20,local=21,p95=30),1.0,rounds(),REQ)
    assert a['eligible'] and a['quality_pass'] and a['latency_pass']

def test_rejects_accuracy_regression_even_when_faster():
    a=gate_v21(measures(),measures(acc=.85,mean=20,local=21,p95=30),.9,rounds(),REQ)
    assert not a['eligible'] and not a['quality_pass']

def test_rejects_p95_regression_despite_faster_mean():
    a=gate_v21(measures(),measures(mean=20,local=21,p95=45),1.0,rounds(),REQ)
    assert not a['eligible'] and not a['p95_pass']

def test_rejects_round_instability():
    a=gate_v21(measures(),measures(mean=20,local=21,p95=30),1.0,rounds()[:2]+[{'baseline_forward_mean_ms':20,'candidate_forward_mean_ms':25}],REQ)
    assert not a['eligible'] and not a['repeatability_pass']

def test_refuses_resource_report_without_cpu_measurements():
    with pytest.raises(ValueError,match='predates'):
        summarize({'comparison':{'sample_count':800,'rounds':3}})

def test_rejects_invented_cpu_reduction():
    rec={'run_id':'heldout_mock','resource_evidence':{'instrumentation_version':'process_cpu_v1',
         'forward_process_cpu_mean_reduction_fraction':.90,'observed_process_peak_rss_mb':500,
         'cpu_scope':'process time','memory_scope':'shared'},
         'optimization':{'approval':'eligible_for_review','export':None,'gate':{}},
         'comparison':{'sample_count':800,'rounds':3,'all_labels_present':True,
         'baseline':{'process_cpu_forward':{'mean_ms':20},'accuracy_on_reference_labels':.9},
         'candidate':{'process_cpu_forward':{'mean_ms':10},'accuracy_on_reference_labels':.9}}}
    with pytest.raises(ValueError,match='inconsistent'):
        summarize(rec)

def test_staged_smoke_fails_on_label_mismatch():
    with pytest.raises(ValueError,match='failed'):
        inspect_results([{'parity_pass':False,'prediction_agreement':False,'positive_probability_delta':.5}])

def test_staged_smoke_fails_without_probes():
    with pytest.raises(ValueError,match='No smoke probes'):
        inspect_results([])

def test_rss_returns_nonnegative_number():
    assert _process_peak_rss_mb()>0


def test_smoke_rejects_nonfinite_probability_delta():
    with pytest.raises(ValueError,match='failed'):
        inspect_results([{'parity_pass':True,'prediction_agreement':True,'positive_probability_delta':float('nan')}])


def test_smoke_does_not_use_old_export_when_latest_run_rejected(tmp_path):
    import json
    file=tmp_path/'runs.jsonl'
    file.write_text('\n'.join(map(json.dumps,[
        {'run_id':'heldout_old','optimization':{'approval':'eligible_for_review','export':{'sha256':'abc'}}},
        {'run_id':'heldout_recent','optimization':{'approval':'rejected','export':None}},
    ]))+'\n')
    with pytest.raises(ValueError,match='Latest heldout'):
        latest_exported_run(file)


def test_validator_records_real_path_cpu_measurements_with_dependency_hooks(tmp_path,monkeypatch):
    """Full generator passes fake inferencers; no torch/ONNX downloads in CI."""
    import sys
    import types
    import contextlib
    from app import nimbo_validate_v21 as validate

    class Predictions:
        def __init__(self, label):self.label=label
        def argmax(self, axis=None):return types.SimpleNamespace(item=lambda: self.label)

    class Tok:
        def __call__(self, text, **kwargs):return {'input_ids':int(text.split('_')[-1])%2}

    class PT:
        def __call__(self, **kwargs):return types.SimpleNamespace(logits=Predictions(kwargs['input_ids']))

    class Session:
        def get_inputs(self):return [types.SimpleNamespace(name='input_ids')]
        def run(self,_,values):return [Predictions(values['input_ids'])]

    torch=types.SimpleNamespace(set_num_threads=lambda n:None,from_numpy=lambda v:v,
        inference_mode=contextlib.nullcontext, argmax=lambda logits,dim:logits.argmax(), __version__='fake')
    numpy=types.SimpleNamespace(int64=int,asarray=lambda v,dtype=None:v)
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(
        AutoModelForSequenceClassification=object,AutoTokenizer=object))
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'numpy',numpy)
    monkeypatch.setitem(sys.modules,'onnxruntime',types.SimpleNamespace(__version__='fake'))
    monkeypatch.setattr(validate,'weight_bytes',lambda x:1234)
    monkeypatch.setattr(validate,'LOG',tmp_path/'log.jsonl')
    graph=tmp_path/'model.onnx'
    graph.write_bytes(b'fake-graph')
    rows=[(f'review_{i}',i%2) for i in range(500)]
    gen=validate.validate_onnx(validate.ValidationRequest(sample_count=500,rounds=2,export_variant=False),
         data_adapter=lambda req:(rows,'fake_disjoint'),
         model_adapter=lambda req:(Tok(),PT(),Session(),graph,'0'*40))
    packets=list(gen)
    report=packets[-1]['report']
    b=report['comparison']['baseline'];c=report['comparison']['candidate']
    for result in (b,c):
        assert result['process_cpu_forward']['count']==1000
        assert result['process_cpu_local']['count']==1000
        assert result['process_cpu_forward']['mean_ms']>=0
    assert report['comparison']['resource_evidence']['observed_process_peak_rss_mb']>0
    assert report['comparison']['baseline']['accuracy_on_reference_labels']==1
    assert report['comparison']['candidate']['accuracy_on_reference_labels']==1
    stored=__import__('json').loads((tmp_path/'log.jsonl').read_text().splitlines()[-1])
    assert stored['resource_evidence']['instrumentation_version']=='process_cpu_v1'
