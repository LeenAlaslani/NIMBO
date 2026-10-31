"""Pure fixture tests: do not download models or touch production adoption state."""
import copy
import hashlib
import json
from pathlib import Path
import pytest

from tools.final_judging_pack_v1 import read_evidence, publish, update_readme


SHA='a'*64
RID='heldout_foobar0123456789'
REV='revision_foo'
MODEL='lvwerra/distilbert-imdb'


def put(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False)+'\n',encoding='utf8')


def fixtures(tmp):
    put(tmp/'results/resource_tuning_v3/latest_development_selection.json',{
        'status':'SELECTED_DEV_ONLY','heldout_evaluated_for_this_selection':True,
        'selected_candidate_threads':2,'model_id':MODEL,'model_revision':REV,
        'development_samples':200,'development_rounds':2,
        'results':[{'candidate_threads':2,'gate':{'eligible_on_development':True,
                    'wall_improvement_fraction':.16,'cpu_time_improvement_fraction':.055}}]})
    run={'run_id':RID,'model_id':MODEL,'model_revision':REV,'sample_count':800,'rounds':3,
         'thread_count':4,'candidate_thread_count':2,'dataset_fingerprint':'abc',
         'optimization':{'approval':'eligible_for_review','export':{'sha256':SHA,
           'onnx_model':'results/staged/model.onnx','manifest':'results/staged/manifest.json'},
           'gate':{'optimization_objective':'cpu_time','eligible':True,
                   'quality_pass':True,'latency_pass':True,'cpu_pass':True,
                   'p95_pass':True,'repeatability_pass':True,
                   'mean_speedup_fraction':.1479,'cpu_time_reduction_fraction':.0254}},
         'comparison':{'sample_count':800,'rounds':3,'prediction_agreement_fraction':1.,
          'baseline':{'accuracy_on_reference_labels':.9175,'correct_unique':734,
                      'warm_forward':{'mean_ms':20},'process_cpu_forward':{'mean_ms':50}},
          'candidate':{'accuracy_on_reference_labels':.9175,'correct_unique':734,
                       'warm_forward':{'mean_ms':17.042},'process_cpu_forward':{'mean_ms':48.73}}}}
    f=tmp/'results/independent_validation_v21.jsonl'
    f.parent.mkdir(parents=True,exist_ok=True)
    f.write_text(json.dumps(run)+'\n',encoding='utf8')
    meta={'run_id':RID,'model_revision':REV,'candidate_threads':2,
          'baseline_threads':4,'production_deployed':False}
    tests=[{'concurrency':w,'repeat':r,'status':'PASS',
            'prediction_parity':{'all_agree':True},'performance_flags':[]}
           for w in (1,2,4) for r in (1,2)]
    put(tmp/'results/concurrency_probe_v1/stress_test.json',{
        **meta,'model_sha256':SHA,'status':'PASS','adoption_state_changed':False,
        'tests':tests})
    put(tmp/'results/http_probe_v1/http_stress_test.json',{
        **meta,'artifact_sha256':SHA,'status':'PASS','failed_conditions':0,
        'flagged_conditions':0,'adoption_state_changed':False,
        'cases':[{'concurrency':t['concurrency'],'repeat':t['repeat'],'status':'PASS',
                  'predictions_match':True,'performance_flags':[]} for t in tests]})
    adopt=dict(meta)
    del adopt['run_id']
    put(tmp/'results/adoption_proof_v1/session_1/report.json',{
        **adopt,'source_run_id':RID,'artifact_sha256':SHA,'status':'PASS',
        'rollback_verified':True,'original_adoption_state_modified':False,
        'test_service_kind':'isolated_actual_adoption_router_proof',
        'shadow_and_canary':{'outcome':'CANARY_PASS','shadow_count':6,
                             'canary_pairs':5,'canary_candidate_only':2},'errors':[]})
    (tmp/'README.md').write_text('# NIMBO\n\n## Verified main proof\n\nOriginal kept.\n',encoding='utf8')
    return tmp


def test_emit_public_evidence_and_readme_idempotent(tmp_path):
    root=fixtures(tmp_path)
    m=publish(root)
    assert m['candidate_cpu_threads']==2
    assert m['mean_process_cpu_time_reduction_fraction']==.0254
    assert all(not k.startswith('/Users/') for k in str(m['sources']))
    snap=(root/'README.md').read_text()
    publish(root)
    assert (root/'README.md').read_text()==snap
    assert snap.count('<!-- NIMBO_FINAL_PROOF_BEGIN -->')==1
    assert 'Original kept.' in snap
    s=json.loads((root/'docs/FINAL_JUDGING_EVIDENCE.json').read_text())
    assert len(s['sources'])==5


@pytest.mark.parametrize('mutator',[
    lambda p: p['optimization']['gate'].update(cpu_pass=False),
    lambda p: p['optimization'].update(approval='rejected'),
    lambda p: p.update(candidate_thread_count=4),
    lambda p: p['comparison']['candidate'].update(correct_unique=700),
    lambda p: p['optimization']['gate'].update(cpu_time_reduction_fraction=.30),
])
def test_invalid_heldout_rejected_without_publication(tmp_path,mutator):
    root=fixtures(tmp_path)
    f=root/'results/independent_validation_v21.jsonl'
    data=json.loads(f.read_text());mutator(data);f.write_text(json.dumps(data)+'\n')
    with pytest.raises(ValueError,match='Evidence rejected'):publish(root)
    assert not (root/'docs/FINAL_JUDGING_EVIDENCE.json').exists()


def test_reject_unrelated_http_artifact(tmp_path):
    root=fixtures(tmp_path)
    f=root/'results/http_probe_v1/http_stress_test.json'
    d=json.loads(f.read_text());d['artifact_sha256']='b'*64;put(f,d)
    with pytest.raises(ValueError,match='no matching evidence'):read_evidence(root)


def test_reject_failed_rollback(tmp_path):
    root=fixtures(tmp_path)
    f=root/'results/adoption_proof_v1/session_1/report.json'
    d=json.loads(f.read_text());d['rollback_verified']=False;put(f,d)
    with pytest.raises(ValueError,match='adoption proof'):read_evidence(root)


def test_dev_selection_locked(tmp_path):
    root=fixtures(tmp_path)
    f=root/'results/resource_tuning_v3/latest_development_selection.json'
    d=json.loads(f.read_text());d['heldout_evaluated_for_this_selection']=False;put(f,d)
    with pytest.raises(ValueError,match='development selection'):read_evidence(root)
