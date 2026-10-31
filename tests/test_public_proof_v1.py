"""Public summary checks are fixture-only and require no model downloads."""
import copy
import json
import pytest
from tools.verify_public_proof_v1 import verify_public_proof


def proof():
    return {'schema_version':'nimbo_final_judging_evidence_v1',
       'source_run_id':'heldout_7830c2ff573a47ae', 'model_revision':'x'*40,
       'artifact_sha256':'a'*64,'candidate_status':'eligible_for_review_only',
       'holdout_examples':800,'holdout_rounds':3,
       'baseline_cpu_threads':4,'candidate_cpu_threads':2,
       'development':{'selection_status':'SELECTED_DEV_ONLY','selected_threads':2,'samples':200},
       'correct_per_path':734,'accuracy_baseline':.9175,'accuracy_candidate':.9175,
       'prediction_agreement_fraction':1.,
       'baseline_warm_forward_ms':20.,'candidate_warm_forward_ms':17.042,
       'mean_warm_forward_latency_reduction_fraction':.1479,
       'baseline_process_cpu_ms':50.,'candidate_process_cpu_ms':48.73,
       'mean_process_cpu_time_reduction_fraction':.0254,
       'concurrency':{'cases_passed':6,'status':'PASS'},
       'http':{'cases_passed':6,'status':'PASS'},
       'adoption':{'status':'PASS','rollback_verified':True,'original_state_untouched':True,
                   'canary_pairs':5,'canary_candidate_only':2,
                   'scope':'isolated_actual_router_http_not_primary_ui'},
       'sources':{k:{'file':'results/'+k+'.json','sha256':'b'*64} for k in ('development','heldout_log','engine_stress','http_stress','adoption')},
       'not_measured':['electrical energy','FLOPs','cloud billing']}


def write(tmp_path, obj):
    path=tmp_path/'proof.json'
    path.write_text(json.dumps(obj),encoding='utf8')
    return path


def test_valid_summary(tmp_path):
    assert verify_public_proof(write(tmp_path,proof()))['correct_per_path']==734


@pytest.mark.parametrize('change',[lambda x:x.update(mean_process_cpu_time_reduction_fraction=.3),
    lambda x:x['adoption'].update(rollback_verified=False),
    lambda x:x['http'].update(cases_passed=5),
    lambda x:x['sources']['heldout_log'].update(file='/Users/private/trace.json'),
    lambda x:x.update(accuracy_candidate=.95)])
def test_invalid_summary_rejected(tmp_path,change):
    x=proof();change(x)
    with pytest.raises(ValueError,match='Public proof rejected'):
        verify_public_proof(write(tmp_path,x))
