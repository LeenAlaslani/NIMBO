"""Pure resource-objective evaluation of an existing heldout run; no new claims."""
from __future__ import annotations
from app.nimbo_validate_v21 import ValidationRequest, gate_v21

def assess_record(rec, *, max_cpu_increase_fraction=0.0):
    comp=rec.get('comparison') or {}
    if comp.get('sample_count')!=800 or comp.get('rounds')!=3 or not comp.get('all_labels_present'):
        raise ValueError('Expected real labeled 800/3 heldout record')
    if not str(rec.get('run_id','')).startswith('heldout_'):
        raise ValueError('Expected heldout run id')
    b,c=comp['baseline'],comp['candidate']
    agreement=comp['prediction_agreement_fraction']
    rr=comp['per_round']
    if len(rr)!=3 or any(not isinstance(v.get('baseline_forward_mean_ms'),(float,int)) or not isinstance(v.get('candidate_forward_mean_ms'),(float,int)) for v in rr):
        raise ValueError('Round-level evidence incomplete')
    common=dict(sample_count=800,rounds=3,max_cpu_increase_fraction=max_cpu_increase_fraction)
    latency=gate_v21(b,c,agreement,rr,ValidationRequest(**common,optimization_objective='latency'))
    cpu=gate_v21(b,c,agreement,rr,ValidationRequest(**common,optimization_objective='cpu_time'))
    return {'run_id':rec['run_id'],'source_approval':rec.get('optimization',{}).get('approval'),
            'measured_wall_latency_improvement_fraction':latency['mean_speedup_fraction'],
            'measured_cpu_time_improvement_fraction':cpu['cpu_time_reduction_fraction'],
            'cpu_time_measured':cpu['cpu_time_measured'],
            'latency_review_pass':latency['eligible'],
            'cpu_time_review_pass':cpu['eligible'],
            'recommended_scope':('LATENCY_ONLY' if latency['eligible'] and not cpu['eligible'] else
                                 'LATENCY_AND_CPU_TIME' if cpu['eligible'] else 'NO_APPROVAL'),
            'cpu_reason':cpu['reason'],
            'constraints':cpu['constraints'],
            'not_measured':['energy_joules','FLOPs','cloud_cost_usd','isolated_per_model_RSS'],
            'deployment_approved':False,
            'not_new_experiment':True}
