"""Offline consistency check for published NIMBO *summary*, not a benchmark rerun."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path, PurePosixPath

DEFAULT = Path('docs/FINAL_JUDGING_EVIDENCE.json')


def verify_public_proof(path=DEFAULT):
    data=json.loads(Path(path).read_text(encoding='utf8'))
    def check(ok, message):
        if not ok:
            raise ValueError('Public proof rejected: '+message)
    def number(key, lo=None, hi=None):
        v=data.get(key)
        check(type(v) in (int,float) and math.isfinite(v), 'invalid '+key)
        if lo is not None:check(v>=lo, 'invalid range '+key)
        if hi is not None:check(v<=hi, 'invalid range '+key)
        return float(v)
    check(data.get('schema_version')=='nimbo_final_judging_evidence_v1','unknown evidence schema')
    check(data.get('candidate_status')=='eligible_for_review_only','invalid candidate status')
    check(data.get('holdout_examples')==800 and data.get('holdout_rounds')==3,'unknown evaluation set')
    check(data.get('baseline_cpu_threads')==4 and data.get('candidate_cpu_threads')==2,'unexpected CPU thread settings')
    check(data.get('development',{}).get('selection_status')=='SELECTED_DEV_ONLY','development selection missing')
    check(data.get('development',{}).get('selected_threads')==2 and data.get('development',{}).get('samples')==200,'development settings differ')
    check(data.get('correct_per_path')==734,'reference-label count inconsistent')
    a=number('accuracy_baseline',0,1);b=number('accuracy_candidate',0,1)
    check(abs(a-734/800)<1e-6 and abs(b-734/800)<1e-6,'accuracy inconsistent with labels')
    check(abs(number('prediction_agreement_fraction',0,1)-1)<1e-10,'agreement not complete')
    wall_b=number('baseline_warm_forward_ms',0);wall_c=number('candidate_warm_forward_ms',0)
    cpu_b=number('baseline_process_cpu_ms',0);cpu_c=number('candidate_process_cpu_ms',0)
    check(wall_b>0 and cpu_b>0,'invalid baseline denominator')
    w=number('mean_warm_forward_latency_reduction_fraction',0,1)
    c=number('mean_process_cpu_time_reduction_fraction',0,1)
    check(abs((wall_b-wall_c)/wall_b-w)<.003,'warm latency reduction does not match times')
    check(abs((cpu_b-cpu_c)/cpu_b-c)<.003,'CPU-time reduction does not match times')
    check(data.get('concurrency',{}).get('cases_passed')==6 and data['concurrency'].get('status')=='PASS','engine concurrency failed')
    check(data.get('http',{}).get('cases_passed')==6 and data['http'].get('status')=='PASS','HTTP probe failed')
    adoption=data.get('adoption',{})
    check(adoption.get('status')=='PASS' and adoption.get('rollback_verified') is True and adoption.get('original_state_untouched') is True, 'adoption/rollback failed')
    check(adoption.get('canary_pairs')==5 and adoption.get('canary_candidate_only')==2,'canary proof missing')
    check(adoption.get('scope')=='isolated_actual_router_http_not_primary_ui','HTTP scope inflated')
    for k in ('artifact_sha256','model_revision','source_run_id'):
        check(isinstance(data.get(k),str) and len(data[k])>8, 'missing '+k)
    check(len(data['artifact_sha256'])==64 and all(c in '0123456789abcdef' for c in data['artifact_sha256']), 'invalid artifact checksum')
    sources=data.get('sources')
    check(isinstance(sources,dict) and set(sources)=={'development','heldout_log','engine_stress','http_stress','adoption'},'incomplete source index')
    for name,rec in sources.items():
        path=rec.get('file');digest=rec.get('sha256')
        check(isinstance(path,str) and path and not PurePosixPath(path).is_absolute() and '..' not in PurePosixPath(path).parts and '\\' not in path, 'unsafe path '+name)
        check(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest), 'invalid source hash '+name)
    exclusions='; '.join(data.get('not_measured',[])).lower()
    check(all(x in exclusions for x in ('energy','flops','billing')),'required claim boundaries absent')
    print('PUBLIC JUDGING EVIDENCE: CONSISTENT (summary-only; raw local logs not reverified).')
    print(f'Heldout 800 x 3 | 91.75% accuracy both | lower warm latency {w*100:.2f}% | lower process CPU time {c*100:.2f}%')
    print('Engine 6/6 | HTTP 6/6 | rollback verified | NOT energy/FLOPs/billing evidence')
    return data


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--path',type=Path,default=DEFAULT)
    args=p.parse_args()
    verify_public_proof(args.path)


if __name__=='__main__': main()
