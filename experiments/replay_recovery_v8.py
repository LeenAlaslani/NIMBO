"""Offline oracle upper-bound simulation, NOT a deployable verifier benchmark.
Uses human review to simulate ideal decisions, expressly prohibited at runtime.
"""
import argparse, csv, json
from pathlib import Path

def yn(s):
    return str(s).strip().lower() in ('true', 'yes', '1', 'pass')

def read(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--benchmark', default='results/benchmark_v2_raw_labeled.csv')
    p.add_argument('--labels', default='results/full_prompt_labels_v6.csv')
    p.add_argument('--output',default='results/recovery_v8_oracle_bound.json')
    a=p.parse_args()
    base=read(a.benchmark)
    labels={r['prompt_id']:r for r in read(a.labels)}
    total=delivered=attempts=unrecoverable=0
    detail=[]
    for row in base:
        lab=labels.get(row['prompt_id'])
        if not lab or not lab['fast_full_prompt_pass'].strip() or not lab['standard_full_prompt_pass'].strip():
            raise ValueError('Missing human Full Prompt label for '+row['prompt_id'])
        fast=yn(lab['fast_full_prompt_pass'])
        standard=yn(lab['standard_full_prompt_pass'])
        if fast:
            path='fast'; duration=float(row['fast_s']); success=True; n=1
        else:
            path='standard' if standard else 'both_fail'
            duration=float(row['fast_s'])+float(row['standard_s']); success=standard; n=2
        total+=duration;delivered+=int(success);attempts+=n;unrecoverable+=int(not success)
        detail.append({'prompt_id':row['prompt_id'],'route_from_oracle':path,'success':success,'generation_time_s':round(duration,4)})
    result={'type':'ORACLE_UPPER_BOUND_NOT_DEPLOYABLE','sample_count':len(detail),'successes':delivered,
            'unrecoverable':unrecoverable,'generation_attempts':attempts,
            'generation_time_s':round(total,4),
            'generation_seconds_per_success':round(total/delivered,4) if delivered else None,
            'excluded_costs':['real automated verifier','verifier failures','end-to-end latency','model loading'],
            'warning':'Human ground-truth used to choose routing; NEVER use as real measured NIMBO performance.',
            'per_prompt':detail}
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='per_prompt'},indent=2))
    print('Saved:',target)
if __name__=='__main__': main()
