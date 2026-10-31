"""NIMBO v6: independent metrics for core-object and full-prompt criteria.
No inference, model tuning or image regeneration is performed here.
"""
import argparse
import csv
import json
from pathlib import Path

def boolean(s):
    v=str(s).strip().lower()
    if v in ('true','1','yes','pass'): return True
    if v in ('false','0','no','fail'): return False
    raise ValueError(f'Invalid boolean label: {s!r}')

def optional_boolean(s):
    return None if str(s).strip()=='' else boolean(s)

def metrics(rows,fast_key,std_key):
    complete=[r for r in rows if r.get(fast_key,'').strip() and r.get(std_key,'').strip()]
    answer={}
    def assess(name, route):
        successes=accepted_bad=unnecessary=escalations=0
        time=0.0
        for r in complete:
            fast=boolean(r[fast_key]); std=boolean(r[std_key])
            escalating=route(r)
            successes+=int(std if escalating else fast)
            accepted_bad+=int(not escalating and not fast)
            unnecessary+=int(escalating and fast)
            escalations+=int(escalating)
            if name=='nimbo_existing':
                time+=float(r['nimbo_s'])
            else:
                time+=float(r['fast_s'])+(float(r['standard_s']) if escalating else 0)
        return dict(successes=successes,rate=round(successes/len(complete),4) if complete else None,
                    false_accepts=accepted_bad,unnecessary_fallbacks=unnecessary,escalations=escalations,
                    total_latency_s=round(time,4),seconds_per_success=round(time/successes,4) if successes else None)
    answer['labeled_samples']=len(complete)
    answer['always_fast']=assess('always_fast',lambda r:False)
    answer['always_standard']=assess('always_standard',lambda r:True)
    answer['nimbo_existing']=assess('nimbo_existing',lambda r:r['route']=='escalated_to_standard')
    return answer

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',default='results/benchmark_v2_raw_labeled.csv')
    ap.add_argument('--full-labels',default='results/full_prompt_labels_v6.csv')
    ap.add_argument('--output',default='results/quality_evaluation_v6.json')
    args=ap.parse_args()
    with open(args.csv,encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows: raise ValueError('CSV empty')
    needed=['prompt_id','human_fast_object_present','human_standard_object_present','route','fast_s','standard_s','nimbo_s']
    for col in needed:
        if col not in rows[0]: raise ValueError(f'Missing column {col}')
    ids=[str(r['prompt_id']) for r in rows]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate prompt_id')
    label_file=Path(args.full_labels)
    if not label_file.exists():
        label_file.parent.mkdir(parents=True,exist_ok=True)
        with label_file.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['prompt_id','prompt','fast_full_prompt_pass','standard_full_prompt_pass','review_notes'])
            w.writeheader()
            for r in rows:
                w.writerow(dict(prompt_id=r['prompt_id'],prompt=r.get('prompt',''),
                                fast_full_prompt_pass='',standard_full_prompt_pass='',review_notes=''))
    with label_file.open(encoding='utf-8-sig',newline='') as f:
        extras=list(csv.DictReader(f))
    extra_by_id={str(r['prompt_id']):r for r in extras}
    for r in rows:
        e=extra_by_id.get(str(r['prompt_id']),{})
        r['fast_full_prompt_pass']=e.get('fast_full_prompt_pass','')
        r['standard_full_prompt_pass']=e.get('standard_full_prompt_pass','')
        for field in ('fast_full_prompt_pass','standard_full_prompt_pass'):
            optional_boolean(r[field])
    core=metrics(rows,'human_fast_object_present','human_standard_object_present')
    full=metrics(rows,'fast_full_prompt_pass','standard_full_prompt_pass')
    summary={
        'scope':'16-image development benchmark; not a held-out validation',
        'quality_definitions':{
            'core_object':'Original human labels: recognizable required object with essential appearance',
            'full_prompt':'Separate human labels: core object plus required attributes, relations, and scene constraints'
        },
        'core_object':core,'full_prompt':full,
        'warnings':['Core-object and full-prompt scores MUST NOT be merged.',
                    'Latency is existing T4 wall-clock proxy, not GPU compute or dollars.',
                    'Full-prompt results appear only for pairs with both human labels.',
                    'Changing ground-truth labels requires visual review.']
    }
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('Full-prompt review template:',label_file)
    print('Metrics saved:',output)

if __name__=='__main__':
    main()
