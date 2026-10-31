"""Honest diagnostics/selection for NIMBO image routing.

Requires human labels for both image variants. Uses a calibrated threshold
for a given set of scores; NEVER interprets CLIP relative scores as probability.
Old evaluation samples may only be used as DEVELOPMENT data. New, untouched
prompts and seeds are mandatory for a final quality claim.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path


def label(v):
    s=str(v).strip().lower()
    if s in {'true','yes','y','1'}:return True
    if s in {'false','no','n','0'}:return False
    raise ValueError(f'Missing or invalid human label: {v!r}')


def read_labeled(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError('No rows')
    required={'prompt_id','object_relative_score','fast_s','standard_s','checker_s','human_fast_object_present','human_standard_object_present'}
    absent=required-set(rows[0]);
    if absent:raise ValueError('Missing columns: '+str(sorted(absent)))
    data=[]
    for row in rows:
        data.append(dict(id=str(row['prompt_id']), score=float(row['object_relative_score']),
           fast=label(row['human_fast_object_present']),standard=label(row['human_standard_object_present']),
           fast_s=float(row['fast_s']),standard_s=float(row['standard_s']),checker_s=float(row['checker_s']),
           category=row.get('category','unknown')))
    if len({r['id'] for r in data})!=len(data):raise ValueError('Duplicate prompt ids')
    return data


def metrics(rows, threshold):
    if not rows:raise ValueError('No evaluation rows')
    n=len(rows); escalation=[r['score'] < threshold for r in rows]
    correct=sum((r['standard'] if esc else r['fast']) for r,esc in zip(rows,escalation))
    seconds=sum(r['fast_s']+r['checker_s']+(r['standard_s'] if esc else 0) for r,esc in zip(rows,escalation))
    # Always-fast and standard baselines do not pay checker cost.
    fast_good=sum(r['fast'] for r in rows);standard_good=sum(r['standard'] for r in rows)
    bad_accept=sum(not r['fast'] and not esc for r,esc in zip(rows,escalation))
    waste=sum(r['fast'] and esc for r,esc in zip(rows,escalation))
    # A positive upgrade is an escalation that turns failure into success.
    useful=sum((not r['fast']) and r['standard'] and esc for r,esc in zip(rows,escalation))
    fail_escalation=sum((not r['standard']) and esc for r,esc in zip(rows,escalation))
    return {'n':n,'threshold':threshold,'successes':correct,'success_rate':round(correct/n,5),
        'fallback_count':sum(escalation),'fallback_rate':round(sum(escalation)/n,5),
        'false_accepts':bad_accept,'unnecessary_fallbacks':waste,
        'useful_escalations':useful,'failed_escalations':fail_escalation,
        'total_latency_s':round(seconds,4), 'mean_latency_s':round(seconds/n,4),
        'latency_per_success_s':round(seconds/correct,4) if correct else None,
        'fast_success_rate':round(fast_good/n,5), 'standard_success_rate':round(standard_good/n,5),
        'fast_total_s':round(sum(r['fast_s'] for r in rows),4),
        'standard_total_s':round(sum(r['standard_s'] for r in rows),4)}


def thresholds(rows):
    scores=sorted({r['score'] for r in rows})
    # strict < threshold. Add both sides of every observed score.
    return sorted(set([0.0,1.0,1.000001]+[min(1.000001,v+0.000001) for v in scores]))


def select_policy(rows, quality_floor=0.95, max_fallback=0.5):
    if not 0<=quality_floor<=1 or not 0<=max_fallback<=1:raise ValueError('constraints outside [0,1]')
    candidates=[metrics(rows,t) for t in thresholds(rows)]
    feasible=[m for m in candidates if m['success_rate']+1e-9>=quality_floor and m['fallback_rate']<=max_fallback+1e-9]
    if not feasible:
        return {'status':'no_feasible_adaptive_policy','selection':None,
                'reason':'No threshold passes both quality and fallback budget on development labels.',
                'candidate_count':len(candidates)}
    best=min(feasible,key=lambda m:(m['total_latency_s'],m['false_accepts'],m['fallback_count']))
    return {'status':'development_candidate_only','selection':best,'candidate_count':len(candidates),
            'reason':'Selected from labeled development examples. Validate on unseen prompts/seeds.'}


def diagnostic(rows, threshold=0.65, quality_floor=0.95, max_fallback=0.5):
    failures=[]
    for r in rows:
        escalated=r['score']<threshold
        if not escalated and not r['fast']:kind='false_accept'
        elif escalated and r['fast']:kind='unnecessary_fallback'
        elif escalated and not r['standard']:kind='fallback_still_fails'
        else:continue
        failures.append({'prompt_id':r['id'],'category':r['category'],'error':kind,
                         'relative_score':r['score'],'fast_object_present':r['fast'],
                         'standard_object_present':r['standard']})
    return {'scope':'development_diagnostics_only','existing_policy':metrics(rows,threshold),
         'errors':failures, 'selection':select_policy(rows,quality_floor,max_fallback),
         'limitations':['Small reviewed development set. Threshold improvement on these records is NOT unseen-test improvement.',
         'One reviewer, binary object presence only; no aesthetic quality or safety evaluation.',
         'New prompts, new seeds, independent labels and deployment latency measurements are required.',
         'Inference latency is not GPU energy, FLOPs or financial cost.']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',required=True)
    p.add_argument('--threshold',type=float,default=.65)
    p.add_argument('--quality-floor',type=float,default=.95)
    p.add_argument('--max-fallback',type=float,default=.5)
    p.add_argument('--output',default='results/image_policy_diagnostics_v3.json')
    args=p.parse_args()
    data=read_labeled(args.csv)
    result=diagnostic(data,args.threshold,args.quality_floor,args.max_fallback)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
