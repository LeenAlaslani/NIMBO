"""NIMBO v4: independent validation-policy selection and held-out test evaluation.

Evaluates an adaptive route only when it actually helps the small model on
calibration. Reports a transparent small-only policy otherwise. Never fits on test.
"""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path
from statistics import mean
from time import perf_counter_ns

from experiments.run_sst2 import SMALL, LARGE
from nimbo.benchmark import macro_f1
from nimbo.text_backend import load_classifier

THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0)


def accuracy(rows, preds):
    if len(rows) != len(preds) or not rows:
        raise ValueError('Expected nonempty, equally sized records and predictions')
    return sum(r['label'] == p['label'] for r,p in zip(rows,preds)) / len(rows)


def select_policy(rows, small_preds, large_preds, max_drop=.01, max_fallback=.25,
                  min_gain=0.005):
    """Select on calibration only; require gain over small and quality floor.

    min_gain defaults to 0.5 percentage points; if not reached, use small-only.
    The confidence threshold is empirically selected, not probability calibrated.
    """
    small_acc=accuracy(rows,small_preds)
    large_acc=accuracy(rows,large_preds)
    candidates=[]
    for threshold in THRESHOLDS:
        mask=[p['confidence'] < threshold for p in small_preds]
        fallback_rate=sum(mask)/len(rows)
        chosen=[l if fallback else s for s,l,fallback in zip(small_preds,large_preds,mask)]
        acc=accuracy(rows,chosen)
        viable=(acc >= large_acc-max_drop-1e-12 and
                fallback_rate <= max_fallback+1e-12 and
                fallback_rate > 0 and
                acc >= small_acc+min_gain-1e-12)
        candidates.append({'threshold':threshold,'accuracy':round(acc,5),
                           'fallback_rate':round(fallback_rate,5),
                           'gain_vs_small':round(acc-small_acc,5),
                           'eligible':viable})
    viable=[c for c in candidates if c['eligible']]
    selected=min(viable,key=lambda x:(x['fallback_rate'],-x['accuracy'])) if viable else None
    policy=({'type':'adaptive','threshold':selected['threshold']} if selected
            else {'type':'small_only','threshold':None,
                  'reason':'No adaptive policy demonstrated enough calibration benefit under constraints'})
    return policy,candidates,{'small':round(small_acc,5),'large':round(large_acc,5)}


def timed(fn):
    start=perf_counter_ns()
    response=fn()
    return response,(perf_counter_ns()-start)/1e6


def summarize(rows, predictions, durations, fallback):
    values=sorted(durations)
    def percentile(q):
        idx=(len(values)-1)*q
        low=int(idx); upper=min(low+1,len(values)-1)
        return values[low]+(values[upper]-values[low])*(idx-low)
    return {'samples':len(rows), 'accuracy':round(accuracy(rows,predictions),5),
            'macro_f1':round(macro_f1([r['label'] for r in rows],
                                      [p['label'] for p in predictions]),5),
            'mean_latency_ms':round(mean(values),4),
            'p50_latency_ms':round(percentile(.5),4),
            'p95_latency_ms':round(percentile(.95),4),
            'fallback_rate':round(fallback/len(rows),5)}


def evaluate(rows, small, large, policy, warmup=3):
    for r in rows[:warmup]:
        small(r['text']); large(r['text'])
    observations={name:{'pred':[],'time':[]} for name in ('small','large','nimbo')}
    fallbacks=0
    for i,row in enumerate(rows):
        text=row['text']
        def adaptive():
            nonlocal fallbacks
            p=small(text)
            if policy['type']=='adaptive' and p['confidence'] < policy['threshold']:
                fallbacks+=1
                return large(text)
            return p
        runners={'small':lambda:small(text),'large':lambda:large(text),'nimbo':adaptive}
        order=('small','nimbo','large') if i%2==0 else ('large','nimbo','small')
        for name in order:
            pred,elapsed=timed(runners[name])
            observations[name]['pred'].append(pred)
            observations[name]['time'].append(elapsed)
    return {name:summarize(rows,obs['pred'],obs['time'],fallbacks if name=='nimbo' else 0)
            for name,obs in observations.items()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--calibration',type=int,default=300)
    p.add_argument('--test',type=int,default=800)
    p.add_argument('--max-drop',type=float,default=.01)
    p.add_argument('--max-fallback',type=float,default=.25)
    p.add_argument('--min-gain',type=float,default=.005)
    p.add_argument('--output',default='results/rotten_tomatoes_v4.json')
    args=p.parse_args()
    if not 20 <= args.calibration <= 1066 or not 20 <= args.test <= 1066:
        p.error('calibration and test must each be between 20 and 1066')
    if not all(0<=x<=1 for x in [args.max_drop,args.max_fallback,args.min_gain]):
        p.error('Constraints must be between 0 and 1')
    from datasets import load_dataset
    ds=load_dataset('cornell-movie-review-data/rotten_tomatoes')
    def records(split,n):
        return [{'text':r['text'],'label':'POSITIVE' if r['label']==1 else 'NEGATIVE'}
                for r in ds[split].select(range(n))]
    # Unlike v3, calibration and test are distinct, named dataset splits.
    calibration=records('validation',args.calibration)
    testing=records('test',args.test)
    small=load_classifier(SMALL,device=-1,sst2_labels=True)
    large=load_classifier(LARGE,device=-1,sst2_labels=True)
    small_cal=[small(r['text']) for r in calibration]
    large_cal=[large(r['text']) for r in calibration]
    policy,candidates,cal_acc=select_policy(calibration,small_cal,large_cal,
        args.max_drop,args.max_fallback,args.min_gain)
    stats=evaluate(testing,small,large,policy)
    large_ms=stats['large']['mean_latency_ms']
    result={
        'evaluation':'Cross-dataset sentiment transfer; calibration on validation, final test on test',
        'data':'cornell-movie-review-data/rotten_tomatoes',
        'calibration_split':'validation','test_split':'test',
        'calibration_samples':len(calibration),'test_samples':len(testing),
        'constraints':{'max_accuracy_drop':args.max_drop,'max_fallback_rate':args.max_fallback,
                       'minimum_adaptive_gain_over_small':args.min_gain},
        'calibration_accuracy':cal_acc,
        'selected_policy':policy,'calibration_candidates':candidates,
        'test_strategies':stats,
        'nimbo_vs_large_latency_reduction_pct':round(100*(1-stats['nimbo']['mean_latency_ms']/large_ms),2)
                if large_ms else None,
        'test_nimbo_quality_constraint_met':stats['nimbo']['accuracy'] >= stats['large']['accuracy']-args.max_drop-1e-12,
        'agent_added_quality_value_on_test':stats['nimbo']['accuracy'] > stats['small']['accuracy'],
        'platform':platform.platform(),'models':{'small':SMALL,'large':LARGE},
        'limitations':['Policy is selected using validation labels only; never tune based on test output.',
                       'If policy is small_only, speed improvement over BERT is not routing intelligence.',
                       'Quality constraint on calibration is empirical, not a statistical guarantee.',
                       'Training-data overlap of either pretrained model with Rotten Tomatoes unverified.',
                       'CPU latency only; no energy, FLOP or billing cost claims.']}
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
