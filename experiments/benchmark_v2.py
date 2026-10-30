"""Real, held-out SST-2 policy evaluation with separate calibration and test sets.

Calibration chooses a threshold under a minimum accuracy constraint. Test is never
used to choose policy. Timings include the actual small->large fallback path.
"""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from nimbo.benchmark import macro_f1, percentile_ms
from nimbo.text_backend import load_classifier
from experiments.run_sst2 import SMALL, LARGE


def choose_threshold(small_records, large_records, labels, *, max_accuracy_drop=0.01, candidates=None):
    """Pick lowest fallback count satisfying baseline accuracy tolerance on calibration data.
    Confidence may be miscalibrated; this is empirical threshold selection, not calibration.
    """
    candidates = candidates or [0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0]
    baseline_acc = mean(int(x['label'] == y) for x,y in zip(large_records,labels))
    viable = []
    for t in candidates:
        choices = [l if s['confidence'] < t else s for s,l in zip(small_records,large_records)]
        acc = mean(int(p['label'] == y) for p,y in zip(choices,labels))
        fallbacks = sum(s['confidence'] < t for s in small_records)
        if acc >= baseline_acc - max_accuracy_drop - 1e-12:
            viable.append((fallbacks,t,acc))
    if not viable:
        return 1.0
    return min(viable)[1]


def classify_summary(truth, predictions, times_ms, routes):
    return {
        'accuracy': round(mean(int(y==p['label']) for y,p in zip(truth,predictions)),5),
        'macro_f1': round(macro_f1(truth,[p['label'] for p in predictions]),5),
        'mean_latency_ms': round(mean(times_ms),3),
        'p50_latency_ms': percentile_ms(times_ms,.5),
        'p95_latency_ms': percentile_ms(times_ms,.95),
        'fallback_rate': round(sum(x=='small_then_large' for x in routes)/len(routes),5),
        'samples':len(truth),
    }


def run_real_benchmark(calibration, testing, small, large, max_drop=.01, warmup=5):
    for row in (calibration + testing)[:warmup]:
        small(row['text']);large(row['text'])
    cal_s=[small(r['text']) for r in calibration]
    cal_l=[large(r['text']) for r in calibration]
    threshold=choose_threshold(cal_s,cal_l,[r['label'] for r in calibration],max_accuracy_drop=max_drop)
    measured={'small':([],[],[]),'large':([],[],[]),'adaptive':([],[],[])}
    # Alternate first method per item to reduce systematic ordering bias.
    for i,r in enumerate(testing):
        text=r['text']
        def timed(call):
            start=perf_counter(); res=call(); return res,(perf_counter()-start)*1000
        def small_call(): return small(text)
        def large_call(): return large(text)
        def adaptive_call():
            s=small(text)
            if s['confidence'] < threshold: return large(text),'small_then_large'
            return s,'small_only'
        funcs={'small':small_call,'large':large_call,'adaptive':adaptive_call}
        order=('large','adaptive','small') if i%2 else ('small','adaptive','large')
        for name in order:
            result,elapsed=timed(funcs[name])
            route='none'
            if name=='adaptive':result,route=result
            measured[name][0].append(result)
            measured[name][1].append(elapsed)
            measured[name][2].append(route)
    truth=[r['label'] for r in testing]
    summaries={name: classify_summary(truth,*values) for name,values in measured.items()}
    return {'calibration_samples':len(calibration),'test_samples':len(testing),
            'threshold':threshold,'max_calibration_accuracy_drop':max_drop,
            'strategies':summaries,
            'adaptive_vs_large_latency_reduction_pct':round(100*(1-summaries['adaptive']['mean_latency_ms']/summaries['large']['mean_latency_ms']),2),
            'adaptive_vs_small_accuracy_delta':round(summaries['adaptive']['accuracy']-summaries['small']['accuracy'],5),
            'adaptive_vs_large_accuracy_delta':round(summaries['adaptive']['accuracy']-summaries['large']['accuracy'],5)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--calibration',type=int,default=200)
    p.add_argument('--test',type=int,default=500)
    p.add_argument('--max-drop',type=float,default=.01)
    p.add_argument('--device',type=int,default=-1)
    p.add_argument('--output',default='results/benchmark_v2.json')
    a=p.parse_args()
    if a.calibration<10 or a.test<10 or a.calibration+a.test>872 or not 0<=a.max_drop<=1:
        p.error('SST-2 validation has 872 examples; use calibration>=10, test>=10, total<=872 and max-drop in [0,1]')
    from datasets import load_dataset
    ds=load_dataset('glue','sst2',split='validation')
    rows=[{'text':r['sentence'],'label':'POSITIVE' if r['label']==1 else 'NEGATIVE'} for r in ds]
    cal=rows[:a.calibration]; test=rows[a.calibration:a.calibration+a.test]
    small=load_classifier(SMALL,device=a.device,sst2_labels=True)
    large=load_classifier(LARGE,device=a.device,sst2_labels=True)
    report=run_real_benchmark(cal,test,small,large,max_drop=a.max_drop)
    report.update({'dataset':'GLUE SST-2 validation disjoint calibration and test slices',
                   'models':{'small':SMALL,'large':LARGE},'platform':platform.platform(),
                   'limitations':['SST-2 validation was divided into disjoint calibration/test subsets; independent official test labels unavailable.',
                                  'Confidence threshold tuned for empirical accuracy, not statistically calibrated probabilities.',
                                  'Timing on one device; no GPU energy or peak-memory savings claimed.',
                                  'Production policies should calibrate on representative workloads, not SST-2 alone.']})
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
