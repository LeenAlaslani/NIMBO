#!/usr/bin/env python3
"""Local CPU *inference-engine* concurrency stress probe (not HTTP, not energy).

Never runs deployment/canary endpoints, changes runtime policy or chooses a new
configuration using heldout scores. Verifies staged ONNX SHA and fixed threads.
Uses synthetic unlabeled input. Tests prediction parity, failures, throughput,
service and burst-response latency at concurrency 1/2/4. No accuracy claims.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path('results/concurrency_probe_v1')
LEVELS = (1, 2, 4)
SOURCE_SELECTION = Path('results/resource_tuning_v3/latest_development_selection.json')

# Fixed, synthetic stress inputs. No ground-truth labels, no reuse of tuned
# development or heldout datasets. Mixed lengths, punctuation and negation.
SEEDS = (
    'I loved the acting and the memorable soundtrack.',
    'A genuinely disappointing film; the ending made no sense.',
    'The camera work was lovely, although the plot was slow.',
    'Not terrible, just unremarkable and somewhat predictable.',
    'Neither the script nor the performance lived up to expectations.',
    'An absorbing film with a strong story and great pacing.',
    'I cannot recommend this movie; the dialogue was awful.',
    'The cast was good but I was bored during the second half.',
    'A moving story. The performances were wonderful!',
    'What looked promising soon became messy and incoherent.',
    'It has an odd tone and that actually works really well.',
    'I expected little, but I was pleasantly surprised.',
    'The performances were fine; nothing else was memorable.',
    'Beautifully directed and consistently interesting.',
    'I disliked the sequel far more than the original film.',
    'It is hardly flawless, but the climax is great.',
)


def make_prompts(n: int) -> list[str]:
    if not isinstance(n, int) or not 8 <= n <= 128:
        raise ValueError('Requests per condition must be 8..128')
    prompts = []
    for i in range(n):
        base = SEEDS[i % len(SEEDS)]
        # Deterministic variation without changing the classification task.
        context = ' A supporting character changes their mind.' * (i % 3)
        prompts.append(base + context + (' Overall, I would watch it again.' if i % 7 == 0 else ''))
    return prompts


def percentile_ms(xs, q=.95):
    if not xs or not 0 <= q <= 1 or not all(isinstance(x,(int,float)) and math.isfinite(x) and x >= 0 for x in xs):
        raise ValueError('Invalid latency measurements')
    ordered=sorted(xs)
    rank=(len(ordered)-1)*q
    lo=int(rank); hi=min(lo+1,len(ordered)-1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(rank-lo)


def summarize_batch(items, *, request_count, elapsed_ms, cpu_ms, concurrency, path):
    """Side-effect-free summary; queueing is included in burst-response timing."""
    if len(items)!=request_count or not 8 <= request_count <= 128 or concurrency not in LEVELS:
        raise ValueError('Incomplete or invalid stress batch')
    if not (math.isfinite(elapsed_ms) and elapsed_ms>0 and math.isfinite(cpu_ms) and cpu_ms>=0):
        raise ValueError('Invalid batch resource metrics')
    if any(type(r.get('index')) is not int or r['index']<0 or r['index']>=request_count for r in items):
        raise ValueError('Invalid request identity')
    if {r['index'] for r in items}!=set(range(request_count)):
        raise ValueError('Missing or repeated request')
    errors=[r for r in items if r['error'] is not None]
    ok=[r for r in items if r['error'] is None]
    if any(r['service_ms']<0 or r['response_ms']<r['service_ms'] for r in ok):
        raise ValueError('Impossible latency')
    for r in ok:
        if r.get('prediction') not in (0,1): raise ValueError('Malformed prediction')
    lat_service=[r['service_ms'] for r in ok]
    lat_response=[r['response_ms'] for r in ok]
    return {'path':path, 'concurrency':concurrency,'requests':request_count,
            'successes':len(ok),'errors':len(errors),
            'throughput_successes_per_sec':round(len(ok)*1000/elapsed_ms,4),
            'batch_elapsed_ms':round(elapsed_ms,4),
            'batch_process_cpu_ms':round(cpu_ms,4),
            'batch_process_cpu_per_request_ms':round(cpu_ms/request_count,4),
            'service_mean_ms':round(statistics.mean(lat_service),4) if ok else None,
            'service_p95_ms':round(percentile_ms(lat_service),4) if ok else None,
            'burst_response_mean_ms':round(statistics.mean(lat_response),4) if ok else None,
            'burst_response_p95_ms':round(percentile_ms(lat_response),4) if ok else None,
            'error_details':[{'index':r['index'],'message':r['error']} for r in errors[:5]]}


def run_batch(path, worker, prompts, concurrency):
    if path not in ('baseline','candidate') or concurrency not in LEVELS:
        raise ValueError('Invalid stress path or concurrency')
    started=time.perf_counter(); cpu_started=time.process_time()
    submissions={}
    results=[]
    def work(i, prompt):
        entered=time.perf_counter()
        try:
            pred=worker(path,prompt)
            if type(pred) is not int or pred not in (0,1): raise ValueError('Invalid prediction from engine')
            error=None
        except Exception as exc:
            pred=None;error=f'{type(exc).__name__}: {str(exc)[:180]}'
        ended=time.perf_counter()
        return i,pred,error,entered,ended
    with ThreadPoolExecutor(max_workers=concurrency,thread_name_prefix='nimbo-probe') as ex:
        futures={}
        for i,prompt in enumerate(prompts):
            submission=time.perf_counter()
            futures[ex.submit(work,i,prompt)]=(i,submission)
        for future in as_completed(futures):
            i,submission=futures[future]
            try:
                idx,pred,error,entered,ended=future.result()
                assert idx==i
                results.append({'index':idx,'prediction':pred,'error':error,
                    'service_ms':(ended-entered)*1000,'response_ms':(ended-submission)*1000})
            except BaseException as exc:
                results.append({'index':i,'prediction':None,'error':f'Worker infrastructure: {type(exc).__name__}: {str(exc)[:160]}',
                                'service_ms':0,'response_ms':0})
    elapsed_ms=(time.perf_counter()-started)*1000
    cpu_ms=(time.process_time()-cpu_started)*1000
    return summarize_batch(results,request_count=len(prompts),elapsed_ms=elapsed_ms,
                            cpu_ms=cpu_ms,concurrency=concurrency,path=path),{
                                r['index']:r['prediction'] for r in results if r['error'] is None}


def compare_predictions(baseline,candidate,n):
    if any(k not in baseline or k not in candidate for k in range(n)):
        return {'compared':len(set(baseline)&set(candidate)),'total':n,'mismatch_indices':[],
                'all_agree':False,'reason':'Missing predictions from failed requests'}
    mismatches=[i for i in range(n) if baseline[i]!=candidate[i]]
    return {'compared':n,'total':n,'mismatch_indices':mismatches[:10],
            'all_agree':not mismatches,'reason':'All predictions agree' if not mismatches else 'Inference output mismatch'}


def validate_lineage(selection, record, review):
    if selection.get('status')!='SELECTED_DEV_ONLY' or selection.get('heldout_evaluated_for_this_selection') is not True:
        raise ValueError('Development selection is not locked and heldout-evaluated')
    chosen=selection.get('selected_candidate_threads')
    if chosen != 2 and chosen not in LEVELS:
        raise ValueError('Unexpected candidate thread count')
    if record.get('candidate_thread_count')!=chosen or record.get('thread_count')!=4:
        raise ValueError('Staged heldout threads differ from development selection')
    if record.get('optimization_objective')!='cpu_time':
        raise ValueError('Heldout run did not gate CPU objective')
    gate=record.get('optimization',{}).get('gate',{})
    if record.get('optimization',{}).get('approval')!='eligible_for_review' or not all(
        gate.get(x) is True for x in ('eligible','quality_pass','latency_pass','cpu_pass','p95_pass','repeatability_pass')):
        raise ValueError('Heldout CPU objective did not pass')
    if record.get('model_revision')!=selection.get('model_revision') or record.get('model_revision')!=review['revision']:
        raise ValueError('Pinned model revision mismatch')
    if review['validation'].get('run_id')!=record.get('run_id') or not review.get('sha256'):
        raise ValueError('Staged candidate lineage mismatch')
    return chosen


def actual_probe(requests, repeats):
    import numpy as np
    import torch
    from app.nimbo_adoption_v21_1 import _review, _prepare_cache
    from tools.staged_candidate_smoke_v1 import latest_exported_run

    selection=json.loads(SOURCE_SELECTION.read_text(encoding='utf8'))
    run_id=latest_exported_run()
    review=_review(run_id) # SHA, pinned revision, manifest checked; no rollout
    record=review['validation']
    chosen=validate_lineage(selection,record,review)
    engine=_prepare_cache(review) # CPU only, warms both paths; no state changes
    inp_names=engine['inputs']
    assert engine['candidate'].get_providers()==['CPUExecutionProvider']

    def worker(which,text):
        values=engine['tokenizer'](text,max_length=256,truncation=True,return_tensors='np')
        prepared={k:np.asarray(v,dtype=np.int64) for k,v in values.items() if k in inp_names}
        if which=='baseline':
            with torch.inference_mode():
                logits=engine['baseline'](**{k:torch.from_numpy(v) for k,v in prepared.items()}).logits
            raw=logits.detach().numpy()
        elif which=='candidate':
            raw=engine['candidate'].run(None,prepared)[0]
        else: raise ValueError('Unexpected inference path')
        scores=np.asarray(raw,dtype=np.float64).reshape(-1)
        if scores.size!=2 or not np.isfinite(scores).all():
            raise ValueError('Non-finite or invalid inference logits')
        return int(scores.argmax())

    prompts=make_prompts(requests)
    seed_digest=hashlib.sha256('\n'.join(prompts).encode('utf8')).hexdigest()
    all_tests=[]
    hard_failures=[]
    performance_flags=[]
    for workers in LEVELS:
        for rep in range(repeats):
            order=('baseline','candidate') if rep % 2==0 else ('candidate','baseline')
            batches={};preds={}
            for which in order:
                stats,output=run_batch(which,worker,prompts,workers)
                batches[which]=stats;preds[which]=output
            parity=compare_predictions(preds['baseline'],preds['candidate'],len(prompts))
            passed=not batches['baseline']['errors'] and not batches['candidate']['errors'] and parity['all_agree']
            flags=[]
            if passed:
                # Descriptive signals only: small synthetic burst is NOT a deployment gate.
                if batches['candidate']['burst_response_p95_ms'] > batches['baseline']['burst_response_p95_ms']:
                    flags.append('candidate_burst_p95_worse')
                if batches['candidate']['throughput_successes_per_sec'] < batches['baseline']['throughput_successes_per_sec']:
                    flags.append('candidate_throughput_worse')
            if flags: performance_flags.append({'concurrency':workers,'repeat':rep+1,'flags':flags})
            all_tests.append({'concurrency':workers,'repeat':rep+1,
                              'baseline':batches['baseline'],'candidate':batches['candidate'],
                              'prediction_parity':parity,'performance_flags':flags,
                              'status':'FAIL' if not passed else ('PERFORMANCE_REVIEW' if flags else 'PASS')})
            if not passed:hard_failures.append({'concurrency':workers,'repeat':rep+1,'parity':parity,
                'baseline_errors':batches['baseline']['errors'],'candidate_errors':batches['candidate']['errors']})
            print(f"Workers {workers}, repeat {rep+1}: {all_tests[-1]['status']} | "
                  f"throughput baseline {batches['baseline']['throughput_successes_per_sec']:.2f}/s, "
                  f"candidate {batches['candidate']['throughput_successes_per_sec']:.2f}/s | "
                  f"p95 response baseline {batches['baseline']['burst_response_p95_ms']}, "
                  f"candidate {batches['candidate']['burst_response_p95_ms']} ms", flush=True)
    report={'schema_version':'stress_v1','timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'run_id':run_id,'model_revision':review['revision'],'model_sha256':review['sha256'],
        'model_id':record['model_id'],'selection_file':str(SOURCE_SELECTION),
        'pytorch_threads':record['thread_count'],'onnx_threads':chosen,
        'requests_per_condition':requests,'repeats_per_concurrency':repeats,
        'concurrency_levels':list(LEVELS),'synthetic_input_sha256':seed_digest,
        'platform':platform.platform(),
        'tests':all_tests,'failure_details':hard_failures,
        'performance_regressions_to_review':performance_flags,
        'status':'FAIL' if hard_failures else ('PERFORMANCE_REVIEW' if performance_flags else 'PASS'),
        'production_deployed':False,'adoption_state_changed':False,
        'measurement_note':('Isolated CPU inference-engine burst, synthetic unlabeled inputs. '
          'Each path uses same inputs and pinned model. Burst response includes executor queueing, '
          'not HTTP/network; warmup and model loading excluded. CPU time is per-process aggregated '
          'and may include other process activity. Both models remain resident; shared peak RSS '
          'cannot be attributed to one path. Throughput and p95 depend on local load. '
          'No labeled accuracy, true request queue, energy, FLOPs, billing or production load proof.')}
    ROOT.mkdir(parents=True,exist_ok=True)
    output=ROOT/f"stress_{run_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('=== NIMBO CONCURRENCY PROBE ===')
    print('STATUS:',report['status'])
    print('RUN:',run_id)
    print('EVIDENCE:',output)
    print('PERFORMANCE FLAGS:',len(performance_flags),'(descriptive, not a statistical gate)')
    print('SCOPE: engine-level CPU concurrency; NO HTTP / accuracy / energy claims')
    if hard_failures:raise RuntimeError('Concurrency probe detected errors or prediction mismatch; no approval')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description='CPU-only inference engine stress probe; does NOT deploy')
    parser.add_argument('--requests',type=int,default=24)
    parser.add_argument('--repeats',type=int,default=2)
    opt=parser.parse_args(argv)
    if not 8 <= opt.requests <= 128 or not 1 <= opt.repeats <= 3:
        parser.error('Requests 8..128 and repeats 1..3; use a separate benchmark for larger production loads')
    actual_probe(opt.requests,opt.repeats)


if __name__=='__main__':
    main()
