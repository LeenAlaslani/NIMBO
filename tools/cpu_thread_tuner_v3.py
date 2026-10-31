#!/usr/bin/env python3
"""NIMBO: tune CPU thread count on disjoint *development* IMDb rows only.

Never tunes using heldout labels/latencies; no deployment or auto-approval.
Paired warm CPU timings use one fixed PyTorch(4 threads) baseline for each
candidate ONNX thread count. Model loading, export, and tokenizer are outside
forward timing. Per-path process CPU measures time across process threads.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.nimbo_adoption_v21_1 import _review
from app.nimbo_optimize_v20 import MODEL_ID, TrialRequest, evaluation_set, p95
from app.nimbo_validate_v21 import ValidationRequest, validate_onnx
from tools.staged_candidate_smoke_v1 import latest_exported_run

OUTPUT = Path('results/resource_tuning_v3')
CANDIDATE_COUNTS = (1, 2, 4)
BASELINE_THREADS = 4
DEV_SAMPLES = 200
DEV_ROUNDS = 2


def digest_texts(rows):
    return hashlib.sha256(('\n'.join(hashlib.sha256(t.encode('utf8')).hexdigest()
                                     for t, _ in rows)).encode('utf8')).hexdigest()


def summarize(values):
    if not values or not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError('Missing, negative or nonfinite timing samples')
    return {'mean_ms': round(statistics.mean(values), 4),
            'p95_ms': round(p95(values), 4), 'count': len(values)}


def score_candidate(baseline, candidate, *, agreement, baseline_accuracy,
                    candidate_accuracy, per_round):
    """Fail closed; measured CPU reduction must be >=2%, wall >=5%.

    2% is an exploration screening rule, NOT a statistical confidence interval.
    Final release uses the existing stricter heldout CPU objective gate.
    """
    reasons = []
    b_wall = baseline['wall']['mean_ms']; c_wall = candidate['wall']['mean_ms']
    b_cpu = baseline['cpu']['mean_ms']; c_cpu = candidate['cpu']['mean_ms']
    b_p95 = baseline['wall']['p95_ms']; c_p95 = candidate['wall']['p95_ms']
    numbers = [b_wall, c_wall, b_cpu, c_cpu, b_p95, c_p95,
               agreement, baseline_accuracy, candidate_accuracy]
    if not all(isinstance(n, (int, float)) and math.isfinite(n) for n in numbers):
        raise ValueError('Invalid or missing numeric measurements')
    if b_wall <= 0 or b_cpu <= 0 or b_p95 <= 0:
        raise ValueError('Invalid baseline measurements')
    wall_gain = (b_wall - c_wall) / b_wall
    cpu_gain = (b_cpu - c_cpu) / b_cpu
    if agreement < .995: reasons.append('prediction agreement below 99.5%')
    if candidate_accuracy < baseline_accuracy - .01 - 1e-12:
        reasons.append('development accuracy drop exceeded 1 percentage point')
    if wall_gain < .05: reasons.append('mean wall-latency improvement below 5%')
    if cpu_gain < .02: reasons.append('process CPU-time improvement below exploratory 2% floor')
    if c_p95 > b_p95: reasons.append('p95 latency regression')
    if len(per_round) != DEV_ROUNDS or any(
       not all(math.isfinite(v) and v > 0 for v in (r['baseline_wall_ms'],r['candidate_wall_ms'],r['baseline_cpu_ms'],r['candidate_cpu_ms']))
       or r['candidate_wall_ms'] >= r['baseline_wall_ms']
       or r['candidate_cpu_ms'] >= r['baseline_cpu_ms']
       for r in per_round):
        reasons.append('not faster and less CPU in every development round')
    return {'eligible_on_development': not reasons,
            'wall_improvement_fraction': round(wall_gain, 5),
            'cpu_time_improvement_fraction': round(cpu_gain, 5),
            'accuracy_drop_fraction': round(baseline_accuracy-candidate_accuracy, 5),
            'prediction_agreement_fraction': round(agreement, 5),
            'rejection_reasons': reasons}


def choose_candidate(candidates):
    """Never choose a failing or missing candidate; minimize measured CPU first."""
    passing = [x for x in candidates if x.get('gate', {}).get('eligible_on_development') is True]
    if not passing:
        return None
    return sorted(passing, key=lambda x: (
        -x['gate']['cpu_time_improvement_fraction'],
        -x['gate']['wall_improvement_fraction'], x['candidate_threads']))[0]


def _timed(call):
    started=time.perf_counter()
    cpu_started=time.process_time()
    prediction=call()
    return (prediction, max(0, (time.perf_counter()-started)*1000),
            max(0, (time.process_time()-cpu_started)*1000))


def development_sweep(review, *, counts=CANDIDATE_COUNTS):
    """Real paired dev measurements; only one ORT session resident at a time."""
    import gc
    import numpy as np
    import torch
    import onnxruntime as ort
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if not counts or any(x not in range(1,9) for x in counts) or len(set(counts))!=len(counts):
        raise ValueError('Candidate thread counts must be distinct integers 1..8')
    torch.set_num_threads(BASELINE_THREADS)
    tok=AutoTokenizer.from_pretrained(str(review['folder']), local_files_only=True, trust_remote_code=False)
    model=AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, revision=review['revision'], trust_remote_code=False).eval().cpu()
    rows, origin=evaluation_set(TrialRequest(phase='development',sample_count=DEV_SAMPLES,seed=20261031))
    if len(rows)!=DEV_SAMPLES: raise ValueError('Development partition mismatch')
    fingerprint=digest_texts(rows)
    results=[]
    for threads in counts:
        opt=ort.SessionOptions()
        opt.intra_op_num_threads=threads
        opt.inter_op_num_threads=1
        opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess=ort.InferenceSession(str(review['folder']/'model.onnx'),sess_options=opt,
                                  providers=['CPUExecutionProvider'])
        if sess.get_providers()!=['CPUExecutionProvider']:
            raise RuntimeError('Unexpected ONNX execution provider')
        names=[x.name for x in sess.get_inputs()]
        if 'input_ids' not in names: raise ValueError('ONNX graph lacks input_ids')
        prepared=[]
        for txt,label in rows:
            x=tok(txt, max_length=256, truncation=True, return_tensors='np')
            n={k:np.asarray(x[k],dtype=np.int64) for k in names}
            pt={k:torch.from_numpy(v) for k,v in n.items()}
            prepared.append((n,pt,int(label)))
        def baseline(pt):
            with torch.inference_mode():
                return int(torch.argmax(model(**pt).logits,dim=-1).item())
        def candidate(np_inputs):
            return int(np.asarray(sess.run(None,np_inputs)[0]).argmax(axis=-1).item())
        for np_inputs,pt,_ in prepared[:4]:
            baseline(pt);candidate(np_inputs)
        timings={name:{'wall':[],'cpu':[]} for name in ('baseline','candidate')}
        first_predictions={}
        per_round=[]
        for rd in range(DEV_ROUNDS):
            part={name:{'wall':[],'cpu':[]} for name in ('baseline','candidate')}
            for idx,(np_inputs,pt,label) in enumerate(prepared):
                # Alternate request order to avoid systematic first-mover bias.
                order=('baseline','candidate') if (rd+idx)%2==0 else ('candidate','baseline')
                preds={}
                for name in order:
                    out,wall,cpu=_timed(lambda: baseline(pt) if name=='baseline' else candidate(np_inputs))
                    preds[name]=out
                    timings[name]['wall'].append(wall)
                    timings[name]['cpu'].append(cpu)
                    part[name]['wall'].append(wall)
                    part[name]['cpu'].append(cpu)
                if rd==0: first_predictions[idx]=(preds['baseline'],preds['candidate'],label)
                elif (preds['baseline'],preds['candidate']) != first_predictions[idx][:2]:
                    raise RuntimeError('Prediction changed between development rounds')
            per_round.append({'round':rd+1,
                'baseline_wall_ms':statistics.mean(part['baseline']['wall']),
                'candidate_wall_ms':statistics.mean(part['candidate']['wall']),
                'baseline_cpu_ms':statistics.mean(part['baseline']['cpu']),
                'candidate_cpu_ms':statistics.mean(part['candidate']['cpu'])})
        b_acc=sum(x==y for x,_,y in first_predictions.values())/DEV_SAMPLES
        c_acc=sum(x==y for _,x,y in first_predictions.values())/DEV_SAMPLES
        agreement=sum(a==b for a,b,_ in first_predictions.values())/DEV_SAMPLES
        b={name:summarize(samples) for name,samples in timings['baseline'].items()}
        c={name:summarize(samples) for name,samples in timings['candidate'].items()}
        gate=score_candidate(b,c,agreement=agreement,baseline_accuracy=b_acc,
                             candidate_accuracy=c_acc,per_round=per_round)
        results.append({'baseline_threads':BASELINE_THREADS,'candidate_threads':threads,
            'development_samples':DEV_SAMPLES,'rounds':DEV_ROUNDS,
            'baseline':b,'candidate':c,'baseline_accuracy':round(b_acc,5),
            'candidate_accuracy':round(c_acc,5),
            'per_round':[{'round':r['round'],**{k:round(v,4) for k,v in r.items() if k!='round'}}
                         for r in per_round], 'gate':gate})
        print(f"ONNX threads {threads}: wall {100*gate['wall_improvement_fraction']:+.2f}%, "
              f"CPU {100*gate['cpu_time_improvement_fraction']:+.2f}%, "
              f"dev gate {'PASS' if gate['eligible_on_development'] else 'FAIL'}",flush=True)
        del sess
        gc.collect()
    return {'model_id':MODEL_ID,'model_revision':review['revision'],
        'source_review_run_id':review['run_id'], 'source_model_sha256':review['sha256'],
        'development_partition':origin,'development_text_digest':fingerprint,
        'development_samples':DEV_SAMPLES,'development_rounds':DEV_ROUNDS,
        'hardware':{'system':platform.system(),'machine':platform.machine()},
        'timing_scope':'pretokenized warm forward; process CPU seconds aggregated across all process threads; single process, alternating pairs',
        'results':results}


def run_development(run_id):
    review=_review(run_id)
    print('=== NIMBO DEVELOPMENT CPU-THREAD TUNING ===')
    print('Source verified staged run:',run_id)
    report=development_sweep(review)
    chosen=choose_candidate(report['results'])
    report.update({'created_utc':datetime.now(timezone.utc).isoformat(),
      'status':'SELECTED_DEV_ONLY' if chosen else 'NO_ELIGIBLE_CANDIDATE',
      'selected_candidate_threads':chosen['candidate_threads'] if chosen else None,
      'selected_dev_cpu_reduction_fraction':chosen['gate']['cpu_time_improvement_fraction'] if chosen else None,
      'heldout_evaluated_for_this_selection':False,
      'production_deployed':False,'energy_joules':None,'flops':None,'cloud_cost_usd':None,
      'caveat':'Repeated historical heldout runs exist; holdout is disjoint from tuning rows but not virgin external test; results vary with load. No auto deployment.'})
    OUTPUT.mkdir(parents=True,exist_ok=True)
    target=OUTPUT/'latest_development_selection.json'
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('Decision:',report['status'])
    if chosen:print('Selected ONNX threads:',chosen['candidate_threads'])
    else:print('No candidate met BOTH latency and CPU constraints; keep baseline. Do NOT run heldout tuning.')
    print('Saved:',target)
    return report


def eligible_heldout_config(saved, review):
    if saved.get('status')!='SELECTED_DEV_ONLY':
        raise ValueError('No passing development selection. Heldout run refused.')
    if saved.get('heldout_evaluated_for_this_selection') is not False:
        raise ValueError('This selection already marked as evaluated; no repeated optimization using heldout')
    if saved.get('model_id')!=MODEL_ID or saved.get('model_revision')!=review['revision']:
        raise ValueError('Model/revision mismatch')
    if saved.get('source_model_sha256')!=review['sha256'] or saved.get('source_review_run_id')!=review['run_id']:
        raise ValueError('Staged artifact lineage mismatch')
    if saved.get('development_samples')!=DEV_SAMPLES or saved.get('development_rounds')!=DEV_ROUNDS:
        raise ValueError('Invalid development partition metadata')
    selected=saved.get('selected_candidate_threads')
    matching=[x for x in saved.get('results',[]) if x.get('candidate_threads')==selected
              and x.get('gate',{}).get('eligible_on_development') is True]
    if len(matching)!=1 or selected not in CANDIDATE_COUNTS:
        raise ValueError('No valid selected candidate in dev record')
    if choose_candidate(saved['results'])!=matching[0]:
        raise ValueError('Selection does not match deterministic rule')
    return selected


def run_heldout(selection_file):
    saved=json.loads(Path(selection_file).read_text(encoding='utf8'))
    review=_review(saved['source_review_run_id'])
    threads=eligible_heldout_config(saved, review)
    # Lock selection before heldout to avoid inadvertently picking again based on scores.
    saved['heldout_evaluated_for_this_selection']=True
    saved['heldout_started_utc']=datetime.now(timezone.utc).isoformat()
    Path(selection_file).write_text(json.dumps(saved,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    req=ValidationRequest(optimization_objective='cpu_time',threads=BASELINE_THREADS,
                          candidate_threads=threads,rounds=3,sample_count=800,
                          max_cpu_increase_fraction=0,export_variant=True)
    print(f'=== LOCKED HELDOUT CHECK: PyTorch {BASELINE_THREADS} threads / ONNX {threads} threads ===')
    print('No tuning on heldout; candidate chosen on 200 development examples.')
    for packet in validate_onnx(req):
        if packet['type']=='event': print(packet['stage'], '|', packet['message'],flush=True)
        elif packet['type']=='result':
            report=packet['report']
            gate=report['optimization']['gate']
            print('HELDOUT DECISION:',report['optimization']['approval'])
            print('QUALITY PASS:',gate['quality_pass'], 'WALL PASS:',gate['latency_pass'],
                  'CPU PASS:',gate['cpu_pass'])
            print('WALL REDUCTION:',round(100*gate['mean_speedup_fraction'],2),'%')
            print('CPU-TIME REDUCTION:',None if gate['cpu_time_reduction_fraction'] is None
                  else round(100*gate['cpu_time_reduction_fraction'],2),'%')
            print('Run evidence: results/independent_validation_v21.jsonl')
    return None


def main(argv=None):
    p=argparse.ArgumentParser(description='CPU thread tuning: development first, locked heldout second')
    p.add_argument('--phase',choices=('development','heldout'),default='development')
    p.add_argument('--run-id',default=None,help='Previously verified, staged ONNX run ID')
    p.add_argument('--selection',type=Path,default=OUTPUT/'latest_development_selection.json')
    a=p.parse_args(argv)
    if a.phase=='development':
        run_development(a.run_id or latest_exported_run())
    else:
        run_heldout(a.selection)

if __name__=='__main__':
    main()
