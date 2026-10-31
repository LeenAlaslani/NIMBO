"""NIMBO Step 22: preregistered cross-domain sentiment robustness trial.

Fixed TweetEval sentiment TEST source, binary positive/negative selection,
800 distinct texts, 3 alternating warm CPU rounds, no tuning on this data.
Locked source model + staged ONNX graph, PyTorch 4 CPU threads / ONNX 2.
Exactly the staged source run is used; NEVER deploys or changes NIMBO.

This tests a NEW evaluation DOMAIN for this NIMBO optimization policy, not
an external certification or a proof that pretrained weights never saw data.
Neutral tweets are excluded because the source IMDb classifier is binary.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import random
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATASET_ID = 'cardiffnlp/tweet_eval'
DATASET_CONFIG = 'sentiment'
DATASET_SPLIT = 'test'
# The older 55864a... revision contains tweet_eval.py, which datasets>=4 refuses.
# Pin the official, converted Parquet snapshot instead; test split and label policy unchanged.
LEGACY_SCRIPT_REVISION = '55864a3201f347e9edfac045e7f772eb2c0ee801'
DATASET_REVISION = 'b0f6bf8397168a983b87508f0dfd24094eb4e360'
DATASET_PARQUET_PATH = 'sentiment/test/0000.parquet'
MODEL_ID = 'lvwerra/distilbert-imdb'
SOURCE_RUN = 'heldout_7830c2ff573a47ae'
SEED = 20261031
PER_CLASS = 400
N_ROUNDS = 3
PT_THREADS = 4
ORT_THREADS = 2
MAX_LENGTH = 256
OUTPUT = Path('results/external_domain_v1/tweet_eval_binary_test_800.json')

def load_pinned_tweet_eval_test(*, downloader=None, parquet_loader=None):
    """Read official pinned TEST Parquet, never execute a legacy dataset script.

    The two optional adapters permit deterministic offline contract tests. Returns
    Dataset + SHA-256 of downloaded source bytes, not a sampled-text digest.
    """
    if downloader is None:
        from huggingface_hub import hf_hub_download
        downloader = hf_hub_download
    if parquet_loader is None:
        from datasets import load_dataset
        parquet_loader = load_dataset
    filename = downloader(repo_id=DATASET_ID, repo_type='dataset',
                          revision=DATASET_REVISION, filename=DATASET_PARQUET_PATH)
    source = Path(filename)
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError('Pinned TweetEval TEST Parquet download is missing or empty')
    sha = hashlib.sha256()
    with source.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            sha.update(chunk)
    ds = parquet_loader('parquet', data_files={DATASET_SPLIT: str(source)},
                        split=DATASET_SPLIT)
    if len(ds) < 2*PER_CLASS:
        raise RuntimeError('TweetEval TEST Parquet has fewer rows than required')
    cols = getattr(ds, 'column_names', None)
    if cols is not None and not {'text', 'label'} <= set(cols):
        raise RuntimeError('Pinned TEST file does not contain text and label')
    return ds, sha.hexdigest()

def pick_preregistered(rows, *, seed=SEED, per_class=PER_CLASS):
    """0=NEGATIVE, 2=POSITIVE, skip neutral 1; class-balanced by design."""
    if not isinstance(per_class,int) or per_class < 1:
        raise ValueError('Sample size must be a positive integer')
    buckets={0:[],2:[]}
    for i, rec in enumerate(rows):
        label=rec.get('label')
        text=rec.get('text')
        if label == 1:continue
        if label not in buckets or not isinstance(text,str) or not text.strip():
            raise ValueError('Unexpected class or empty TweetEval text')
        buckets[label].append((i,text,label//2))
    if any(len(items)<per_class for items in buckets.values()):
        raise ValueError('Insufficient examples in positive/negative classes')
    rng=random.Random(seed)
    selected=[]
    for label in (0,2):
        indices=list(buckets[label]);rng.shuffle(indices)
        selected.extend(indices[:per_class])
    rng.shuffle(selected)
    if len({i for i,_,_ in selected}) != 2*per_class:
        raise ValueError('Duplicate dataset positions')
    return selected

def row_digest(selected):
    # Canonical digest binds both text and gold label; do not publish raw rows.
    serial=json.dumps([[idx,txt,label] for idx,txt,label in selected],ensure_ascii=False,separators=(',',':'))
    return hashlib.sha256(serial.encode('utf8')).hexdigest()

def summary(values):
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Timing samples unavailable/nonfinite')
    v=sorted(values)
    idx=min(len(v)-1,math.ceil(.95*len(v))-1)
    return {'mean_ms':round(statistics.mean(v),4),'p95_ms':round(v[idx],4),'count':len(v)}

def reduction(base,candidate):
    if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (base,candidate)) or base<=0 or candidate<0:
        raise ValueError('Invalid timing summary')
    return (base-candidate)/base

def assess_cross_domain(baseline,candidate,*,agreement,rounds):
    """Predeclared robustness gates; report failures, never retune model/threads."""
    a=baseline['accuracy'];b=candidate['accuracy']
    w=reduction(baseline['wall']['mean_ms'],candidate['wall']['mean_ms'])
    c=reduction(baseline['cpu']['mean_ms'],candidate['cpu']['mean_ms'])
    p=candidate['wall']['p95_ms'] <= baseline['wall']['p95_ms']
    parity=agreement >= .995 and b >= a-.01
    keys=('baseline_wall_mean_ms','candidate_wall_mean_ms','baseline_cpu_mean_ms','candidate_cpu_mean_ms')
    valid_rounds=(len(rounds)==N_ROUNDS and all(
        isinstance(r,dict) and all(isinstance(r.get(k),(int,float)) and
        math.isfinite(r[k]) and r[k]>0 for k in keys) for r in rounds))
    wall_ok=bool(valid_rounds and w>=.05 and p and all(
        x['candidate_wall_mean_ms']<x['baseline_wall_mean_ms'] for x in rounds))
    cpu_ok=bool(valid_rounds and c>=.02 and all(
        x['candidate_cpu_mean_ms']<x['baseline_cpu_mean_ms'] for x in rounds))
    # An absolute floor is diagnostic; the source model was trained for film reviews.
    domain_floor=(a>=.75 and b>=.75)
    return {'prediction_parity_pass':parity,'latency_gate_pass':wall_ok,'cpu_gate_pass':cpu_ok,
            'absolute_domain_accuracy_75pct_diagnostic':domain_floor,
            'mean_wall_reduction_fraction':round(w,5),'mean_cpu_reduction_fraction':round(c,5),
            'p95_pass':p, 'all_cross_domain_review_gates_pass':bool(parity and wall_ok and cpu_ok and domain_floor),
            'interpretation':'REVIEW_ONLY_CROSS_DOMAIN_PASS' if parity and wall_ok and cpu_ok and domain_floor
            else 'CROSS_DOMAIN_LIMITATION_RECORDED; retain baseline or require workload-specific validation'}

def _peek_memory():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(value/(1024*1024 if platform.system()=='Darwin' else 1024),2)

def verify_review(review):
    manifest=review['manifest'];rec=review['validation']
    if review['run_id']!=SOURCE_RUN or manifest.get('candidate_cpu_threads') != ORT_THREADS:
        raise RuntimeError('Candidate does not match locked two-thread selection')
    if manifest.get('cpu_threads') != PT_THREADS:
        raise RuntimeError('Baseline CPU thread count changed')
    if manifest.get('max_length') != MAX_LENGTH:
        raise RuntimeError('Input truncation policy changed')
    gate=rec['optimization']['gate']
    if gate.get('optimization_objective')!='cpu_time' or gate.get('eligible') is not True or gate.get('cpu_pass') is not True:
        raise RuntimeError('Selected source was not an eligible CPU objective trial')
    if rec.get('dataset_fingerprint') is None or len(review['sha256']) != 64:
        raise RuntimeError('Source lineage missing')
    return True

def run(*, output=OUTPUT):
    if output.exists():
        raise RuntimeError('External test already completed. No repeat as a "fresh" holdout: '+str(output))
    from app.nimbo_adoption_v21_1 import _review
    review=_review(SOURCE_RUN)  # verifies manifest and ONNX SHA before any fresh data download
    verify_review(review)
    import numpy as np
    import torch
    import onnxruntime as ort
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print('EXTERNAL DATASET: TweetEval sentiment TEST, fixed revision', flush=True)
    print('LABELS: 400 negative + 400 positive; neutral excluded by fixed protocol',flush=True)
    print('LOCKED SETTINGS: PyTorch 4 threads; ONNX 2 threads; 3 paired rounds; no retuning',flush=True)
    ds, parquet_digest = load_pinned_tweet_eval_test()
    print('PARQUET PIN:', DATASET_REVISION, DATASET_PARQUET_PATH, flush=True)
    print('PARQUET SHA256:', parquet_digest, flush=True)
    selected=pick_preregistered(ds)
    fingerprint=row_digest(selected)
    torch.set_num_threads(PT_THREADS)
    tok=AutoTokenizer.from_pretrained(str(review['folder']),local_files_only=True,trust_remote_code=False)
    model=AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,revision=review['revision'],trust_remote_code=False).eval().cpu()
    id2label={int(k):str(v).upper() for k,v in model.config.id2label.items()}
    if len(id2label)!=2 or id2label.get(0) not in ('NEGATIVE','LABEL_0') or id2label.get(1) not in ('POSITIVE','LABEL_1'):
        raise RuntimeError('Binary model label order not established; refusing evaluation')
    opt=ort.SessionOptions();opt.intra_op_num_threads=ORT_THREADS;opt.inter_op_num_threads=1
    opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session=ort.InferenceSession(str(review['folder']/'model.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
    if session.get_providers()!=['CPUExecutionProvider']:
        raise RuntimeError('Unexpected ONNX provider')
    input_names=[x.name for x in session.get_inputs()]
    if 'input_ids' not in input_names:raise RuntimeError('ONNX schema missing input_ids')
    prepared=[]
    for _,text,label in selected:
        x=tok(text,max_length=MAX_LENGTH,truncation=True,return_tensors='np')
        np_input={k:np.asarray(x[k],dtype=np.int64) for k in input_names}
        pt_input={k:torch.from_numpy(v) for k,v in np_input.items()}
        prepared.append((np_input,pt_input,label))
    def infer(inputs,which):
        if which=='baseline':
            with torch.inference_mode():
                return int(torch.argmax(model(**inputs).logits,dim=-1).item())
        return int(np.asarray(session.run(None,inputs)[0]).argmax(axis=-1).item())
    for n,p,_ in prepared[:4]:infer(p,'baseline');infer(n,'candidate')
    observations={which:{'wall':[],'cpu':[]} for which in ('baseline','candidate')}
    per_round=[];first={}
    for rd in range(N_ROUNDS):
        part={which:{'wall':[],'cpu':[]} for which in ('baseline','candidate')}
        for i,(n,p,y) in enumerate(prepared):
            predictions={}
            order=('baseline','candidate') if (i+rd)%2==0 else ('candidate','baseline')
            for which in order:
                inputs=p if which=='baseline' else n
                start_wall=time.perf_counter_ns();start_cpu=time.process_time_ns()
                pred=infer(inputs,which)
                cpu=(time.process_time_ns()-start_cpu)/1e6
                wall=(time.perf_counter_ns()-start_wall)/1e6
                predictions[which]=pred
                part[which]['cpu'].append(cpu);part[which]['wall'].append(wall)
                observations[which]['cpu'].append(cpu);observations[which]['wall'].append(wall)
            if rd==0:first[i]=(predictions['baseline'],predictions['candidate'],y)
            elif first[i][:2]!=(predictions['baseline'],predictions['candidate']):
                raise RuntimeError('Cross-round prediction instability')
        row={'round':rd+1,
             'baseline_wall_mean_ms':round(statistics.mean(part['baseline']['wall']),4),
             'candidate_wall_mean_ms':round(statistics.mean(part['candidate']['wall']),4),
             'baseline_cpu_mean_ms':round(statistics.mean(part['baseline']['cpu']),4),
             'candidate_cpu_mean_ms':round(statistics.mean(part['candidate']['cpu']),4)}
        per_round.append(row)
        print(f"ROUND {rd+1}/3: wall {row['baseline_wall_mean_ms']:.2f} -> {row['candidate_wall_mean_ms']:.2f} ms; "
              f"CPU {row['baseline_cpu_mean_ms']:.2f} -> {row['candidate_cpu_mean_ms']:.2f} ms",flush=True)
    per_path={}
    for which,position in (('baseline',0),('candidate',1)):
        accuracy=sum(int(vals[position]==vals[2]) for vals in first.values())/len(prepared)
        recall={}
        for label,name in ((0,'negative'),(1,'positive')):
            targets=[vals for vals in first.values() if vals[2]==label]
            recall[name]=round(sum(int(v[position]==label) for v in targets)/len(targets),5)
        per_path[which]={'accuracy':round(accuracy,5),'class_recall':recall,
                         'wall':summary(observations[which]['wall']),
                         'cpu':summary(observations[which]['cpu'])}
    agreement=sum(int(a==b) for a,b,_ in first.values())/len(prepared)
    decision=assess_cross_domain(per_path['baseline'],per_path['candidate'],agreement=agreement,rounds=per_round)
    record={'schema_version':'nimbo.external_domain.v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'source_run_id':SOURCE_RUN,'source_onnx_sha256':review['sha256'],'source_model_revision':review['revision'],
            'dataset':{'id':DATASET_ID,'config':DATASET_CONFIG,'split':DATASET_SPLIT,'revision':DATASET_REVISION,'parquet_path':DATASET_PARQUET_PATH,'parquet_sha256':parquet_digest,'previous_loader_script_revision':LEGACY_SCRIPT_REVISION,
                       'selection_seed':SEED,'selection_digest':fingerprint,'count':len(prepared),
                       'negative_count':PER_CLASS,'positive_count':PER_CLASS,'neutral_included':False,
                       'note':'Binary conditional subset of 3-way TweetEval; not 3-class accuracy'},
            'setup':{'baseline':'PyTorch FP32','baseline_threads':PT_THREADS,'candidate':'ONNX FP32',
                     'candidate_threads':ORT_THREADS,'rounds':N_ROUNDS,'max_length':MAX_LENGTH,
                     'provider':'CPUExecutionProvider','hardware':platform.platform()},
            'metrics':per_path,'prediction_agreement_fraction':round(agreement,5),
            'per_round':per_round,'decision':decision,'shared_process_peak_rss_mb':_peek_memory(),
            'deployment':False,'quality_audited_externally':False,'energy_joules':None,'flops':None,
            'cloud_cost_usd':None,'limits':[
             'New task domain for this NIMBO execution-policy test; no proof pretrained model never encountered source data',
             'TweetEval neutral items excluded by precommitted task mapping; accuracy not general sentiment performance',
             'Candidate frozen before evaluating TweetEval; not for tuning threads or model selection',
             'Process CPU time includes all process threads and may vary with background load; 2% floor is not statistical certainty',
             'Memory is shared-process peak, not memory saving between paths',
             'Neither HTTP overhead nor energy nor money measured; not production approved']}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('=== NIMBO NEW-DOMAIN EVALUATION ===')
    print('DATASET: TweetEval sentiment TEST (negative/positive only; 800 unique rows)')
    print('ACCURACY:',f"{per_path['baseline']['accuracy']*100:.2f}% -> {per_path['candidate']['accuracy']*100:.2f}%")
    print('PREDICTION AGREEMENT:',f'{agreement*100:.2f}%')
    print('LATENCY CHANGE:',f"{decision['mean_wall_reduction_fraction']*100:+.2f}% reduction")
    print('CPU-TIME CHANGE:',f"{decision['mean_cpu_reduction_fraction']*100:+.2f}% reduction")
    print('DOMAIN ACCURACY FLOOR (75%):', 'PASS' if decision['absolute_domain_accuracy_75pct_diagnostic'] else 'FAIL')
    print('REVIEW:',decision['interpretation'])
    print('REPORT:',output)
    print('Not deployed. Do not retune using this test set.')
    return record

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    args=parser.parse_args(argv)
    run()

if __name__=='__main__': main()
