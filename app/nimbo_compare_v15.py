"""Measured, local classification comparison. No inferred GPU FLOPs or financial claims."""
from __future__ import annotations
import json, os, re, statistics, time
from datetime import datetime, timezone
from pathlib import Path
from functools import lru_cache
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix='/nimbo/bench', tags=['model-comparison'])
SMALL = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
LARGE = 'textattack/bert-base-uncased-SST-2'

def allowed_models():
    base=[SMALL,LARGE]
    extra=os.environ.get('NIMBO_EXTRA_CLASSIFIER_MODEL','').strip()
    if extra and re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',extra) and extra not in base:
        base.append(extra)
    return base

class Example(BaseModel):
    text:str=Field(min_length=1,max_length=2000)
    expected_label:Optional[str]=None

class CompareRequest(BaseModel):
    baseline_model:str=LARGE
    candidate_model:str=SMALL
    examples:list[Example]=Field(min_length=1,max_length=30)

@router.get('/catalog')
def catalog():
    return {'models':[{'id':m,'source':'local configured' if i<2 else 'admin-approved external model',
                        'ready_on_first_load':True} for i,m in enumerate(allowed_models())],
            'workload':'classification','max_examples':30,
            'notes':'One local model pair per evaluation; no arbitrary URLs or credentials accepted.'}

@lru_cache(maxsize=4)
def load_pipeline(model_id:str):
    from nimbo.text_backend import load_classifier
    return load_classifier(model_id,device=-1,sst2_labels=True)

def compare_records(task:CompareRequest, get_model=None, timer=None):
    if get_model is None: get_model=load_pipeline
    if timer is None: timer=time.perf_counter
    valid=set(allowed_models())
    if task.baseline_model not in valid or task.candidate_model not in valid:
        raise ValueError('Model must be registered by the backend administrator.')
    if task.baseline_model==task.candidate_model:
        raise ValueError('Baseline and candidate must be different models.')
    if any(x.expected_label is not None and x.expected_label.upper() not in {'POSITIVE','NEGATIVE'} for x in task.examples):
        raise ValueError('Expected labels must be POSITIVE, NEGATIVE or empty.')
    began=timer()
    def evt(stage,message,**other):
        return {'type':'event','stage':stage,'message':message,
                'at_ms':round((timer()-began)*1000,2),**other}
    yield evt('analyze',f'{len(task.examples)} evaluation input(s) received; content not persisted')
    yield evt('route','Comparing registered candidate and baseline on identical inputs',
              candidate=task.candidate_model,baseline=task.baseline_model)
    # Loading and warmup are excluded from the paired inference timings, but included
    # in the request end-to-end measurement and reported explicitly.
    yield evt('execute','Loading baseline and candidate models and performing warmup')
    load_start=timer()
    base=get_model(task.baseline_model)
    cand=get_model(task.candidate_model)
    warmup_text='A normal short text for warmup.'
    base(warmup_text); cand(warmup_text)
    load_warmup_s=timer()-load_start
    times={'baseline':[],'candidate':[]}
    all_labels=all(x.expected_label is not None for x in task.examples)
    correct={'baseline':0,'candidate':0}
    predictions=[]
    for i,x in enumerate(task.examples):
        # Alternate execution order to temper a systematic first-run ordering bias.
        calls=[('baseline',base),('candidate',cand)] if i%2==0 else [('candidate',cand),('baseline',base)]
        outcomes={}
        for name,fn in calls:
            t=timer()
            output=fn(x.text)
            measured=(timer()-t)*1000
            if not isinstance(output,dict) or 'label' not in output:
                raise ValueError(f'{name} model must return a label dictionary')
            label=str(output['label']).upper()
            if label not in {'POSITIVE','NEGATIVE'}:
                raise ValueError('This evaluation requires an SST-2-compatible positive/negative model.')
            times[name].append(measured)
            outcomes[name]=label
            if x.expected_label is not None and label==x.expected_label.upper():correct[name]+=1
        predictions.append({'index':i,'baseline_prediction':outcomes['baseline'],
                            'candidate_prediction':outcomes['candidate'],
                            'expected_label':x.expected_label.upper() if x.expected_label else None})
        yield evt('execute',f'Evaluated input {i+1}/{len(task.examples)}',
                  completed=i+1,total=len(task.examples))
    def summarize(name):
        xs=times[name]
        return {'mean_inference_ms':round(statistics.mean(xs),3),
                'median_inference_ms':round(statistics.median(xs),3),
                'inference_total_ms':round(sum(xs),3),
                'accuracy_on_supplied_labels':round(correct[name]/len(xs),4) if all_labels else None,
                'successes':correct[name] if all_labels else None,
                'inference_seconds_per_labeled_success':round(sum(xs)/(1000*correct[name]),5) if all_labels and correct[name] else None}
    b=summarize('baseline');c=summarize('candidate')
    speed_ratio=(b['mean_inference_ms']-c['mean_inference_ms'])/b['mean_inference_ms'] if b['mean_inference_ms'] else None
    yield evt('verify','Supplied labels compared' if all_labels else 'No complete independent labels provided; accuracy not evaluated')
    result={
        'workload':'classification','route':'paired_baseline_candidate',
        'result':{'label':predictions[0]['candidate_prediction'] if len(predictions)==1 else f'{len(predictions)} inputs evaluated'},
        'metrics':{'end_to_end_ms':round((timer()-began)*1000,2),
                   'load_warmup_ms':round(load_warmup_s*1000,2),
                   'baseline_mean_inference_ms':b['mean_inference_ms'],
                   'candidate_mean_inference_ms':c['mean_inference_ms']},
        'comparison':{'baseline_model':task.baseline_model,'candidate_model':task.candidate_model,
                      'sample_count':len(task.examples),'baseline':b,'candidate':c,
                      'mean_latency_reduction_fraction':round(speed_ratio,4) if speed_ratio is not None else None,
                      'all_labels_present':all_labels,
                      'predictions':predictions,
                      'criterion':'warm CPU inference latency, not energy, FLOPs or dollars',
                      'scope':'local paired development evaluation; external labels not independently audited'},
        'guardrail':{'action':'LABELS_SUPPLIED_UNAUDITED' if all_labels else 'UNVERIFIED',
                     'reason':'Provided labels are not a certified quality audit.' if all_labels else 'No complete reference labels supplied.'},
        'measurement_note':'Warm local CPU timings; model load and warmup excluded from pair metrics and separately reported. Not a hardware-compute or cost estimate.'
    }
    # Persist comparison metadata only, never the uploaded prompts or text.
    record={
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'workload':'classification','evidence_class':'local_paired_cpu_benchmark',
        'baseline_model':task.baseline_model,'candidate_model':task.candidate_model,
        'sample_count':len(task.examples),
        'baseline_mean_inference_ms':b['mean_inference_ms'],
        'candidate_mean_inference_ms':c['mean_inference_ms'],
        'candidate_latency_reduction_fraction':result['comparison']['mean_latency_reduction_fraction'],
        'quality_labels_supplied':all_labels,
        'baseline_accuracy_supplied_labels':b['accuracy_on_supplied_labels'],
        'candidate_accuracy_supplied_labels':c['accuracy_on_supplied_labels'],
        'note':'Warm CPU inference timings; uploaded content not stored; no FLOPs, power or financial cost measured.'
    }
    try:
        path=Path('results/baseline_comparisons_v15.jsonl')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a',encoding='utf8') as output:
            output.write(json.dumps(record,ensure_ascii=False)+'\n')
        result['comparison']['evidence_recorded']=True
    except OSError:
        result['comparison']['evidence_recorded']=False
    yield evt('deliver','Paired measurements ready; quality and compute claims restricted to measured evidence')
    yield {'type':'result','report':result}

@router.post('/compare')
def compare(payload:CompareRequest):
    if payload.baseline_model not in allowed_models() or payload.candidate_model not in allowed_models() or payload.baseline_model==payload.candidate_model:
        raise HTTPException(422,'Select two distinct backend-registered classifiers.')
    if any(x.expected_label is not None and x.expected_label.upper() not in {'POSITIVE','NEGATIVE'} for x in payload.examples):
        raise HTTPException(422,'Expected labels must be POSITIVE, NEGATIVE or omitted.')
    def stream():
        try:
            for packet in compare_records(payload):yield json.dumps(packet,ensure_ascii=False)+'\n'
        except Exception as ex:
            yield json.dumps({'type':'error','message':f'{type(ex).__name__}: {ex}'},ensure_ascii=False)+'\n'
    return StreamingResponse(stream(),media_type='application/x-ndjson',headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})

@router.get('/recent')
def recent(limit:int=20):
    limit=max(1,min(limit,50))
    path=Path('results/baseline_comparisons_v15.jsonl')
    if not path.is_file():return {'records':[],'evidence_class':'local_paired_cpu_benchmark'}
    try:
        lines=path.read_text(encoding='utf8').splitlines()[-limit:]
        return {'records':[json.loads(x) for x in lines if x.strip()],
                'evidence_class':'local_paired_cpu_benchmark'}
    except (OSError,ValueError):
        return {'records':[],'evidence_class':'local_paired_cpu_benchmark','warning':'Evidence log could not be read.'}
