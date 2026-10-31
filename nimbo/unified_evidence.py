"""Unified, read-only evidence catalog. Never mixes benchmark, runtime and oracle statistics."""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter

FILES={
 'image_quality':'quality_evaluation_v6.json',
 'image_policy':'routing_v7_development.json',
 'image_oracle':'recovery_v8_oracle_bound.json',
 'context_qwen':'context_qwen_quality_v23.json',
 'live_guardrails':'live_guardrails_v2.json',
}
KINDS={
 'image_quality':('image_generation','offline_labeled_benchmark'),
 'image_policy':('image_generation','offline_replay'),
 'image_oracle':('image_generation','oracle_upper_bound'),
 'context_qwen':('text_generation','local_development_benchmark'),
 'live_guardrails':('text_generation','local_development_run'),
}

def read_json(path):
 try:
  return json.loads(path.read_text(encoding='utf-8'))
 except (OSError,ValueError):
  return None

def collect(results='results',limit=100):
 base=Path(results)
 # A runtime JSONL event stores timing + route, not human verification or financial cost.
 path=base/'runtime_evidence_v5.jsonl';runtime=[];invalid=0
 if path.exists():
  with path.open(encoding='utf-8') as f:
   for line in f:
    if not line.strip():continue
    try:
     item=json.loads(line)
     if isinstance(item,dict):runtime.append(item)
     else:invalid+=1
    except ValueError:invalid+=1
 runtime=runtime[-max(1,min(500,int(limit))):]
 records=[]
 for i,r in enumerate(runtime):
  records.append({'id':'runtime:'+str(i),'source':'runtime_evidence_v5.jsonl',
   'evidence_kind':'runtime_measurement','workload':r.get('workload','unknown'),
   'route':r.get('route'), 'latency_ms':r.get('end_to_end_ms'),
   'quality_certified':r.get('quality_certified') is True,
   'decision':r.get('action'), 'timestamp_utc':r.get('timestamp_utc'),
   'success_label':None, 'financial_cost':None})
 for name,filename in FILES.items():
  raw=read_json(base/filename)
  if raw is None:continue
  task,kind=KINDS[name]
  abstract={'id':name,'source':filename,'workload':task,'evidence_kind':kind,
   'quality_certified':False,'timestamp_utc':None,'financial_cost':None}
  if name=='image_quality':
   for criterion in ('core_object','full_prompt'):
    for route,measures in raw.get(criterion,{}).items():
     if isinstance(measures,dict):
      records.append(dict(abstract,id=name+':'+criterion+':'+route,
       criterion=criterion,route=route, sample_count=raw[criterion].get('labeled_samples'),
       successes=measures.get('successes'),success_rate=measures.get('rate'),
       seconds_per_success=measures.get('seconds_per_success')))
  elif name=='image_policy':
   m=raw.get('summary',{})
   records.append(dict(abstract,route='clip_policy',sample_count=m.get('count'),
    successes=m.get('offline_successes'),success_rate=m.get('offline_success_rate'),
    seconds_per_success=m.get('proxy_seconds_per_success')))
  elif name=='image_oracle':
   records.append(dict(abstract,route='perfect_human_oracle',sample_count=raw.get('sample_count'),
    successes=raw.get('successes'),seconds_per_success=raw.get('generation_seconds_per_success')))
  elif name=='context_qwen':
   records.append(dict(abstract,route='context_optimization',sample_count=raw.get('tasks'),
    baseline_input_tokens=raw.get('baseline_input_tokens'),
    optimized_input_tokens=raw.get('optimized_input_tokens_including_retries'),
    optimized_generation_s=raw.get('optimized_generation_s_including_retries'),
    comparison_warning='Includes abstentions; not a quality-controlled speedup estimate'))
  elif name=='live_guardrails':
   records.append(dict(abstract,route='qwen_live_guardrails',sample_count=len(raw.get('tasks',[])),
    attempts=raw.get('total_attempts'),generation_s=raw.get('total_generation_s')))
 counts=dict(Counter(r.get('workload','unknown') for r in records if r['evidence_kind']=='runtime_measurement'))
 return {'schema_version':'1.0','runtime_summary':{'request_count':len(runtime),'by_workload':counts,
             'invalid_lines':invalid,'no_quality_rate_claim':True},
         'records':records,'limitations':[
          'Runtime events do not contain independently verified task success labels.',
          'Benchmarks, development runs and oracle bounds are different evidence classes: never aggregate their success or timing.',
          'Latency is not GPU FLOPs, power consumption or financial cost.',
          'Images are offline only until a working image gateway is connected.',
          'Inputs and outputs are not saved in this evidence catalog.']}
