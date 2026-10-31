"""Validate published cross-domain aggregate fields only; not a model benchmark."""
import json, math
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'docs/CROSS_DOMAIN_PUBLIC_EVIDENCE.json'

def verify(data):
    def req(cond,msg):
        if not cond: raise ValueError(msg)
    req(data.get('schema_version')=='nimbo.public_cross_domain.v1','schema')
    req(data.get('source_run_id')=='heldout_7830c2ff573a47ae','source ID')
    req(data.get('dataset',{}).get('revision')=='b0f6bf8397168a983b87508f0dfd24094eb4e360','dataset revision')
    req(data.get('dataset',{}).get('count')==800 and data['dataset'].get('negative_count')==400 and data['dataset'].get('positive_count')==400 and data['dataset'].get('neutral_included') is False,'test split selection')
    req(data.get('setup',{}).get('baseline_threads')==4 and data['setup'].get('candidate_threads')==2 and data['setup'].get('rounds')==3,'threads/rounds')
    req(data.get('decision')=='REVIEW_ONLY_CROSS_DOMAIN_PASS' and data.get('review_only') is True and data.get('production_deployed') is False,'decision/deployment')
    req(data.get('independent_external_audit') is False,'audit')
    req(all(data.get(k) is None for k in ('energy_joules','flops','cloud_cost_usd')),'unsupported resource claim')
    req(data.get('prediction_agreement_fraction')==1.0 and data.get('baseline_accuracy')==data.get('candidate_accuracy'),'accuracy agreement')
    req(0.75<=data['candidate_accuracy']<=1.0, 'accuracy floor')
    for measurement,b,c,delta in [('wall','baseline_wall_mean_ms','candidate_wall_mean_ms','warm_forward_latency_reduction_fraction'),('cpu','baseline_process_cpu_mean_ms','candidate_process_cpu_mean_ms','process_cpu_time_reduction_fraction')]:
        x,y,z=data.get(b),data.get(c),data.get(delta)
        req(all(isinstance(q,(int,float)) and math.isfinite(q) for q in (x,y,z)) and x>0 and y>0,measurement+' numbers')
        req(abs((x-y)/x-z)<0.0002,measurement+' arithmetic')
    for k in ('parquet_sha256','selection_digest'):
        req(len(str(data['dataset'].get(k,'')))==64,'digest '+k)
    req(len(str(data.get('onnx_artifact_sha256','')))==64,'ONNX digest')
    return True

def main():
    verify(json.loads(PATH.read_text(encoding='utf8')))
    print('PUBLIC CROSS-DOMAIN SUMMARY: CONSISTENT (not a rerun)')

if __name__=='__main__':main()
