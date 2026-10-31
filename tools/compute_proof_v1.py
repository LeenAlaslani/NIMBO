"""Summarize only actually measured CPU/resource data in a v21 heldout record."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

LOG=Path('results/independent_validation_v21.jsonl')


def summarize(record):
    c=record['comparison']
    r=record.get('resource_evidence') or c.get('resource_evidence')
    if not r or r.get('instrumentation_version')!='process_cpu_v1':
        raise ValueError('This run predates actual process CPU instrumentation; do not infer resource savings from latency')
    if c.get('sample_count')!=800 or c.get('rounds')!=3 or not c.get('all_labels_present'):
        raise ValueError('Expected labeled 800 input / 3 round holdout')
    b,a=c['baseline'],c['candidate']
    cpu_b, cpu_a=b['process_cpu_forward']['mean_ms'],a['process_cpu_forward']['mean_ms']
    if not 0<cpu_b or not 0<=cpu_a: raise ValueError('Invalid measured CPU times')
    reduction=1-cpu_a/cpu_b
    measured=r.get('forward_process_cpu_mean_reduction_fraction')
    if measured is None or abs(reduction-measured)>0.0002:
        raise ValueError('Process CPU reduction inconsistent with recorded values')
    return {'run_id':record['run_id'],'baseline_cpu_ms':cpu_b,'candidate_cpu_ms':cpu_a,
            'process_cpu_reduction_fraction':measured, 'baseline_accuracy':b['accuracy_on_reference_labels'],
            'candidate_accuracy':a['accuracy_on_reference_labels'],
            'peak_process_rss_mb':r['observed_process_peak_rss_mb'],
            'scope':r['cpu_scope'], 'memory_scope':r['memory_scope'],
            'artifact':record['optimization'].get('export'),
            'gate':record['optimization']['gate'],'approval':record['optimization']['approval'],
            'energy_measured':False,'ram_reduction_measured':False,
            'not_proof_of':['FLOPs','energy','cloud bill','per-model memory savings','independent quality certification']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--log',type=Path,default=LOG)
    args=parser.parse_args()
    records=[json.loads(line) for line in args.log.read_text(encoding='utf8').splitlines() if line.strip()]
    if not records: raise ValueError('No heldout runs')
    report=summarize(records[-1])
    dest=Path('results/compute_proof_v1')
    dest.mkdir(parents=True,exist_ok=True)
    target=dest/('resources_'+report['run_id']+'.json')
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('=== NIMBO COMPUTE PROOF ===')
    print('Run:',report['run_id'],'| gate:',report['approval'])
    print(f"Process CPU per warm forward: {report['baseline_cpu_ms']:.3f} ms -> {report['candidate_cpu_ms']:.3f} ms")
    print(f"Measured process CPU time change: {report['process_cpu_reduction_fraction']*100:+.2f}% reduction (negative means more CPU)")
    print('Process peak RSS (shared, NOT savings):',report['peak_process_rss_mb'],'MB')
    print('Exported review artifact:', bool(report['artifact']))
    print('Energy/cost/per-path RAM reduction: NOT measured')
    print('Record:',target)

if __name__=='__main__':main()
