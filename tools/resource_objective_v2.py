#!/usr/bin/env python3
"""Reassess the latest actual Step 11 record without mutating the historic log."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
# Allow both `python -m tools.<name>` and direct `python tools/<name>.py`
# from the repository root; direct execution otherwise omits the repo from sys.path.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nimbo.resource_policy_v2 import assess_record

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--log',type=Path,default=Path('results/independent_validation_v21.jsonl'))
    p.add_argument('--max-cpu-increase',type=float,default=0.0)
    a=p.parse_args()
    if not 0 <= a.max_cpu_increase <= 0.5:
        p.error('max-cpu-increase must be between 0 and 0.5')
    if not a.log.exists():
        raise ValueError('Missing real heldout log, no result can be claimed')
    records=[json.loads(line) for line in a.log.read_text(encoding='utf8').splitlines() if line.strip()]
    if not records:
        raise ValueError('No measured heldout runs')
    latest=records[-1]
    result=assess_record(latest,max_cpu_increase_fraction=a.max_cpu_increase)
    print('=== NIMBO RESOURCE OBJECTIVE AUDIT ===')
    print('Run:', result['run_id'])
    print('Latency review:', 'PASS' if result['latency_review_pass'] else 'FAIL')
    print('CPU-time review:', 'PASS' if result['cpu_time_review_pass'] else 'FAIL')
    print('Latency change: {:+.2f}% reduction'.format(100*result['measured_wall_latency_improvement_fraction']))
    gain=result['measured_cpu_time_improvement_fraction']
    print('CPU-time change:', 'NOT MEASURED' if gain is None else '{:+.2f}% reduction'.format(100*gain))
    print('Recommendation:',result['recommended_scope'])
    print('Reason:',result['cpu_reason'])
    print('Still not measured: energy, FLOPs, billing, per-path RAM')
    out=Path('results/resource_objective_v2')/('audit_'+result['run_id']+'.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('Saved:',out)

if __name__=='__main__':main()
