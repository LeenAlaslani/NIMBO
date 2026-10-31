"""Deterministic integration-contract demonstrations, not a quality benchmark."""
import json
from pathlib import Path
from dataclasses import asdict
from nimbo.guardrails_core import Policy, Observation, Evidence, decide

def main():
    rows = [
        ('text_validated', Policy('text_generation',5), Observation(Evidence.PASS,.5,1,.96,'demo_calibrated',True)),
        ('text_unknown', Policy('text_generation',5), Observation(Evidence.PASS,.5,1,.96,'unvalidated_rule',False)),
        ('image_escalate', Policy('image_generation',8), Observation(Evidence.FAIL,1,1,escalation_available=True)),
        ('classification_budget', Policy('text_classification',2), Observation(Evidence.UNCERTAIN,1.99,1)),
        ('image_no_certification', Policy('image_generation',8), Observation(Evidence.PASS,1,2,.94,'clip_development_only',False,already_escalated=True)),
    ]
    report = {'mode':'synthetic_contract_demo_NOT_empirical_performance','items':[]}
    for label,p,o in rows:
        d = decide(p,o)
        report['items'].append({'case':label,'task_type':p.task_type,**asdict(d),'action':d.action.value})
        print(label,'->',d.action.value,'|',d.reason)
    out = Path('results/guardrails_core_v3_demo.json'); out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print('Saved',out)
if __name__=='__main__':main()
