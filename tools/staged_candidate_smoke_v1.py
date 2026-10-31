"""Independent-process smoke recheck of a *staged* NIMBO ONNX artifact.

Checks the existing v21.1 adoption gate, SHA-256, the pinned model revision,
and live paired predictions on a small fixed diagnostic set. NEVER deploys.
The diagnostics are NOT a new independent labeled holdout evaluation.
"""
from __future__ import annotations
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys
# Allow both `python -m tools.<name>` and direct `python tools/<name>.py`
# from the repository root; direct execution otherwise omits the repo from sys.path.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.nimbo_adoption_v21_1 import _review, _prepare_cache, _parity_probe, VALIDATIONS

PROMPTS = (
    'I really enjoyed this film and would watch it again.',
    'Terrible acting, slow story, and an awful ending.',
    'A thoughtful performance with convincing dialogue.',
    'The film is a waste of time and money.',
    'Solid performances, though the plot has some weak spots.',
    'Nothing about this story works, despite the strong cast.',
    'What a delightful, emotional and beautifully shot film.',
    'The pacing was dreadful and the characters were flat.',
)


def latest_exported_run(path=VALIDATIONS):
    if not path.is_file(): raise ValueError('Heldout validation log is unavailable')
    lines=[line for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
    if not lines: raise ValueError('No heldout records found')
    rec=json.loads(lines[-1])
    opt=rec.get('optimization',{})
    if opt.get('approval')=='eligible_for_review' and (opt.get('export') or {}).get('sha256'):
        return rec['run_id']
    raise ValueError('Latest heldout run did not stage an eligible ONNX candidate. Do not fall back to an older run.')


def inspect_results(results, *, probability_delta_limit=.025):
    """Fail closed on missing, mislabeled, drifted, or not-run probes."""
    if not results: raise ValueError('No smoke probes executed')
    if any(not isinstance(x,dict) or x.get('parity_pass') is not True or
           x.get('prediction_agreement') is not True or
           not isinstance(x.get('positive_probability_delta'),(int,float)) or
           not math.isfinite(x['positive_probability_delta']) or
           x['positive_probability_delta'] < 0 or
           x['positive_probability_delta'] > probability_delta_limit
           for x in results):
        raise ValueError('Staged candidate smoke failed: prediction mismatch or score drift')
    return {'tested_prompts':len(results), 'matched_predictions':len(results),
            'max_positive_probability_delta':max(x['positive_probability_delta'] for x in results),
            'parity_threshold':probability_delta_limit,'status':'PASS',
            'scope':'post-export smoke only; no ground-truth labels, not independent quality certification'}


def run_smoke(run_id):
    review=_review(run_id)  # verifies heldout decision, manifest lineage, and exact model SHA-256
    engine=_prepare_cache(review)  # loads staged ONNX + original pinned PyTorch checkpoint
    probes=[]
    for index,prompt in enumerate(PROMPTS,1):
        result=_parity_probe(prompt,review,engine=engine)
        probes.append(result)
        print(f"Prompt {index}/{len(PROMPTS)}: {result['status']}; probability delta={result['positive_probability_delta']}",flush=True)
    summary=inspect_results(probes)
    summary.update({'source_run_id':run_id,'model_sha256':review['sha256'],
                    'model_revision':review['revision'],
                    'timestamp_utc':datetime.now(timezone.utc).isoformat(),
                    'production_deployed':False, 'shadow_or_canary_activated':False})
    destination=Path('results/compute_proof_v1')
    destination.mkdir(parents=True,exist_ok=True)
    out=destination/('staged_smoke_'+run_id+'.json')
    out.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('STAGED SMOKE: PASS (not deployed). Evidence:',out)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',default=None,help='Exact heldout ID; omitted uses latest eligible exported record')
    args=parser.parse_args()
    run_smoke(args.run_id or latest_exported_run())

if __name__=='__main__':
    main()
