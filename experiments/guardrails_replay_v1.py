"""Offline audit of EXISTING Qwen v2.3 outputs. No model run, no new savings claims."""
import argparse,csv,json
from pathlib import Path
from nimbo.quality_latency_guardrails import verify,decision
from experiments.context_v2_diagnostics import CASES
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='results/context_qwen_quality_v23.csv')
    p.add_argument('--output',default='results/guardrails_replay_v1.json')
    args=p.parse_args()
    cases={name:(q,ctx) for name,q,ctx,required in CASES}
    with open(args.csv,newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    output=[]
    for r in rows:
        if r['mode']!='optimized' or r['context_kind']=='abstain':continue
        q,ctx=cases[r['task_id']]
        # For multihop the selected evidence is known explicitly from the case,
        # but for arbitrary questions replaying full context is NOT the same as runtime selected context.
        if r['task_id']=='multihop':
            evidence='Project Atlas is owned by Orion team.\nOrion team approvals go through Maya.'
        else: evidence=ctx
        verdict=verify(q,evidence,r['answer'])
        output.append(dict(task=r['task_id'],answer=r['answer'],status=verdict.status,
                           reason=verdict.reason,expected_if_rule=verdict.expected,
                           possible_action=decision(verdict,1,2,2.0,1.0),
                           caveat='Offline diagnostic; no retry performed'))
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps({'mode':'offline_replay','records':output,
              'limitations':['Pattern-based rule only for explicit two-hop approver cases',
                             'UNVERIFIED does not mean incorrect; SUPPORTED_BY_RULE not universal factual proof',
                             'No live retry or latency measurement performed']},ensure_ascii=False,indent=2))
    for item in output:print(item['task'],item['status'],item['reason'])
    print('Saved:',dest)
if __name__=='__main__':main()
