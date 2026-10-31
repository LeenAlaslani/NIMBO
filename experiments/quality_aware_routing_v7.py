"""NIMBO v7 — auditable routing simulation and independent evaluation.
Never uses human labels to make a route decision; labels are evaluation-only.
No generated images/models are changed. A post-standard checker is NOT simulated.
"""
import argparse,csv,json,math
from pathlib import Path

def truth(v):
    t=str(v).strip().lower()
    if t in ('true','1','yes','pass'): return True
    if t in ('false','0','no','fail'): return False
    raise ValueError('Invalid quality label: '+repr(v))

def read_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def route(score,threshold):
    return 'escalate' if score<threshold else 'accept_fast'

def evaluate(rows,labels,threshold,checker_mode):
    """checker_mode='none' uses recorded CLIP cost. 'manual_review' intentionally returns pending state."""
    count=success=fail_accept=unnecessary=fallbacks=unresolved=0
    elapsed=0.0
    per=[]
    for r in rows:
        pid=str(r['prompt_id']); label=labels.get(pid)
        if label is None: continue
        a,b=label
        score=float(r['object_relative_score'])
        action=route(score,threshold)
        fast=float(r['fast_s'])
        check=float(r['checker_s'])
        # Reuse timing of the same original model settings, not predictions of new hardware
        standard=float(r['standard_s'])
        if action=='escalate':
            fallbacks+=1
            delivered=b
            unnecessary+=int(a)
            latency=fast+check+standard
            # Standard output has not been runtime-verified, even if ground truth is known offline
            status='NEEDS_POST_STANDARD_VERIFICATION' if checker_mode=='manual_review' else 'DELIVERED_UNVERIFIED'
            unresolved+=int(checker_mode=='manual_review')
        else:
            delivered=a
            fail_accept+=int(not a)
            latency=fast+check
            status='ACCEPT_FAST'
        success+=int(delivered)
        count+=1;elapsed+=latency
        per.append(dict(prompt_id=pid,score=score,action=action,
                        offline_success=delivered,observed_latency_s=round(latency,5),
                        delivery_status=status))
    return dict(count=count,threshold=threshold,offline_successes=success,
                offline_success_rate=round(success/count,5) if count else None,
                false_accepts=fail_accept,unnecessary_fallbacks=unnecessary,
                fallbacks=fallbacks,pending_second_stage_verification=unresolved,
                proxy_total_seconds=round(elapsed,4),
                proxy_seconds_per_success=round(elapsed/success,4) if success else None,
                predictions=per)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--benchmark',default='results/benchmark_v2_raw_labeled.csv')
    ap.add_argument('--labels',default='results/full_prompt_labels_v6.csv')
    ap.add_argument('--threshold',type=float,default=0.65,
                    help='Pre-registered CLIP threshold; do not tune on test set')
    ap.add_argument('--output',default='results/routing_v7_development.json')
    ap.add_argument('--mode',choices=['development','holdout'],default='development')
    ap.add_argument('--post-standard',choices=['none','manual_review'],default='manual_review')
    args=ap.parse_args()
    if not (0<=args.threshold<=1): ap.error('threshold must be in [0,1]')
    rows=read_csv(args.benchmark)
    raw=read_csv(args.labels)
    if len(set(r['prompt_id'] for r in rows))!=len(rows): raise ValueError('Duplicate benchmark ids')
    if len(set(r['prompt_id'] for r in raw))!=len(raw): raise ValueError('Duplicate label ids')
    labels={}
    for r in raw:
        if r.get('fast_full_prompt_pass','').strip() and r.get('standard_full_prompt_pass','').strip():
            labels[str(r['prompt_id'])]=(truth(r['fast_full_prompt_pass']),truth(r['standard_full_prompt_pass']))
    if args.mode=='holdout':
        # User provides new benchmark/features and newly completed human labels
        # This script does not assert any independent generalization without a new dataset.
        if {r['prompt_id'] for r in rows}==set(str(i) for i in range(16)):
            raise ValueError('Holdout must use genuinely NEW cases, not the 16 development prompts')
    result=evaluate(rows,labels,args.threshold,args.post_standard)
    predictions=result.pop('predictions')
    report=dict(mode=args.mode,policy_name='CLIP score only + optional post-standard gate',
                policy_frozen=dict(threshold=args.threshold,decision='fallback if score < threshold'),
                limitations=['Offline counterfactual from previously measured generation timings; not new wall clock runtime.',
                             'Full prompt quality comes from human labels used ONLY for scoring.',
                             'Post-standard check is an UNIMPLEMENTED gate, not an automatic quality guarantee.',
                             'A pending post-standard verification must not be presented as a verified PASS.',
                             'Any thresholds picked from these 16 images are development-only, not externally validated.'],
                summary=result,predictions=predictions)
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    print('Saved:',dest)
if __name__=='__main__': main()
