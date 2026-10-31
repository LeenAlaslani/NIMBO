"""Evaluate manually annotated NIMBO image routing; no dependencies beyond stdlib.

Run: python -m experiments.evaluate_image_review_v2 --csv path/to/human_review_v2.csv
Human labels: 1=object clearly visible, 0=absent or unclear. Empty=unreviewed.
"""
import argparse
import csv
import json
from pathlib import Path


def summarize(rows):
    annotated=[]
    for row in rows:
        fast=str(row.get('human_fast_object_present','')).strip()
        standard=str(row.get('human_standard_object_present','')).strip()
        if fast not in ('0','1') or standard not in ('0','1'):
            continue
        accepted=row['route']=='fast_accepted'
        annotated.append((row,accepted,fast=='1',standard=='1'))
    n=len(annotated)
    if not n:
        return {'labeled':0,'status':'Need human 0/1 labels for BOTH fast and standard images.'}
    # CLIP decision interpreted as fast image passing a defined object-presence criterion.
    tp=sum(a and f for _,a,f,_ in annotated)
    tn=sum(not a and not f for _,a,f,_ in annotated)
    fp=sum(a and not f for _,a,f,_ in annotated)  # dangerous accept
    fn=sum(not a and f for _,a,f,_ in annotated) # unnecessary escalation
    final_success=sum((f if a else s) for _,a,f,s in annotated)
    standard_success=sum(s for _,a,f,s in annotated)
    fast_success=sum(f for _,a,f,s in annotated)
    r={'labeled':n,'checker_accuracy':round((tp+tn)/n,4),
       'precision_of_fast_accept':round(tp/(tp+fp),4) if tp+fp else None,
       'recall_of_fast_acceptable':round(tp/(tp+fn),4) if tp+fn else None,
       'confusion':{'correct_accept':tp,'correct_escalate':tn,
                    'unsafe_accept':fp,'unnecessary_escalate':fn},
       'human_object_success_rate':{'always_fast':round(fast_success/n,4),
                                    'always_standard':round(standard_success/n,4),
                                    'nimbo':round(final_success/n,4)},
       'note':'Small manually labeled set; object visibility is not overall image quality.'}
    return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--output')
    args=p.parse_args()
    with open(args.csv,encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    result=summarize(rows)
    dest=Path(args.output) if args.output else Path(args.csv).with_name('human_review_evaluation_v2.json')
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print('Saved:',dest)
if __name__=='__main__':main()
