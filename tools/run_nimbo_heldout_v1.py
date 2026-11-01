"""Verify the existing M2 candidate on new labeled examples; no deployment."""
from __future__ import annotations
import argparse
import json
import sys
from app.nimbo_heldout_validation_v1 import validate_existing_candidate


def main(argv=None):
    p=argparse.ArgumentParser(description='NIMBO M3 local heldout candidate re-validation')
    p.add_argument('--m2-report',required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--preprocess',choices=('identity','cardiff_tweet'),default='identity')
    p.add_argument('--root',default='.')
    args=p.parse_args(argv)
    try:
        path,report=validate_existing_candidate(args.m2_report,args.dataset,
                           root=args.root,rounds=args.rounds,preprocess=args.preprocess)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f'NIMBO M3 FAILED: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 1
    print('STATUS: HELDOUT_VALIDATION_COMPLETED — REVIEW ONLY')
    print('REPORT:',path)
    print(json.dumps({'baseline':report['baseline'],'candidate':report['candidate'],
                     'decision':report['decision']},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
