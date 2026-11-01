"""M6: evidence-driven model candidate fallback selection; never production approval."""
from __future__ import annotations
import argparse
import subprocess
import json
import sys
from app.nimbo_auto_decision_v1 import select_verified_fallback


def main(argv=None):
    p=argparse.ArgumentParser(description='Choose the verified FP32 package when every INT8 development variant was rejected')
    p.add_argument('--package',required=True)
    p.add_argument('--package-sha256',required=True)
    p.add_argument('--m3-report',required=True)
    p.add_argument('--m3-dataset',required=True)
    p.add_argument('--m5-report',required=True)
    p.add_argument('--search-report',required=True)
    p.add_argument('--dev-dataset',required=True)
    p.add_argument('--root',default='.')
    a=p.parse_args(argv)
    try:
        path,result=select_verified_fallback(a.package,a.package_sha256,a.m3_report,a.m3_dataset,
           a.m5_report,a.search_report,a.dev_dataset,root=a.root)
    except (ValueError,KeyError,TypeError,RuntimeError,OSError,ImportError,subprocess.TimeoutExpired) as exc:
        print('NIMBO M6 SELECTION FAILED:',type(exc).__name__,str(exc),file=sys.stderr)
        return 1
    print('STATUS: VERIFIED_FALLBACK_SELECTED — REVIEW ONLY')
    print('DECISION:',result['decision'])
    print('SELECTED:',result['selected_package']['path'])
    print('RECEIPT:',path)
    print(json.dumps({'int8_candidates_rejected':result['evidence']['int8_candidates_rejected'],
                      'offline_reload':result['evidence']['offline_reload'],
                      'eligible_for_deployment':result['eligible_for_deployment']},indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
