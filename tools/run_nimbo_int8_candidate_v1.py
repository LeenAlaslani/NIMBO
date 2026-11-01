"""M5: run a genuine dynamic INT8 dev-only trial using the previously verified M4 ZIP."""
from __future__ import annotations
import argparse
import json
import sys
from app.nimbo_int8_candidate_v1 import run_int8_trial


def main(argv=None):
    p=argparse.ArgumentParser(description='NIMBO M5 size-aware INT8 candidate; NO automatic export/deployment')
    p.add_argument('--package',required=True,help='Previously verified M4 review ZIP')
    p.add_argument('--package-sha256',required=True,help='Printed SHA-256 from successful M4 run')
    p.add_argument('--m3-report',required=True)
    p.add_argument('--m3-dataset',required=True)
    p.add_argument('--dev-dataset',required=True,help='200-800 LABELED new development rows; do not reuse M3 test set')
    p.add_argument('--objective',choices=('size','balanced'),default='size')
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--root',default='.')
    args=p.parse_args(argv)
    try:
        report,result=run_int8_trial(args.package,args.package_sha256,args.m3_report,
            args.m3_dataset,args.dev_dataset,root=args.root,objective=args.objective,rounds=args.rounds)
    except (ValueError,RuntimeError,OSError,ImportError,KeyError) as error:
        print(f'NIMBO M5 FAILED: {type(error).__name__}: {error}',file=sys.stderr)
        return 1
    print('STATUS: INT8_CANDIDATE_EVALUATED — DEV ONLY')
    print('REPORT:',report)
    print('INT8_GRAPH:',report.parent/'model.int8.onnx')
    print(json.dumps({'fp32_graph_bytes':result['fp32_graph_bytes'],
                      'int8_graph_bytes':result['int8_graph_bytes'],
                      'fp32':result['fp32_metrics'],'int8':result['int8_metrics'],
                      'decision':result['decision']},indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
