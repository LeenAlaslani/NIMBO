"""NIMBO M5.1 bounded selective INT8 candidate search (development data only)."""
from __future__ import annotations
import argparse
import json
import sys
from app.nimbo_quality_quantization_v1 import DEFAULT_PROFILES, PROFILES, run_quality_search


def main(argv=None):
    p=argparse.ArgumentParser(description='Quality-aware selective INT8; never auto-deploy or export')
    p.add_argument('--package',required=True,help='M4 verified FP32 ONNX archive')
    p.add_argument('--package-sha256',required=True)
    p.add_argument('--m5-report',required=True,help='Original rejected M5 report')
    p.add_argument('--m3-report',required=True)
    p.add_argument('--m3-dataset',required=True)
    p.add_argument('--dev-dataset',required=True)
    p.add_argument('--root',default='.')
    p.add_argument('--objective',choices=['size','balanced'],default='size')
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--profiles',default=','.join(DEFAULT_PROFILES),
                   help='Comma-separated profiles, permitted: '+','.join(PROFILES))
    args=p.parse_args(argv)
    try:
        path,result=run_quality_search(args.package,args.package_sha256,args.m5_report,
                 args.m3_report,args.m3_dataset,args.dev_dataset,
                 root=args.root,objective=args.objective,rounds=args.rounds,
                 profiles=tuple(x.strip() for x in args.profiles.split(',')))
    except (ValueError,RuntimeError,OSError,ImportError,KeyError) as exc:
        print(f'NIMBO M5.1 FAILED: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 1
    print('STATUS: QUALITY_AWARE_INT8_SEARCH_COMPLETED — DEV ONLY')
    print('REPORT:',path)
    print('DECISION:',result['decision'])
    print('SELECTED:',json.dumps(result['selected'],ensure_ascii=False))
    print(json.dumps({'trials':[{'profile':r['profile'],'status':r['status'],
                 'graph_bytes':r.get('int8_graph_bytes'),
                 'accuracy':(r.get('int8_metrics') or {}).get('accuracy'),
                 'agreement':(r.get('decision') or {}).get('prediction_agreement'),
                 'decision':(r.get('decision') or {}).get('decision'),
                 'error':r.get('error')} for r in result['trials']]},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
