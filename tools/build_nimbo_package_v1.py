"""M4: build a review-only offline ONNX model ZIP, gated on M2 and M3 evidence."""
from __future__ import annotations
import argparse,json,sys
from app.nimbo_package_v1 import package_model

def main(argv=None):
    p=argparse.ArgumentParser(description='NIMBO M4 build and independently verify ONNX review package')
    p.add_argument('--m3-report',required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--root',default='.')
    args=p.parse_args(argv)
    try:
        path, report=package_model(args.m3_report,args.dataset,root=args.root)
    except (ValueError,OSError,RuntimeError,ImportError,KeyError) as exc:
        print(f'NIMBO M4 FAILED: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 1
    print('STATUS: RUNNABLE_PACKAGE_VERIFIED — REVIEW ONLY')
    print('PACKAGE:',path)
    print(json.dumps(report,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
