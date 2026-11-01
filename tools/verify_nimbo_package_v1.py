"""Independently reload extracted NIMBO package without Hub or project-owned model files."""
from __future__ import annotations
import argparse,json,sys
from app.nimbo_package_v1 import verify_package

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--package',required=True)
    p.add_argument('--dataset',required=True)
    args=p.parse_args(argv)
    try:
        report=verify_package(args.package,args.dataset)
    except (ValueError,OSError,RuntimeError,ImportError,KeyError) as exc:
        print(f'NIMBO M4 VERIFY FAILED: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 1
    print('STATUS: OFFLINE_BUNDLE_RELOAD_VERIFIED_REVIEW_ONLY')
    print(json.dumps(report,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
