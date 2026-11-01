"""Run an accepted local M7 job in its own Python process."""
from __future__ import annotations
import argparse
from pathlib import Path
from app.nimbo_auto_pipeline_jobs_v1 import run_worker, job_folder


def main(argv=None):
    p=argparse.ArgumentParser(description='Internal M7 worker; use /nimbo/auto/jobs to submit')
    p.add_argument('--root',required=True)
    p.add_argument('--job-id',required=True)
    a=p.parse_args(argv)
    root=Path(a.root).expanduser().resolve(strict=True)
    job_folder(root,a.job_id)
    result=run_worker(root,a.job_id)
    return 0 if result['state'] in ('COMPLETED_REVIEW_ONLY','COMPLETED_REJECTED') else 1

if __name__=='__main__':
    raise SystemExit(main())
