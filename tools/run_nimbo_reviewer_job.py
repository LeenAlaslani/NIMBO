"""Run the existing optimization engine with the hosted resource limits."""
import resource
import sys
from pathlib import Path

from app.nimbo_auto_pipeline_jobs_v1 import run_worker
from app.nimbo_reviewer_catalog import RUNTIME


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (240, 245))
    resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    root, job_id = sys.argv[1:]
    license_file = Path(__file__).resolve().parents[1] / 'deploy' / 'LICENSE-Apache-2.0.txt'
    result = run_worker(Path(root), job_id, runtime=RUNTIME, license_file=license_file)
    return int(result['state'] not in {'COMPLETED_REVIEW_ONLY', 'COMPLETED_REJECTED'})


if __name__ == '__main__':
    sys.exit(main())
