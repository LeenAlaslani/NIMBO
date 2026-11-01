"""CLI milestone 1: load an external model, run a measured CPU baseline."""
from __future__ import annotations
import argparse
import json
import sys
from app.nimbo_model_intake_v1 import run_real_baseline


def main(argv=None):
    parser = argparse.ArgumentParser(description="NIMBO real external model baseline (no optimization yet)")
    parser.add_argument("--model-id", required=True, help="Public Hugging Face owner/repository")
    parser.add_argument("--dataset", required=True, help="UTF-8 labeled CSV or JSONL, text and optional label")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output-dir", default="results/auto_optimizer_baselines")
    args = parser.parse_args(argv)
    try:
        path, result = run_real_baseline(args.model_id, args.dataset, cpu_threads=args.cpu_threads,
                                         max_length=args.max_length, output_dir=args.output_dir)
    except (ValueError, ImportError, RuntimeError, OSError) as exc:
        print(f"NIMBO BASELINE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"STATUS: BASELINE_MEASURED — NOT OPTIMIZED\nREPORT: {path}")
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
