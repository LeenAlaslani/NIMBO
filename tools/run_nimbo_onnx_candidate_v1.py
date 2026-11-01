"""Execute an M2 exploratory ONNX FP32 paired benchmark for an external model."""
from __future__ import annotations
import argparse
import json
import sys
from app.nimbo_onnx_candidate_v1 import run_paired_candidate


def main(argv=None):
    p = argparse.ArgumentParser(description='NIMBO M2 real same-model PyTorch/ONNX benchmark (NOT an exportable deployment package)')
    p.add_argument('--model-id', required=True)
    p.add_argument('--revision', default=None, help='Immutable 40-digit Hugging Face revision SHA')
    p.add_argument('--dataset', required=True)
    p.add_argument('--baseline-report', default=None, help='Verify identity with a previous M1 baseline report')
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--output-root', default='results/auto_optimizer_candidates_v1')
    args = p.parse_args(argv)
    try:
        path, result = run_paired_candidate(args.model_id, args.dataset, revision=args.revision,
            baseline_report=args.baseline_report, threads=args.cpu_threads, max_length=args.max_length,
            rounds=args.rounds, output_root=args.output_root)
    except (ValueError, ImportError, RuntimeError, OSError) as exc:
        print(f'NIMBO M2 FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    print('STATUS: ONNX_CANDIDATE_BENCHMARKED — EXPERIMENTAL ONLY')
    print('REPORT:', path)
    print('MODEL_GRAPH:', result['artifact']['path'])
    print(json.dumps({'baseline':result['baseline'], 'candidate':result['onnx_candidate'],
                      'decision':result['decision']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
