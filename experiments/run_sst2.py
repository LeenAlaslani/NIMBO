"""Run genuine SST-2 inference; uses real pretrained checkpoints and held-out labels.

Example:
  python -m experiments.run_sst2 --samples 100 --threshold 0.9

NOTE: threshold is illustrative unless selected with a separate validation split.
"""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path

from nimbo.benchmark import evaluate
from nimbo.text_backend import load_classifier

SMALL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
LARGE = "textattack/bert-base-uncased-SST-2"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--output", default="results/sst2_real.json")
    p.add_argument("--device", type=int, default=-1, help="-1 CPU; 0 NVIDIA GPU")
    args = p.parse_args()
    if args.samples < 5: p.error("--samples must be at least 5")
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", split=f"validation[:{args.samples}]")
    rows = [{"text": row["sentence"], "label": "POSITIVE" if row["label"] == 1 else "NEGATIVE"} for row in ds]
    small = load_classifier(SMALL, device=args.device, sst2_labels=True)
    large = load_classifier(LARGE, device=args.device, sst2_labels=True)
    report = evaluate(rows, small, large, threshold=args.threshold, warmup=3)
    report["model_ids"] = {"small": SMALL, "large": LARGE}
    report["dataset"] = "GLUE SST-2 validation (first N examples)"
    report["device"] = args.device
    report["platform"] = platform.platform()
    report["caveats"] = ["Threshold 0.9 is not calibrated; use a separate calibration split before optimization claims.", "Two models may differ in training data and quality; accuracy is measured on the same labeled records.", "Runs baseline then adaptive; repeat alternating order before final latency claims.", "No peak memory or energy figures are reported."]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
