"""NIMBO routing diagnostics. Real inference; no simulated performance data.

Run: python -m experiments.diagnose_routing --calibration 200 --test 500
Move this file to experiments/diagnose_routing.py in the repository.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean
from datasets import load_dataset
from nimbo.text_backend import load_classifier
from experiments.run_sst2 import SMALL, LARGE


def accuracy(rows):
    return mean(int(r['prediction'] == r['truth']) for r in rows)


def analyze(rows, thresholds):
    small_acc = mean(int(r['small']['label'] == r['truth']) for r in rows)
    large_acc = mean(int(r['large']['label'] == r['truth']) for r in rows)
    categories = {
        'both_correct': 0,
        'only_small_correct': 0,
        'only_large_correct': 0,
        'both_wrong': 0,
    }
    for row in rows:
        s = row['small']['label'] == row['truth']
        l = row['large']['label'] == row['truth']
        categories['both_correct' if s and l else 'only_small_correct' if s else 'only_large_correct' if l else 'both_wrong'] += 1
    sweep = []
    for threshold in thresholds:
        n_fallback = sum(r['small']['confidence'] < threshold for r in rows)
        correct = sum((r['large'] if r['small']['confidence'] < threshold else r['small'])['label'] == r['truth'] for r in rows)
        sweep.append({'threshold': threshold, 'fallback_count': n_fallback,
                      'fallback_rate': round(n_fallback / len(rows), 4),
                      'adaptive_accuracy': round(correct / len(rows), 4)})
    return {'n':len(rows), 'small_accuracy':round(small_acc,4), 'large_accuracy':round(large_acc,4),
            'error_overlap':categories, 'threshold_sweep':sweep}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration', type=int, default=200)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--output', default='results/routing_diagnostic.json')
    args = parser.parse_args()
    if args.calibration < 1 or args.test < 1 or args.calibration + args.test > 872:
        parser.error('calibration and test must be positive, with sum <= 872')
    ds = load_dataset('glue', 'sst2', split='validation')
    small = load_classifier(SMALL, device=args.device, sst2_labels=True)
    large = load_classifier(LARGE, device=args.device, sst2_labels=True)
    records = []
    for sample in list(ds)[:args.calibration + args.test]:
        sentence = sample['sentence']
        records.append({'truth':'POSITIVE' if sample['label'] == 1 else 'NEGATIVE',
                        'small':small(sentence), 'large':large(sentence)})
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0]
    report = {
        'method':'Diagnostics only: test sweep must NOT be used to select production policy',
        'calibration':analyze(records[:args.calibration], thresholds),
        'test':analyze(records[args.calibration:], thresholds),
        'models':{'small':SMALL,'large':LARGE},
        'dataset':'GLUE SST-2 validation, disjoint partitions',
    }
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
