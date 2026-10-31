"""NIMBO independent-domain evaluation on Rotten Tomatoes sentiment dataset.

The selected threshold uses calibration only. This is cross-dataset evaluation,
not proof of zero training-set overlap. Never optimize using test reports.
"""
import argparse
import json
import platform
from pathlib import Path
from statistics import mean
from time import perf_counter_ns

from nimbo.benchmark import macro_f1, percentile_ms
from nimbo.text_backend import load_classifier
from experiments.run_sst2 import SMALL, LARGE

THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0]


def summarize(rows, preds, elapsed_ns, fallback_count=0):
    actual = [r['label'] for r in rows]
    predicted = [p['label'] for p in preds]
    times = [n / 1e6 for n in elapsed_ns]
    return {
        'samples': len(rows),
        'accuracy': round(mean(a == b for a, b in zip(actual, predicted)), 5),
        'macro_f1': round(macro_f1(actual, predicted), 5),
        'mean_latency_ms': round(mean(times), 4),
        'p50_latency_ms': percentile_ms(times, .5),
        'p95_latency_ms': percentile_ms(times, .95),
        'fallback_rate': round(fallback_count / len(rows), 5),
    }


def select_policy(rows, small_preds, large_preds, max_drop=.01, max_fallback=.25):
    """Minimize fallback count while preserving near-large accuracy on calibration.

    If the constrained target is infeasible, return an explicitly marked
    large-only fallback policy, rather than falsely claiming guaranteed quality.
    """
    large_acc = mean(p['label'] == r['label'] for p, r in zip(large_preds, rows))
    options = []
    for threshold in THRESHOLDS:
        mask = [p['confidence'] < threshold for p in small_preds]
        fallback = sum(mask) / len(rows)
        chosen = [l if use_large else s for s, l, use_large in zip(small_preds, large_preds, mask)]
        acc = mean(p['label'] == r['label'] for p, r in zip(chosen, rows))
        options.append({'threshold': threshold, 'accuracy': round(acc, 5),
                        'fallback_rate': round(fallback, 5),
                        'meets_target': acc >= large_acc - max_drop - 1e-12 and fallback <= max_fallback})
    viable = [o for o in options if o['meets_target']]
    selected = min(viable, key=lambda o: (o['fallback_rate'], -o['accuracy'])) if viable else None
    return selected, options, round(large_acc, 5)


def timed(predict):
    """Time a zero-argument runner that already captures the input text."""
    start = perf_counter_ns()
    prediction = predict()
    return prediction, perf_counter_ns() - start


def evaluate(test_rows, small, large, threshold, warmup=4):
    for item in test_rows[:warmup]:
        small(item['text']); large(item['text'])
    observed = {key: ([], []) for key in ('small', 'large', 'adaptive')}
    escalations = 0
    for i, row in enumerate(test_rows):
        def adaptive():
            nonlocal escalations
            p = small(row['text'])
            if threshold is None or p['confidence'] < threshold:
                escalations += 1
                return large(row['text'])
            return p
        runners = {'small': lambda: small(row['text']),
                   'large': lambda: large(row['text']), 'adaptive': adaptive}
        order = ('small', 'adaptive', 'large') if i % 2 == 0 else ('large', 'adaptive', 'small')
        for route in order:
            pred, elapsed = timed(runners[route])
            observed[route][0].append(pred)
            observed[route][1].append(elapsed)
    return {name: summarize(test_rows, p, t, escalations if name == 'adaptive' else 0)
            for name, (p, t) in observed.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration', type=int, default=200)
    parser.add_argument('--test', type=int, default=800)
    parser.add_argument('--max-drop', type=float, default=.01)
    parser.add_argument('--max-fallback', type=float, default=.25)
    parser.add_argument('--output', default='results/rotten_tomatoes_v3.json')
    args = parser.parse_args()
    if args.calibration < 20 or args.test < 20 or args.calibration + args.test > 1066:
        parser.error('Choose calibration >=20, test >=20, and total <=1066.')
    if not 0 <= args.max_drop <= 1 or not 0 <= args.max_fallback <= 1:
        parser.error('Drops and fallback budget must be between 0 and 1')
    from datasets import load_dataset
    data = load_dataset('cornell-movie-review-data/rotten_tomatoes', split='test')
    rows = [{'text': r['text'], 'label': 'POSITIVE' if r['label'] == 1 else 'NEGATIVE'}
            for r in data]
    calibration = rows[:args.calibration]
    testing = rows[args.calibration:args.calibration + args.test]
    small = load_classifier(SMALL, device=-1, sst2_labels=True)
    large = load_classifier(LARGE, device=-1, sst2_labels=True)
    small_cal = [small(r['text']) for r in calibration]
    large_cal = [large(r['text']) for r in calibration]
    policy, candidates, baseline_acc = select_policy(
        calibration, small_cal, large_cal, args.max_drop, args.max_fallback)
    threshold = policy['threshold'] if policy else None
    stats = evaluate(testing, small, large, threshold)
    result = {'evaluation': 'cross-dataset sentiment transfer, not an in-domain SST-2 test',
              'dataset': 'cornell-movie-review-data/rotten_tomatoes test',
              'calibration_samples': len(calibration), 'test_samples': len(testing),
              'calibration_large_accuracy': baseline_acc,
              'quality_constraint': {'max_accuracy_drop': args.max_drop,
                                     'max_fallback_rate': args.max_fallback},
              'policy': policy or {'type': 'large_only_safety_fallback',
                                   'reason': 'No candidate met quality and fallback constraints'},
              'candidate_summary_calibration': candidates,
              'test_strategies': stats,
              'latency_reduction_vs_large_pct': round(100 * (1 - stats['adaptive']['mean_latency_ms'] /
                                               stats['large']['mean_latency_ms']), 2)
                                               if stats['large']['mean_latency_ms'] > 0 else None,
              'models': {'small': SMALL, 'large': LARGE}, 'platform': platform.platform(),
              'caveats': ['Labels are from a different sentiment dataset; domain shift is expected.',
                          'Model training overlap with this external corpus has not been verified.',
                          'The test set must not be used to tune threshold or model selection.',
                          'Model quality is assessed by classification accuracy, not confidence calibration.',
                          'CPU latency, not GPU compute, energy, or billing cost.']}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
