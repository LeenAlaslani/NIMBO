"""Reproducible benchmark: matched label-space real text classifiers.

No synthetic samples and no invented savings. CSV must contain text,label columns.
Run from CLI after installing the optional text dependencies.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Callable

from nimbo.core import Modality, NimboAgent, Request


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not {'text', 'label'}.issubset(reader.fieldnames or []):
            raise ValueError('CSV requires text,label columns')
        rows = [{'text': r['text'].strip(), 'label': r['label'].strip()} for r in reader]
    if not rows or any(not r['text'] or not r['label'] for r in rows):
        raise ValueError('CSV must contain non-empty text and label values')
    return rows


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    scores = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return mean(scores)


def percentile_ms(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low, high = int(index), min(len(ordered)-1, int(index)+1)
    return round(ordered[low] + (ordered[high]-ordered[low]) * (index-low), 3)


def assess(rows, predictor: Callable[[str], dict], *, warmup: int = 2):
    if not 0 <= warmup:
        raise ValueError('warmup must be nonnegative')
    for row in rows[:warmup]:
        predictor(row['text'])
    labels, latencies, routes = [], [], []
    for row in rows:
        start = perf_counter()
        pred = predictor(row['text'])
        elapsed = (perf_counter()-start)*1000
        if 'label' not in pred:
            raise ValueError('predictor must return label')
        labels.append(str(pred['label']))
        latencies.append(elapsed)
        routes.append(str(pred.get('route', 'single-model')))
    truths = [r['label'] for r in rows]
    return {
        'samples': len(rows),
        'accuracy': round(sum(a==b for a,b in zip(truths,labels))/len(rows), 5),
        'macro_f1': round(macro_f1(truths,labels), 5),
        'latency_p50_ms': percentile_ms(latencies, .5),
        'latency_p95_ms': percentile_ms(latencies, .95),
        'mean_latency_ms': round(mean(latencies), 3),
        'routes': dict(Counter(routes)),
    }


def evaluate(rows, small, large, threshold: float = .8, warmup: int = 2):
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be between 0 and 1')
    agent = NimboAgent(small, large, lambda _: None, lambda _: None, threshold)
    def adaptive(text):
        trace = agent.run(Request(Modality.TEXT, text))
        return {**trace.result, 'route': trace.decision['route']}
    baseline = assess(rows, large, warmup=warmup)
    optimized = assess(rows, adaptive, warmup=warmup)
    return {
        'benchmark': 'text-classification',
        'note': 'Same input set and runtime; run multiple repetitions for robust measurements. No training/peak-memory claims.',
        'threshold': threshold,
        'baseline_always_large': baseline,
        'nimbo_adaptive': optimized,
        'latency_reduction_pct': round(100*(1-optimized['mean_latency_ms']/baseline['mean_latency_ms']),2) if baseline['mean_latency_ms']>0 else None,
        'macro_f1_delta': round(optimized['macro_f1']-baseline['macro_f1'],5),
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark actual NIMBO text inference')
    parser.add_argument('--data', required=True, help='Held-out CSV with text,label columns')
    parser.add_argument('--small', required=True, help='HF sequence classification model ID')
    parser.add_argument('--large', required=True, help='HF sequence classification model ID with same labels')
    parser.add_argument('--threshold', type=float, default=.8, help='Calibrated on separate validation data')
    parser.add_argument('--output', default='results/text_benchmark.json')
    parser.add_argument('--warmup', type=int, default=2)
    args = parser.parse_args()
    from nimbo.text_backend import load_classifier
    small = load_classifier(args.small)
    large = load_classifier(args.large)
    rows = load_rows(args.data)
    # Validate every class is represented using the same label identifiers in both models.
    expected = set(r['label'] for r in rows)
    sample = rows[0]['text']
    s, l = str(small(sample)['label']), str(large(sample)['label'])
    if s not in expected or l not in expected:
        raise ValueError('Model label IDs do not match CSV labels; align label maps before benchmarking')
    result = evaluate(rows, small, large, args.threshold, args.warmup)
    result['model_ids'] = {'small': args.small, 'large': args.large}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
