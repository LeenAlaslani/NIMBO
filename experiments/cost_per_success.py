"""Evidence-based latency per successful image task, using labeled paired runs.

Run: python -m experiments.cost_per_success --csv results/benchmark_v2_raw.csv
A success is defined ONLY as human-labeled critical-object presence. This does
not imply overall image quality or safety. Time is NOT equivalent to GPU energy.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def label(value):
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {'true', '1', 'yes', 'y'}:
        return True
    if s in {'false', '0', 'no', 'n'}:
        return False
    return None


def number(row, key):
    try:
        value = float(row[key])
        if not (0 <= value < float('inf')):
            raise ValueError()
        return value
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric field {key!r} for prompt {row.get('prompt_id')}") from exc


def evaluate(rows):
    if not rows:
        raise ValueError('No records')
    strategies = {name: {'total_s': 0.0, 'successes': 0, 'known': 0, 'latencies_s': []} for name in ('always_fast', 'always_standard', 'nimbo')}
    pending = []
    checks = 0
    escalations = 0
    false_accepts = 0
    unnecessary_escalations = 0
    for row in rows:
        route = row['route']
        if route not in {'fast_accepted', 'escalated_to_standard'}:
            raise ValueError(f'Unknown route: {route}')
        fast_time = number(row, 'fast_s')
        standard_time = number(row, 'standard_s')
        check_time = number(row, 'checker_s')
        adaptive_time = number(row, 'nimbo_s')
        # Validate accounting: a fallback must include fast + checker + standard.
        expected = fast_time + check_time + (standard_time if route == 'escalated_to_standard' else 0)
        if abs(expected-adaptive_time) > max(0.03, expected*0.02):
            raise ValueError(f"Inconsistent NIMBO latency for prompt {row.get('prompt_id')}")
        checks += 1
        if route == 'escalated_to_standard':
            escalations += 1
        fast_label = label(row.get('human_fast_object_present'))
        standard_label = label(row.get('human_standard_object_present'))
        if fast_label is None or standard_label is None:
            pending.append(str(row.get('prompt_id', '?')))
        if route == 'fast_accepted' and fast_label is False:
            false_accepts += 1
        if route == 'escalated_to_standard' and fast_label is True:
            unnecessary_escalations += 1
        chosen = standard_label if route == 'escalated_to_standard' else fast_label
        for name, time_s, success in (
            ('always_fast', fast_time, fast_label),
            ('always_standard', standard_time, standard_label),
            ('nimbo', adaptive_time, chosen)):
            rec = strategies[name]
            rec['total_s'] += time_s
            rec['latencies_s'].append(time_s)
            if success is not None:
                rec['known'] += 1
                rec['successes'] += int(success)
    all_labeled = not pending
    output = {}
    for name, rec in strategies.items():
        # Only compute per-success on a completely labeled cohort.
        output[name] = {
            'mean_latency_s': round(mean(rec['latencies_s']), 4),
            'total_latency_s': round(rec['total_s'], 4),
            'labeled_count': rec['known'],
            'successful_tasks': rec['successes'] if all_labeled else None,
            'success_rate': round(rec['successes']/len(rows), 4) if all_labeled else None,
            'latency_seconds_per_success': round(rec['total_s']/rec['successes'], 4) if all_labeled and rec['successes'] else None,
        }
    return {
        'sample_count':len(rows),
        'quality_metric':'human-rated presence of required object (not full image quality)',
        'quality_validated': all_labeled,
        'pending_human_labels':pending,
        'fallback_count':escalations,
        'fallback_rate':round(escalations/len(rows),4),
        'checker_total_s':round(sum(number(row,'checker_s') for row in rows),4),
        'false_accept_count':false_accepts if all_labeled else None,
        'unnecessary_fallback_count':unnecessary_escalations if all_labeled else None,
        'strategies':output,
        'notes':['Seconds of wall-clock inference per successful task is a latency-based proxy, NOT GPU compute, FLOPs, energy or financial cost.',
                 'Standard-only latency excludes quality checks; NIMBO includes CLIP and fallback.',
                 'All quality labels must be completed before reporting success-based metrics.',
                 'Model loads, idle GPU power, and switching/throughput costs are excluded.']
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--output', default='results/cost_per_success_image.json')
    args = parser.parse_args()
    with open(args.csv, newline='', encoding='utf-8') as source:
        report = evaluate(list(csv.DictReader(source)))
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report['quality_validated']:
        print(f"\nQUALITY NOT VALIDATED: label both image types for {len(report['pending_human_labels'])} prompts in the CSV.")

if __name__ == '__main__':
    main()
