"""Build a claim-limited judging summary from an actual local held-out run.

No model inference is performed. No invented energy/FLOPs/cost claims.
Run from repo root: python tools/judging_evidence.py
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

DEFAULT_LOG = Path('results/independent_validation_v21.jsonl')
DEFAULT_DEST = Path('results/judging')


def summarize(record):
    c = record['comparison']
    opt = record['optimization']
    gate = opt['gate']
    if (opt.get('validation') != 'heldout_repeated_v21'
            or c.get('sample_count') != 800
            or c.get('rounds') != 3
            or not c.get('all_labels_present')):
        raise ValueError('Record is not the 800-example, three-round labeled heldout validation')
    b, a = c['baseline'], c['candidate']
    for x in (b, a):
        if not isinstance(x.get('accuracy_on_reference_labels'), (float, int)):
            raise ValueError('Recorded accuracy is missing; do not declare quality success')
        if not isinstance(x.get('mean_inference_ms'), (float, int)) or x['mean_inference_ms'] <= 0:
            raise ValueError('Recorded mean inference latency is invalid')
    measured = 1 - (a['mean_inference_ms'] / b['mean_inference_ms'])
    recorded = c['mean_latency_reduction_fraction']
    if abs(measured-recorded) > 0.0001:
        raise ValueError('Measured and recorded speedup do not agree')
    if opt.get('approval') == 'eligible_for_review' and not all((gate.get('eligible'), gate.get('quality_pass'), gate.get('latency_pass'), gate.get('p95_pass'), gate.get('repeatability_pass'))):
        raise ValueError('An eligible run has failing or missing gates')
    return {
        'run_id': record['run_id'],
        'timestamp_utc': record.get('timestamp_utc'),
        'validation':'heldout_repeated_v21',
        'model_id':record.get('model_id'),
        'model_revision':record.get('model_revision'),
        'partition_digest':c.get('partition_digest'),
        'hardware':record.get('hardware'),
        'python_version':record.get('python_version'),
        'pytorch_version':record.get('pytorch_version'),
        'onnxruntime_version':record.get('onnxruntime_version'),
        'inputs':c['sample_count'],
        'rounds':c['rounds'],
        'measurements_per_path':c['measurement_count'],
        'baseline_mean_ms':b['mean_inference_ms'],
        'candidate_mean_ms':a['mean_inference_ms'],
        'baseline_p95_ms':b.get('p95_ms'),
        'candidate_p95_ms':a.get('p95_ms'),
        'baseline_accuracy':b['accuracy_on_reference_labels'],
        'candidate_accuracy':a['accuracy_on_reference_labels'],
        'prediction_agreement_fraction':c['prediction_agreement_fraction'],
        'mean_latency_reduction_fraction':recorded,
        'local_pipeline_speedup_fraction':c['local_pipeline_speedup_fraction'],
        'rounds_detail':c['per_round'],
        'approval':opt['approval'],
        'gate':gate,
        'exported_variant':bool(opt.get('export')),
        'claims_not_supported':['lower FLOPs','less electrical energy','reduced cloud spend','third-party audited quality','deployed in production'],
    }


def markdown(x):
    q=lambda v: f'{v*100:.2f}%'
    lines=[
        '# NIMBO — held-out judging evidence (local run)',
        '',
        f'- **Run ID:** `{x["run_id"]}`',
        f'- **Recorded UTC:** {x["timestamp_utc"]}',
        f'- **Model:** `{x["model_id"]}`',
        f'- **Model revision:** `{x["model_revision"]}`',
        f'**Held-out digest:** `{x["partition_digest"]}`',
        '',
        '| Metric | PyTorch FP32 | ONNX FP32 |',
        '|---|---:|---:|',
        f'| Warm mean inference | {x["baseline_mean_ms"]:.4f} ms | {x["candidate_mean_ms"]:.4f} ms |',
        f'| Warm p95 inference | {x["baseline_p95_ms"]:.4f} ms | {x["candidate_p95_ms"]:.4f} ms |',
        f'| Reference-label accuracy | {q(x["baseline_accuracy"])} | {q(x["candidate_accuracy"])} |',
        '',
        f'- **Mean warm CPU inference reduction:** {q(x["mean_latency_reduction_fraction"])}',
        f'- **Tokenization + forward local reduction:** {q(x["local_pipeline_speedup_fraction"])}',
        f'- **Prediction agreement:** {q(x["prediction_agreement_fraction"])}',
        f'- **Paired test:** {x["inputs"]} disjoint IMDb records, {x["rounds"]} rounds, {x["measurements_per_path"]} measured passes per path.',
        f'- **Gate decision:** `{x["approval"]}`; not a production deployment.',
        '',
        '## Robustness / repeatability across rounds',
        '',
        '| Round | Baseline mean (ms) | ONNX mean (ms) |',
        '|---:|---:|---:|',
    ]
    for row in x['rounds_detail']:
        lines.append(f'| {row["round"]} | {row["baseline_forward_mean_ms"]:.4f} | {row["candidate_forward_mean_ms"]:.4f} |')
    lines.extend(['','## Trust boundary','',
        'Reference labels are the local IMDb evaluation labels on a disjoint holdout, **not third-party audited quality**.',
        'Measurements are **warm CPU inference latency**, not reduced FLOPs, electrical energy or dollar costs.',
        'Setup/model loading and HTTP/browser time are **not** part of the warm inference numbers.',
        'Passing the gate means **eligible for manual review**, not deployed or production certified.',
        '',f'Environment snapshot: `{json.dumps(x["hardware"],ensure_ascii=False)}`; Python {x["python_version"]}, torch {x["pytorch_version"]}, onnxruntime {x["onnxruntime_version"]}.',
        ''])
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--log',type=Path,default=DEFAULT_LOG)
    p.add_argument('--dest',type=Path,default=DEFAULT_DEST)
    args=p.parse_args()
    records=[json.loads(l) for l in args.log.read_text(encoding='utf-8').splitlines() if l.strip()]
    if not records: raise ValueError('No heldout validation records found')
    data=summarize(records[-1]);args.dest.mkdir(parents=True,exist_ok=True)
    (args.dest/'LATEST_HELDOUT_EVIDENCE.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (args.dest/'LATEST_HELDOUT_EVIDENCE.md').write_text(markdown(data),encoding='utf-8')
    print(f"Saved judging evidence for {data['run_id']}: {data['mean_latency_reduction_fraction']*100:.2f}% lower mean warm CPU inference; accuracy {data['baseline_accuracy']*100:.2f}% to {data['candidate_accuracy']*100:.2f}%.")
    print(f'See {args.dest}/LATEST_HELDOUT_EVIDENCE.md')

if __name__=='__main__': main()
