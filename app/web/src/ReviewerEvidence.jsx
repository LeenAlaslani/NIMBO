import React from 'react';

const metrics = [
  ['Mean warm forward latency (ms)', 'mean_warm_forward_ms', 2],
  ['p95 warm forward latency (ms)', 'p95_warm_forward_ms', 2],
  ['Total process CPU time (ms)', 'process_cpu_forward_total_ms', 2],
  ['Labeled accuracy (%)', 'accuracy', 2, 100],
];
const number = (value, digits, multiplier = 1) =>
  Number.isFinite(value) ? (value * multiplier).toFixed(digits) : 'Unavailable';
const result = value => value === true ? 'Passed' : value === false ? 'Failed' : 'Unavailable';

export default function ReviewerEvidence({job}) {
  if (!job?.reason) return null;
  const comparison = job.comparison;
  return <section className="reviewerEvidence" aria-label="Live optimization result" aria-live="polite">
    <h2>Live optimization result</h2>
    <p>{job.reason}</p>
    {comparison && <>
      <p>{comparison.scope === 'heldout' ? 'Held-out' : 'Pilot'} comparison on {comparison.examples} labeled examples over {comparison.rounds} paired rounds.</p>
      <div className="reviewerEvidenceTable">
        <table>
          <thead><tr><th scope="col">Metric</th><th scope="col">PyTorch FP32</th><th scope="col">ONNX Runtime FP32</th></tr></thead>
          <tbody>{metrics.map(([label, key, digits, multiplier]) => <tr key={key}>
            <th scope="row">{label}</th>
            <td>{number(comparison.baseline[key], digits, multiplier)}</td>
            <td>{number(comparison.candidate[key], digits, multiplier)}</td>
          </tr>)}</tbody>
        </table>
      </div>
      <p>Quality gate {result(comparison.gates.quality_pass ?? comparison.gates.quality_gate)} · Performance gate {result(comparison.gates.performance_pass ?? comparison.gates.performance_gate)} · Prediction agreement {number(comparison.gates.prediction_agreement, 2, 100)}%</p>
    </>}
    {job.package && <p>Offline reload passed. Download the review ZIP from the mission controls.</p>}
  </section>;
}
