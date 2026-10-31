import test from 'node:test';
import assert from 'node:assert/strict';
import {readableRoute,latestWorkspacePair,latencyPercent} from '../src/evidencePresentation.mjs';

test('display route never breaks machine identifier',()=>{
  assert.equal(readableRoute('paired_baseline_candidate'),'Paired baseline comparison');
  assert.equal(readableRoute('same_model_onnx_fp32_heldout'),'same model onnx fp32 heldout');
  assert.equal(readableRoute(undefined),'Not recorded');
});
test('only real workspace pairs and newest timestamp selected',()=>{
  const rows=[
    {id:'older',route:'paired_baseline_candidate',evidence_kind:'local_paired_cpu_benchmark',timestamp_utc:'2026-10-30T00:00:00Z'},
    {id:'latest',route:'paired_baseline_candidate',evidence_kind:'local_paired_cpu_benchmark',timestamp_utc:'2026-10-31T12:00:00Z'},
    {id:'heldout',route:'same_model_onnx_fp32_heldout',evidence_kind:'heldout_repeated_cpu_validation',timestamp_utc:'2026-10-31T13:00:00Z'},
    {id:'runtime',route:'paired_baseline_candidate',evidence_kind:'runtime_measurement',timestamp_utc:'2026-10-31T14:00:00Z'},
  ];
  assert.equal(latestWorkspacePair(rows).id,'latest');
  assert.equal(latestWorkspacePair([]),null);
});
test('latency formatting does not turn missing evidence into zero savings',()=>{
  assert.equal(latencyPercent(0.48),'48.0%');
  assert.equal(latencyPercent(-0.2),'20.0%');
  assert.equal(latencyPercent(null),'Not measured');
  assert.equal(latencyPercent(NaN),'Not measured');
});
