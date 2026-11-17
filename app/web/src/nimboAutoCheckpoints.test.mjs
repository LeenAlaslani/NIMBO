import test from 'node:test';
import assert from 'node:assert/strict';
import {recordedCheckpoints,jobHeadline} from './nimboAutoCheckpoints.mjs';
const e=message=>({message});
test('completed engine evidence is recognized independently of agent failure',()=>{
 const s=recordedCheckpoints([
  e('M7 engine: PyTorch baseline measured'),
  e('M7 engine: ONNX paired benchmark passed local gates'),
  e('M7 engine: Heldout quality and performance gates passed'),
  e('M7 engine: Offline model package verified; review only'),
  e('Agent execution failed; inspect protected local log (AgentDecisionError)'),
 ]);
 assert.deepEqual(s.map(x=>x.state),['RECORDED','RECORDED','RECORDED','RECORDED']);
 assert.equal(jobHeadline({state:'FAILED'},'agentic'),'AGENT / ENGINE FAILURE');
});
test('recorded rejection cannot be falsely presented as a success',()=>{
 const s=recordedCheckpoints([
  e('M7 engine: Starting ONNX FP32 graph export and paired benchmark'),
  e('M7 engine: ONNX pilot failed required quality/mean-speed checks; no heldout promotion')
 ]);
 assert.equal(s[1].state,'REJECTED');
 assert.equal(s[2].state,'PENDING');
 assert.equal(s[3].state,'PENDING');
});
test('no evidence is no progress; replay is labelled historical',()=>{
 assert.deepEqual(recordedCheckpoints([]).map(x=>x.state),['PENDING','PENDING','PENDING','PENDING']);
 assert.match(jobHeadline({state:'RUNNING'},'historical'),/NOT LIVE/);
});
