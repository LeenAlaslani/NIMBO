import test from 'node:test';
import assert from 'node:assert/strict';
import { sceneFromTimeline, realEvents, eventWorker } from './autoOptimizer3DState.mjs';
const events = [
  {seq:1,stage:'queued',state:'QUEUED',message:'Accepted'},
  {seq:2,stage:'analyze',state:'RUNNING',message:'Baseline'},
  {seq:3,stage:'optimize',state:'RUNNING',message:'ONNX export'},
  {seq:4,stage:'verify',state:'RUNNING',message:'Heldout'},
  {seq:5,stage:'deliver',state:'COMPLETED_REVIEW_ONLY',message:'Verified offline'},
];
test('No job or events -> no fabricated animation', () => {
 const s = sceneFromTimeline(null, [], null);
 assert.equal(s.worker,null);assert.equal(s.running,false);assert.equal(s.verifiedDelivery,false);
});
test('Live event maps to same named real worker', () => {
 const s = sceneFromTimeline({job_id:'j',state:'RUNNING'},events.slice(0,3));
 assert.equal(s.worker,'model');assert.equal(s.event.message,'ONNX export');assert.equal(s.running,true);
});
test('Offline reload is required to celebrate verified delivery', () => {
 assert.equal(sceneFromTimeline({job_id:'j',state:'COMPLETED_REVIEW_ONLY'},events).verifiedDelivery,true);
 assert.equal(sceneFromTimeline({job_id:'j',state:'COMPLETED_REVIEW_ONLY'},events,3).verifiedDelivery,false);
 assert.equal(sceneFromTimeline({job_id:'j',state:'COMPLETED_REJECTED'},events).verifiedDelivery,false);
});
test('Historical selection is not live, even when job running', () => {
 const s=sceneFromTimeline({job_id:'j',state:'RUNNING'},events,2);
 assert.equal(s.worker,'captain');assert.equal(s.historical,true);assert.equal(s.running,false);
});
test('Rejected at VERIFY stays VERA and never celebrates', () => {
 const e=[...events.slice(0,4),{seq:5,stage:'verify',state:'COMPLETED_REJECTED',message:'Rejected gates'}];
 const s=sceneFromTimeline({job_id:'j',state:'COMPLETED_REJECTED'},e);
 assert.equal(s.worker,'vera');assert.equal(s.verifiedDelivery,false);assert.equal(s.outcome,'COMPLETED_REJECTED');
});
test('Generic error remains at last observed stage; no invented worker', () => {
 const e=[...events.slice(0,3),{seq:4,stage:'error',state:'FAILED',message:'Execution failed'}];
 assert.equal(eventWorker(e.at(-1),e),'model');
 assert.equal(sceneFromTimeline({job_id:'j',state:'FAILED'},e).verifiedDelivery,false);
});
test('Only real, uniquely sequenced records are accepted', () => {
 const s=realEvents([{seq:3,stage:'optimize',message:'Actual'}, {seq:3,stage:'deliver',message:'Duplicate'},
   {seq:4,stage:'deliver',message:2},{seq:-4,stage:'analyze',message:'Bad'},
   {seq:1,stage:'analyze',message:'Before'}]);
 assert.deepEqual(s.map(x=>x.seq),[1,3]);
});
test('Unknown stage does not invent a station or progress', () => {
 const s=sceneFromTimeline({job_id:'j',state:'RUNNING'},[{seq:1,stage:'nonsense',state:'RUNNING',message:'Observed'}]);
 assert.equal(s.worker,null);assert.equal(s.running,true);assert.equal(s.verifiedDelivery,false);
});
