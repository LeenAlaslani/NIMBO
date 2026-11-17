
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  nextRecordedSeq,m12PickWorker
} from './nimboM12Replay.mjs';

const events=[
  {seq:1,stage:'queued',message:'Mission accepted'},
  {seq:2,stage:'analyze',
   message:'M7 engine: PyTorch baseline measured'},
  {seq:3,stage:'verify',
   message:'VERA decision: approve_for_review'},
  {seq:4,stage:'verify',
   message:'METRI decision: record_evidence',
   evidence:{agent_role:'METRI'}},
  {seq:5,stage:'deliver',message:'Package verified'}
];

test('Replay follows actual event sequence',()=>{
  assert.equal(nextRecordedSeq(events,null),1);
  assert.equal(nextRecordedSeq(events,1),2);
  assert.equal(nextRecordedSeq(events,4),5);
  assert.equal(nextRecordedSeq(events,5),null);
});

test('Replay never invents missing events',()=>{
  assert.equal(nextRecordedSeq([],null),null);
  assert.equal(nextRecordedSeq(events,2),3);
});

test('METRI is not incorrectly mapped to VERA',()=>{
  assert.equal(
    m12PickWorker(events[3],events,()=> 'vera'),'metri'
  );
});

test('VERA and baseline map to their correct stations',()=>{
  assert.equal(
    m12PickWorker(events[2],events,()=> 'captain'),'vera'
  );
  assert.equal(
    m12PickWorker(events[1],events,()=> 'captain'),'model'
  );
});

test('Unknown events preserve existing mapping',()=>{
  assert.equal(
    m12PickWorker(events[4],events,()=> 'bundle'),'bundle'
  );
});
