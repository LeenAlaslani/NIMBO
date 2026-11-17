import {test} from 'node:test';
import assert from 'node:assert/strict';
import {ROOM_ORDER,roomForMission,workerLocations,eventWorker,outcome,workerDialogue,finalJob} from './livingLabPlan.mjs';
test('three rooms for task families, not for individual model IDs',()=>{
 assert.deepEqual(ROOM_ORDER,['execution','forge','vision']);
 assert.equal(roomForMission('auto_onnx'),'forge');
 assert.equal(roomForMission('compare'),'execution');
 assert.equal(roomForMission('live','image'),'vision');
});
test('PICO and workers have unique stations in one world',()=>{
 for(const room of ROOM_ORDER){const w=workerLocations(room);assert.equal(Object.keys(w).length,8);assert.notEqual(w.captain.x,w.vera.x);assert.equal(w.captain.room,room)}
});
test('all stage mappings come only from observed events',()=>{
 const t=[{seq:1,stage:'analyze',message:'started'},{seq:2,stage:'optimize',message:'built'},{seq:3,stage:'verify',message:'checked'},{seq:4,stage:'error',message:'failed'}];
 assert.equal(eventWorker(t[0],t),'captain');assert.equal(eventWorker(t[1],t),'forge');assert.equal(eventWorker(t[2],t),'vera');assert.equal(eventWorker(t[3],t),'vera');assert.equal(eventWorker(null),null);
});
test('rejection cannot masquerade as success or download eligibility',()=>{
 assert.equal(outcome({state:'COMPLETED_REJECTED'}).kind,'rejected');
 assert.equal(outcome({state:'FAILED'}).kind,'failed');
 assert.equal(outcome({state:'COMPLETED_REVIEW_ONLY'}).kind,'verified');
 assert.equal(finalJob({state:'RUNNING'}),false);
});
test('dialogue must label unrecorded briefings',()=>{
 const intro=workerDialogue('forge',null,null);assert.equal(intro.recorded,false);
 const ev={seq:1,stage:'optimize',state:'RUNNING',message:'ONNX export initiated'};
 const reply=workerDialogue('forge',ev,{state:'RUNNING'});assert.equal(reply.recorded,true);assert.match(reply.message,/ONNX export/);
});
