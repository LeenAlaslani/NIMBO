import test from 'node:test';
import assert from 'node:assert/strict';
import {replayEventsUntil,isAtRecordedEnd} from './nimboGardenReplay.mjs';
const events=[{seq:1,message:'queued'},{seq:4,message:'FORGE started'},
 {seq:9,message:'PyTorch baseline measured'},
 {seq:17,message:'Agent execution failed'}];
test('Replay at event 4 cannot display future benchmark or failure',()=> {
 assert.deepEqual(replayEventsUntil(events,4,true).map(x=>x.seq),[1,4]);
});
test('Live uses all real recorded events so far',()=> {
 assert.deepEqual(replayEventsUntil(events,null,false),events);
});
test('Replay before first event shows no future evidence',()=> {
 assert.deepEqual(replayEventsUntil(events,null,true),[]);
});
test('Final job outcome only at the final recorded frame',()=> {
 assert.equal(isAtRecordedEnd(events,9,true),false);
 assert.equal(isAtRecordedEnd(events,17,true),true);
});
test('Malformed inputs never introduce future evidence',()=> {
 assert.deepEqual(replayEventsUntil(null,5,true),[]);
 assert.deepEqual(replayEventsUntil(events,'17',true),[]);
});
