import {test} from 'node:test';
import {strict as assert} from 'node:assert';
import {missionWorkerForEvent,MISSION_FLOW} from './unifiedWorkerMap.mjs';
const ev=(seq,stage,state='RUNNING')=>({seq,stage,state,message:stage});
test('Optimize mission uses only selected real task roles, adds FORGE and BUNDLE',()=>{
 assert.deepEqual(MISSION_FLOW,['captain','forge','vera','bundle']);
 assert.equal(missionWorkerForEvent(ev(1,'analyze')),'captain');
 assert.equal(missionWorkerForEvent(ev(2,'optimize')),'forge');
 assert.equal(missionWorkerForEvent(ev(3,'verify')),'vera');
 assert.equal(missionWorkerForEvent(ev(4,'deliver')),'bundle');
});
test('Error/unknown event inherits last recorded station only',()=>{
 const t=[ev(1,'analyze'),ev(2,'verify'),ev(3,'error','FAILED')];
 assert.equal(missionWorkerForEvent(t[2],t),'vera');
 assert.equal(missionWorkerForEvent(ev(1,'unknown'),[]),null);
});
test('No fake station when missing events',()=>{assert.equal(missionWorkerForEvent(null),null)});
