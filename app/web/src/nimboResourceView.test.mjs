import test from 'node:test';
import assert from 'node:assert/strict';
import {measured,cpuLabel,mibLabel,fill,points} from './nimboResourceView.mjs';
test('unknown resource data is never shown as 0',()=>{
 assert.equal(measured(null),false);assert.equal(measured(undefined),false);
 assert.equal(cpuLabel(null),'—');assert.equal(mibLabel(NaN),'—');
});
test('CPU over 100 is retained for multicore processes',()=>{
 assert.equal(cpuLabel(235.5),'235.5%');assert.equal(fill(235,100),100);
});
test('history requires two measured points',()=>{
 assert.equal(points([{cpu:null},{cpu:0}], 'cpu'),'');
 assert.match(points([{cpu:0},{cpu:100},{cpu:50}],'cpu'), /^0\.00,48\.00/);
});
