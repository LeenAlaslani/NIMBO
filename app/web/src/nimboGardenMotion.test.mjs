import test from 'node:test';
import assert from 'node:assert/strict';
import {facingYaw, walkStep, shortestAngle, stationFacingYaw, dampYaw} from './nimboGardenMotion.mjs';

test('PICO turns east instead of always looking into the camera',()=> {
  const p=walkStep({x:0,z:0,yaw:0},{x:8,z:0},.05);
  assert.ok(p.x>0 && Math.abs(p.z)<.00001);
  assert.ok(p.yaw>0);
});
test('PICO turns west on a reverse journey with shortest angle',()=> {
  const p=walkStep({x:0,z:0,yaw:0},{x:-8,z:0},.05);
  assert.ok(p.x<0);
  assert.ok(p.yaw<0);
});
test('Walking does not teleport or overshoot the destination',()=> {
  let p={x:0,z:0,yaw:0};
  for(let i=0;i<60;i++)p=walkStep(p,{x:1,z:1},.05);
  assert.ok(Math.hypot(p.x-1,p.z-1)<.075);
  assert.ok(p.x<=1 && p.z<=1);
});
test('No movement when already at the destination',()=> {
  const p=walkStep({x:2,z:3,yaw:1},{x:2,z:3},.016);
  assert.equal(p.moving,false);
  assert.equal(p.stepDistance,0);
});
test('Large browser frame gaps cannot teleport the character',()=> {
  const p=walkStep({x:0,z:0,yaw:0},{x:10,z:10},30);
  assert.ok(Math.hypot(p.x,p.z)<.3);
});
test('Yaw damping chooses the shortest wraparound turn',()=> {
  assert.ok(Math.abs(shortestAngle(3.12,-3.12))<.1);
  assert.ok(Math.abs(dampYaw(3.12,-3.12,.05)-3.12)<.1);
});
test('Interaction orientation points toward recorded peer',()=> {
  assert.equal(stationFacingYaw(-2,0,0,null),-2);
  assert.ok(Math.abs(stationFacingYaw(-2,0,0,{x:2,z:0})-Math.PI/2)<.001);
});
test('Facing calculation uses local +Z as forward',()=> {
  assert.equal(facingYaw(0,0,0,1),0);
  assert.ok(Math.abs(facingYaw(0,0,1,0)-Math.PI/2)<.001);
});
