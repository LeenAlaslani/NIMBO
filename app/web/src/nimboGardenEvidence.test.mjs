import test from 'node:test';
import assert from 'node:assert/strict';
import {gardenEvidence, gardenHandoffs} from './nimboGardenEvidence.mjs';

test('DistilBERT engine success remains separate from agent failure', () => {
  const result = gardenEvidence([
    {message:'PyTorch baseline measured'},
    {message:'ONNX paired benchmark passed local gates'},
    {message:'Heldout quality and performance gates passed'},
    {message:'Offline model package verified; review only'},
    {message:'Agent execution failed (AgentDecisionError)'}
  ]);
  assert.equal(result.package, true);
  assert.equal(result.agentFailed, true);
});

test('Starting a stage is not evidence of success', () => {
  const result = gardenEvidence([
    {message:'Starting new heldout paired evaluation'},
    {message:'Creating and independently offline-verifying ONNX FP32 package'}
  ]);
  assert.equal(result.heldout, false);
  assert.equal(result.package, false);
});

test('Handoffs only use explicit recorded worker names', () => {
  assert.deepEqual(gardenHandoffs([
    {message:'CAPTAIN accepted a mission'},
    {message:'FORGE selected the authorized tool'},
    {message:'VERA checking evidence'}
  ]), [['captain','forge'],['forge','vera']]);
});
