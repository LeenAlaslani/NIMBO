import test from 'node:test';
import assert from 'node:assert/strict';
import { stageState, canReviewDownload } from './autoOptimizerJobView.mjs';
const id='a5a70c3cfd64405a918fa419985c9339';
const event=(stage,state='RUNNING')=>({stage,state});
test('no backend events means no progress',()=>{
  for(const stage of ['analyze','optimize','verify','deliver']) assert.equal(stageState(stage,[], 'QUEUED'),'pending');
});
test('stages follow genuine events, not a progress timer',()=>{
  const events=[event('analyze'),event('optimize'),event('verify')];
  assert.equal(stageState('analyze',events,'RUNNING'),'observed');
  assert.equal(stageState('verify',events,'RUNNING'),'running');
  assert.equal(stageState('deliver',events,'RUNNING'),'pending');
});
test('terminal failure in verify never appears verified',()=>{
  const events=[event('analyze'),event('optimize'),event('verify','COMPLETED_REJECTED')];
  assert.equal(stageState('verify',events,'COMPLETED_REJECTED'),'rejected');
  assert.equal(stageState('deliver',events,'COMPLETED_REJECTED'),'pending');
});
test('delivery is verified only if backend produced the terminal state',()=>{
  assert.equal(stageState('deliver',[event('deliver')],'RUNNING'),'running');
  assert.equal(stageState('deliver',[event('deliver','COMPLETED_REVIEW_ONLY')],'COMPLETED_REVIEW_ONLY'),'verified');
});
test('no download without a real passed review-only state and flag',()=>{
  assert.equal(canReviewDownload(null),false);
  assert.equal(canReviewDownload({job_id:id,state:'QUEUED',eligible_for_review_download:true}),false);
  assert.equal(canReviewDownload({job_id:id,state:'COMPLETED_REJECTED',eligible_for_review_download:true}),false);
  assert.equal(canReviewDownload({job_id:id,state:'COMPLETED_REVIEW_ONLY',eligible_for_review_download:false}),false);
  assert.equal(canReviewDownload({job_id:'../secret',state:'COMPLETED_REVIEW_ONLY',eligible_for_review_download:true}),false);
  assert.equal(canReviewDownload({job_id:id,state:'COMPLETED_REVIEW_ONLY',eligible_for_review_download:true}),true);
});
