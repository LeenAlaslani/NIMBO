/** Pure derivations of M7 backend evidence; never manufacture progress. */
const TERMINAL = new Set(['COMPLETED_REVIEW_ONLY', 'COMPLETED_REJECTED', 'FAILED', 'INTERRUPTED']);
const JOB_ID = /^[0-9a-f]{32}$/;
export function stageState(stage, events, currentState) {
  const matching = events.filter(e => e?.stage === stage);
  if (!matching.length) return 'pending';
  if (matching.some(e => e.state === 'COMPLETED_REJECTED' || e.state === 'FAILED' || e.state === 'INTERRUPTED')) return 'rejected';
  if (stage === 'deliver' && currentState === 'COMPLETED_REVIEW_ONLY') return 'verified';
  const last = events[events.length - 1];
  if (last?.stage === stage && !TERMINAL.has(currentState)) return 'running';
  return 'observed';
}
export function canReviewDownload(job) {
  return job?.state === 'COMPLETED_REVIEW_ONLY' && job?.eligible_for_review_download === true && JOB_ID.test(job.job_id || '');
}
