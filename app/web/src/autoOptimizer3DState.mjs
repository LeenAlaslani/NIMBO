/** Strict presentation projection of REAL persisted M7 job events.
 *  No fake progress, inferred agent reasoning, or synthetic completion. */
export const STAGE_WORKER = Object.freeze({
  queued: 'captain', analyze: 'captain', optimize: 'model',
  verify: 'vera', deliver: 'metri',
});
export const FINAL_STATES = Object.freeze([
  'COMPLETED_REVIEW_ONLY', 'COMPLETED_REJECTED', 'FAILED', 'INTERRUPTED',
]);

export function realEvents(events) {
  if (!Array.isArray(events)) return [];
  const seen = new Set();
  return events.filter(e => {
    if (!e || !Number.isSafeInteger(e.seq) || e.seq <= 0 ||
        typeof e.message !== 'string' || typeof e.stage !== 'string' ||
        seen.has(e.seq)) return false;
    seen.add(e.seq);
    return true;
  }).sort((a, b) => a.seq - b.seq);
}

export function eventWorker(event, timeline = []) {
  if (!event) return null;
  if (Object.hasOwn(STAGE_WORKER, event.stage)) return STAGE_WORKER[event.stage];
  // 'error' may not name a worker; stay at the last OBSERVED station.
  const preceding = timeline.filter(e => e.seq < event.seq &&
    Object.hasOwn(STAGE_WORKER, e.stage));
  return preceding.length ? STAGE_WORKER[preceding[preceding.length - 1].stage] : null;
}

export function sceneFromTimeline(job, events, selectedSeq = null) {
  const timeline = realEvents(events);
  const latest = timeline.at(-1) || null;
  const historical = selectedSeq !== null;
  const selected = historical ? timeline.find(e => e.seq === selectedSeq) || null : latest;
  const event = selected || (!historical ? latest : null);
  const finished = Boolean(job && FINAL_STATES.includes(job.state));
  const validJob = Boolean(job && typeof job.job_id === 'string');
  return {
    event, latest, timeline, worker: eventWorker(event, timeline),
    historical, validJob, finished,
    running: Boolean(validJob && !historical && job.state === 'RUNNING' && event),
    verifiedDelivery: Boolean(validJob && job.state === 'COMPLETED_REVIEW_ONLY' &&
       event?.stage === 'deliver' && event?.state === 'COMPLETED_REVIEW_ONLY' && !historical),
    outcome: validJob ? job.state : 'NO_JOB',
  };
}
