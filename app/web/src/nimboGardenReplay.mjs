/** Historical evidence must be scoped to the selected replay event. */
export function replayEventsUntil(events, selectedSeq, historical) {
  const list = Array.isArray(events) ? events : [];
  if (!historical) return list;
  if (!Number.isInteger(selectedSeq) || selectedSeq < 1) return [];
  return list.filter(e => Number.isInteger(e?.seq) && e.seq <= selectedSeq);
}
export function isAtRecordedEnd(events, selectedSeq, historical) {
  if (!historical) return true;
  const list = Array.isArray(events) ? events : [];
  return !!list.length && list.at(-1)?.seq === selectedSeq;
}
