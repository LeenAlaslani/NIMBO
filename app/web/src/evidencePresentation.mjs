/** Display-only helpers. No inference or evidence records are modified. */
export function readableRoute(route) {
  if (typeof route !== 'string' || !route.trim()) return 'Not recorded';
  if (route === 'paired_baseline_candidate') return 'Paired baseline comparison';
  return route.trim().replaceAll('_', ' ');
}

export function latestWorkspacePair(records) {
  if (!Array.isArray(records)) return null;
  const paired = records.filter(r => r && r.evidence_kind === 'local_paired_cpu_benchmark' && r.route === 'paired_baseline_candidate');
  if (!paired.length) return null;
  // Use explicit timestamps, not the collector's grouping/order, to pick the newest.
  return paired.reduce((best, entry) => {
    const when = Date.parse(entry.timestamp_utc || '');
    const prev = Date.parse(best.timestamp_utc || '');
    return (Number.isFinite(when) ? when : -Infinity) >= (Number.isFinite(prev) ? prev : -Infinity) ? entry : best;
  });
}

export function latencyPercent(fraction) {
  return typeof fraction === 'number' && Number.isFinite(fraction)
    ? `${Math.abs(fraction * 100).toFixed(1)}%` : 'Not measured';
}
