export function gardenEvidence(events = []) {
  const messages = events.map(e => String(e?.message || ''));
  const has = pattern => messages.some(m => pattern.test(m));
  return {
    baseline: has(/PyTorch baseline measured/i),
    onnx: has(/ONNX paired benchmark passed local gates/i),
    heldout: has(/Heldout quality and performance gates passed/i),
    package: has(/Offline model package verified; review only/i),
    agentFailed: has(/Agent execution failed/i),
    agentReviewed: has(/VERA decision: approve_for_review/i),
    count: events.length
  };
}

const ROLES = /\b(CAPTAIN|FORGE|MAKO|VERA|METRI|BUNDLE|TOKI|ROUTA)\b/i;

export function gardenHandoffs(events = []) {
  const roles = events
    .map(e => String(e?.message || '').match(ROLES)?.[1]?.toLowerCase())
    .filter(Boolean);
  const links = [];
  for (let i = 1; i < roles.length; i++) {
    if (roles[i] !== roles[i - 1]) {
      links.push([roles[i - 1], roles[i]]);
    }
  }
  return links.slice(-5);
}
