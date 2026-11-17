/**
 * NIMBO M13.3 — pure, deterministic 3D motion math.
 * Positions are Three.js world X/Z; characters look along local +Z.
 * Animation is presentation only, never a model-execution claim.
 */
export function shortestAngle(from, to) {
  if (!Number.isFinite(from) || !Number.isFinite(to)) return 0;
  return Math.atan2(Math.sin(to - from), Math.cos(to - from));
}

export function facingYaw(fromX, fromZ, toX, toZ) {
  return Math.atan2(toX - fromX, toZ - fromZ);
}

export function dampYaw(current, desired, dt, turnRate = 7) {
  const step = Math.min(1, Math.max(0, dt) * turnRate);
  return current + shortestAngle(current, desired) * step;
}

/** Frame-rate independent, speed-limited movement with no teleport/overshoot. */
export function walkStep(pose, dest, dt, speed = 3.1, turnRate = 8) {
  const x = Number(pose.x) || 0;
  const z = Number(pose.z) || 0;
  const yaw = Number(pose.yaw) || 0;
  const dx = dest.x - x;
  const dz = dest.z - z;
  const distance = Math.hypot(dx, dz);
  const seconds = Math.max(0, Math.min(.08, Number(dt) || 0));
  if (!Number.isFinite(distance) || distance < .075 || seconds === 0) {
    return {x, z, yaw, moving: false, stepDistance: 0};
  }
  const desiredYaw = facingYaw(x, z, dest.x, dest.z);
  const nextYaw = dampYaw(yaw, desiredYaw, seconds, turnRate);
  // Arrival ease: natural deceleration near the station.
  const easedSpeed = speed * Math.min(1, Math.max(.35, distance / 1.1));
  const stepDistance = Math.min(distance, easedSpeed * seconds);
  return {
    x: x + dx / distance * stepDistance,
    z: z + dz / distance * stepDistance,
    yaw: nextYaw,
    moving: true,
    stepDistance
  };
}

/** Local station actors face screens or current recorded peer handoffs. */
export function stationFacingYaw(workYaw, agentX, agentZ, peer) {
  if (!peer) return workYaw;
  return facingYaw(agentX, agentZ, peer.x, peer.z);
}
