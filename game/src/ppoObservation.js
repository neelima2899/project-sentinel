const clamp = value => Math.max(-1, Math.min(1, value));

export function buildObservation(s) {
  const px = s.p.x - s.e.x;
  const pz = s.p.z - s.e.z;
  const gx = s.g.x - s.e.x;
  const gz = s.g.z - s.e.z;

  const playerDistance = Math.hypot(px, pz);
  const guardianDistance = Math.hypot(gx, gz);

  const steps = s.aiSteps ?? 0;
  const nextAttackStep = s.nextAttackStep ?? 1;

  return [
    px / 16,
    pz / 16,
    gx / 16,
    gz / 16,
    s.player / 100,
    s.guardian / 120,
    s.enemy / 140,
    playerDistance / 16,
    guardianDistance / 16,
    steps / 160,
    Number(steps >= nextAttackStep),
    Math.min(1, Math.max(0, nextAttackStep - steps) / 3),
    Number(Boolean(s.powerStrikeActive)),
    (s.powerStrikeCountdown ?? 0) / 2,
    (s.powerStrikeCooldown ?? 0) / 8,
  ].map(clamp);
}
