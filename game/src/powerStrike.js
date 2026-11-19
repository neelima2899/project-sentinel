export function updatePowerStrike(s) {
  const radius = 2.5;
  const damage = 35;
  const cooldown = 8;
  const distance = Math.hypot(s.g.x - s.e.x, s.g.z - s.e.z);

  if (s.player <= 0 || s.guardian <= 0 || s.enemy <= 0) {
    return;
  }

  if (s.powerStrikeActive) {
    s.powerStrikeCountdown -= 1;

    if (s.powerStrikeCountdown <= 0) {
      if (distance <= radius) {
        const dealt = Math.min(damage, s.enemy);
        s.enemy = Math.max(0, s.enemy - damage);
        if (s.damageStats) {
          s.damageStats.guardian += dealt;
          if (dealt > 0) s.damageStats.powerStrikes += 1;
        }
        s.message = "Guardian Power Strike hit Adversary for 35 damage";
      } else {
        s.message = "Adversary escaped Guardian Power Strike";
      }

      s.powerStrikeActive = false;
      s.powerStrikeCountdown = 0;
      s.powerStrikeCooldown = cooldown;
    }

    return;
  }

  if (s.powerStrikeCooldown > 0) {
    s.powerStrikeCooldown -= 1;
    return;
  }

  if (distance <= radius) {
    s.powerStrikeActive = true;
    s.powerStrikeCountdown = 2;
    s.message = "WARNING: Guardian charging Power Strike!";
  }
}
