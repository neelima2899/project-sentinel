
import numpy as np
from environment_v4 import SentinelEnv

# Test 1: A normal battle can naturally trigger Power Strike.
env = SentinelEnv()
obs, _ = env.reset(seed=42)

warnings = 0
strikes = 0

for step in range(160):
    # Simple baseline: approach and attack the Player.
    distance = np.linalg.norm(env.p - env.e)
    action = 2 if distance <= env.ATTACK_RANGE else 0

    old_hp = env.hp[2]
    obs, reward, terminated, truncated, info = env.step(action)

    if env.power_strike_active:
        warnings += 1

    # Detect a completed Power Strike by its cooldown reset.
    if env.power_strike_cooldown == env.POWER_STRIKE_COOLDOWN:
        strikes += 1

    if terminated or truncated:
        break

print("Battle winner:", info["winner"])
print("Battle length:", info["steps"])
print("Warning observations:", warnings)
print("Completed Power Strikes:", strikes)
print("Adversary remaining HP:", env.hp[2])

assert obs.shape == (15,)
assert warnings > 0, "No Power Strike warning occurred"
print("PASS: Natural Power Strike warning")

# Test 2: A defeated Guardian must not complete its strike.
env.reset(seed=42)
env.hp[1] = 0
env.power_strike_active = True
env.power_strike_countdown = 1

before = env.hp[2]
damage = env._update_power_strike()

assert damage == 0
assert env.hp[2] == before
print("PASS: Defeated Guardian cannot deal Power Strike damage")

print("\nSENTINEL v4 battle tests passed!")
