
import numpy as np
from environment_v4 import SentinelEnv
from stable_baselines3.common.env_checker import check_env

# Test 1: Gymnasium compatibility
env = SentinelEnv()
check_env(env)
print("PASS: Gymnasium environment validation")

# Test 2: Observation dimensions
obs, _ = env.reset(seed=42)
assert obs.shape == (15,)
print("PASS: 15-dimensional observations")

# Test 3: Guardian Power Strike
env.reset(seed=42)
env.g = np.array([0., 0.], dtype=np.float32)
env.e = np.array([1., 0.], dtype=np.float32)

assert env._update_power_strike() == 0
assert env.power_strike_countdown == 2
assert env._update_power_strike() == 0
assert env.power_strike_countdown == 1
assert env._update_power_strike() == 35
assert env.hp[2] == 105
print("PASS: 35-HP Power Strike")

# Test 4: Cooldown
assert env.power_strike_cooldown == 8

for _ in range(8):
    assert env._update_power_strike() == 0

assert env.power_strike_cooldown == 0
assert not env.power_strike_active
print("PASS: Power Strike cooldown")

# Test 5: Tactical retreat
env.reset(seed=42)
env.p = np.array([-6., -6.], dtype=np.float32)
env.g = np.array([0., 0.], dtype=np.float32)
env.e = np.array([2., 0.], dtype=np.float32)
env.power_strike_active = True
env.power_strike_countdown = 2

for _ in range(2):
    _, _, terminated, truncated, _ = env.step(4)
    assert not (terminated or truncated)

assert env.hp[2] == 140
assert np.linalg.norm(env.g - env.e) > env.POWER_STRIKE_RADIUS
print("PASS: Tactical retreat avoids Power Strike")

print("\nALL SENTINEL v4 TESTS PASSED!")
