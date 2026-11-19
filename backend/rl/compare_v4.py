
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from environment_v4 import SentinelEnv

ROOT = Path(__file__).resolve().parent.parent
model = PPO.load(str(ROOT / "models" / "sentinel_ppo_v4.zip"))

EPISODES = 500

def aggressive(env, obs):
    distance = np.linalg.norm(env.p - env.e)
    return 2 if distance <= env.ATTACK_RANGE else 0

def tactical(env, obs):
    if env.power_strike_active and env.hp[1] > 0:
        if np.linalg.norm(env.g - env.e) <= env.POWER_STRIKE_RADIUS:
            return 4
    return aggressive(env, obs)

policies = {
    "Aggressive baseline": aggressive,
    "Tactical baseline": tactical,
    "PPO v4": lambda env, obs: int(model.predict(obs, deterministic=True)[0]),
}

for name, policy in policies.items():
    wins = 0
    health = []
    retreats = []
    warning_retreats = []
    lengths = []
    power_hits = 0

    for seed in range(10000, 10000 + EPISODES):
        env = SentinelEnv()
        obs, _ = env.reset(seed=seed)
        done = False
        episode_warning_retreats = 0

        while not done:
            action = policy(env, obs)

            if env.power_strike_active and action == 4:
                episode_warning_retreats += 1

            old_hp = env.hp[2]
            obs, _, terminated, truncated, info = env.step(action)

            # A completed strike that deals damage resets cooldown.
            # Check its damage using the Guardian damage difference,
            # while acknowledging normal Guardian attacks may also occur.
            if (
                env.power_strike_cooldown == env.POWER_STRIKE_COOLDOWN
                and old_hp - env.hp[2] >= env.POWER_STRIKE_DAMAGE
            ):
                power_hits += 1

            done = terminated or truncated

        wins += info["winner"] == "adversary"
        health.append(env.hp[2])
        retreats.append(info["action_counts"][4])
        warning_retreats.append(episode_warning_retreats)
        lengths.append(info["steps"])

    print("\n" + name)
    print(f"Win rate: {wins / EPISODES:.1%}")
    print(f"Mean remaining health: {np.mean(health):.1f}")
    print(f"Mean retreats: {np.mean(retreats):.2f}")
    print(f"Mean warning retreats: {np.mean(warning_retreats):.2f}")
    print(f"Mean battle length: {np.mean(lengths):.1f}")
    print(f"Estimated Power Strike hits: {power_hits}")
