
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from environment_v4 import SentinelEnv

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sentinel_ppo_v4.zip"
EPISODES = 500

model = PPO.load(str(MODEL_PATH))

wins = 0
health_remaining = []
retreats = []
warning_retreats = []
battle_lengths = []

for i in range(EPISODES):
    env = SentinelEnv()
    obs, _ = env.reset(seed=10000 + i)

    done = False
    episode_warning_retreats = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        # Count retreats specifically during Power Strike warnings.
        if env.power_strike_active and action == 4:
            episode_warning_retreats += 1

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    wins += info["winner"] == "adversary"
    health_remaining.append(env.hp[2])
    retreats.append(info["action_counts"][4])
    warning_retreats.append(episode_warning_retreats)
    battle_lengths.append(info["steps"])

print("\nSENTINEL PPO v4 ? 500 BATTLE EVALUATION")
print(f"Win rate: {wins / EPISODES:.1%}")
print(f"Mean remaining health: {np.mean(health_remaining):.1f}")
print(f"Mean retreats: {np.mean(retreats):.2f}")
print(f"Mean warning retreats: {np.mean(warning_retreats):.2f}")
print(f"Mean battle length: {np.mean(battle_lengths):.1f}")
