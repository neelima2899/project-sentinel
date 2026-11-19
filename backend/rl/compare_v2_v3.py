
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from environment_v3 import SentinelEnv

ROOT = Path(__file__).resolve().parent.parent
EPISODES = 500

models = {
    "PPO v2": PPO.load(str(ROOT / "models" / "sentinel_ppo_v2.zip")),
    "PPO v3": PPO.load(str(ROOT / "models" / "sentinel_ppo_v3.zip")),
}

for name, model in models.items():
    wins = 0
    health_remaining = []
    retreats = []
    lengths = []

    for i in range(EPISODES):
        env = SentinelEnv()
        obs, _ = env.reset(seed=10000 + i)

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        wins += info["winner"] == "adversary"
        health_remaining.append(env.hp[2])
        retreats.append(info["action_counts"][4])
        lengths.append(info["steps"])

    print(f"\n{name}")
    print(f"Win rate: {wins / EPISODES:.1%}")
    print(f"Mean remaining health: {np.mean(health_remaining):.1f}")
    print(f"Mean retreats: {np.mean(retreats):.2f}")
    print(f"Mean battle length: {np.mean(lengths):.1f}")
