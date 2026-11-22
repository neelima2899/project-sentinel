
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from backend.rl.environment_v4 import SentinelEnv


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "sentinel_ppo_v4.zip"
)

EPISODES = 100


def evaluate(policy_name, model=None):
    wins = Counter()
    actions = Counter()
    rewards = []
    damages = []
    rng = np.random.default_rng(2026)

    for episode in range(EPISODES):
        env = SentinelEnv()
        obs, _ = env.reset(seed=1000 + episode)

        done = False
        total_reward = 0

        while not done:
            if model is None:
                action = int(rng.integers(env.action_space.n))
            else:
                action, _ = model.predict(
                    obs, deterministic=True
                )
                action = int(action)

            actions[action] += 1

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward
            done = terminated or truncated

        wins[info["winner"]] += 1
        rewards.append(total_reward)
        damages.append(info["damage"]["enemy"])

        env.close()

    total_actions = sum(actions.values())

    print(f"\n{policy_name}")
    print("-" * 35)
    print(f"Adversary wins: {wins['adversary']}")
    print(f"Allies win:     {wins['allies']}")
    print(f"Timeouts:       {wins['timeout']}")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Average damage: {np.mean(damages):.2f}")

    print("\nAction distribution:")
    for action in range(5):
        percent = 100 * actions[action] / max(1, total_actions)
        print(f"Action {action}: {percent:.1f}%")


if __name__ == "__main__":
    model = PPO.load(str(MODEL_PATH), device="cpu")

    evaluate("Trained PPO", model)
    evaluate("Random baseline")