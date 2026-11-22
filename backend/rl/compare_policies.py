
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
SEED_START = 3000


def rule_based_action(env):
    """Simple tactical baseline using available state."""
    player_distance = np.linalg.norm(env.p - env.e)
    guardian_distance = np.linalg.norm(env.g - env.e)

    # Escape an imminent Guardian Power Strike.
    if (
        env.power_strike_active
        and guardian_distance <= env.POWER_STRIKE_RADIUS
    ):
        return 4

    # Attack the Player when within range and ready.
    if (
        player_distance <= env.ATTACK_RANGE
        and env.steps + 1 >= env.next_attack_step
    ):
        return 2

    # Otherwise approach the Player.
    return 0


def evaluate(name, policy, seeds):
    wins = Counter()
    actions = Counter()
    rewards = []
    steps_taken = []
    strike_hits = 0
    strike_damage = 0
    successful_attacks = []

    for seed in seeds:
        env = SentinelEnv()
        observation, _ = env.reset(seed=seed)
        total_reward = 0

        while True:
            action = int(policy(env, observation))
            actions[action] += 1

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward
            event = info["power_strike"]

            if event["resolved"] and event["hit"]:
                strike_hits += 1
                strike_damage += event["damage"]

            if terminated or truncated:
                break

        wins[info["winner"]] += 1
        rewards.append(total_reward)
        steps_taken.append(info["steps"])
        successful_attacks.append(info["successful_attacks"])
        env.close()

    print(f"\n{name}")
    print("-" * 40)
    print(f"Adversary wins: {wins['adversary']}/{len(seeds)}")
    print(f"Allies wins: {wins['allies']}/{len(seeds)}")
    print(f"Timeouts: {wins['timeout']}/{len(seeds)}")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Average battle steps: {np.mean(steps_taken):.1f}")
    print(f"Average successful attacks: {np.mean(successful_attacks):.1f}")
    print(f"Power Strike hits: {strike_hits}")
    print(f"Power Strike damage: {strike_damage}")

    total = sum(actions.values())
    print("Action distribution:")
    for action in range(5):
        print(f"  {action}: {100 * actions[action] / max(1, total):.1f}%")


def main():
    model = PPO.load(str(MODEL_PATH), device="cpu")
    seeds = list(range(SEED_START, SEED_START + EPISODES))
    rng = np.random.default_rng(2026)

    evaluate(
        "Trained PPO",
        lambda env, obs: model.predict(
            obs, deterministic=True
        )[0],
        seeds,
    )

    evaluate(
        "Rule-based tactical baseline",
        lambda env, obs: rule_based_action(env),
        seeds,
    )

    evaluate(
        "Random baseline",
        lambda env, obs: rng.integers(env.action_space.n),
        seeds,
    )


if __name__ == "__main__":
    main()
