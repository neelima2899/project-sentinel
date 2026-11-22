
from collections import Counter
from pathlib import Path

from stable_baselines3 import PPO

from backend.rl.environment_v4 import SentinelEnv


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "sentinel_ppo_v4.zip"
)

EPISODES = 100


def evaluate_power_strike():
    model = PPO.load(str(MODEL_PATH), device="cpu")

    actions = Counter()
    strikes_started = 0
    strikes_resolved = 0
    strikes_hit = 0
    strikes_missed = 0
    strikes_interrupted = 0
    total_strike_damage = 0
    warning_decisions = 0
    episodes_with_warning = 0

    for episode in range(EPISODES):
        env = SentinelEnv()
        obs, _ = env.reset(seed=2000 + episode)

        saw_warning = False
        done = False

        while not done:
            charging_before = env.power_strike_active

            action, _ = model.predict(
                obs, deterministic=True
            )
            action = int(action)

            if charging_before:
                saw_warning = True
                warning_decisions += 1
                actions[action] += 1

            obs, _, terminated, truncated, info = (
                env.step(action)
            )

            event = info["power_strike"]

            if event["started"]:
                strikes_started += 1

            if event["resolved"]:
                strikes_resolved += 1

                if event["hit"]:
                    strikes_hit += 1
                else:
                    strikes_missed += 1

                total_strike_damage += event["damage"]

            done = terminated or truncated

        if saw_warning:
            episodes_with_warning += 1

        # An active strike at episode end never resolved.
        if env.power_strike_active:
            strikes_interrupted += 1

        env.close()

    print("\nSENTINEL — VERIFIED POWER STRIKE EVALUATION")
    print("-" * 45)

    print(f"Episodes: {EPISODES}")
    print(f"Episodes with warning: {episodes_with_warning}")
    print(f"Warning-time decisions: {warning_decisions}")

    print("\nActions during warning:")
    for action in range(5):
        count = actions[action]
        percentage = (
            100 * count / warning_decisions
            if warning_decisions else 0
        )
        print(f"Action {action}: {count} ({percentage:.1f}%)")

    print("\nPower Strike outcomes:")
    print(f"Started: {strikes_started}")
    print(f"Resolved: {strikes_resolved}")
    print(f"Hits: {strikes_hit}")
    print(f"Misses: {strikes_missed}")
    print(f"Interrupted: {strikes_interrupted}")
    print(f"Total Power Strike damage: {total_strike_damage}")

    if strikes_resolved:
        hit_rate = 100 * strikes_hit / strikes_resolved
        print(f"Hit rate: {hit_rate:.1f}%")

    accounted = (
        strikes_resolved + strikes_interrupted
    )
    print(
        "Event accounting consistent:",
        strikes_started == accounted
    )


if __name__ == "__main__":
    evaluate_power_strike()
