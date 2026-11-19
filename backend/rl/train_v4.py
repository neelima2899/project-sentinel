from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment_v4 import SentinelEnv

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sentinel_ppo_v4.zip"

def main():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    env = Monitor(SentinelEnv())

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=42,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )

    print("Training SENTINEL PPO v4...")
    model.learn(total_timesteps=200_000)

    model.save(str(MODEL_PATH))
    print("Saved model:", MODEL_PATH)

    env.close()

if __name__ == "__main__":
    main()
