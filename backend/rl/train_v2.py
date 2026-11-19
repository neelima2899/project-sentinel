import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment_v2 import SentinelEnv

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=200_000)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
(root/'models').mkdir(exist_ok=True)
env = Monitor(SentinelEnv())
model = PPO('MlpPolicy', env, seed=args.seed, verbose=1,
            policy_kwargs={'net_arch': [64, 64]}, n_steps=1024, batch_size=64)
model.learn(total_timesteps=args.steps)
model.save(str(root/'models'/'sentinel_ppo_v2'))
env.close()
print('Saved:', root/'models'/'sentinel_ppo_v2.zip')
