from pathlib import Path
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment import SentinelEnv

parser=argparse.ArgumentParser()
parser.add_argument('--steps',type=int,default=100_000)
parser.add_argument('--seed',type=int,default=42)
args=parser.parse_args()
out=Path(__file__).resolve().parent.parent/'models'
out.mkdir(exist_ok=True)
env=Monitor(SentinelEnv())
model=PPO('MlpPolicy',env,seed=args.seed,verbose=1,policy_kwargs={'net_arch':[32,32]})
model.learn(total_timesteps=args.steps)
model.save(str(out/'sentinel_ppo'))
print('Saved model:',out/'sentinel_ppo.zip')
