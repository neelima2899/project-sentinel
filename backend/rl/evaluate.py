import argparse,csv
from pathlib import Path
import numpy as np
from environment import SentinelEnv

parser=argparse.ArgumentParser()
parser.add_argument('--episodes',type=int,default=100)
parser.add_argument('--seed',type=int,default=2026)
args=parser.parse_args()
try:
    from stable_baselines3 import PPO
    model_path=Path(__file__).resolve().parent.parent/'models'/'sentinel_ppo.zip'
    model=PPO.load(str(model_path)) if model_path.exists() else None
except ImportError:
    model=None

# Fixed baseline: approach nearest living target, attack if in range.
def baseline(env):
    choices=[]
    if env.hp[0]>0: choices.append((np.linalg.norm(env.p-env.e),0,2))
    if env.hp[1]>0: choices.append((np.linalg.norm(env.g-env.e),1,3))
    dist,move,attack=min(choices)
    return attack if dist<=1.4 else move

def run(label,policy):
    rows=[]
    for i in range(args.episodes):
        env=SentinelEnv()
        obs,_=env.reset(seed=args.seed+i)
        done=False
        while not done:
            action=policy(env,obs)
            obs,_,terminated,truncated,info=env.step(action)
            done=terminated or truncated
        rows.append({'policy':label,'episode':i,'winner':info['winner'], 'steps':info['steps'],
                     'enemy_damage':info['damage']['enemy'],'player_damage':info['damage']['player'],
                     'guardian_damage':info['damage']['guardian'],
                     **{f'action_{j}':v for j,v in enumerate(info['action_counts'])}})
    wins=sum(r['winner']=='adversary' for r in rows)
    print(f'{label}: {wins}/{len(rows)} adversary wins ({wins/len(rows):.1%}); mean steps {np.mean([r["steps"] for r in rows]):.1f}')
    return rows

rows=run('rule_based',lambda e,o:baseline(e))
if model is not None:
    rows+=run('ppo',lambda e,o:int(model.predict(o,deterministic=True)[0]))
else:
    print('PPO model not found; run train.py to include learned-policy results.')
out=Path(__file__).resolve().parent.parent/'results'
out.mkdir(exist_ok=True)
with (out/'evaluation.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
print('Wrote',out/'evaluation.csv')
