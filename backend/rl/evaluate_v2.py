import argparse
import csv
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from environment_v2 import SentinelEnv

parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=100)
parser.add_argument('--seed', type=int, default=2026)
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
model_path = root/'models'/'sentinel_ppo_v2.zip'
if not model_path.exists():
    raise SystemExit('Missing v2 model. Run python rl/train_v2.py first.')
model = PPO.load(str(model_path))

def baseline(env, obs):
    # Attack player whenever possible; otherwise close distance.
    if env.hp[0] > 0:
        return 2 if np.linalg.norm(env.p-env.e) <= env.ATTACK_RANGE and env.steps + 1 >= env.next_attack_step else 0
    return 3 if np.linalg.norm(env.g-env.e) <= env.ATTACK_RANGE and env.steps + 1 >= env.next_attack_step else 1

def run(name, policy):
    records = []
    for i in range(args.episodes):
        env = SentinelEnv()
        obs, _ = env.reset(seed=args.seed+i)
        done = False
        while not done:
            action = policy(env, obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        records.append({
            'policy': name, 'episode': i, 'winner': info['winner'],
            'steps': info['steps'], 'enemy_damage': info['damage']['enemy'],
            'player_damage': info['damage']['player'],
            'guardian_damage': info['damage']['guardian'],
            'successful_attacks': info['successful_attacks'],
            **{f'action_{j}': n for j, n in enumerate(info['action_counts'])},
        })
    wins = sum(r['winner'] == 'adversary' for r in records)
    print(f'{name}: {wins}/{len(records)} adversary wins ({wins/len(records):.1%}); '
          f'mean steps {np.mean([r["steps"] for r in records]):.1f}; '
          f'mean successful attacks {np.mean([r["successful_attacks"] for r in records]):.1f}')
    return records

rows = run('rule_based_v2', baseline)
rows += run('ppo_v2', lambda env, obs: int(model.predict(obs, deterministic=True)[0]))
rows += run(
    'random_v2',
    lambda env, obs: int(env.np_random.integers(env.action_space.n))
)
(root/'results').mkdir(exist_ok=True)
output = root/'results'/'evaluation_v2.csv'
with output.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print('Wrote', output)
