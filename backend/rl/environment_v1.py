"""Simplified SENTINEL combat task. This is NOT identical to browser physics."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

ACTIONS = ('approach_player', 'approach_guardian', 'attack_player', 'attack_guardian', 'retreat')

class SentinelEnv(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self, seed=None):
        super().__init__()
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.reset(seed=seed)

    def _obs(self):
        p, g, e = self.p, self.g, self.e
        return np.array([*( (p-e)/8 ), *((g-e)/8), self.hp[0]/100, self.hp[1]/120,
                         self.hp[2]/140, np.linalg.norm(p-e)/12, np.linalg.norm(g-e)/12,
                         self.steps/160], dtype=np.float32).clip(-1, 1)

    @staticmethod
    def _move(a,b,speed):
        delta = b-a
        dist = np.linalg.norm(delta)
        if dist > 1e-6:
            a += delta / dist * min(dist, speed)
        np.clip(a, -8, 8, out=a)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.p = np.array([0.,3.], dtype=np.float32)
        self.g = np.array([-4.,-3.], dtype=np.float32)
        self.e = np.array([4.,-3.], dtype=np.float32)
        self.hp = [100,120,140]
        self.steps = 0
        self.damage = {'player':0,'guardian':0,'enemy':0}
        self.action_counts = np.zeros(len(ACTIONS),dtype=int)
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        assert self.action_space.contains(action)
        self.steps += 1
        self.action_counts[action] += 1
        old_hp = self.hp.copy()
        # One decision every 0.4 simulated seconds. Player approaches and attacks
        # when near; Guardian approaches and attacks. Enemy chooses an action.
        if self.hp[0] > 0:
            self._move(self.p,self.e,0.45)
            if np.linalg.norm(self.p-self.e) < 2.1 and self.steps % 2 == 0:
                self.hp[2] = max(0,self.hp[2]-18)
                self.damage['player'] += 18
        if self.hp[1] > 0 and self.hp[2] > 0:
            self._move(self.g,self.e,0.65)
            if np.linalg.norm(self.g-self.e) < 1.55 and self.steps % 4 == 0:
                self.hp[2] = max(0,self.hp[2]-8)
                self.damage['guardian'] += 8
        if self.hp[2] > 0:
            if action in (0,1):
                target = self.p if action == 0 else self.g
                if self.hp[0 if action==0 else 1] > 0:
                    self._move(self.e,target,0.55)
            elif action in (2,3) and self.steps % 3 == 0:
                index = 0 if action == 2 else 1
                target = self.p if index == 0 else self.g
                if self.hp[index] > 0 and np.linalg.norm(target-self.e) <= 1.4:
                    damage = min(10,self.hp[index])
                    self.hp[index] -= damage
                    self.damage['enemy'] += damage
            elif action == 4:
                target = self.p if self.hp[0] > 0 else self.g
                self._move(self.e,self.e+(self.e-target),0.55)
        terminated = self.hp[2] <= 0 or self.hp[0] <= 0
        truncated = self.steps >= 160 and not terminated
        # Dense reward for effective damage; penalty for being damaged.
        reward = 0.06*(old_hp[0]-self.hp[0]) + 0.03*(old_hp[1]-self.hp[1]) - 0.04*(old_hp[2]-self.hp[2]) - 0.01
        if terminated:
            reward += 12 if self.hp[0] <= 0 else -12
        info = {'winner': 'adversary' if self.hp[0]<=0 else 'allies' if self.hp[2]<=0 else 'timeout',
                'steps':self.steps, 'damage':dict(self.damage), 'action_counts':self.action_counts.tolist()}
        return self._obs(),float(reward),bool(terminated),bool(truncated),info
