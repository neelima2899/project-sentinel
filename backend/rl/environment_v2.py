"""SENTINEL v2: simplified, seeded tactical combat simulator (not browser physics)."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

ACTIONS = ('approach_player', 'approach_guardian', 'attack_player', 'attack_guardian', 'retreat')

class SentinelEnv(gym.Env):
    metadata = {'render_modes': []}
    MAX_STEPS = 160
    ATTACK_RANGE = 1.8

    def __init__(self, seed=None):
        super().__init__()
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        self.reset(seed=seed)

    @staticmethod
    def _move(a, b, speed):
        delta = b - a
        dist = np.linalg.norm(delta)
        if dist > 1e-6:
            a += delta / dist * min(dist, speed)
        np.clip(a, -8, 8, out=a)

    def _obs(self):
        p, g, e = self.p, self.g, self.e
        return np.array([
            *((p-e)/16), *((g-e)/16),
            self.hp[0]/100, self.hp[1]/120, self.hp[2]/140,
            np.linalg.norm(p-e)/16, np.linalg.norm(g-e)/16,
            self.steps/self.MAX_STEPS,
            float(self.steps >= self.next_attack_step),
            min(1.0, max(0, self.next_attack_step-self.steps)/3),
        ], dtype=np.float32).clip(-1, 1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.p = np.array([0., 3.], dtype=np.float32) + self.np_random.uniform(-1.5, 1.5, 2).astype(np.float32)
        self.g = np.array([-4., -3.], dtype=np.float32) + self.np_random.uniform(-1.5, 1.5, 2).astype(np.float32)
        self.e = np.array([4., -3.], dtype=np.float32) + self.np_random.uniform(-1.5, 1.5, 2).astype(np.float32)
        self.hp = [100, 120, 140]
        self.steps = 0
        self.next_attack_step = 1
        self.damage = {'player': 0, 'guardian': 0, 'enemy': 0}
        self.action_counts = np.zeros(len(ACTIONS), dtype=int)
        self.successful_attacks = 0
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action: {action}')
        self.steps += 1
        self.action_counts[action] += 1
        old_hp = self.hp.copy()
        old_distance = np.linalg.norm(self.p-self.e)

        # Enemy acts first, making the attack decision meaningful.
        if action in (0, 1):
            idx = action
            if self.hp[idx] > 0:
                self._move(self.e, self.p if idx == 0 else self.g, 0.65)
        elif action in (2, 3):
            idx = action - 2
            target = self.p if idx == 0 else self.g
            if (self.hp[idx] > 0 and self.steps >= self.next_attack_step
                    and np.linalg.norm(target-self.e) <= self.ATTACK_RANGE):
                damage = min(20, self.hp[idx])
                self.hp[idx] -= damage
                self.damage['enemy'] += damage
                self.successful_attacks += 1
                self.next_attack_step = self.steps + 2
        elif action == 4:
            target = self.p if self.hp[0] > 0 else self.g
            self._move(self.e, self.e + (self.e-target), 0.75)

        # Player and guardian respond only if player is still alive.
        if self.hp[0] > 0:
            self._move(self.p, self.e, 0.45)
            if np.linalg.norm(self.p-self.e) < 1.75 and self.steps % 3 == 0:
                damage = min(12, self.hp[2])
                self.hp[2] -= damage
                self.damage['player'] += damage
        if self.hp[1] > 0 and self.hp[2] > 0 and self.hp[0] > 0:
            self._move(self.g, self.e, 0.55)
            if np.linalg.norm(self.g-self.e) < 1.45 and self.steps % 4 == 0:
                damage = min(6, self.hp[2])
                self.hp[2] -= damage
                self.damage['guardian'] += damage

        terminated = self.hp[0] <= 0 or self.hp[2] <= 0
        truncated = self.steps >= self.MAX_STEPS and not terminated
        dealt_player = old_hp[0]-self.hp[0]
        dealt_guardian = old_hp[1]-self.hp[1]
        taken = old_hp[2]-self.hp[2]
        # Target is defeating player; guardian damage helps, but is less valuable.
        reward = 0.10*dealt_player + 0.025*dealt_guardian - 0.055*taken - 0.02
        if action == 0 and self.hp[0] > 0:
            reward += 0.05*np.clip(old_distance-np.linalg.norm(self.p-self.e), -1, 1)
        if terminated:
            reward += 30 if self.hp[0] <= 0 else -30
        elif truncated:
            reward -= 12
        info = {
            'winner': 'adversary' if self.hp[0] <= 0 else 'allies' if self.hp[2] <= 0 else 'timeout',
            'steps': self.steps, 'damage': dict(self.damage),
            'action_counts': self.action_counts.tolist(),
            'successful_attacks': self.successful_attacks,
        }
        return self._obs(), float(reward), bool(terminated), bool(truncated), info
