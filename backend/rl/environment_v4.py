"""SENTINEL v4: simplified, seeded tactical combat simulator (not browser physics)."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

ACTIONS = ('approach_player', 'approach_guardian', 'attack_player', 'attack_guardian', 'retreat')

class SentinelEnv(gym.Env):
    metadata = {'render_modes': []}
    MAX_STEPS = 160
    ATTACK_RANGE = 1.8
    POWER_STRIKE_DAMAGE = 35
    POWER_STRIKE_WARNING = 2
    POWER_STRIKE_RADIUS = 2.5
    POWER_STRIKE_COOLDOWN = 8

    def __init__(self, seed=None):
        super().__init__()
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(15,), dtype=np.float32)
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
            float(self.power_strike_active),
            self.power_strike_countdown / self.POWER_STRIKE_WARNING,
            self.power_strike_cooldown / self.POWER_STRIKE_COOLDOWN,
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
        self.power_strike_countdown = 0
        self.power_strike_cooldown = 0
        self.power_strike_active = False
        self.power_strike_event = {"started": False,"resolved": False,"hit": False,"damage": 0,}
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action: {action}')
        self.steps += 1
        self.action_counts[action] += 1
        self.power_strike_event = {"started": False,"resolved": False,"hit": False,"damage": 0,}
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
            # Escape the Guardian's Power Strike when charging.
            if self.power_strike_active and self.hp[1] > 0:
                target = self.g
            else:
                target = self.p if self.hp[0] > 0 else self.g

            escape_direction = self.e - target

            # Handle the rare case of identical positions.
            if np.linalg.norm(escape_direction) < 1e-6:
                escape_direction = np.array([1.0, 0.0], dtype=np.float32)

            self._move(self.e,self.e + escape_direction,0.75)


        # Player and guardian respond only if player is still alive.
        if self.hp[0] > 0:
            self._move(self.p, self.e, 0.45)
            if np.linalg.norm(self.p-self.e) < 1.75 and self.steps % 3 == 0:
                damage = min(12, self.hp[2])
                self.hp[2] -= damage
                self.damage['player'] += damage
        if self.hp[1] > 0 and self.hp[2] > 0 and self.hp[0] > 0:
            # Guardian stays stationary while charging its Power Strike.
            if not self.power_strike_active:
                self._move(self.g, self.e, 0.55)

            if (
                not self.power_strike_active
                and np.linalg.norm(self.g - self.e) < 1.45
                and self.steps % 4 == 0
            ):
                damage = min(6, self.hp[2])
                self.hp[2] -= damage
                self.damage['guardian'] += damage
                # Update Guardian Power Strike after normal combat.
        if self.hp[0] > 0 and self.hp[1] > 0 and self.hp[2] > 0:
            power_damage = self._update_power_strike()

            if power_damage > 0:
                self.damage['guardian'] += power_damage
        terminated = self.hp[0] <= 0 or self.hp[2] <= 0
        truncated = self.steps >= self.MAX_STEPS and not terminated
        dealt_player = old_hp[0]-self.hp[0]
        dealt_guardian = old_hp[1]-self.hp[1]
        taken = old_hp[2]-self.hp[2]
        # Target is defeating player; guardian damage helps, but is less valuable.
        # Survival Tactician: damage becomes more costly
        # as the Adversary loses health.

        health_ratio = self.hp[2] / 140.0
        risk_multiplier = 1.0 + (1.0 - health_ratio)
        damage_penalty = 0.055 * taken * risk_multiplier
        reward = (0.10 * dealt_player+ 0.025 * dealt_guardian- damage_penalty- 0.02)
        if action == 0 and self.hp[0] > 0:
            reward += 0.05*np.clip(old_distance-np.linalg.norm(self.p-self.e), -1, 1)
        if terminated:
               if self.hp[0] <= 0:
                 remaining_health_ratio = self.hp[2] / 140.0
                 reward += 30 + 10 * remaining_health_ratio
               else:
                  reward -= 30
        elif truncated:
            reward -= 12
        info = {
            'winner': 'adversary' if self.hp[0] <= 0 else 'allies' if self.hp[2] <= 0 else 'timeout',
            'steps': self.steps, 'damage': dict(self.damage),
            'action_counts': self.action_counts.tolist(),
            'successful_attacks': self.successful_attacks,
            'power_strike': dict(self.power_strike_event),
        }
        return self._obs(), float(reward), bool(terminated), bool(truncated), info
    
    def _update_power_strike(self):
        """Advance the Guardian Power Strike by one simulation step."""

        damage = 0

        # Resolve an existing Power Strike.
        if self.power_strike_active:
            self.power_strike_countdown -= 1

            if self.power_strike_countdown <= 0:
                distance = np.linalg.norm(self.g - self.e)

                self.power_strike_event["resolved"] = True

                if (
                    self.hp[1] > 0
                    and distance <= self.POWER_STRIKE_RADIUS
                ):
                    damage = min(
                        self.POWER_STRIKE_DAMAGE,
                        self.hp[2]
                    )

                    self.hp[2] -= damage

                    self.power_strike_event["hit"] = True
                    self.power_strike_event["damage"] = damage

                self.power_strike_active = False
                self.power_strike_cooldown = (
                    self.POWER_STRIKE_COOLDOWN
                )

            return damage

        # Wait for the cooldown.
        if self.power_strike_cooldown > 0:
            self.power_strike_cooldown -= 1
            return damage

        # Start charging if the Guardian is nearby.
        distance = np.linalg.norm(self.g - self.e)

        if (
            self.hp[1] > 0
            and distance <= self.POWER_STRIKE_RADIUS
        ):
            self.power_strike_active = True
            self.power_strike_countdown = (
                self.POWER_STRIKE_WARNING
            )

            self.power_strike_event["started"] = True

        return damage
