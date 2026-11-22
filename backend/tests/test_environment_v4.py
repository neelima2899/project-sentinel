
import numpy as np
import pytest

from backend.rl.environment_v4 import SentinelEnv


def test_observation_shape_and_bounds():
    env = SentinelEnv()
    observation, _ = env.reset(seed=42)

    assert observation.shape == (15,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert np.all(observation >= -1)
    assert np.all(observation <= 1)


def test_action_space():
    env = SentinelEnv()
    assert env.action_space.n == 5


def test_seed_reproducibility():
    env1 = SentinelEnv()
    env2 = SentinelEnv()

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)

    np.testing.assert_array_equal(obs1, obs2)


def test_invalid_action_rejected():
    env = SentinelEnv()
    env.reset(seed=42)

    with pytest.raises(ValueError):
        env.step(99)


def test_step_output():
    env = SentinelEnv()
    env.reset(seed=42)

    obs, reward, terminated, truncated, info = env.step(0)

    assert obs.shape == (15,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "winner" in info
    assert "damage" in info


def test_retreat_increases_distance_in_controlled_state():
    env = SentinelEnv()
    env.reset(seed=42)

    # Controlled positions, with no active Power Strike.
    env.p = np.array([0.0, 0.0], dtype=np.float32)
    env.g = np.array([-5.0, -5.0], dtype=np.float32)
    env.e = np.array([2.0, 0.0], dtype=np.float32)

    before = np.linalg.norm(env.e - env.p)
    env.step(4)
    after = np.linalg.norm(env.e - env.p)

    assert after > before
