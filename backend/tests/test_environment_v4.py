
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

def test_power_strike_emits_explicit_hit_event():
    env = SentinelEnv()
    env.reset(seed=42)

    # Keep the player away and position the enemy
    # within the Guardian's Power Strike radius.
    env.p[:] = [7.0, 7.0]
    env.g[:] = [0.0, 0.0]
    env.e[:] = [0.5, 0.0]

    _, _, _, _, info1 = env.step(2)
    assert info1["power_strike"]["started"] is True

    _, _, _, _, info2 = env.step(2)
    assert info2["power_strike"]["resolved"] is False

    _, _, _, _, info3 = env.step(2)

    event = info3["power_strike"]
    assert event["resolved"] is True
    assert event["hit"] is True
    assert event["damage"] == 35

    env.close()


def test_power_strike_event_resets_each_step():
    env = SentinelEnv()
    env.reset(seed=42)

    env.p[:] = [7.0, 7.0]
    env.g[:] = [0.0, 0.0]
    env.e[:] = [0.5, 0.0]

    _, _, _, _, info1 = env.step(2)
    assert info1["power_strike"]["started"] is True

    _, _, _, _, info2 = env.step(2)
    assert info2["power_strike"]["started"] is False
    assert info2["power_strike"]["resolved"] is False
    assert info2["power_strike"]["damage"] == 0

    env.close()
