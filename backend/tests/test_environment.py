from backend.rl.environment import SentinelEnv

def test_rollout():
    env=SentinelEnv()
    obs,_=env.reset(seed=7)
    assert obs.shape==(10,)
    for _ in range(200):
        obs,reward,terminated,truncated,info=env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert isinstance(reward,float)
        if terminated or truncated: break
    assert terminated or truncated
    assert info['winner'] in ('adversary','allies','timeout')
