import numpy as np
import pytest


def make_env():
    from dqn.headless import HeadlessFlappyEnv
    return HeadlessFlappyEnv()


def test_action_advances_physics_and_cooldown_rejects_repeated_flap():
    env = make_env()
    initial, _ = env.reset(seed=12)
    state, _, done, info = env.step(1)
    assert not done
    assert state[0] > initial[0]
    assert state[1] > 0
    assert state[4] > 0
    second, _, _, _ = env.step(1)
    assert second[1] < state[1]
    assert info['score'] == 0


def test_reset_is_seeded_and_clears_episode_state():
    env = make_env()
    initial, _ = env.reset(seed=123)
    states = [env.step(0)[0] for _ in range(5)]
    env.reset(seed=123)
    repeated = [env.step(0)[0] for _ in range(5)]
    np.testing.assert_array_equal(states, repeated)
    reset, info = env.reset(seed=124)
    assert reset[0] == initial[0] == 0
    assert reset[1] == reset[4] == 0
    assert info['score'] == 0


def test_death_is_terminal_and_requires_reset():
    env = make_env()
    env.reset(seed=1)
    for _ in range(200):
        _, reward, done, info = env.step(0)
        if done:
            break
    assert done
    assert reward < 0
    assert info['score'] == 0
    with pytest.raises(RuntimeError, match='reset'):
        env.step(0)


def test_score_counts_each_real_trigger_entry_only_once():
    env = make_env()
    env.reset(seed=7)
    # Put one real pipe close enough to cross its sensor in a few ticks.
    pipe = env.pipes[0]
    pipe.position = (-1.9, 0)
    env.bird.position = (-2, 0)
    env.bird.linearVelocity = (0, 0)
    scores = [env.step(0)[3]['score'] for _ in range(5)]
    assert scores == [1, 1, 1, 1, 1]


def test_fatal_contact_and_sensor_overlap_do_not_award_a_pipe():
    env = make_env()
    env.reset(seed=7)
    env.pipes[0].position = (-2.4, 0)
    env.bird.position = (-2, 1.5)
    _, reward, done, info = env.step(0)
    assert done and reward == -5
    assert info['score'] == 0


def test_flap_impulse_is_integrated_before_upward_velocity_cap():
    env = make_env()
    env.reset(seed=3)
    state, _, _, _ = env.step(1)
    # 12 m/s flap, -29.5281 m/s^2 gravity, 20ms semi-implicit tick.
    assert float(state[0] * 5) == pytest.approx(.22818876, abs=.0001)
    assert float(state[1]) == pytest.approx(1.)
