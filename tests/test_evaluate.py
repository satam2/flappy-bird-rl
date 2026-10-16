import numpy as np
import pytest


class ScriptedEnvironment:
    backend = 'test-fixture'

    def reset(self, seed=None):
        self.score = seed % 3
        self.ticks = 0
        return np.zeros(5, dtype=np.float32), {'score': 0}

    def step(self, action):
        self.ticks += 1
        return np.zeros(5), 999., self.ticks == 2, {'score': self.score}


class FrozenPolicy:
    def select_action(self, state, explore=True):
        assert explore is False
        return 0


def test_evaluation_uses_pipe_count_all_seeds_and_no_exploration():
    from dqn.evaluate import evaluate
    report = evaluate(FrozenPolicy(), ScriptedEnvironment(), episodes=3, seed_start=10)
    assert report['scores'] == [1, 2, 0]
    assert report['mean_pipes'] == 1
    assert report['best_pipes'] == 2
    assert not report['target_met']
    assert report['seeds'] == [10, 11, 12]


def test_truncated_episodes_cannot_satisfy_performance_target():
    from dqn.evaluate import evaluate
    report = evaluate(FrozenPolicy(), ScriptedEnvironment(), episodes=1, seed_start=74, max_steps=1)
    assert report['truncated_episodes'] == 1
    assert not report['target_met']


def test_zero_episode_evaluation_is_rejected():
    from dqn.evaluate import evaluate
    with pytest.raises(ValueError):
        evaluate(FrozenPolicy(), ScriptedEnvironment(), episodes=0)


def test_report_hash_identifies_loaded_weights_if_checkpoint_is_replaced(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys
    from dqn.dqn_agent import DQNAgent
    from dqn import evaluate as module
    checkpoint = tmp_path / 'best.pth'
    DQNAgent(capacity=8).save(checkpoint)
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    class ReplacingEnvironment(ScriptedEnvironment):
        def close(self):
            checkpoint.write_bytes(b'checkpoint replaced while evaluation was running')

    monkeypatch.setattr(module, 'make_env', lambda backend, port: ReplacingEnvironment())
    report = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['evaluate', str(checkpoint), '--episodes', '1',
                                     '--output', str(report)])
    module.main()
    assert json.loads(report.read_text())['checkpoint_sha256'] == expected
