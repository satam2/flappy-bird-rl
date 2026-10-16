import sys


def test_training_seed_never_enters_validation_or_final_ranges():
    from dqn.train import training_seed
    assert training_seed(42, 1) == 43
    assert training_seed(9999, 1) == 10100
    assert training_seed(19899, 1) == 20100
    for episode in range(25000):
        seed = training_seed(42, episode)
        assert not 10000 <= seed <= 10099
        assert not 20000 <= seed <= 20099


def test_resume_at_completed_budget_is_a_valid_noop(tmp_path, monkeypatch):
    from dqn.dqn_agent import DQNAgent
    from dqn.train import main
    checkpoint = tmp_path / 'done.pth'
    agent = DQNAgent(capacity=8)
    agent.environment_steps = 10
    agent.save(checkpoint, {'episode': 1, 'backend': 'box2d-proxy-v2'})
    run = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['train', '--resume', str(checkpoint),
                                     '--steps', '10', '--run-dir', str(run)])
    main()
    restored = DQNAgent.load(run / 'latest.pth')
    assert restored.environment_steps == 10
