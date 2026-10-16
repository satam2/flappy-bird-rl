import numpy as np
import torch


def test_greedy_evaluation_does_not_consume_exploration_or_update_weights():
    from dqn.dqn_agent import DQNAgent
    agent = DQNAgent(seed=7)
    agent.epsilon = 1.0
    state = np.zeros(5, dtype=np.float32)
    weights = {k: v.clone() for k, v in agent.policy_net.state_dict().items()}
    actions = [agent.select_action(state, explore=False) for _ in range(10)]
    assert len(set(actions)) == 1
    assert agent.epsilon == 1.0
    assert all(torch.equal(weights[k], v) for k, v in agent.policy_net.state_dict().items())


def test_checkpoint_round_trip_restores_optimizer_and_exploration(tmp_path):
    from dqn.dqn_agent import DQNAgent
    agent = DQNAgent(seed=17, batch_size=4, warmup=4)
    for i in range(8):
        agent.store(np.full(5, i / 10), i % 2, 1, np.zeros(5), i % 3 == 0)
        agent.train_step()
    path = tmp_path / 'agent.pth'
    agent.save(path, metadata={'episode': 8})
    restored = DQNAgent.load(path)
    assert restored.steps == agent.steps
    assert restored.environment_steps == agent.environment_steps
    assert restored.epsilon == agent.epsilon
    assert restored.metadata == {'episode': 8}
    assert restored.optimizer.state_dict()['state']
    assert len(restored.buffer) == len(agent.buffer)
    for k, v in agent.policy_net.state_dict().items():
        assert torch.equal(v, restored.policy_net.state_dict()[k])


def test_double_dqn_uses_online_action_and_target_value():
    from dqn.dqn_agent import double_dqn_targets
    online = torch.tensor([[5., 1.], [1., 5.]])
    target = torch.tensor([[2., 9.], [8., 3.]])
    values = double_dqn_targets(torch.tensor([1., -1.]), torch.tensor([0., 1.]), online, target, .5)
    torch.testing.assert_close(values, torch.tensor([2., -1.]))
