import torch

from dqn.dqn_agent import DQNAgent
import dqn.dqn_agent as dqn_agent_module


class TableQNet(torch.nn.Module):
    def __init__(self, table):
        super().__init__()
        self.q_table = torch.nn.Parameter(torch.tensor(table, dtype=torch.float32))

    def forward(self, x):
        idx = x[:, 0].long()
        return self.q_table[idx]


class CaptureLoss:
    def __init__(self, store):
        self.store = store

    def __call__(self, q_values, target):
        self.store["target"] = target.detach().cpu()
        return ((q_values - target) ** 2).mean()


def _state(index):
    return [float(index), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_train_step_uses_policy_argmax_and_target_gather_for_bootstrap_target(monkeypatch):
    captured = {}
    monkeypatch.setattr(dqn_agent_module.nn, "MSELoss", lambda: CaptureLoss(captured))
    monkeypatch.setattr(dqn_agent_module.nn, "SmoothL1Loss", lambda: CaptureLoss(captured))

    agent = DQNAgent(state_dim=7, action_dim=2)
    agent.batch_size = 2
    agent.min_buffer_size = 2
    agent.gamma = 0.99

    agent.policy_net = TableQNet(
        [
            [5.0, 1.0],
            [0.0, 4.0],
        ]
    ).to(agent.device)
    agent.target_net = TableQNet(
        [
            [2.0, 9.0],
            [7.0, 3.0],
        ]
    ).to(agent.device)
    agent.optimizer = torch.optim.SGD(agent.policy_net.parameters(), lr=0.01)

    agent.store(_state(0), 0, 1.0, _state(0), False)
    agent.store(_state(1), 1, 1.0, _state(1), True)

    agent.train_step()

    assert "target" in captured
    expected = torch.tensor([1.0 + 0.99 * 2.0, 1.0], dtype=torch.float32)
    assert torch.allclose(
        torch.sort(captured["target"]).values,
        torch.sort(expected).values,
        atol=1e-6,
    )


def test_epsilon_decay_is_step_based_and_respects_minimum():
    agent = DQNAgent(state_dim=7, action_dim=2)
    agent.batch_size = 1
    agent.min_buffer_size = 1
    agent.epsilon = 1.0
    agent.epsilon_decay = 0.5
    agent.epsilon_min = 0.2

    agent.policy_net = TableQNet([[0.0, 0.0]]).to(agent.device)
    agent.target_net = TableQNet([[0.0, 0.0]]).to(agent.device)
    agent.optimizer = torch.optim.SGD(agent.policy_net.parameters(), lr=0.01)

    transition = (_state(0), 0, 0.0, _state(0), False)
    for _ in range(4):
        agent.store(*transition)

    agent.train_step()
    assert agent.epsilon == 0.5
    agent.train_step()
    assert agent.epsilon == 0.25
    agent.train_step()
    assert agent.epsilon == 0.2
    agent.train_step()
    assert agent.epsilon == 0.2
