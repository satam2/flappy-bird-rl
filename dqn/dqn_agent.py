from pathlib import Path

import numpy as np
import torch
from torch import nn

from .replay_buffer import ReplayBuffer

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


def double_dqn_targets(rewards, dones, online_next, target_next, gamma):
    actions = online_next.argmax(dim=1, keepdim=True)
    values = target_next.gather(1, actions).squeeze(1)
    return rewards + gamma * (1 - dones) * values


class DQNAgent:
    def __init__(self, state_dim=5, action_dim=2, seed=42, batch_size=128,
                 warmup=2000, train_every=4, learning_rate=3e-4,
                 target_update_freq=1000, epsilon_decay_steps=100000,
                 capacity=100000, gamma=.99):
        self.config = dict(state_dim=state_dim, action_dim=action_dim, seed=seed,
                           batch_size=batch_size, warmup=warmup, train_every=train_every,
                           learning_rate=learning_rate, target_update_freq=target_update_freq,
                           epsilon_decay_steps=epsilon_decay_steps, capacity=capacity, gamma=gamma)
        torch.manual_seed(seed)
        self.device = torch.device('cpu')
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(capacity, state_dim, seed)
        self.rng = np.random.default_rng(seed)
        self.epsilon = 1.
        self.environment_steps = 0
        self.steps = 0
        self.metadata = {}

    def select_action(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.random() < .1)
        with torch.inference_mode():
            return int(self.policy_net(torch.as_tensor(state, dtype=torch.float32)).argmax())

    def store(self, *transition):
        self.buffer.push(*transition)
        self.environment_steps += 1
        progress = max(0, self.environment_steps - self.config['warmup'])
        self.epsilon = max(.02, 1 - .98 * progress / self.config['epsilon_decay_steps'])

    def train_step(self):
        if len(self.buffer) < max(self.config['batch_size'], self.config['warmup']):
            return None
        if self.environment_steps % self.config['train_every']:
            return None
        states, actions, rewards, next_states, dones = (
            torch.from_numpy(x) for x in self.buffer.sample(self.config['batch_size']))
        values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            targets = double_dqn_targets(rewards, dones, self.policy_net(next_states),
                                         self.target_net(next_states), self.config['gamma'])
        loss = nn.functional.smooth_l1_loss(values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.)
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.config['target_update_freq'] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.detach())

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if metadata is not None:
            self.metadata = metadata
        checkpoint = dict(
            format_version=1, config=self.config, policy=self.policy_net.state_dict(),
            target=self.target_net.state_dict(), optimizer=self.optimizer.state_dict(),
            epsilon=self.epsilon, environment_steps=self.environment_steps, steps=self.steps,
            rng=self.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            replay=self.buffer.state_dict(), metadata=self.metadata,
        )
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state.get('format_version') != 1:
            raise ValueError('Expected a version-1 checkpoint; legacy four-input weights need retraining')
        agent = cls(**state['config'])
        agent.policy_net.load_state_dict(state['policy'])
        agent.target_net.load_state_dict(state['target'])
        agent.optimizer.load_state_dict(state['optimizer'])
        agent.buffer.load_state_dict(state['replay'])
        agent.epsilon = state['epsilon']
        agent.environment_steps = state['environment_steps']
        agent.steps = state['steps']
        agent.rng.bit_generator.state = state['rng']
        torch.set_rng_state(state['torch_rng'])
        agent.metadata = state['metadata']
        return agent
