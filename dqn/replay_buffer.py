import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity=100000, state_dim=5, seed=42):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.size = 0
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        i = self.position
        self.states[i], self.next_states[i] = state, next_state
        self.actions[i], self.rewards[i], self.dones[i] = action, reward, done
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = self.rng.choice(self.size, batch_size, replace=False)
        return tuple(getattr(self, key)[indices] for key in
                     ('states', 'actions', 'rewards', 'next_states', 'dones'))

    def state_dict(self):
        return {
            'position': self.position, 'size': self.size,
            'rng': self.rng.bit_generator.state,
            'data': {key: torch.from_numpy(getattr(self, key)[:self.size].copy())
                     for key in ('states', 'actions', 'rewards', 'next_states', 'dones')},
        }

    def load_state_dict(self, state):
        self.position, self.size = state['position'], state['size']
        self.rng.bit_generator.state = state['rng']
        for key, value in state['data'].items():
            getattr(self, key)[:self.size] = value.cpu().numpy()

    def __len__(self):
        return self.size
