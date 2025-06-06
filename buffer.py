import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        """
        Args:
            state_dim (tuple): State shape, e.g., (4, 84, 84)
            action_dim (int or tuple): Action shape (usually scalar for discrete actions)
            max_size (int): Maximum buffer size
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.s = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.a = np.zeros((max_size, *action_dim) if isinstance(action_dim, tuple) else (max_size, 1), dtype=np.int64)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.s_prime = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

    def update(self, s, a, r, s_prime, terminated):
        """
        Store a transition in the buffer.
        """
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_prime[self.ptr] = s_prime
        self.terminated[self.ptr] = terminated

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        Returns:
            tuple of tensors: (states, actions, rewards, next_states, dones)
        """
        ind = np.random.randint(0, self.size, size=batch_size)
        return (self.s[ind],self.a[ind],self.r[ind],self.s_prime[ind],self.terminated[ind],)

    def __len__(self):
        return self.size

class PrioritizedReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5), alpha=0.6, beta=0.4, beta_anneal_steps=1e5, eps=1e-6):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.alpha = alpha
        self.beta = beta
        self.beta_anneal = (1.0 - beta) / beta_anneal_steps
        self.eps = eps

        self.priorities = np.zeros((max_size,), dtype=np.float32)

        self.s = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.a = np.zeros((max_size, 1), dtype=np.int64)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.s_prime = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

    def update(self, s, a, r, s_prime, terminated, td_error=None):
        idx = self.ptr
        self.s[idx] = s
        self.a[idx] = a
        self.r[idx] = r
        self.s_prime[idx] = s_prime
        self.terminated[idx] = terminated

        if td_error is None:
            priority = 1.0  # maximum priority on new samples
        else:
            priority = (abs(td_error) + self.eps) ** self.alpha
        self.priorities[idx] = priority

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer!")

        probs = self.priorities[:self.size] + self.eps
        probs = probs ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        weights = (1.0 / (self.size * probs[indices])) ** self.beta
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_anneal)

        batch = (
            self.s[indices],
            self.a[indices],
            self.r[indices],
            self.s_prime[indices],
            self.terminated[indices],
            indices,
            weights.astype(np.float32).reshape(-1, 1),
        )
        return batch

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = (abs(err) + self.eps) ** self.alpha

    def __len__(self):
        return self.size