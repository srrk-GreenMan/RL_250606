import numpy as np

class ExplorationStrategy:
    def select_action(self, q_values, training: bool):
        raise NotImplementedError

    def update(self, step: int):
        pass

class EpsilonGreedy(ExplorationStrategy):
    def __init__(self, action_dim, epsilon=1.0, epsilon_min=0.1, decay_steps=1e5):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (epsilon - epsilon_min) / decay_steps

    def select_action(self, q_values):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        return int(np.argmax(q_values))

    def update(self, step):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

class SoftmaxExploration(ExplorationStrategy):
    def __init__(self, action_dim, temperature=1.0, temperature_min=0.1, decay_steps=1e5):
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.temperature_decay = (temperature - temperature_min) / decay_steps

    def select_action(self, q_values):
        q = q_values / self.temperature
        q = q - np.max(q) + 1e-9  # for numerical stability
        probs = np.exp(q)
        probs /= np.sum(probs)
        return int(np.random.choice(len(probs), p=probs))

    def update(self, step):
        self.temperature = max(self.temperature - self.temperature_decay, self.temperature_min)
