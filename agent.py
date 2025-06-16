import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from model import QNetwork
from buffer import ReplayBuffer, PrioritizedReplayBuffer
from exploration import EpsilonGreedy, SoftmaxExploration

class Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        exploration_strategy=None,
        lr=0.00025,
        gamma=0.99,
        batch_size=64,
        warmup_steps=5000,
        buffer_size=int(1e5),
        target_update_interval=5000,
        use_double_q=False,
        prioritized=True,
        alpha=0.6,
        beta=0.4,
        beta_increment=1e-6,
        epsilon=1e-6
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval
        self.use_double_q = use_double_q
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.prioritized_epsilon = epsilon

        self.network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        self.target_network.to(self.device)

        obs_space = spaces.Box(low=0.0, high=1.0, shape=state_dim, dtype=np.float32)
        act_space = spaces.Discrete(action_dim)
        if self.prioritized:
            self.buffer = PrioritizedReplayBuffer(
                buffer_size,
                obs_space,
                act_space,
                device=self.device,
                use_uint8=True,
                alpha=alpha,
                beta=beta,
                beta_increment=beta_increment,
                epsilon=epsilon,
            )
        else:
            self.buffer = ReplayBuffer(buffer_size, obs_space, act_space, device=self.device, use_uint8=True)

        self.total_steps = 0
        self.exploration_strategy = exploration_strategy
        self.exploration_strategy.action_dim = action_dim  # if needed
        
    def summary(self):
        print("=== Agent Configuration Summary ===")
        print(f"Exploration: {self.exploration_strategy.__class__.__name__}")
        print(f"LR: {self.optimizer.param_groups[0]['lr']}")
        print(f"Gamma: {self.gamma}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Warmup Steps: {self.warmup_steps}")
        print(f"Target Update Interval: {self.target_update_interval}")
        print(f"Double Q: {self.use_double_q}")
        print(f"Prioritized Replay: {self.prioritized}")
        if self.prioritized:
            print(f"Alpha: {self.alpha}, Beta: {self.beta}")
        print("===================================")

    @torch.no_grad()
    def act(self, state):
        if self.total_steps < self.warmup_steps:
            return np.random.randint(0, self.action_dim)

        x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q = self.network(x)
        q_np = q.cpu().numpy().squeeze()
        return self.exploration_strategy.select_action(q_np)
            
    @torch.no_grad()
    def select_best_action(self, state):
        x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q = self.network(x)
        return int(torch.argmax(q, dim=1).item())
    
    def learn(self):
        samples = self.buffer.sample(self.batch_size)
        s = samples.observations
        a = samples.actions.long()
        r = samples.rewards
        s_prime = samples.next_observations
        terminated = samples.dones
        weights = getattr(samples, "weights", None)
        batch_indices = getattr(samples, "batch_indices", None)

        with torch.no_grad():
            next_q = self.target_network(s_prime)
            if self.use_double_q:
                best_actions = torch.argmax(self.network(s_prime), dim=1, keepdim=True)
                max_next_q = next_q.gather(1, best_actions)
            else:
                max_next_q = next_q.max(dim=1, keepdim=True).values
            td_target = r + (1. - terminated) * self.gamma * max_next_q

        q_values = self.network(s).gather(1, a)
        if weights is not None:
            loss_elements = F.mse_loss(q_values, td_target, reduction="none")
            loss = (loss_elements * weights).mean()
        else:
            loss = F.mse_loss(q_values, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if batch_indices is not None:
            td_errors = (q_values.detach() - td_target).abs().cpu().numpy().squeeze()
            self.buffer.update_priorities(batch_indices, td_errors + self.prioritized_epsilon)
        return {"total_steps": self.total_steps, "value_loss": loss.item()}

    def process(self, transition):
        result = {}
        self.total_steps += 1
        s, a, r, s_prime, done = transition
        self.buffer.add(s[None], s_prime[None], np.array([a]), np.array([r]), np.array([done]), infos=[{}])

        if self.total_steps > self.warmup_steps:
            result = self.learn()

        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        self.exploration_strategy.update(self.total_steps)
        return result

    def train_epoch(self, env, epoch_steps):
        states, _ = env.reset(seed=42)
        num_envs = env.num_envs
        rets = np.zeros(num_envs)

        step_count = 0
        while step_count < epoch_steps:  # safety cap
            actions = [self.act(states[i]) for i in range(num_envs)]
            next_states, rewards, terminateds, truncateds, infos = env.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            for i in range(num_envs):
                transition = (
                    states[i],
                    actions[i],
                    rewards[i],
                    next_states[i],
                    float(dones[i])
                )
                self.process(transition)
                rets[i] += rewards[i]

            if np.any(dones):
                states, _ = env.reset(seed=42)
                rets = np.zeros(num_envs)
            else:
                states = next_states

            step_count += num_envs  # because num_envs transitions per step
