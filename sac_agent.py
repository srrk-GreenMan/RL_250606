import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from buffer import ReplayBuffer
from model import AtariCNN
from gymnasium import spaces


class Actor(nn.Module):
    def __init__(self, observation_space: spaces.Box, action_dim: int):
        super().__init__()
        self.features = AtariCNN(observation_space.shape)
        self.mu = nn.Linear(512, action_dim)
        self.log_std = nn.Linear(512, action_dim)

    def forward(self, obs):
        x = self.features(obs)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, observation_space: spaces.Box, action_dim: int):
        super().__init__()
        self.features = AtariCNN(observation_space.shape)
        self.q = nn.Linear(512 + action_dim, 1)

    def forward(self, obs, action):
        x = self.features(obs)
        x = torch.cat([x, action], dim=1)
        return self.q(x)


def soft_update(source: nn.Module, target: nn.Module, tau: float):
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.mul_(1 - tau)
        tgt_param.data.add_(tau * src_param.data)


class SACAgent:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        batch_size: int = 256,
        buffer_size: int = int(1e5),
        warmup_steps: int = 1000,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_dim = action_space.shape[0]
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.total_steps = 0

        self.actor = Actor(observation_space, self.action_dim).to(self.device)
        self.critic1 = Critic(observation_space, self.action_dim).to(self.device)
        self.critic2 = Critic(observation_space, self.action_dim).to(self.device)
        self.critic1_target = Critic(observation_space, self.action_dim).to(self.device)
        self.critic2_target = Critic(observation_space, self.action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_size, observation_space, action_space, device=self.device)

    def act(self, obs, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mu, log_std = self.actor(obs_t)
        if deterministic:
            action = torch.tanh(mu)
        else:
            std = log_std.exp()
            dist = Normal(mu, std)
            z = dist.sample()
            action = torch.tanh(z)
        return action.cpu().numpy()[0]

    def select_best_action(self, obs):
        return self.act(obs, deterministic=True)

    def learn(self):
        if self.buffer.size() < self.batch_size:
            return {}
        samples = self.buffer.sample(self.batch_size)
        obs = samples.observations
        actions = samples.actions
        next_obs = samples.next_observations
        dones = samples.dones
        rewards = samples.rewards

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            q1_next = self.critic1_target(next_obs, next_action)
            q2_next = self.critic2_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = rewards + (1 - dones) * self.gamma * q_next

        q1 = self.critic1(obs, actions)
        q2 = self.critic2(obs, actions)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        critic_loss = critic1_loss + critic2_loss

        self.critic1_opt.zero_grad()
        self.critic2_opt.zero_grad()
        critic_loss.backward()
        self.critic1_opt.step()
        self.critic2_opt.step()

        action_pi, log_pi = self.actor.sample(obs)
        q1_pi = self.critic1(obs, action_pi)
        q2_pi = self.critic2(obs, action_pi)
        actor_loss = (self.alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        soft_update(self.critic1, self.critic1_target, self.tau)
        soft_update(self.critic2, self.critic2_target, self.tau)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "total_steps": self.total_steps,
        }

    def process(self, transition):
        obs, action, reward, next_obs, done = transition
        self.total_steps += 1
        self.buffer.add(obs[None], next_obs[None], np.array([action]), np.array([reward]), np.array([done]), infos=[{}])

        if self.total_steps > self.warmup_steps:
            return self.learn()
        return {}

    def train_epoch(self, env, epoch_steps):
        states, _ = env.reset(seed=42)
        num_envs = env.num_envs
        step_count = 0
        while step_count < epoch_steps:
            actions = [self.act(states[i]) for i in range(num_envs)]
            next_states, rewards, terminateds, truncateds, infos = env.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            for i in range(num_envs):
                transition = (
                    states[i],
                    actions[i],
                    rewards[i],
                    next_states[i],
                    float(dones[i]),
                )
                self.process(transition)

            states = next_states
            step_count += num_envs

        return {}
