import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from model import ActorCritic

class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        update_epochs=4,
        batch_size=64,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

        # Add total_steps tracking
        self.total_steps = 0

    def summary(self):
        print("=== PPOAgent Configuration Summary ===")
        print(f"LR: {self.optimizer.param_groups[0]['lr']}")
        print(f"Gamma: {self.gamma}")
        print(f"GAE Lambda: {self.gae_lambda}")
        print(f"Clip Eps: {self.clip_eps}")
        print(f"Update Epochs: {self.update_epochs}")
        print(f"Batch Size: {self.batch_size}")
        print("======================================")

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        logits, value = self.network(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    @torch.no_grad()
    def select_best_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        logits, _ = self.network(state)
        return int(torch.argmax(logits, dim=1).item())

    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def finish_path(self, last_value=0):

        states = torch.from_numpy(np.stack(self.states)).float().to(self.device)
        #states = torch.tensor(self.states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.actions).to(self.device)
        log_probs = torch.tensor(self.log_probs).to(self.device)
        values = torch.tensor(self.values + [last_value]).to(self.device)

        rewards = []
        gae = 0
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + self.gamma * values[i + 1] * (1 - self.dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[i]) * gae
            rewards.insert(0, gae + values[i])
        returns = torch.tensor(rewards).to(self.device)
        advantages = returns - values[:-1]

        self.states, self.actions, self.log_probs, self.rewards, self.dones, self.values = [], [], [], [], [], []
        return states, actions, log_probs, returns.detach(), advantages.detach()

    def update(self, states, actions, old_log_probs, returns, advantages):
        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.update_epochs):
            for s, a, old_log_p, ret, adv in loader:
                logits, value = self.network(s)
                dist = Categorical(logits=logits)
                log_p = dist.log_prob(a)
                ratio = (log_p - old_log_p).exp()

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (ret - value.squeeze()).pow(2).mean()
                entropy = dist.entropy().mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item()}

    def train_epoch(self, env, epoch_steps):
        states, _ = env.reset(seed=42)
        num_envs = env.num_envs
        step_count = 0
        while step_count < epoch_steps:
            actions = []
            log_probs = []
            values = []
            for i in range(num_envs):
                a, log_p, v = self.act(states[i])
                actions.append(a)
                log_probs.append(log_p)
                values.append(v)
            next_states, rewards, terminateds, truncateds, _ = env.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            for i in range(num_envs):
                self.store(states[i], actions[i], log_probs[i], rewards[i], float(dones[i]), values[i])

            states = next_states
            step_count += num_envs
            # Update total_steps
            self.total_steps += num_envs

            if np.any(dones):
                if len(self.states) >= self.batch_size:
                    print(f" Episode ended, updating with {len(self.states)} experiences")
                last_vals = []
                for i in range(num_envs):
                    with torch.no_grad():
                        state_tensor = torch.from_numpy(states[i]).float().unsqueeze(0).to(self.device)
                        _, last_val = self.network(state_tensor)
                        last_vals.append(last_val.item())
                s, a, log_p, ret, adv = self.finish_path(np.mean(last_vals))
                if len(s) > 0:
                    self.update(s, a, log_p, ret, adv)

        # after loop handle leftover steps
        if len(self.states) > 0:
            print(f"Epoch ended, final update with {len(self.states)} remaining experiences")

            last_vals = []
            for i in range(num_envs):
                with torch.no_grad():
                    state_tensor = torch.from_numpy(states[i]).float().unsqueeze(0).to(self.device)
                    _, last_val = self.network(state_tensor)
                    last_vals.append(last_val.item())

            s, a, log_p, ret, adv = self.finish_path(np.mean(last_vals))
            if len(s) > 0:
                last_losses = self.update(s, a, log_p, ret, adv)
        else:
            print('warning: No experiences collected during this epoch!')
            last_losses = {"policy_loss": 0.0, "value_loss": 0.0}
        return last_losses

    def analyze_action_distribution(self, env, num_steps=200):
        """현재 정책의 액션 분포 분석"""
        states, _ = env.reset()
        action_counts = np.zeros(env.single_action_space.n)
        entropy_sum = 0.0

        print("=== Action Distribution Analysis ===")

        for step in range(num_steps):
            actions = []
            entropies = []

            for i in range(env.num_envs):
                # 정책의 확률 분포 직접 확인
                state = torch.from_numpy(states[i]).float().unsqueeze(0).to(self.device)
                logits, _ = self.network(state)
                dist = Categorical(logits=logits)

                action = dist.sample()
                actions.append(action.item())
                action_counts[action.item()] += 1
                entropies.append(dist.entropy().item())

            entropy_sum += np.mean(entropies)
            states, _, _, _, _ = env.step(actions)

    # 결과 출력
        total_actions = num_steps * env.num_envs
        print("Action probabilities:")
        for i, count in enumerate(action_counts):
            prob = count / total_actions
            print(f"  Action {i}: {prob:.3f} ({int(count)} times)")

        avg_entropy = entropy_sum / num_steps
        print(f"Average entropy: {avg_entropy:.3f}")

    # 문제 진단
        if avg_entropy < 0.5:
            print("⚠️  LOW ENTROPY: Policy is too deterministic!")
            print("   → Increase entropy coefficient in loss function")

        max_prob = max(action_counts) / total_actions
        if max_prob > 0.8:
            print(f"⚠️  DOMINANT ACTION: Action dominates {max_prob:.1%}")
            print("   → Policy is stuck, increase exploration")

        return action_counts, avg_entropy
