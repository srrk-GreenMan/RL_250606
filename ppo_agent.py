import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque

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
        normalize_advantages=True,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        target_kl=0.01,  # Early stopping based on KL divergence
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.normalize_advantages = normalize_advantages
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Buffer for storing trajectories
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

        # Tracking
        self.total_steps = 0

        # Debugging metrics
        self.debug_metrics = {
            'policy_losses': deque(maxlen=100),
            'value_losses': deque(maxlen=100),
            'entropies': deque(maxlen=100),
            'kl_divs': deque(maxlen=100),
            'grad_norms': deque(maxlen=100),
            'clip_fractions': deque(maxlen=100),
            'value_predictions': deque(maxlen=100),
            'returns': deque(maxlen=100),
            'advantages': deque(maxlen=100),
        }

    def summary(self):
        print("=== PPOAgent Configuration Summary ===")
        print(f"Device: {self.device}")
        print(f"LR: {self.optimizer.param_groups[0]['lr']}")
        print(f"Gamma: {self.gamma}")
        print(f"GAE Lambda: {self.gae_lambda}")
        print(f"Clip Eps: {self.clip_eps}")
        print(f"Update Epochs: {self.update_epochs}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Normalize Advantages: {self.normalize_advantages}")
        print(f"Value Loss Coef: {self.value_loss_coef}")
        print(f"Entropy Coef: {self.entropy_coef}")
        print(f"Max Grad Norm: {self.max_grad_norm}")
        print(f"Target KL: {self.target_kl}")
        print("======================================")

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():  # 추가: inference 시 gradient 계산 방지
            logits, value = self.network(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # value도 detach하고 squeeze하여 반환
        return action.item(), log_prob, value.squeeze()

    @torch.no_grad()
    def select_best_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        logits, _ = self.network(state)
        return int(torch.argmax(logits, dim=1).item())

    def store(self, state, action, log_prob, reward, done, value):
        state_tensor = torch.from_numpy(state).float().to(self.device)
        self.states.append(state_tensor)
        self.actions.append(action)
        self.log_probs.append(log_prob.detach())  # 올바른 detach 사용
        self.rewards.append(reward)
        self.dones.append(done)
        # value가 텐서인 경우 스칼라로 변환
        if isinstance(value, torch.Tensor):
            self.values.append(value.detach().squeeze().item())
        else:
            self.values.append(value)

    def compute_gae(self, rewards, values, dones, last_value, num_envs):
        """개선된 GAE 계산 - 1D 배열을 2D로 reshape하여 처리"""
        # Convert to numpy arrays
        rewards = np.array(rewards)
        values = np.array(values)
        dones = np.array(dones)

        # Reshape from 1D to 2D (num_steps, num_envs)
        # 데이터가 [env0_step0, env1_step0, ..., env0_step1, env1_step1, ...] 형태로 저장되어 있다고 가정
        num_steps = len(rewards) // num_envs

        # Reshape to (num_steps, num_envs)
        rewards = rewards.reshape(num_steps, num_envs)
        values = values.reshape(num_steps, num_envs)
        dones = dones.reshape(num_steps, num_envs)

        # Ensure last_value is array
        if isinstance(last_value, (int, float)):
            last_value = np.array([last_value] * num_envs)
        elif isinstance(last_value, list):
            last_value = np.array(last_value)

        # Initialize returns and advantages
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)

        # Compute GAE for each environment separately
        for env_idx in range(num_envs):
            gae = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_value = last_value[env_idx]
                else:
                    next_value = values[t + 1, env_idx]

                delta = rewards[t, env_idx] + self.gamma * next_value * (1 - dones[t, env_idx]) - values[t, env_idx]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t, env_idx]) * gae
                returns[t, env_idx] = gae + values[t, env_idx]
                advantages[t, env_idx] = gae

        # Flatten back to 1D in the same order as input
        returns = returns.flatten()
        advantages = advantages.flatten()

        return returns, advantages

    def finish_path(self, last_values, num_envs):
        """개선된 finish_path - num_envs 파라미터 추가"""
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, dtype=torch.long).to(self.device)
        log_probs = torch.stack(self.log_probs).squeeze()

        # Compute returns and advantages with num_envs
        returns, advantages = self.compute_gae(
            self.rewards, self.values, self.dones, last_values, num_envs
        )

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        # Normalize advantages
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Store debugging info
        self.debug_metrics['value_predictions'].extend(self.values)
        self.debug_metrics['returns'].extend(returns.cpu().numpy().tolist())
        self.debug_metrics['advantages'].extend(advantages.cpu().numpy().tolist())

        # Clear buffers
        self.states, self.actions, self.log_probs, self.rewards, self.dones, self.values = [], [], [], [], [], []

        return states, actions, log_probs, returns, advantages

    def update(self, states, actions, old_log_probs, returns, advantages):
        """개선된 update with debugging metrics"""
        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        total_clip_fraction = 0
        total_grad_norm = 0
        num_batches = 0

        for epoch in range(self.update_epochs):
            epoch_kl = 0

            for s, a, old_log_p, ret, adv in loader:
                logits, value = self.network(s)
                dist = Categorical(logits=logits)
                log_p = dist.log_prob(a)

                # Calculate ratio and KL divergence
                ratio = (log_p - old_log_p).exp()
                with torch.no_grad():
                    kl = (old_log_p - log_p).mean()
                    epoch_kl += kl.item()

                # PPO loss
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = (ret - value.squeeze()).pow(2).mean()

                # Entropy
                entropy = dist.entropy().mean()

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping and norm tracking
                grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)

                self.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    clip_fraction = ((ratio - 1).abs() > self.clip_eps).float().mean()
                    total_clip_fraction += clip_fraction.item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl += kl.item()
                total_grad_norm += grad_norm.item()
                num_batches += 1

            # Early stopping based on KL
            if self.target_kl is not None and epoch_kl / len(loader) > self.target_kl:
                print(f"Early stopping at epoch {epoch} due to KL divergence")
                break


        # Store debugging metrics
        metrics = {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "entropy": total_entropy / num_batches,
            "kl_div": total_kl / num_batches,
            "clip_fraction": total_clip_fraction / num_batches,
            "grad_norm": total_grad_norm / num_batches,
        }

        # Update debug metrics
        self.debug_metrics['policy_losses'].append(metrics['policy_loss'])
        self.debug_metrics['value_losses'].append(metrics['value_loss'])
        self.debug_metrics['entropies'].append(metrics['entropy'])
        self.debug_metrics['kl_divs'].append(metrics['kl_div'])
        self.debug_metrics['clip_fractions'].append(metrics['clip_fraction'])
        self.debug_metrics['grad_norms'].append(metrics['grad_norm'])

        return metrics

    def train_epoch(self, env, epoch_steps, min_batch_size=None):
        """개선된 train_epoch - 데이터를 올바른 순서로 저장"""
        seed = np.random.randint(0, 10000)
        states, _ = env.reset(seed=seed)
        num_envs = env.num_envs
        step_count = 0

        if min_batch_size is None:
            min_batch_size = self.batch_size

        episode_rewards = []
        episode_lengths = []
        current_episode_rewards = np.zeros(num_envs)
        current_episode_lengths = np.zeros(num_envs)

        while step_count < epoch_steps:
            # 각 환경에 대해 행동 선택 및 저장
            actions = []
            for i in range(num_envs):
                a, log_p, v = self.act(states[i])
                actions.append(a)
                # 바로 store하여 올바른 순서 유지
                self.store(states[i], a, log_p, 0, False, v)  # reward와 done은 나중에 업데이트

            # 환경 스텝
            next_states, rewards, terminateds, truncateds, _ = env.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            # 저장된 데이터의 reward와 done 업데이트
            start_idx = len(self.rewards) - num_envs
            for i in range(num_envs):
                self.rewards[start_idx + i] = rewards[i]
                self.dones[start_idx + i] = dones[i]

            # 에피소드 통계 업데이트
            current_episode_rewards += rewards
            current_episode_lengths += 1

            for i in range(num_envs):
                if dones[i]:
                    episode_rewards.append(current_episode_rewards[i])
                    episode_lengths.append(current_episode_lengths[i])
                    current_episode_rewards[i] = 0
                    current_episode_lengths[i] = 0

            states = next_states
            step_count += num_envs
            self.total_steps += num_envs

            # 충분한 데이터가 모이면 업데이트
            if len(self.states) >= min_batch_size and (np.any(dones) or step_count >= epoch_steps):
                # 마지막 value 계산
                last_vals = []
                with torch.no_grad():
                    for i in range(num_envs):
                        state_tensor = torch.from_numpy(states[i]).float().unsqueeze(0).to(self.device)
                        _, last_val = self.network(state_tensor)
                        last_vals.append(last_val.item())

                # 업데이트
                s, a, log_p, ret, adv = self.finish_path(last_vals, num_envs)
                if len(s) > 0:
                    last_losses = self.update(s, a, log_p, ret, adv)
                    print(f"Updated with {len(s)} experiences | " +
                          f"Policy Loss: {last_losses['policy_loss']:.4f} | " +
                          f"Value Loss: {last_losses['value_loss']:.4f} | " +
                          f"KL: {last_losses['kl_div']:.4f}")

        # 남은 경험 처리
        if len(self.states) > 0:
            # 마지막 value 계산
            last_vals = []
            with torch.no_grad():
                for i in range(num_envs):
                    state_tensor = torch.from_numpy(states[i]).float().unsqueeze(0).to(self.device)
                    _, last_val = self.network(state_tensor)
                    last_vals.append(last_val.item())

            s, a, log_p, ret, adv = self.finish_path(last_vals, num_envs)
            if len(s) > 0:
                last_losses = self.update(s, a, log_p, ret, adv)
            else:
                last_losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        else:
            last_losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        stats = last_losses.copy()
        if episode_rewards:
            stats.update({
                "mean_episode_reward": np.mean(episode_rewards),
                "mean_episode_length": np.mean(episode_lengths),
                "episodes_completed": len(episode_rewards)
            })

        return stats

    def diagnose_performance(self):
        """성능 문제 진단"""
        print("\n=== PPO Performance Diagnosis ===")

        # 1. Policy Loss
        if len(self.debug_metrics['policy_losses']) > 0:
            recent_policy_loss = np.mean(list(self.debug_metrics['policy_losses'])[-10:])
            print(f"Recent Policy Loss: {recent_policy_loss:.4f}")
            if recent_policy_loss > 0.1:
                print("  ⚠️  High policy loss - consider reducing learning rate")

        # 2. Value Loss
        if len(self.debug_metrics['value_losses']) > 0:
            recent_value_loss = np.mean(list(self.debug_metrics['value_losses'])[-10:])
            print(f"Recent Value Loss: {recent_value_loss:.4f}")
            if recent_value_loss > 0.5:
                print("  ⚠️  High value loss - value function not learning well")

        # 3. Entropy
        if len(self.debug_metrics['entropies']) > 0:
            recent_entropy = np.mean(list(self.debug_metrics['entropies'])[-10:])
            print(f"Recent Entropy: {recent_entropy:.4f}")
            if recent_entropy < 0.1:
                print("  ⚠️  Low entropy - increase entropy coefficient")
            elif recent_entropy > 2.0:
                print("  ⚠️  High entropy - policy too random")

        # 4. KL Divergence
        if len(self.debug_metrics['kl_divs']) > 0:
            recent_kl = np.mean(list(self.debug_metrics['kl_divs'])[-10:])
            print(f"Recent KL Divergence: {recent_kl:.4f}")
            if recent_kl > 0.02:
                print("  ⚠️  High KL - updates too aggressive")

        # 5. Clip Fraction
        if len(self.debug_metrics['clip_fractions']) > 0:
            recent_clip = np.mean(list(self.debug_metrics['clip_fractions'])[-10:])
            print(f"Recent Clip Fraction: {recent_clip:.4f}")
            if recent_clip > 0.2:
                print("  ⚠️  High clipping - reduce learning rate or clip_eps")

        # 6. Gradient Norm
        if len(self.debug_metrics['grad_norms']) > 0:
            recent_grad = np.mean(list(self.debug_metrics['grad_norms'])[-10:])
            print(f"Recent Gradient Norm: {recent_grad:.4f}")
            if recent_grad > 0.5:
                print("  ⚠️  High gradients - potential instability")


        print("=====================================\n")

    def plot_debugging_metrics(self, save_path=None):
        """디버깅 메트릭 시각화"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('PPO Debugging Metrics', fontsize=16)

        metrics_to_plot = [
            ('policy_losses', 'Policy Loss'),
            ('value_losses', 'Value Loss'),
            ('entropies', 'Entropy'),
            ('kl_divs', 'KL Divergence'),
            ('clip_fractions', 'Clip Fraction'),
            ('grad_norms', 'Gradient Norm'),
        ]

        for idx, (metric_name, title) in enumerate(metrics_to_plot):
            if idx < 7:
                ax = axes[idx // 3, idx % 3]
                data = list(self.debug_metrics[metric_name])
                if data:
                    ax.plot(data)
                    ax.set_title(title)
                    ax.set_xlabel('Updates')
                    ax.grid(True)

        # Value predictions vs Returns
        ax = axes[2, 1]
        if self.debug_metrics['value_predictions'] and self.debug_metrics['returns']:
            values = list(self.debug_metrics['value_predictions'])[-1000:]
            returns = list(self.debug_metrics['returns'])[-1000:]
            ax.scatter(values, returns, alpha=0.5)
            ax.plot([min(values), max(values)], [min(values), max(values)], 'r--')
            ax.set_xlabel('Value Predictions')
            ax.set_ylabel('Actual Returns')
            ax.set_title('Value Function Accuracy')
            ax.grid(True)

        # Advantage distribution
        ax = axes[2, 2]
        if self.debug_metrics['advantages']:
            advantages = list(self.debug_metrics['advantages'])[-1000:]
            ax.hist(advantages, bins=50, alpha=0.7)
            ax.set_xlabel('Advantages')
            ax.set_ylabel('Frequency')
            ax.set_title('Advantage Distribution')
            ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()

    def analyze_action_distribution(self, env, num_steps=200):
        """현재 정책의 액션 분포 분석"""
        seed = np.random.randint(0, 10000)
        states, _ = env.reset(seed=seed)
        action_counts = np.zeros(env.single_action_space.n)
        entropy_sum = 0.0
        action_probs_history = []

        print("=== Action Distribution Analysis ===")

        for step in range(num_steps):
            actions = []
            entropies = []
            step_probs = np.zeros(env.single_action_space.n)

            for i in range(env.num_envs):
                state = torch.from_numpy(states[i]).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits, _ = self.network(state)
                dist = Categorical(logits=logits)
                probs = dist.probs.cpu().numpy()[0]

                action = dist.sample()
                actions.append(action.item())
                action_counts[action.item()] += 1
                entropies.append(dist.entropy().item())
                step_probs += probs

            step_probs /= env.num_envs
            action_probs_history.append(step_probs)
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
            print("   → Increase entropy coefficient")
            print(f"   → Current entropy_coef: {self.entropy_coef}")

        max_prob = max(action_counts) / total_actions
        if max_prob > 0.8:
            print(f"⚠️  DOMINANT ACTION: Action {np.argmax(action_counts)} dominates {max_prob:.1%}")
            print("   → Policy is stuck, increase exploration")
            print("   → Consider resetting optimizer or reducing learning rate")

        # 시간에 따른 액션 확률 변화 확인
        action_probs_history = np.array(action_probs_history)
        prob_std = np.std(action_probs_history, axis=0)
        print(f"\nAction probability std over time: {prob_std}")
        if np.max(prob_std) < 0.01:
            print("⚠️  STATIC POLICY: Action probabilities not changing")
            print("   → Policy might be stuck in local optimum")

        return action_counts, avg_entropy, action_probs_history
