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
        normalize_advantages=True,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.normalize_advantages = normalize_advantages

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # 수정: tensor 형태로 저장하여 gradient tracking 보존
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
        print(f"Normalize Advantages: {self.normalize_advantages}")
        print("======================================")

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        logits, value = self.network(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

    @torch.no_grad()
    def select_best_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        logits, _ = self.network(state)
        return int(torch.argmax(logits, dim=1).item())

    def store(self, state, action, log_prob, reward, done, value):
        # 수정: state를 tensor로 변환하여 저장 (효율성 개선)
        state_tensor = torch.from_numpy(state).float().to(self.device)
        self.states.append(state_tensor)
        self.actions.append(action)
        # 수정: log_prob와 value를 tensor 형태로 저장 (gradient tracking 보존)
        self.log_probs.append(log_prob.detach())
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value.detach())

    def finish_path(self, last_value=0):
        # 수정: 이미 tensor로 저장되어 있으므로 stack만 하면 됨
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, dtype=torch.long).to(self.device)
        log_probs = torch.stack(self.log_probs).squeeze()
        
        # values는 이미 tensor이므로 stack하고 last_value 추가
        values_tensor = torch.stack(self.values).squeeze()
        last_value_tensor = torch.tensor(last_value, dtype=torch.float32).to(self.device)
        values = torch.cat([values_tensor, last_value_tensor.unsqueeze(0)])

        # GAE 계산
        rewards = []
        gae = 0
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + self.gamma * values[i + 1] * (1 - self.dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[i]) * gae
            rewards.insert(0, gae + values[i])
        
        returns = torch.stack(rewards)
        advantages = returns - values[:-1]
        
        # 수정: advantage normalization 추가
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 메모리 초기화
        self.states, self.actions, self.log_probs, self.rewards, self.dones, self.values = [], [], [], [], [], []
        
        # 수정: 이미 적절히 detach되어 있으므로 중복 detach 제거
        return states, actions, log_probs, returns, advantages

    def update(self, states, actions, old_log_probs, returns, advantages):
        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_batches = 0

        for epoch in range(self.update_epochs):
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
                # 수정: gradient clipping 추가 (학습 안정성)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_batches += 1

        return {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "entropy": total_entropy / num_batches
        }

    def train_epoch(self, env, epoch_steps, min_batch_size=None):
        # 수정: 시드 랜덤화로 overfitting 방지
        seed = np.random.randint(0, 10000)
        states, _ = env.reset(seed=seed)
        num_envs = env.num_envs
        step_count = 0
        
        # 수정: 최소 배치 크기 설정
        if min_batch_size is None:
            min_batch_size = self.batch_size
        
        episode_rewards = []
        episode_lengths = []
        current_episode_rewards = np.zeros(num_envs)
        current_episode_lengths = np.zeros(num_envs)
        
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

            # 에피소드 통계 업데이트
            current_episode_rewards += rewards
            current_episode_lengths += 1

            for i in range(num_envs):
                self.store(states[i], actions[i], log_probs[i], rewards[i], float(dones[i]), values[i])
                
                # 에피소드 종료 시 통계 기록
                if dones[i]:
                    episode_rewards.append(current_episode_rewards[i])
                    episode_lengths.append(current_episode_lengths[i])
                    current_episode_rewards[i] = 0
                    current_episode_lengths[i] = 0

            states = next_states
            step_count += num_envs
            self.total_steps += num_envs

            # 수정: 충분한 경험이 쌓였거나 에피소드가 끝났을 때 업데이트
            if len(self.states) >= min_batch_size and (np.any(dones) or step_count >= epoch_steps):
                # 마지막 value 계산
                last_vals = []
                for i in range(num_envs):
                    with torch.no_grad():
                        state_tensor = torch.from_numpy(states[i]).float().unsqueeze(0).to(self.device)
                        _, last_val = self.network(state_tensor)
                        last_vals.append(last_val.item())
                
                s, a, log_p, ret, adv = self.finish_path(np.mean(last_vals))
                if len(s) > 0:
                    last_losses = self.update(s, a, log_p, ret, adv)
                    print(f"Updated with {len(s)} experiences")

        # 수정: 남은 경험이 있다면 처리
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
                last_losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        else:
            print('Warning: No experiences collected during this epoch!')
            last_losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        # 에피소드 통계 추가
        stats = last_losses.copy()
        if episode_rewards:
            stats.update({
                "mean_episode_reward": np.mean(episode_rewards),
                "mean_episode_length": np.mean(episode_lengths),
                "episodes_completed": len(episode_rewards)
            })
        
        return stats

    def analyze_action_distribution(self, env, num_steps=200):
        """현재 정책의 액션 분포 분석"""
        # 수정: 시드 랜덤화
        seed = np.random.randint(0, 10000)
        states, _ = env.reset(seed=seed)
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
