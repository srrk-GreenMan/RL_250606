# train.py

import os
import argparse
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch

from agent import Agent
from car_racing_env import CarRacingEnv
from gymnasium.vector import AsyncVectorEnv

from exploration import EpsilonGreedy, SoftmaxExploration

# === 1. Config 로딩 ===
def load_config(path):
    with open(path, "r") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.safe_load(f)
        elif path.endswith(".json"):
            return json.load(f)
        else:
            raise ValueError("Unsupported config format")

# === 2. Exploration Strategy 생성 ===
EXPLORATION_MAP = {
    "EpsilonGreedy": EpsilonGreedy,
    "SoftmaxExploration": SoftmaxExploration,
}

def build_exploration_strategy(cfg, action_dim):
    name = cfg.get("name", "SoftmaxExploration")
    params = cfg.get("params", {})

    cls = EXPLORATION_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown exploration strategy: {name}")

    return cls(action_dim=action_dim, **params)

# === 3. 평가 함수 ===
def evaluate(agent, n_evals):
    eval_env = CarRacingEnv()
    total_score = 0
    for i in range(n_evals):
        s, _ = eval_env.reset(seed=i)
        done, score = False, 0
        while not done:
            a = agent.select_best_action(s)
            s, r, terminated, truncated, _ = eval_env.step(a)
            score += r
            done = terminated or truncated
        total_score += score
    return np.round(total_score / n_evals, 4)

# === 4. 벡터 환경 생성 함수 ===
def make_env():
    def _thunk():
        return CarRacingEnv()
    return _thunk

# === 5. 메인 학습 함수 ===
def main(config):
    os.makedirs(config["model_dir"], exist_ok=True)

    env_fns = [make_env() for _ in range(config["num_envs"])]
    parallel_env = AsyncVectorEnv(env_fns)

    state_dim = parallel_env.single_observation_space.shape
    action_dim = parallel_env.single_action_space.n

    # Agent 생성
    agent_cfg = config["agent"]
    exploration_cfg = agent_cfg.get("exploration_strategy", {})
    exploration_strategy = build_exploration_strategy(exploration_cfg, action_dim)

    agent = Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        exploration_strategy=exploration_strategy,
        lr=agent_cfg.get("lr", 0.00025),
        gamma=agent_cfg.get("gamma", 0.99),
        batch_size=agent_cfg.get("batch_size", 64),
        warmup_steps=agent_cfg.get("warmup_steps", 5000),
        buffer_size=agent_cfg.get("buffer_size", int(1e5)),
        target_update_interval=agent_cfg.get("target_update_interval", 5000),
        use_double_q=agent_cfg.get("use_double_q", False),
        use_per=agent_cfg.get("use_per", False)
    )
    agent.summary()

    history = {"Epoch": [], "AvgReturn": []}
    best_return = -float('inf')
    best_model = None

    for epoch in range(1, config["num_epochs"] + 1):
        agent.train_epoch(parallel_env, config["epoch_steps"])

        if epoch % config["eval_interval"] == 0:
            avg_return = evaluate(agent, config["eval_episodes"])
            history["Epoch"].append(epoch)
            history["AvgReturn"].append(avg_return)
            print(f"[Epoch {epoch}] Steps: {agent.total_steps} | Eval Score: {avg_return}")

            if avg_return > best_return:
                best_return = avg_return
                best_model = agent.network.state_dict()
                print(f">>> New best model saved at epoch {epoch} with return {best_return:.2f}")
                torch.save(best_model, os.path.join(config["model_dir"], "best_model.pth"))

    # 저장
    plt.figure(figsize=(6, 4))
    plt.plot(history["Epoch"], history["AvgReturn"], marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Average Return")
    plt.title("Training Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config["model_dir"], "training_curve.png"))
    plt.close()

    torch.save(best_model, os.path.join(config["model_dir"], "best_model.pth"))

# === 6. Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CarRacing Agent (Config Only)")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml or config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)