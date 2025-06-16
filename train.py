# train.py

import os
import argparse
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
import wandb

from utils import set_seed

from agent import Agent
from ppo_agent import PPOAgent
from sac_agent import SACAgent
from car_racing_env import CarRacingEnv
from gymnasium.vector import AsyncVectorEnv
from ray_env import RayVectorEnv
from ray import tune

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

AGENT_MAP = {
    "DQN": Agent,
    "PPO": PPOAgent,
    "SAC": SACAgent,
}

# Recommended hyperparameter ranges for grid search
SEARCH_SPACE = {
    "DQN": {
        "lr": [1e-4, 5e-4],
        "gamma": [0.98, 0.999],
        "batch_size": [32, 64],
        "warmup_steps": [5000, 10000],
        "buffer_size": [100000, 500000],
        "target_update_interval": [5000, 10000],
        "use_double_q": [True, False],
    },
    "PPO": {
        "lr": [1e-4, 3e-4],
        "gamma": [0.98, 0.99],
        "gae_lambda": [0.9, 0.95],
        "clip_eps": [0.1, 0.2],
        "update_epochs": [4, 10],
        "batch_size": [32, 64],
    },
    "SAC": {
        "lr": [1e-4, 3e-4],
        "gamma": [0.98, 0.99],
        "tau": [0.005, 0.01],
        "alpha": [0.1, 0.2],
        "batch_size": [256, 512],
        "buffer_size": [100000, 1000000],
        "warmup_steps": [1000, 5000],
    },
}

def build_exploration_strategy(cfg, action_dim):
    name = cfg.get("name", "SoftmaxExploration")
    params = cfg.get("params", {})

    cls = EXPLORATION_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown exploration strategy: {name}")

    return cls(action_dim=action_dim, **params)

# === 3. 평가 함수 ===
def evaluate(agent, n_evals, env_kwargs):
    eval_env = CarRacingEnv(**env_kwargs)
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
    eval_env.close()
    return np.round(total_score / n_evals, 4)

# === 4. 기록 저장 함수 ===
def save_history(history, model_dir):
    """Save training history as JSON."""
    path = os.path.join(model_dir, "history.json")
    with open(path, "w") as f:
        json.dump(history, f)

# === 5. 벡터 환경 생성 함수 ===
def make_env(env_kwargs):
    def _thunk():
        return CarRacingEnv(**env_kwargs)
    return _thunk

# === 6. Hyperparameter Optimization ===

def _tune_train(config, base_config=None, project=None):
    cfg = copy.deepcopy(base_config)
    cfg["agent"].update(config["agent"])
    trial_name = tune.get_trial_name()
    cfg["model_dir"] = os.path.join(base_config["model_dir"], trial_name)
    score = main(cfg, project=project, run_name=trial_name)
    tune.report(score=score)


def run_hpo(base_config, project=None):
    agent_type = base_config.get("agent_type", "DQN")
    space = SEARCH_SPACE.get(agent_type)
    if space is None:
        raise ValueError(f"No search space defined for {agent_type}")

    param_space = {"agent": {k: tune.grid_search(v) for k, v in space.items()}}

    tuner = tune.Tuner(
        tune.with_parameters(_tune_train, base_config=base_config, project=project),
        param_space=param_space,
    )
    results = tuner.fit()
    best = results.get_best_result(metric="score", mode="max")
    print("Best Params", best.config["agent"], "Score", best.metrics.get("score"))

# === 6. 메인 학습 함수 ===
def main(config, project=None, run_name=None):
    os.makedirs(config["model_dir"], exist_ok=True)

    set_seed(config.get("seed", 42))

    run = None
    if project:
        run = wandb.init(project=project, name=run_name, config=config)

    env_kwargs = config.get("env", {})
    if config.get("agent_type") == "SAC":
        # SAC requires a continuous action space
        env_kwargs["continuous"] = True

    if config.get("use_ray", False):
        parallel_env = RayVectorEnv(config["num_envs"])
    else:
        env_fns = [make_env(env_kwargs) for _ in range(config["num_envs"])]
        parallel_env = AsyncVectorEnv(env_fns)

    state_dim = parallel_env.single_observation_space.shape
    action_space = parallel_env.single_action_space
    if hasattr(action_space, "n"):
        action_dim = action_space.n
    else:
        action_dim = int(np.prod(action_space.shape))

    # Agent 생성
    agent_cfg = config["agent"]
    agent_type = config.get("agent_type", "DQN")

    if agent_type == "PPO":
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=agent_cfg.get("lr", 3e-4),
            gamma=agent_cfg.get("gamma", 0.99),
            gae_lambda=agent_cfg.get("gae_lambda", 0.95),
            clip_eps=agent_cfg.get("clip_eps", 0.2),
            update_epochs=agent_cfg.get("update_epochs", 4),
            batch_size=agent_cfg.get("batch_size", 64),
        )
    elif agent_type == "SAC":
        agent = SACAgent(
            observation_space=parallel_env.single_observation_space,
            action_space=parallel_env.single_action_space,
            lr=agent_cfg.get("lr", 3e-4),
            gamma=agent_cfg.get("gamma", 0.99),
            tau=agent_cfg.get("tau", 0.005),
            alpha=agent_cfg.get("alpha", 0.2),
            batch_size=agent_cfg.get("batch_size", 256),
            buffer_size=agent_cfg.get("buffer_size", int(1e5)),
            warmup_steps=agent_cfg.get("warmup_steps", 1000),
        )
    else:
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
            prioritized=agent_cfg.get("prioritized", True),
            alpha=agent_cfg.get("alpha", 0.6),
            beta=agent_cfg.get("beta", 0.4),
            beta_increment=agent_cfg.get("beta_increment", 1e-6),
            epsilon=agent_cfg.get("epsilon", 1e-6),
        )
    agent.summary()

    history = {"Epoch": [], "AvgReturn": []}
    best_return = -float('inf')
    best_model = None

    for epoch in range(1, config["num_epochs"] + 1):
        agent.train_epoch(parallel_env, config["epoch_steps"])

        if epoch % config["eval_interval"] == 0:
            avg_return = evaluate(agent, config["eval_episodes"], env_kwargs)
            history["Epoch"].append(epoch)
            history["AvgReturn"].append(avg_return)
            print(f"[Epoch {epoch}] Steps: {agent.total_steps} | Eval Score: {avg_return}")
            if run:
                wandb.log({"epoch": epoch, "avg_return": avg_return, "total_steps": agent.total_steps})

            if avg_return > best_return:
                best_return = avg_return
                if hasattr(agent, "actor"):
                    best_model = agent.actor.state_dict()
                else:
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
    save_history(history, config["model_dir"])

    torch.save(best_model, os.path.join(config["model_dir"], "best_model.pth"))

    parallel_env.close()
    if run:
        run.finish()

    return best_return

# === 7. Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CarRacing Agent (Config Only)")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml or config.json")
    parser.add_argument("--hpo", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--wandb_project", type=str, default=None, help="Weights & Biases project name")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.hpo:
        run_hpo(config, project=args.wandb_project)
    else:
        main(config, project=args.wandb_project, run_name="single_run")
