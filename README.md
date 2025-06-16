# 🏎️ CarRacing-v3 Agents

This project provides both a Deep Q-Network (DQN) agent and a lightweight PPO actor-critic agent. Each learns from stacked pixel observations, and training can optionally use a Ray-based vector environment. The DQN implementation now relies on an Atari-style CNN backbone for processing frames.

---

## 📂 Project Structure
```text
.
├── train.py            # Main training script
├── agent.py            # DQN agent logic
├── ppo_agent.py        # PPO actor-critic agent
├── buffer.py           # Replay buffer for experience replay
├── car_racing_env.py   # Preprocessing wrapper for CarRacing-v3
├── exploration.py      # Exploration strategies
├── model.py            # Networks (Q-network and Actor-Critic)
├── ray_env.py          # Optional Ray-based vector environment
└── config.yaml         # Training configuration
```
---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install swig gymnasium[box2d] imageio imageio-ffmpeg ray
```

### 2. Train the Agent

Choose the appropriate config file for the algorithm you want to run:

```bash
python train.py --config dqn_config.yaml  # Deep Q-Network
python train.py --config ppo_config.yaml  # Proximal Policy Optimization
python train.py --config sac_config.yaml  # Soft Actor-Critic
```

This will:
- Train the agent using `AsyncVectorEnv` (default) or `RayVectorEnv` when `use_ray: true`
- Evaluate the policy every epoch
- Save the best model as `best_model` variable
- Save a video of the best-performing episode in the `videos/` folder

### 3. Environment Details
- Environment: CarRacing-v3 (discrete mode)
- Observation: 4 stacked grayscale frames (4x84×84)
- Action space: 5 discrete actions (left, right, gas, brake, no-op)
- Preprocessing: crop, resize, grayscale, normalize

### 4. Evaluation
- After training, the script evaluates the best model on 10 random seeds and saves the best rollout as an .mp4 file for qualitative analysis.

### 5. Recommended Hyperparameters

```yaml
DQN:
  lr: [1e-4, 5e-4]
  gamma: [0.98, 0.999]
  batch_size: [32, 64]
  warmup_steps: [5000, 10000]
  buffer_size: [100000, 500000]
  target_update_interval: [5000, 10000]
  use_double_q: [true, false]

PPO:
  lr: [1e-4, 3e-4]
  gamma: [0.98, 0.99]
  gae_lambda: [0.9, 0.95]
  clip_eps: [0.1, 0.2]
  update_epochs: [4, 10]
  batch_size: [32, 64]

SAC:
  lr: [1e-4, 3e-4]
  gamma: [0.98, 0.99]
  tau: [0.005, 0.01]
  alpha: [0.1, 0.2]
  batch_size: [256, 512]
  buffer_size: [100000, 1000000]
  warmup_steps: [1000, 5000]
```

📌 Acknowledgments
- Built using OpenAI Gymnasium
- Inspired by DQN and classic Atari RL pipelines
