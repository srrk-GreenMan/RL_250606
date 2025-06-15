# 🏎️ CarRacing-v3 Agents

This project provides both a Deep Q-Network (DQN) agent and a lightweight PPO actor-critic agent. Each learns from stacked pixel observations, and training can optionally use a Ray-based vector environment.

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

```bash
python train.py --config config.yaml
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

📌 Acknowledgments
- Built using OpenAI Gymnasium
- Inspired by DQN and classic Atari RL pipelines
