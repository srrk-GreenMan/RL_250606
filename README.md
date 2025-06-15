# 🏎️ CarRacing-v3 Agent with Deep Q-Learning

This project implements a Deep Q-Network (DQN) agent to solve the [CarRacing-v3](https://www.gymlibrary.dev/environments/box2d/car_racing/) environment from Gymnasium. The agent learns from pixel observations using frame stacking, reward clipping, experience replay, and softmax or epsilon-greedy exploration strategies.
For scalable training, the code also provides an optional Ray-based vector environment.

---

## 📂 Project Structure
```text
.
├── train.py            # Main training script
├── agent.py            # Agent logic (Deep Q-learning)
├── buffer.py           # Replay buffer for experience replay
├── car_racing_env.py   # Preprocessing wrapper for CarRacing-v3
├── exploration.py      # Exploration strategies
├── model.py            # Q-network
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
