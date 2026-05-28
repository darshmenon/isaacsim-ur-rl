# isaacsim-ur-rl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.x-76b900?logo=nvidia)](https://developer.nvidia.com/isaac-sim)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)

Reinforcement learning for UR arm reach and pick-and-place tasks in **NVIDIA Isaac Sim**, using **SAC** (Stable Baselines 3) with a custom gym environment backed by the Isaac Sim physics engine.

---

## What it does

- Trains a SAC policy to control a UR10 arm to reach randomized 3-D targets in Isaac Sim
- Multi-arm variant (`ur_reach_multi_env.py`) for parallel training across several robot instances
- Saves checkpoints every 50k steps; best model saved automatically via `EvalCallback`
- Runs trained policies in visual or headless mode

---

## Structure

```
isaacsim-ur-rl/
├── train.py              # SAC training entry point
├── run_policy.py         # Load and visualize a trained policy
├── envs/
│   ├── ur_reach_env.py       # Single UR10 gym environment
│   └── ur_reach_multi_env.py # Multi-robot variant
├── configs/
│   └── sac_reach.yaml    # Hyperparameters and env config
├── models/               # Saved policy checkpoints (gitignored)
└── logs/                 # TensorBoard training logs
```

---

## Requirements

- NVIDIA Isaac Sim 4.x (via Omniverse launcher or pip install)
- Python 3.10+
- CUDA-capable GPU

```bash
pip install -r requirements.txt
```

---

## Train

```bash
# Headless (faster)
python train.py --headless --timesteps 1000000

# With GUI
python train.py --timesteps 500000

# Custom config
python train.py --headless --config configs/sac_reach.yaml
```

Checkpoints saved to `models/`. Best model: `models/best_model.zip`.

---

## Run a Trained Policy

```bash
python run_policy.py --model models/best_model.zip --episodes 10

# Headless inference
python run_policy.py --model models/best_model.zip --headless
```

---

## Config

Edit `configs/sac_reach.yaml` to tune hyperparameters:

```yaml
policy: MlpPolicy
learning_rate: 0.0003
buffer_size: 300000
batch_size: 256
gamma: 0.99
tau: 0.005
ent_coef: auto
train_freq: 1
gradient_steps: 1
learning_starts: 5000
eval_freq: 20000
n_eval_episodes: 10

env:
  action_scale: 0.3
  target_radius: 0.05
  reach_bonus: 10.0
  max_episode_steps: 500
```

---

## Author

**darshmenon** — [github.com/darshmenon](https://github.com/darshmenon)
