#!/usr/bin/env python3
"""Train a SAC policy on the UR10 reach task in Isaac Sim.

Usage
-----
    python train.py [--headless] [--timesteps 1000000] [--config configs/sac_reach.yaml]

Isaac Sim's SimulationApp MUST be created before any omni/isaacsim imports,
so this file handles that at the very top.
"""

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--timesteps", type=int, default=1_000_000)
parser.add_argument("--config", default="configs/sac_reach.yaml")
args, unknown = parser.parse_known_args()

# --- Launch Isaac Sim (must happen before all other imports) ---------------
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})
# ---------------------------------------------------------------------------

import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from envs.ur_reach_env import URReachEnv

LOG_DIR = "logs"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Load config
with open(args.config) as f:
    cfg = yaml.safe_load(f)
env_cfg = cfg.get("env", {})

# Build envs
env = URReachEnv(
    render_mode="human" if not args.headless else None,
    action_scale=env_cfg.get("action_scale", 0.3),
    target_radius=env_cfg.get("target_radius", 0.05),
    reach_bonus=env_cfg.get("reach_bonus", 10.0),
    max_episode_steps=env_cfg.get("max_episode_steps", 500),
)

print("Checking environment...")
check_env(env, warn=True)

# Isaac Sim's World is a process-wide singleton (one USD stage), so a second
# URReachEnv can't build its own robot/ground without colliding with the
# training env's prims. Reuse the same env for evaluation instead.
eval_env = env

callbacks = [
    EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=cfg.get("eval_freq", 20_000),
        n_eval_episodes=cfg.get("n_eval_episodes", 10),
        deterministic=True,
        verbose=1,
    ),
    CheckpointCallback(
        save_freq=50_000,
        save_path=MODEL_DIR,
        name_prefix="ur10_reach",
    ),
]

model = SAC(
    cfg.get("policy", "MlpPolicy"),
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=cfg.get("learning_rate", 3e-4),
    buffer_size=cfg.get("buffer_size", 300_000),
    batch_size=cfg.get("batch_size", 256),
    gamma=cfg.get("gamma", 0.99),
    tau=cfg.get("tau", 0.005),
    ent_coef=cfg.get("ent_coef", "auto"),
    train_freq=cfg.get("train_freq", 1),
    gradient_steps=cfg.get("gradient_steps", 1),
    learning_starts=cfg.get("learning_starts", 5_000),
)

print(f"Starting SAC training — {args.timesteps:,} steps on UR10 reach task...")
model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
model.save(f"{MODEL_DIR}/ur10_reach_final")
print("Training complete. Model saved to models/ur10_reach_final.zip")

env.close()
eval_env.close()
simulation_app.close()
