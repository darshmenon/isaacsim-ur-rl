#!/usr/bin/env python3
"""Run a trained SAC policy in Isaac Sim (visual inference).

Usage
-----
    python run_policy.py --model models/best_model.zip [--headless]
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/best_model.zip")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--episodes", type=int, default=10)
args, unknown = parser.parse_known_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
from stable_baselines3 import SAC
from envs.ur_reach_env import URReachEnv

env = URReachEnv(render_mode="human" if not args.headless else None)
model = SAC.load(args.model, env=env)

print(f"Loaded policy from {args.model}")
successes = 0

for ep in range(args.episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    if info.get("reached"):
        successes += 1
    print(f"  Episode {ep + 1:3d} | reward={total_reward:8.2f} | dist={info['dist']:.4f}m | reached={info['reached']}")

print(f"\nSuccess rate: {successes}/{args.episodes} ({100 * successes / args.episodes:.1f}%)")

env.close()
simulation_app.close()
