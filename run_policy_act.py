#!/usr/bin/env python3
"""Run a fine-tuned ACT policy in Isaac Sim (closed-loop visual inference).

Usage
-----
    python run_policy_act.py --checkpoint outputs/act_v0/checkpoints/last [--headless]
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="outputs/act_v0/checkpoints/last")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--config", default="configs/env_force_reach.yaml")
args, unknown = parser.parse_known_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
import torch
import yaml

from envs.ur_force_env import URForceReachEnv

with open(args.config) as f:
    cfg = yaml.safe_load(f)
env_cfg = cfg.get("env", {})

env = URForceReachEnv(
    render_mode="human" if not args.headless else None,
    action_scale=env_cfg.get("action_scale", 0.3),
    target_radius=env_cfg.get("target_radius", 0.05),
    reach_bonus=env_cfg.get("reach_bonus", 10.0),
    max_episode_steps=env_cfg.get("max_episode_steps", 500),
    control_mode=env_cfg.get("control_mode", "position"),
    impedance_gain=env_cfg.get("impedance_gain", 0.0),
    impedance_kp=env_cfg.get("impedance_kp", 400.0),
    impedance_kd=env_cfg.get("impedance_kd", 40.0),
    max_effort=env_cfg.get("max_effort", 150.0),
    force_clip=env_cfg.get("force_clip", 200.0),
    enable_camera=env_cfg.get("enable_camera", True),
    camera_resolution=tuple(env_cfg.get("camera_resolution", (224, 224))),
)

from lerobot.policies.act.modeling_act import ACTPolicy

policy = ACTPolicy.from_pretrained(args.checkpoint)
policy.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
policy.to(device)

print(f"Loaded ACT policy from {args.checkpoint}")
successes = 0

for ep in range(args.episodes):
    obs, _ = env.reset()
    policy.reset()
    done = False
    total_reward = 0.0

    while not done:
        batch = {
            "observation.state": torch.from_numpy(obs).float().unsqueeze(0).to(device),
        }
        if env_cfg.get("enable_camera", True):
            img = env._get_wrist_image()
            img_t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            batch["observation.images.wrist"] = img_t.to(device)

        with torch.no_grad():
            action = policy.select_action(batch)
        action = action.squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    if info.get("reached"):
        successes += 1
    print(f"  Episode {ep + 1:3d} | reward={total_reward:8.2f} | dist={info['dist']:.4f}m | reached={info['reached']}")

print(f"\nSuccess rate: {successes}/{args.episodes} ({100 * successes / args.episodes:.1f}%)")

env.close()
simulation_app.close()
