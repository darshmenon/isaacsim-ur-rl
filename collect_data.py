#!/usr/bin/env python3
"""Roll out episodes in Isaac Sim and record them into a LeRobotDataset.

Motion source is the existing trained SAC checkpoint from isaacsim-ur-rl
(read-only), driving the new force-augmented env so we get contact-rich
trajectories without having to script one by hand for v0.

Usage
-----
    python collect_data.py [--headless] [--config configs/env_force_reach.yaml]

Isaac Sim's SimulationApp MUST be created before any omni/isaacsim imports,
so this file handles that at the very top (same constraint as train.py).
"""

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--config", default="configs/env_force_reach.yaml")
args, unknown = parser.parse_known_args()

# --- Launch Isaac Sim (must happen before all other imports) ---------------
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})
# ---------------------------------------------------------------------------

import numpy as np
import yaml

from envs.ur_force_env import URForceReachEnv

with open(args.config) as f:
    cfg = yaml.safe_load(f)
env_cfg = cfg.get("env", {})
col_cfg = cfg.get("collection", {})

env = URForceReachEnv(
    render_mode="human" if not args.headless else None,
    action_scale=env_cfg.get("action_scale", 0.3),
    target_radius=env_cfg.get("target_radius", 0.05),
    reach_bonus=env_cfg.get("reach_bonus", 10.0),
    max_episode_steps=env_cfg.get("max_episode_steps", 500),
    impedance_gain=env_cfg.get("impedance_gain", 0.0),
    force_clip=env_cfg.get("force_clip", 200.0),
    enable_camera=env_cfg.get("enable_camera", True),
    camera_resolution=tuple(env_cfg.get("camera_resolution", (224, 224))),
)

# --- Motion source: existing SAC checkpoint from the reach task -----------
source_policy_path = col_cfg.get(
    "source_policy", "models/ur10_reach_final.zip"
)
policy = None
if os.path.exists(source_policy_path):
    from stable_baselines3 import SAC
    policy = SAC.load(source_policy_path, env=None)
    print(f"Loaded motion-source policy from {source_policy_path}")
else:
    print(
        f"WARNING: source_policy '{source_policy_path}' not found — "
        "falling back to random actions for this collection run."
    )

# --- LeRobotDataset setup ---------------------------------------------------
from lerobot.datasets.lerobot_dataset import LeRobotDataset

cam_h, cam_w = tuple(env_cfg.get("camera_resolution", (224, 224)))
features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (21,),
        "names": {
            "axes": [
                "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
                "joint_vel_0", "joint_vel_1", "joint_vel_2", "joint_vel_3", "joint_vel_4", "joint_vel_5",
                "target_x", "target_y", "target_z",
                "force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z",
            ],
        },
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": {"axes": [f"joint_delta_{i}" for i in range(6)]},
    },
}
if env_cfg.get("enable_camera", True):
    features["observation.images.wrist"] = {
        "dtype": "video",
        "shape": (cam_h, cam_w, 3),
        "names": ["height", "width", "channels"],
    }

dataset = LeRobotDataset.create(
    repo_id=col_cfg.get("repo_id", "ur10-force-vla/reach-contact-v0"),
    fps=col_cfg.get("fps", 50),
    root=col_cfg.get("output_root", "./datasets/reach-contact-v0"),
    robot_type="ur10",
    features=features,
    use_videos=env_cfg.get("enable_camera", True),
)

task_instruction = col_cfg.get("task_instruction", "reach the target and make light contact")
n_episodes = col_cfg.get("n_episodes", 20)

for ep in range(n_episodes):
    obs, _ = env.reset()
    done = False
    step_i = 0
    while not done:
        if policy is not None:
            action, _ = policy.predict(obs[:15], deterministic=True)
        else:
            action = env.action_space.sample()

        frame = {
            "observation.state": obs.astype(np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "task": task_instruction,
        }
        if env_cfg.get("enable_camera", True):
            frame["observation.images.wrist"] = env._get_wrist_image()

        dataset.add_frame(frame)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_i += 1

    dataset.save_episode()
    print(f"Episode {ep + 1}/{n_episodes} recorded ({step_i} steps, reached={info['reached']})")

dataset.finalize()
print(f"Dataset saved to {col_cfg.get('output_root', './datasets/reach-contact-v0')}")

env.close()
simulation_app.close()
