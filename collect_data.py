#!/usr/bin/env python3
"""Roll out episodes in Isaac Sim and dump them to raw .npz files on disk.

Isaac Sim requires Python 3.10 (this repo's .venv), but `lerobot` requires
Python >=3.12 -- the two can't be imported in the same process. So this
script does NOT depend on lerobot at all; it just writes raw per-episode
arrays (state, action, wrist frames, task string). A separate script,
package_dataset.py, run under .venv312 (which has lerobot installed),
reads these raw files and builds the actual LeRobotDataset.

Motion source is a simple proportional/bang-bang reach controller (NOT
Isaac Sim's RMPFlowController -- diagnosed via an isolated live check that
RMPFlowController.forward() returns bit-identical output regardless of the
target position with attach_gripper=False; likely its preset "UR10"/
"RMPflow" config's internal kinematic model doesn't match our gripper-less
robot structure). Base-joint yaw is driven proportionally by the azimuthal
angle error to the target; shoulder/elbow use the same periodic sign-flip
(track dist-to-target, flip if not improving, lock once close) validated
working for milestone-1 contact verification. Cruder than real IK, but
target-differentiated and doesn't depend on RMPFlow's config matching this
robot. NOTE: the old SAC checkpoint from ur_reach_env.py was trained under
teleport-based position control, so its actions don't transfer meaningfully
to this env's real-dynamics control modes either -- hence a fresh scripted
source rather than reusing it.

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
    control_mode=env_cfg.get("control_mode", "position"),
    impedance_gain=env_cfg.get("impedance_gain", 0.0),
    impedance_kp=env_cfg.get("impedance_kp", 400.0),
    impedance_kd=env_cfg.get("impedance_kd", 40.0),
    max_effort=env_cfg.get("max_effort", 150.0),
    force_clip=env_cfg.get("force_clip", 200.0),
    enable_camera=env_cfg.get("enable_camera", True),
    camera_resolution=tuple(env_cfg.get("camera_resolution", (224, 224))),
    attach_gripper=env_cfg.get("attach_gripper", False),
)

# --- Motion source: proportional yaw + bang-bang shoulder/elbow reach -----
CHECK_INTERVAL = 150
LOCK_DIST = 0.15  # once this close, stop sign-flipping and just hold direction


def azimuth_error(ee_pos, target_pos):
    cur = np.arctan2(ee_pos[1], ee_pos[0])
    tgt = np.arctan2(target_pos[1], target_pos[0])
    err = tgt - cur
    return (err + np.pi) % (2 * np.pi) - np.pi


print("Using scripted proportional-yaw + bang-bang-reach as the demonstration source")

raw_dir = col_cfg.get("raw_output_dir", "./datasets/raw_reach-contact-v0")
os.makedirs(raw_dir, exist_ok=True)

task_instruction = col_cfg.get("task_instruction", "reach the target and make light contact")
n_episodes = col_cfg.get("n_episodes", 20)
enable_camera = env_cfg.get("enable_camera", True)

for ep in range(n_episodes):
    obs, _ = env.reset()
    done = False
    states, actions, frames = [], [], []
    sign = 1.0
    best_dist = None
    locked = False
    step_i = 0

    while not done:
        ee_pos = env._get_ee_pos()
        az_err = azimuth_error(ee_pos, env._target_pos)
        dist = float(np.linalg.norm(ee_pos - env._target_pos))

        if not locked and dist < LOCK_DIST:
            locked = True
        if not locked and step_i % CHECK_INTERVAL == 0:
            if best_dist is not None and dist >= best_dist - 0.01:
                sign *= -1.0
            best_dist = dist

        action = np.array(
            [np.clip(az_err / (np.pi / 2), -1.0, 1.0), sign, sign, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

        states.append(obs.astype(np.float32))
        actions.append(np.asarray(action, dtype=np.float32))
        if enable_camera:
            frames.append(env._get_wrist_image().copy())

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_i += 1

    ep_path = os.path.join(raw_dir, f"ep_{ep:04d}.npz")
    save_kwargs = dict(
        states=np.stack(states),
        actions=np.stack(actions),
        task=task_instruction,
    )
    if enable_camera:
        save_kwargs["frames"] = np.stack(frames)
    np.savez_compressed(ep_path, **save_kwargs)
    print(f"Episode {ep + 1}/{n_episodes} recorded ({len(states)} steps, reached={info['reached']}) -> {ep_path}")

print(f"\nRaw episodes written to {raw_dir}")
print("Next: package into a LeRobotDataset with (from .venv312):")
print(f"  .venv312/bin/python package_dataset.py --config {args.config}")

env.close()
simulation_app.close()
