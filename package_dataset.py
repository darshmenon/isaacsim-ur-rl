#!/usr/bin/env python3
"""Package raw episodes (written by collect_data.py) into a LeRobotDataset.

Run this under .venv312 (Python 3.12, lerobot installed) -- NOT the Isaac
Sim .venv (Python 3.10). It has no Isaac Sim dependency at all, only reads
the .npz files collect_data.py wrote to disk.

Usage
-----
    .venv312/bin/python package_dataset.py --config configs/env_force_reach.yaml
"""

import argparse
import glob
import os

import numpy as np
import yaml

from lerobot.datasets.lerobot_dataset import LeRobotDataset

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/env_force_reach.yaml")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)
env_cfg = cfg.get("env", {})
col_cfg = cfg.get("collection", {})

raw_dir = col_cfg.get("raw_output_dir", "./datasets/raw_reach-contact-v0")
ep_paths = sorted(glob.glob(os.path.join(raw_dir, "ep_*.npz")))
if not ep_paths:
    raise SystemExit(f"No raw episodes found under {raw_dir} -- run collect_data.py first.")

enable_camera = env_cfg.get("enable_camera", True)
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
if enable_camera:
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
    use_videos=enable_camera,
)

for ep_path in ep_paths:
    data = np.load(ep_path, allow_pickle=True)
    states, actions = data["states"], data["actions"]
    task = str(data["task"])
    frames = data["frames"] if enable_camera else None

    for i in range(len(states)):
        frame = {
            "observation.state": states[i],
            "action": actions[i],
            "task": task,
        }
        if enable_camera:
            frame["observation.images.wrist"] = frames[i]
        dataset.add_frame(frame)

    dataset.save_episode()
    print(f"Packaged {ep_path} ({len(states)} steps)")

dataset.finalize()
print(f"\nDataset saved to {col_cfg.get('output_root', './datasets/reach-contact-v0')}")
