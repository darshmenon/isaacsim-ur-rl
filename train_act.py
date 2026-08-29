#!/usr/bin/env python3
"""Fine-tune ACT on the collected force-augmented dataset.

Thin wrapper around LeRobot's real `lerobot-train` CLI (draccus-style flags,
e.g. `--dataset.repo_id=... --policy.type=act`). No LeRobot core changes are
needed: ACT accepts `observation.state` as an arbitrary-length vector, so the
wrist force/torque channels we added ride along as extra state dims.

Usage
-----
    .venv312/bin/python train_act.py [--config configs/act_train.yaml]

Must run under .venv312 (Python 3.12, lerobot installed) -- NOT the Isaac
Sim .venv (Python 3.10, doesn't have lerobot). This script has no Isaac Sim
dependency itself.

Confirm exact flag names against `.venv312/bin/lerobot-train --help` if
this repo's lerobot version has drifted from what's encoded here.
"""

import argparse
import os
import subprocess
import sys

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/act_train.yaml")
parser.add_argument("--dry-run", action="store_true", help="Print the command without running it")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

ds = cfg["dataset"]
pol = cfg["policy"]
tr = cfg["training"]

lerobot_train_bin = os.path.join(os.path.dirname(sys.executable), "lerobot-train")
if not os.path.exists(lerobot_train_bin):
    raise SystemExit(
        f"lerobot-train not found next to {sys.executable}. "
        "Run this script with .venv312/bin/python, not the Isaac Sim venv."
    )

cmd = [
    lerobot_train_bin,
    f"--dataset.repo_id={ds['repo_id']}",
    f"--dataset.root={ds['root']}",
    f"--policy.type={pol['type']}",
    f"--policy.chunk_size={pol.get('chunk_size', 100)}",
    f"--output_dir={tr['output_dir']}",
    f"--batch_size={tr.get('batch_size', 8)}",
    f"--steps={tr.get('steps', 20000)}",
    f"--save_freq={tr.get('save_freq', 5000)}",
    f"--log_freq={tr.get('log_freq', 200)}",
    f"--num_workers={tr.get('num_workers', 4)}",
]

print("Running:", " ".join(cmd))
if args.dry_run:
    sys.exit(0)

subprocess.run(cmd, check=True)
