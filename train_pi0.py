#!/usr/bin/env python3
"""Fine-tune π₀ / π₀.₅ on the local UR10 reach-contact dataset via OpenPI.

Uses the OpenPI checkout at /home/asimov/openpi (its own .venv — not Isaac
Sim's .venv and not .venv312/lerobot). The UR10 data transforms live in
openpi's ur10_policy.py + TrainConfig names pi0_ur10_reach_lora / pi0_ur10_reach.

Usage
-----
    python train_pi0.py [--config configs/pi0_train.yaml]
    python train_pi0.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/pi0_train.yaml")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

openpi_root = os.path.expanduser(cfg.get("openpi_root", "/home/asimov/openpi"))
config_name = cfg["config_name"]
exp_name = cfg.get("exp_name", "ur10_reach_v0")
overwrite = bool(cfg.get("overwrite", True))

python = os.path.join(openpi_root, ".venv", "bin", "python")
train_py = os.path.join(openpi_root, "scripts", "train.py")
compute_py = os.path.join(openpi_root, "scripts", "compute_norm_stats.py")

if not os.path.isfile(python):
    raise SystemExit(
        f"OpenPI venv not found at {python}. "
        "Install with: cd /home/asimov/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync"
    )

env = os.environ.copy()
# Keep checkpoints under this repo by default.
env.setdefault("OPENPI_DATA_HOME", os.path.abspath("./outputs/openpi"))

cmds = []
if cfg.get("compute_norm_stats"):
    cmds.append([python, compute_py, f"--config-name={config_name}"])

train_cmd = [
    python,
    train_py,
    config_name,
    f"--exp-name={exp_name}",
]
if overwrite:
    train_cmd.append("--overwrite")
cmds.append(train_cmd)

for cmd in cmds:
    print("Running:", " ".join(cmd))
    print(f"  cwd={openpi_root}")
    if args.dry_run:
        continue
    subprocess.run(cmd, check=True, cwd=openpi_root, env=env)

if args.dry_run:
    sys.exit(0)
