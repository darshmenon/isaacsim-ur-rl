#!/usr/bin/env python3
"""Impedance motor / motion analysis in Isaac Sim.

Commands a multi-joint sine reference under true effort-mode impedance,
logs motor torque, tracking error, and mechanical power, then compares
stiff vs soft gains. Writes plots + a .npz for further analysis.

Usage
-----
    OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python -u analyze_impedance_motors.py --headless
"""

from __future__ import annotations

import argparse
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--seconds", type=float, default=6.0)
parser.add_argument("--out-dir", default="docs")
parser.add_argument("--npz", default="docs/impedance_motor_analysis.npz")
parser.add_argument("--plot", default="docs/impedance_motor_analysis.png")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs.ur_force_env import URForceReachEnv

# Control / logging rate matches rendering_dt (50 Hz default in env)
RENDER_HZ = 50.0
N_STEPS = int(args.seconds * RENDER_HZ)
JOINT_NAMES = [f"J{i+1}" for i in range(6)]

# Absolute joint reference: slow multi-frequency sine around a home pose
Q_HOME = np.array([0.0, -0.9, 1.1, -1.4, -1.57, 0.0], dtype=np.float64)
AMP = np.array([0.25, 0.18, 0.22, 0.15, 0.12, 0.30], dtype=np.float64)
FREQ = np.array([0.35, 0.45, 0.40, 0.55, 0.50, 0.70], dtype=np.float64)  # Hz


def q_ref_at(t: float) -> np.ndarray:
    return Q_HOME + AMP * np.sin(2.0 * np.pi * FREQ * t)


TRIALS = [
    {"name": "stiff", "kp": 250.0, "kd": 35.0, "color": "#1f77b4"},
    {"name": "soft", "kp": 40.0, "kd": 12.0, "color": "#d62728"},
]

env = URForceReachEnv(
    render_mode="human" if not args.headless else None,
    action_scale=1.0,  # wide enough for absolute-ref delta commands
    target_radius=0.05,
    reach_bonus=0.0,
    max_episode_steps=N_STEPS + 5,
    control_mode="impedance",
    impedance_kp=TRIALS[0]["kp"],
    impedance_kd=TRIALS[0]["kd"],
    max_effort=400.0,  # analysis headroom so stiff tracking isn't clip-limited
    enable_camera=False,
    attach_gripper=False,
)

results = {}

for trial in TRIALS:
    env._impedance_kp = trial["kp"]
    env._impedance_kd = trial["kd"]
    obs, _ = env.reset()

    # Park near home with a few high-authority steps before logging
    for _ in range(40):
        q = env._robot.get_joint_positions()[:6]
        action = np.clip((Q_HOME - q) / env._action_scale, -1.0, 1.0)
        obs, _, term, trunc, info = env.step(action.astype(np.float32))

    t = np.zeros(N_STEPS)
    q = np.zeros((N_STEPS, 6))
    q_des = np.zeros((N_STEPS, 6))
    qd = np.zeros((N_STEPS, 6))
    tau = np.zeros((N_STEPS, 6))
    err = np.zeros((N_STEPS, 6))
    power = np.zeros((N_STEPS, 6))
    force = np.zeros((N_STEPS, 3))

    print(f"\n=== Trial '{trial['name']}'  Kp={trial['kp']}  Kd={trial['kd']} ===")
    for i in range(N_STEPS):
        ti = i / RENDER_HZ
        q_ref = q_ref_at(ti)
        q_now = env._robot.get_joint_positions()[:6]
        # Absolute reference via delta action API
        action = np.clip((q_ref - q_now) / env._action_scale, -1.0, 1.0)
        obs, _, term, trunc, info = env.step(action.astype(np.float32))

        t[i] = ti
        q[i] = info["q"]
        q_des[i] = info["q_des"]
        qd[i] = info["qd"]
        tau[i] = info["tau"]
        err[i] = info["q_des"] - info["q"]
        power[i] = info["tau"] * info["qd"]  # W per joint (approx)
        force[i] = info["wrench"][:3]

    rms_err = np.sqrt(np.mean(err**2, axis=0))
    peak_tau = np.max(np.abs(tau), axis=0)
    mean_abs_power = np.mean(np.abs(power), axis=0)
    print(f"  RMS tracking error (rad): {np.array2string(rms_err, precision=3)}")
    print(f"  Peak |tau| (N·m):         {np.array2string(peak_tau, precision=1)}")
    print(f"  Mean |power| (W):         {np.array2string(mean_abs_power, precision=2)}")

    results[trial["name"]] = dict(
        t=t, q=q, q_des=q_des, qd=qd, tau=tau, err=err, power=power, force=force,
        kp=trial["kp"], kd=trial["kd"],
        rms_err=rms_err, peak_tau=peak_tau, mean_abs_power=mean_abs_power,
    )

os.makedirs(args.out_dir, exist_ok=True)
np.savez_compressed(args.npz, **{f"{k}_{sk}": v for k, d in results.items() for sk, v in d.items()})
print(f"\nSaved telemetry: {args.npz}")

# --- Plots ---------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
fig.suptitle(
    "UR10 impedance motor / motion analysis\n"
    r"$\tau = K_p(q_{des}-q) + K_d(0-\dot q)$  ·  stiff vs soft gains",
    fontsize=12,
)

# Joint 2 (shoulder) tracking — most loaded for this motion
j = 1
ax = axes[0, 0]
for trial in TRIALS:
    r = results[trial["name"]]
    ax.plot(r["t"], r["q_des"][:, j], color=trial["color"], ls="--", alpha=0.7, label=f"{trial['name']} $q_{{des}}$")
    ax.plot(r["t"], r["q"][:, j], color=trial["color"], lw=1.5, label=f"{trial['name']} $q$")
ax.set_title(f"Tracking — {JOINT_NAMES[j]} (shoulder)")
ax.set_ylabel("rad")
ax.legend(fontsize=8, loc="best")
ax.grid(True, alpha=0.3)

# Tracking error norms
ax = axes[0, 1]
for trial in TRIALS:
    r = results[trial["name"]]
    ax.plot(r["t"], np.linalg.norm(r["err"], axis=1), color=trial["color"], label=trial["name"])
ax.set_title(r"Joint tracking error $\|q_{des}-q\|_2$")
ax.set_ylabel("rad")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Motor torques (stiff)
ax = axes[1, 0]
r = results["stiff"]
for ji in range(6):
    ax.plot(r["t"], r["tau"][:, ji], lw=1.0, label=JOINT_NAMES[ji])
ax.set_title(f"Motor torque — stiff (Kp={TRIALS[0]['kp']})")
ax.set_ylabel("N·m")
ax.legend(fontsize=7, ncol=3)
ax.grid(True, alpha=0.3)

# Motor torques (soft)
ax = axes[1, 1]
r = results["soft"]
for ji in range(6):
    ax.plot(r["t"], r["tau"][:, ji], lw=1.0, label=JOINT_NAMES[ji])
ax.set_title(f"Motor torque — soft (Kp={TRIALS[1]['kp']})")
ax.set_ylabel("N·m")
ax.legend(fontsize=7, ncol=3)
ax.grid(True, alpha=0.3)

# Mechanical power sum
ax = axes[2, 0]
for trial in TRIALS:
    r = results[trial["name"]]
    ax.plot(r["t"], np.sum(r["power"], axis=1), color=trial["color"], label=trial["name"])
ax.set_title(r"Instantaneous mech. power $\sum_i \tau_i \dot q_i$")
ax.set_xlabel("t (s)")
ax.set_ylabel("W")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# RMS error bar chart
ax = axes[2, 1]
x = np.arange(6)
w = 0.35
ax.bar(x - w / 2, results["stiff"]["rms_err"], w, color=TRIALS[0]["color"], label="stiff")
ax.bar(x + w / 2, results["soft"]["rms_err"], w, color=TRIALS[1]["color"], label="soft")
ax.set_xticks(x)
ax.set_xticklabels(JOINT_NAMES)
ax.set_title("RMS tracking error by joint")
ax.set_ylabel("rad")
ax.set_xlabel("joint")
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.3)

fig.savefig(args.plot, dpi=160)
print(f"Saved plot: {args.plot}")

# Console summary table
print("\nSummary")
print(f"{'trial':<8} {'Kp':>6} {'Kd':>5} {'mean RMS':>10} {'peak|τ|':>10} {'mean|P|':>10}")
for trial in TRIALS:
    r = results[trial["name"]]
    print(
        f"{trial['name']:<8} {trial['kp']:6.0f} {trial['kd']:5.0f} "
        f"{float(np.mean(r['rms_err'])):10.4f} "
        f"{float(np.max(r['peak_tau'])):10.1f} "
        f"{float(np.mean(r['mean_abs_power'])):10.2f}"
    )

env.close()
simulation_app.close()
