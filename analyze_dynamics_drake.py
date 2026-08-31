#!/usr/bin/env python3
"""Drake reference dynamics for UR10 — gravity, IK, torque envelopes.

Runs outside Isaac (CPU). Uses the UR10 URDF bundled with Isaac Sim packages
and Drake's MultibodyPlant for trustworthy gravity / inverse-dynamics numbers.

Usage
-----
    .venv/bin/python -u analyze_dynamics_drake.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    InverseKinematics,
    Parser,
    Solve,
)

OUT = Path("docs")
OUT.mkdir(exist_ok=True)

URDF_CANDIDATES = [
    Path("assets/ur10_drake_nomesh.urdf"),
    Path(".venv/lib/python3.10/site-packages/isaacsim/exts/"
         "isaacsim.robot_motion.motion_generation/motion_policy_configs/"
         "universal_robots/ur10/ur10_robot.urdf"),
]
urdf = next((p for p in URDF_CANDIDATES if p.is_file()), None)
if urdf is None:
    raise SystemExit("No UR10 URDF found under .venv Isaac packages")

print(f"Loading {urdf}")

builder = DiagramBuilder()
plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
parser = Parser(plant)
# Mesh paths in URDF may be missing; disable mesh strictness where possible
try:
    parser.package_map().PopulateFromFolder(str(urdf.parent.parent))
except Exception:
    pass
model_ids = parser.AddModels(str(urdf))
plant.Finalize()
diagram = builder.Build()
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)

# Find arm joints (skip fixed / mimic if any)
joint_indices = []
joint_names = []
for jid in plant.GetJointIndices(model_ids[0] if model_ids else plant.GetModelInstances()[0]):
    joint = plant.get_joint(jid)
    if joint.num_positions() == 1:
        joint_indices.append(joint.position_start())
        joint_names.append(joint.name())
print("joints:", joint_names)
nq = plant.num_positions()
assert len(joint_names) >= 6, joint_names
# Use first 6 revolute arm joints
pos_starts = joint_indices[:6]
joint_names = joint_names[:6]

# Approximate continuous torque limits (UR10 datasheet-ish; matches PhysX asset)
MAX_TAU = np.array([330.0, 330.0, 150.0, 56.0, 56.0, 56.0])

Q_HOME = np.array([0.0, -1.0, 1.2, -1.4, -1.57, 0.0])
AMP = np.array([0.15, 0.10, 0.12, 0.08, 0.06, 0.15])
FREQ = np.array([0.25, 0.30, 0.28, 0.35, 0.32, 0.40])
HZ, T = 50.0, 4.0
N = int(T * HZ)


def set_q(q6: np.ndarray):
    q = plant.GetPositions(plant_context).copy()
    for i, s in enumerate(pos_starts):
        q[s] = q6[i]
    plant.SetPositions(plant_context, q)


def get_gravity(q6: np.ndarray) -> np.ndarray:
    set_q(q6)
    tau = plant.CalcGravityGeneralizedForces(plant_context)
    return np.array([tau[plant.GetJointByName(n).velocity_start()] for n in joint_names])


def inverse_dynamics(q6, qd6, qdd6) -> np.ndarray:
    """tau = M qdd + C(q,qd) - g(q)  (actuator torque to realize qdd)."""
    set_q(q6)
    v = np.zeros(plant.num_velocities())
    vd = np.zeros(plant.num_velocities())
    for j, jname in enumerate(joint_names):
        js = plant.GetJointByName(jname).velocity_start()
        v[js] = qd6[j]
        vd[js] = qdd6[j]
    plant.SetVelocities(plant_context, v)
    M = plant.CalcMassMatrix(plant_context)
    Cv = plant.CalcBiasTerm(plant_context)
    g = plant.CalcGravityGeneralizedForces(plant_context)
    tau_full = M @ vd + Cv - g
    return np.array([tau_full[plant.GetJointByName(n).velocity_start()] for n in joint_names])


def ee_body():
    # Prefer tool0 / ee_link / wrist_3_link
    for name in ("tool0", "ee_link", "wrist_3_link", "flange"):
        try:
            return plant.GetBodyByName(name)
        except RuntimeError:
            continue
    # last link
    return plant.GetBodyByName(plant.GetBodyIndices()[-1])


body_ee = ee_body()
print("EE body:", body_ee.name())


def fk(q6: np.ndarray) -> np.ndarray:
    set_q(q6)
    X = plant.EvalBodyPoseInWorld(plant_context, body_ee)
    return np.array(X.translation())


# Gravity at home
g_home = get_gravity(Q_HOME)
print(f"Drake gravity @ home: {np.round(g_home, 2)}  (|g|={np.linalg.norm(g_home):.1f} N·m)")

# Trajectory inverse dynamics: PD+FF+g style reference torque
t = np.arange(N) / HZ
q = Q_HOME + AMP * np.sin(2 * np.pi * FREQ * t[:, None])
qd = AMP * (2 * np.pi * FREQ) * np.cos(2 * np.pi * FREQ * t[:, None])
qdd = -AMP * (2 * np.pi * FREQ) ** 2 * np.sin(2 * np.pi * FREQ * t[:, None])

tau_id = np.zeros((N, 6))
for i in range(N):
    tau_id[i] = inverse_dynamics(q[i], qd[i], qdd[i])

peak = np.max(np.abs(tau_id), axis=0)
rms = np.sqrt(np.mean(tau_id**2, axis=0))
print("ID peak |τ|:", np.round(peak, 1))
print("ID RMS  τ :", np.round(rms, 1))
print("limits   :", MAX_TAU)
print("within limits?", np.all(peak <= MAX_TAU + 1e-6))

# IK via Drake InverseKinematics
ik_errs = []
targets = [
    np.array([0.45, 0.10, 0.35]),
    np.array([0.50, -0.15, 0.40]),
    np.array([0.40, 0.20, 0.50]),
]
for tgt in targets:
    ik = InverseKinematics(plant, plant_context)
    ik.AddPositionConstraint(
        plant.GetFrameByName(body_ee.name()),
        np.zeros(3),
        plant.world_frame(),
        tgt,
        tgt,
    )
    prog = ik.prog()
    q0 = plant.GetPositions(plant_context).copy()
    for i, s in enumerate(pos_starts):
        q0[s] = Q_HOME[i]
    prog.SetInitialGuess(ik.q(), q0)
    result = Solve(prog)
    ok = result.is_success()
    qsol = result.GetSolution(ik.q())
    q6 = np.array([qsol[s] for s in pos_starts])
    ee = fk(q6)
    err = float(np.linalg.norm(ee - tgt))
    ik_errs.append(err)
    print(f"IK target={np.round(tgt,3)} success={ok} |err|={err*1000:.2f} mm")

report = {
    "backend": "drake",
    "urdf": str(urdf),
    "joint_names": joint_names,
    "gravity_home_Nm": g_home.tolist(),
    "id_peak_tau_Nm": peak.tolist(),
    "id_rms_tau_Nm": rms.tolist(),
    "limits_Nm": MAX_TAU.tolist(),
    "within_limits": bool(np.all(peak <= MAX_TAU + 1e-6)),
    "ik_errors_m": ik_errs,
    "suggest_cont_Nm": (1.5 * rms).clip(max=MAX_TAU).tolist(),
    "suggest_peak_Nm": (1.2 * peak).clip(max=MAX_TAU).tolist(),
}
with open(OUT / "drake_dynamics_report.json", "w") as f:
    json.dump(report, f, indent=2)

fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
fig.suptitle("Drake UR10 reference — gravity / inverse dynamics / IK")
ax = axes[0, 0]
for j, name in enumerate(joint_names):
    ax.plot(t, tau_id[:, j], label=name)
ax.set_title("Inverse-dynamics τ(t)")
ax.set_ylabel("N·m")
ax.legend(fontsize=7, ncol=3)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
x = np.arange(6)
ax.bar(x - 0.2, peak, 0.4, label="peak ID")
ax.bar(x + 0.2, rms, 0.4, label="RMS ID")
ax.plot(x, MAX_TAU, "k--", label="limit")
ax.set_xticks(x)
ax.set_xticklabels([f"J{i+1}" for i in range(6)])
ax.set_title("Torque envelope vs limits")
ax.set_ylabel("N·m")
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.3)

ax = axes[1, 0]
ax.bar(np.arange(6), g_home)
ax.set_xticks(np.arange(6))
ax.set_xticklabels([f"J{i+1}" for i in range(6)])
ax.set_title("Gravity torque @ home")
ax.set_ylabel("N·m")
ax.grid(True, axis="y", alpha=0.3)

ax = axes[1, 1]
ax.bar(np.arange(len(ik_errs)), [e * 1000 for e in ik_errs], color="#9467bd")
ax.set_xticks(np.arange(len(ik_errs)))
ax.set_xticklabels([f"T{i+1}" for i in range(len(ik_errs))])
ax.set_ylabel("mm")
ax.set_title("Drake IK position error")
ax.grid(True, axis="y", alpha=0.3)

fig.savefig(OUT / "drake_dynamics_analysis.png", dpi=160)
print(f"Saved {OUT/'drake_dynamics_analysis.png'}")
print(f"Saved {OUT/'drake_dynamics_report.json'}")
print("DONE Drake reference")
