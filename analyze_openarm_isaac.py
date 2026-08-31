#!/usr/bin/env python3
"""OpenArm custom motor-kit check in Isaac Sim (PhysX).

Imports Enactic OpenArm v1 URDF (Damiao effort limits in the file), enables
gravity, switches to effort mode, and compares the same motor kits as the
MuJoCo script: Damiao v2 / equal-10 / weak / strong.

Usage
-----
    OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python -u analyze_openarm_isaac.py --headless
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--seconds", type=float, default=4.0)
parser.add_argument("--out-dir", default="docs")
parser.add_argument(
    "--urdf",
    default=str(Path(__file__).resolve().parent / "assets" / "openarm_v1_isaac.urdf"),
)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

# hide_ui avoids URDF-importer UI widgets crashing in headless
simulation_app = SimulationApp({"headless": args.headless, "hide_ui": True})

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import omni.kit.commands
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.prims import define_prim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
from pxr import PhysxSchema, Usd, UsdPhysics

from openarm_motor_common import (
    JOINT_NAMES,
    MOTOR_KITS,
    N_ARM,
    Q_HOME,
    pd_torque,
    q_ref_at,
)

enable_extension("isaacsim.asset.importer.urdf")
# Give the extension a moment to register kit commands
simulation_app.update()
simulation_app.update()

ROOT = Path(__file__).resolve().parent
URDF = Path(args.urdf)
assert URDF.is_file(), f"URDF missing: {URDF}"

LEFT_JOINTS = [f"openarm_left_joint{i}" for i in range(1, N_ARM + 1)]
JOINT_LABELS = JOINT_NAMES

HZ = 50.0
PHYS_DT = 1.0 / 120.0
DT = 1.0 / HZ
STEPS_PER = max(1, int(round(DT / PHYS_DT)))
N = int(args.seconds * HZ)


def enable_body_gravity_usd(root_path: str) -> int:
    """Force PhysX rigid-body gravity on for every body under the articulation."""
    stage = omni.usd.get_context().get_stage()
    n = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if not prim.IsA(UsdPhysics.RigidBodyAPI):
            continue
        rb = UsdPhysics.RigidBodyAPI(prim)
        # USD Physics + PhysX schema both matter depending on importer version
        if rb.GetRigidBodyEnabledAttr():
            rb.GetRigidBodyEnabledAttr().Set(True)
        try:
            physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            attr = physx_rb.GetDisableGravityAttr()
            if not attr:
                attr = physx_rb.CreateDisableGravityAttr(False)
            attr.Set(False)
            n += 1
        except Exception:
            pass
    return n


os.makedirs(args.out_dir, exist_ok=True)
world = World(stage_units_in_meters=1.0, physics_dt=PHYS_DT, rendering_dt=DT)
assets = get_assets_root_path()
define_prim("/World/Ground", "Xform").GetReferences().AddReference(
    assets + "/Isaac/Environments/Grid/default_environment.usd"
)

# --- Import OpenArm URDF -------------------------------------------------
status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False
import_config.fix_base = True
import_config.make_default_prim = False
import_config.create_physics_scene = False
import_config.default_drive_type = import_config.default_drive_type  # keep
status, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=str(URDF),
    import_config=import_config,
    get_articulation_root=True,
)
print(f"URDF import status={status} prim={prim_path}")
if not prim_path:
    # Fallback common root name from robot name=
    prim_path = "/openarm"
elif Path(str(prim_path)).name == "root_joint":
    # fix_base=True makes the importer return the fixed world->base JOINT
    # prim as "articulation root", not the actual body/Xform. Pointing
    # SingleArticulation at a joint (rather than the Xform carrying
    # PhysicsArticulationRootAPI) leaves get_generalized_gravity_forces()
    # returning near-zero noise even though DOFs/max-efforts still resolve.
    prim_path = str(Path(str(prim_path)).parent)
print(f"Using articulation root: {prim_path}")

robot = world.scene.add(SingleArticulation(prim_path=str(prim_path), name="openarm"))
world.reset()
robot.initialize()

view = robot._articulation_view
# Enable gravity via articulation view + USD PhysX flags (URDF imports often disable it)
try:
    n_bodies = int(view.get_body_disable_gravity().shape[1])
    view.set_body_disable_gravity(np.zeros((1, n_bodies), dtype=bool))
    print(f"articulation_view: enabled gravity on {n_bodies} bodies")
except Exception as e:
    print(f"WARN gravity enable (view): {e}")
n_usd = enable_body_gravity_usd(str(prim_path).rsplit("/", 1)[0] if "/" in str(prim_path) else str(prim_path))
# Also walk from /openarm if present
n_usd += enable_body_gravity_usd("/openarm")
print(f"USD PhysxRigidBodyAPI: cleared disableGravity on ~{n_usd} bodies")
if hasattr(robot, "enable_gravity"):
    try:
        robot.enable_gravity()
    except Exception:
        pass
for _ in range(30):
    world.step(render=False)

dof_names = list(robot.dof_names)
print("dof_names:", dof_names)
left_idx = []
for jn in LEFT_JOINTS:
    if jn not in dof_names:
        raise SystemExit(f"Missing joint {jn} in DOFs: {dof_names}")
    left_idx.append(dof_names.index(jn))
left_idx = np.asarray(left_idx, dtype=np.int64)
print("left joint indices:", left_idx)

# Effort mode on whole articulation (same pattern as UR force env)
robot.get_articulation_controller().switch_control_mode("effort")

# PhysX max efforts from URDF (should match Damiao)
try:
    max_eff = np.asarray(view.get_max_efforts(), dtype=np.float64).reshape(-1)
    print("PhysX max_efforts (all DOF):", np.round(max_eff, 1))
    print("PhysX max_efforts (left):", np.round(max_eff[left_idx], 1))
except Exception as e:
    print(f"WARN max_efforts: {e}")
    max_eff = None


def get_q() -> np.ndarray:
    return np.asarray(robot.get_joint_positions(), dtype=np.float64).reshape(-1)[left_idx]


def get_qd() -> np.ndarray:
    return np.asarray(robot.get_joint_velocities(), dtype=np.float64).reshape(-1)[left_idx]


def get_g() -> np.ndarray:
    try:
        g = np.asarray(view.get_generalized_gravity_forces(), dtype=np.float64).reshape(-1)
        return np.nan_to_num(g[left_idx], nan=0.0)
    except Exception:
        return np.zeros(N_ARM)


def apply_tau(tau7: np.ndarray) -> None:
    """Apply torque on left arm DOFs; zero elsewhere."""
    n = len(dof_names)
    efforts = np.zeros(n, dtype=np.float64)
    efforts[left_idx] = tau7
    robot.get_articulation_controller().apply_action(
        ArticulationAction(joint_efforts=efforts)
    )


def hold_at(q_des: np.ndarray, kit: dict, n: int = STEPS_PER) -> None:
    """Recompute PD+gravity torque every physics substep (not once and held
    across STEPS_PER substeps) — a stale qfrc_bias/velocity-feedback torque
    held across substeps is what caused the MuJoCo OpenArm PD loop to go
    numerically unstable on the low-inertia wrist joints; same fix here."""
    for _ in range(n):
        tau = pd_torque(get_q(), get_qd(), q_des, get_g())
        tau = np.clip(tau, -kit["tau_lim"], kit["tau_lim"])
        apply_tau(tau)
        world.step(render=not args.headless)


g0 = get_g()
print(f"gravity @ reset (left): {np.round(g0, 2)}  |g|={np.linalg.norm(g0):.2f}")
if np.linalg.norm(g0) < 0.5:
    print("WARNING: gravity still near zero — τ absolute values may be soft")

# Park near home
for _ in range(int(1.5 * HZ)):
    hold_at(Q_HOME, MOTOR_KITS["damiao_v2"])
print("parked q:", np.round(get_q(), 3))


def run_kit(name: str, kit: dict) -> dict:
    # re-park lightly
    for _ in range(int(0.5 * HZ)):
        hold_at(Q_HOME, kit)

    t = np.zeros(N)
    q_log = np.zeros((N, N_ARM))
    tau_log = np.zeros((N, N_ARM))
    err_log = np.zeros((N, N_ARM))
    sat_log = np.zeros((N, N_ARM), dtype=bool)

    sub_dt = 1.0 / HZ / STEPS_PER
    for i in range(N):
        ti = i / HZ
        # Recompute torque every physics substep (not once and held across
        # STEPS_PER substeps) — see hold_at().
        for k in range(STEPS_PER):
            qdes = q_ref_at(ti + k * sub_dt)
            q = get_q()
            qd = get_qd()
            tau_cmd = pd_torque(q, qd, qdes, get_g())
            tau = np.clip(tau_cmd, -kit["tau_lim"], kit["tau_lim"])
            sat = np.abs(tau_cmd) > kit["tau_lim"] + 1e-9
            apply_tau(tau)
            world.step(render=not args.headless)

        t[i] = ti
        q_log[i] = q
        tau_log[i] = tau
        err_log[i] = qdes - q
        sat_log[i] = sat

    rms = np.sqrt(np.mean(err_log**2, axis=0))
    peak = np.max(np.abs(tau_log), axis=0)
    sat_pct = 100.0 * np.mean(sat_log, axis=0)
    return {
        "t": t,
        "q": q_log,
        "tau": tau_log,
        "err": err_log,
        "rms_err": rms,
        "peak_tau": peak,
        "sat_pct": sat_pct,
        "tau_lim": kit["tau_lim"].copy(),
        "mean_rms": float(np.mean(rms)),
        "mean_sat": float(np.mean(sat_pct)),
    }


results = {}
for name, kit in MOTOR_KITS.items():
    print(f"\n=== Isaac {kit['label']} ===")
    out = run_kit(name, kit)
    results[name] = out
    print(f"  mean RMS err: {out['mean_rms']:.4f}")
    print(f"  peak |tau|:   {np.array2string(out['peak_tau'], precision=2)}")
    print(f"  sat %:        {np.array2string(out['sat_pct'], precision=1)}")
    print(f"  mean sat %:   {out['mean_sat']:.1f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
ax_err, ax_tau, ax_sat = axes
x = np.arange(N_ARM)
width = 0.18
for i, (name, kit) in enumerate(MOTOR_KITS.items()):
    r = results[name]
    off = (i - 1.5) * width
    ax_err.bar(x + off, r["rms_err"], width, label=kit["label"], color=kit["color"])
    ax_tau.bar(x + off, r["peak_tau"], width, color=kit["color"], alpha=0.85)
    ax_tau.plot(x + off, r["tau_lim"], "k_", markersize=10, mew=1.5)
    ax_sat.bar(x + off, r["sat_pct"], width, color=kit["color"], alpha=0.85)

ax_err.set_title("RMS tracking error")
ax_err.set_ylabel("rad")
ax_tau.set_title("Peak |τ| (ticks = kit limit)")
ax_tau.set_ylabel("N·m")
ax_sat.set_title("Saturation time")
ax_sat.set_ylabel("%")
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_LABELS)
    ax.grid(True, axis="y", alpha=0.3)
ax_err.legend(fontsize=7, loc="upper right")
fig.suptitle(
    "OpenArm motor kits — Isaac Sim PhysX (soft motion, per-joint PD+g)",
    fontsize=12,
)
plot_path = os.path.join(args.out_dir, "openarm_isaac_motor_comparison.png")
fig.savefig(plot_path, dpi=140)
plt.close(fig)
print(f"\nWrote {plot_path}")

# Markdown
md_path = os.path.join(args.out_dir, "openarm_isaac_motor_comparison.md")
lines = [
    "# OpenArm motor kits — Isaac Sim check",
    "",
    f"URDF: `{URDF}` (paths rewritten from `openarm_description`).",
    f"Controller: PD + gravity FF (soft motion, per-joint Kp), {HZ:.0f} Hz, {args.seconds:.1f} s.",
    f"Articulation: `{prim_path}` left arm only driven.",
    f"Gravity sample |g|={np.linalg.norm(g0):.2f} N·m after USD/view enable.",    "",
    "## Summary",
    "",
    "| kit | mean RMS (rad) | mean sat % |",
    "|---|---:|---:|",
]
for name, kit in MOTOR_KITS.items():
    r = results[name]
    lines.append(f"| {kit['label']} | {r['mean_rms']:.4f} | {r['mean_sat']:.1f} |")

r = results["damiao_v2"]
lines += [
    "",
    "## Damiao v2 per-joint (Isaac)",
    "",
    "| joint | peak τ | limit | sat % |",
    "|---|---:|---:|---:|",
]
for i, jn in enumerate(JOINT_LABELS):
    lines.append(
        f"| {jn} | {r['peak_tau'][i]:.2f} | {r['tau_lim'][i]:.1f} | {r['sat_pct'][i]:.1f} |"
    )
lines += [
    "",
    f"Plot: `{os.path.basename(plot_path)}`",
    "",
    "Compare with MuJoCo: `docs/openarm_motor_comparison.md`.",
    "",
]
Path(md_path).write_text("\n".join(lines))
print(f"Wrote {md_path}")

summary = {
    name: {
        "mean_rms": results[name]["mean_rms"],
        "mean_sat": results[name]["mean_sat"],
        "peak_tau": results[name]["peak_tau"].tolist(),
        "sat_pct": results[name]["sat_pct"].tolist(),
        "tau_lim": results[name]["tau_lim"].tolist(),
    }
    for name in MOTOR_KITS
}
if max_eff is not None:
    summary["physx_max_efforts_left"] = max_eff[left_idx].tolist()
summary["gravity_sample"] = g0.tolist()
summary["prim_path"] = str(prim_path)
Path(os.path.join(args.out_dir, "openarm_isaac_motor_comparison.json")).write_text(
    json.dumps(summary, indent=2)
)

simulation_app.close()
print("Isaac OpenArm motor check DONE")
