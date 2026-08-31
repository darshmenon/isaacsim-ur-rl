#!/usr/bin/env python3
"""Full impedance + motor analysis pack in one Isaac session.

Fixes vs earlier runs:
  - enable body gravity (USD default often has it disabled)
  - gravity from articulation_view.get_generalized_gravity_forces() each step
  - clip / log torque to PhysX max efforts (UR10 datasheet-like limits)
  - moderate gains so τ stays realistic
  - contact probe lowers EE until force > 0
  - IK reports teleport FK error (mm-scale) separately from impedance hold

Usage
-----
    OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python -u analyze_impedance_full_pack.py --headless
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--out-dir", default="docs")
parser.add_argument("--seconds", type=float, default=4.0)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualSphere
from isaacsim.core.api.sensors import RigidContactView
from isaacsim.core.utils.prims import define_prim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.universal_robots.ur10 import UR10 as _UR10
from isaacsim.storage.native import get_assets_root_path


class UR10(_UR10):
    def post_reset(self) -> None:
        super(_UR10, self).post_reset()
        self._end_effector.post_reset()
        if self._gripper is not None:
            self._gripper.post_reset()


HZ = 50.0
DT = 1.0 / HZ
PHYS_DT = 1.0 / 120.0
STEPS_PER = max(1, int(round(DT / PHYS_DT)))
N = int(args.seconds * HZ)
JOINTS = [f"J{i+1}" for i in range(6)]
# Milder motion so τ stays inside UR limits
Q_HOME = np.array([0.0, -1.0, 1.2, -1.4, -1.57, 0.0], dtype=np.float64)
AMP = np.array([0.15, 0.10, 0.12, 0.08, 0.06, 0.15], dtype=np.float64)
FREQ = np.array([0.25, 0.30, 0.28, 0.35, 0.32, 0.40], dtype=np.float64)


def q_traj(t: float):
    q = Q_HOME + AMP * np.sin(2.0 * np.pi * FREQ * t)
    qd = AMP * (2.0 * np.pi * FREQ) * np.cos(2.0 * np.pi * FREQ * t)
    return q, qd


os.makedirs(args.out_dir, exist_ok=True)
world = World(stage_units_in_meters=1.0, physics_dt=PHYS_DT, rendering_dt=DT)
assets = get_assets_root_path()
define_prim("/World/Ground", "Xform").GetReferences().AddReference(
    assets + "/Isaac/Environments/Grid/default_environment.usd"
)
robot = world.scene.add(
    UR10(prim_path="/World/UR10", name="ur10", position=np.zeros(3), attach_gripper=False)
)
from isaacsim.core.api.objects import FixedCuboid

pad = world.scene.add(
    FixedCuboid(
        prim_path="/World/ContactPad",
        name="contact_pad",
        position=np.array([0.55, 0.0, 0.20]),
        scale=np.array([0.30, 0.30, 0.05]),
        color=np.array([0.2, 0.6, 0.9]),
    )
)
marker = world.scene.add(
    VisualSphere(
        prim_path="/World/IKTarget",
        name="ik_target",
        position=np.array([0.5, 0.15, 0.4]),
        radius=0.025,
        color=np.array([1.0, 0.3, 0.1]),
    )
)
world.reset()

view = robot._articulation_view
# USD UR10 often ships with body gravity disabled — turn it on
n_bodies = int(view.get_body_disable_gravity().shape[1])
view.set_body_disable_gravity(np.zeros((1, n_bodies), dtype=bool))
if hasattr(robot, "enable_gravity"):
    robot.enable_gravity()
for _ in range(20):
    world.step(render=False)

ctrl = robot.get_articulation_controller()
ctrl.switch_control_mode("effort")
idx = np.arange(6)

# Official-ish PhysX effort limits for this asset: [330, 330, 150, 56, 56, 56]
MAX_TAU = np.asarray(view.get_max_efforts(), dtype=np.float64).reshape(-1)[:6]
print("max_efforts (N·m):", np.round(MAX_TAU, 1))


def get_q():
    return robot.get_joint_positions()[:6].astype(np.float64)


def get_qd():
    return robot.get_joint_velocities()[:6].astype(np.float64)


def get_ee():
    return robot.end_effector.get_world_pose()[0].astype(np.float64)


def get_gravity() -> np.ndarray:
    g = np.asarray(view.get_generalized_gravity_forces(), dtype=np.float64).reshape(-1)[:6]
    return np.nan_to_num(g, nan=0.0)


def apply_tau(tau: np.ndarray) -> np.ndarray:
    """Clip to UR max efforts, apply, step. Returns the clipped command."""
    tau_c = np.clip(tau, -MAX_TAU, MAX_TAU)
    ctrl.apply_action(
        ArticulationAction(joint_efforts=tau_c.astype(np.float32), joint_indices=idx)
    )
    for _ in range(STEPS_PER):
        world.step(render=False)
    return tau_c


g0 = get_gravity()
print(f"gravity @ current q: {np.round(g0, 2)}  (|g|={np.linalg.norm(g0):.1f} N·m)")
if np.linalg.norm(g0) < 1e-3:
    print("WARNING: gravity still near zero — results may be unrealistic")


def run_controller(mode: str, kp: float, kd: float, use_g: bool):
    for _ in range(60):
        q, qd = get_q(), get_qd()
        g = get_gravity() if use_g else 0.0
        apply_tau(kp * (Q_HOME - q) + kd * (0.0 - qd) + g)

    t = np.zeros(N)
    q_log = np.zeros((N, 6))
    qdes_log = np.zeros((N, 6))
    tau_log = np.zeros((N, 6))
    err_log = np.zeros((N, 6))
    power_log = np.zeros((N, 6))
    g_log = np.zeros((N, 6))

    for i in range(N):
        ti = i * DT
        q_ref, qd_ref = q_traj(ti)
        q, qd = get_q(), get_qd()
        g = get_gravity()
        if mode == "baseline":
            tau = kp * (q_ref - q) + kd * (0.0 - qd)
        elif mode == "ff":
            tau = kp * (q_ref - q) + kd * (qd_ref - qd)
        else:
            tau = kp * (q_ref - q) + kd * (qd_ref - qd) + g
        tau_c = apply_tau(tau)
        q2, qd2 = get_q(), get_qd()
        t[i] = ti
        q_log[i] = q2
        qdes_log[i] = q_ref
        tau_log[i] = tau_c  # clipped only
        err_log[i] = q_ref - q2
        power_log[i] = tau_c * qd2
        g_log[i] = g

    rms = np.sqrt(np.mean(err_log**2, axis=0))
    peak_tau = np.max(np.abs(tau_log), axis=0)
    rms_tau = np.sqrt(np.mean(tau_log**2, axis=0))
    energy = np.sum(np.abs(power_log), axis=0) * DT
    mean_p = np.mean(np.abs(power_log), axis=0)
    sat_frac = float(np.mean(np.any(np.abs(tau_log) >= MAX_TAU * 0.99, axis=1)))
    return dict(
        t=t, q=q_log, qdes=qdes_log, tau=tau_log, err=err_log, power=power_log, g=g_log,
        rms=rms, peak_tau=peak_tau, rms_tau=rms_tau, energy=energy, mean_p=mean_p,
        mean_rms=float(np.mean(rms)), peak_tau_max=float(np.max(peak_tau)),
        mean_power=float(np.mean(mean_p)), sat_frac=sat_frac,
    )


print("\n=== Controller comparison (Kp=80, Kd=18) ===")
modes = [
    ("baseline", False, "PD (no ff, no g)"),
    ("ff", False, "PD + velocity FF"),
    ("ff_g", True, "PD + FF + gravity"),
]
ctrl_results = {}
for mode, use_g, label in modes:
    r = run_controller(mode, kp=80.0, kd=18.0, use_g=use_g)
    ctrl_results[mode] = r
    print(
        f"  {label:22s}  meanRMS={r['mean_rms']:.4f}  peak|τ|={r['peak_tau_max']:.1f}  "
        f"mean|P|={r['mean_power']:.1f}W  sat={100*r['sat_frac']:.0f}%"
    )

print("\n=== Gain sweep (ff + gravity) ===")
KP_GRID = [40.0, 60.0, 80.0, 120.0, 160.0]
sweep = {}
for kp in KP_GRID:
    kd = max(12.0, 2.0 * np.sqrt(kp))
    r = run_controller("ff_g", kp=kp, kd=kd, use_g=True)
    sweep[kp] = r
    print(
        f"  Kp={kp:6.0f} Kd={kd:5.1f}  meanRMS={r['mean_rms']:.4f}  "
        f"peak|τ|={r['peak_tau_max']:.1f}  mean|P|={r['mean_power']:.1f}W  sat={100*r['sat_frac']:.0f}%"
    )

# Prefer lowest RMS among runs that stay mostly unsaturated
cands = [k for k in KP_GRID if sweep[k]["sat_frac"] < 0.25] or list(KP_GRID)
best_kp = min(cands, key=lambda k: sweep[k]["mean_rms"])
best = sweep[best_kp]
print(f"Recommended Kp={best_kp:.0f} (unsaturated preference)")

print("\n=== Contact compliance probe (vs pad) ===")
# Pose with EE above the pad (~z=0.35–0.45)
q_reach = np.array([0.0, -1.2, 1.6, -1.9, -1.57, 0.0], dtype=np.float64)
for _ in range(100):
    apply_tau(120.0 * (q_reach - get_q()) + 28.0 * (0.0 - get_qd()) + get_gravity())
print(f"  EE above pad at z={get_ee()[2]:.3f}m")

wrist = RigidContactView(
    prim_paths_expr="/World/UR10/wrist_3_link",
    filter_paths_expr=["/World/ContactPad"],
)
wrist.initialize()

compliance = {}
for name, kp_c in [("soft", 35.0), ("stiff", 120.0)]:
    for _ in range(40):
        apply_tau(120.0 * (q_reach - get_q()) + 28.0 * (0.0 - get_qd()) + get_gravity())
    wrist.initialize()
    zs, fs = [], []
    for k in range(100):
        q_cmd = q_reach.copy()
        q_cmd[1] = q_reach[1] - 0.01 * k  # press into pad
        apply_tau(kp_c * (q_cmd - get_q()) + 20.0 * (0.0 - get_qd()) + get_gravity())
        ee = get_ee()
        force = np.asarray(wrist.get_net_contact_forces(), dtype=np.float64).reshape(-1)[:3]
        zs.append(float(ee[2]))
        fs.append(float(np.linalg.norm(force)))
    zs, fs = np.array(zs), np.array(fs)
    print(f"  {name:5s} min_z={zs.min():.3f}m  peak|F|={fs.max():.1f}N")
    hit = np.where(fs > 5.0)[0]
    if len(hit) > 5:
        z0 = zs[hit[0]]
        defl = np.maximum(z0 - zs, 0.0)
        mask = fs > 5.0
        k_est = float(np.linalg.lstsq(defl[mask].reshape(-1, 1), fs[mask], rcond=None)[0][0])
    else:
        defl = np.zeros_like(zs)
        k_est = float("nan")
    compliance[name] = dict(z=zs, f=fs, defl=defl, k_est=k_est, kp=kp_c)
    print(f"         est k≈{k_est:.1f} N/m")

print("\n=== Numerical IK vs teleport FK ===")


def fk_teleport(q):
    robot.set_joint_positions(q.astype(np.float32))
    world.step(render=False)
    return get_ee().copy()


def numerical_ik(target, q0, iters=40):
    q = q0.astype(np.float64).copy()
    eps = 1e-3
    for _ in range(iters):
        p0 = fk_teleport(q)
        err = target - p0
        if np.linalg.norm(err) < 0.003:
            break
        J = np.zeros((3, 6))
        for j in range(6):
            dq = np.zeros(6)
            dq[j] = eps
            J[:, j] = (fk_teleport(q + dq) - p0) / eps
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
        q = q + np.clip(0.7 * dq, -0.3, 0.3)
    ee = fk_teleport(q)
    return q, ee


targets = [
    np.array([0.45, 0.10, 0.35]),
    np.array([0.50, -0.15, 0.40]),
    np.array([0.40, 0.20, 0.50]),
]
ik_rows = []
for tgt in targets:
    marker.set_world_pose(position=tgt)
    q_sol, ee = numerical_ik(tgt, Q_HOME)
    err = float(np.linalg.norm(ee - tgt))
    ik_rows.append({"target": tgt.tolist(), "ee": ee.tolist(), "err_m": err, "q": q_sol.tolist()})
    print(f"  target={np.round(tgt,3)}  ee={np.round(ee,3)}  |err|={err*1000:.1f} mm")

# restore effort mode after teleports
ctrl.switch_control_mode("effort")

sizing = {
    "max_efforts_Nm": MAX_TAU.tolist(),
    "recommended_kp": best_kp,
    "recommended_kd": float(max(12.0, 2.0 * np.sqrt(best_kp))),
    "joints": {},
    "controller_comparison": {
        m: {
            "mean_rms_rad": ctrl_results[m]["mean_rms"],
            "peak_tau_Nm": ctrl_results[m]["peak_tau_max"],
            "mean_abs_power_W": ctrl_results[m]["mean_power"],
            "saturation_frac": ctrl_results[m]["sat_frac"],
        }
        for m, _, _ in modes
    },
    "gain_sweep": {
        str(k): {
            "mean_rms_rad": sweep[k]["mean_rms"],
            "peak_tau_Nm": sweep[k]["peak_tau_max"],
            "mean_abs_power_W": sweep[k]["mean_power"],
            "saturation_frac": sweep[k]["sat_frac"],
        }
        for k in KP_GRID
    },
    "compliance_k_est_N_per_m": {k: compliance[k]["k_est"] for k in compliance},
    "ik_errors_m": [r["err_m"] for r in ik_rows],
    "gravity_sample_Nm": get_gravity().tolist(),
    "notes": "Torques are clipped to PhysX max_efforts. Prefer runs with low saturation_frac.",
}
for j, name in enumerate(JOINTS):
    sizing["joints"][name] = {
        "peak_tau_Nm": float(best["peak_tau"][j]),
        "rms_tau_Nm": float(best["rms_tau"][j]),
        "energy_J": float(best["energy"][j]),
        "rms_tracking_rad": float(best["rms"][j]),
        "limit_Nm": float(MAX_TAU[j]),
        "suggest_cont_Nm": float(min(MAX_TAU[j], 1.5 * best["rms_tau"][j])),
        "suggest_peak_Nm": float(min(MAX_TAU[j], 1.2 * best["peak_tau"][j])),
    }

report_path = os.path.join(args.out_dir, "motor_sizing_report.json")
with open(report_path, "w") as f:
    json.dump(sizing, f, indent=2)

md_path = os.path.join(args.out_dir, "motor_sizing_report.md")
with open(md_path, "w") as f:
    f.write("# UR10 impedance motor sizing report\n\n")
    f.write(
        f"Limits from PhysX: `{np.round(MAX_TAU,1).tolist()}` N·m  \n"
        f"Recommended: **Kp={best_kp:.0f}**, Kd≈{max(12, 2*np.sqrt(best_kp)):.1f} (ff+gravity, low saturation)\n\n"
    )
    f.write("## Controller comparison\n\n")
    f.write("| mode | mean RMS | peak |τ| | mean |P| | sat % |\n|---|---:|---:|---:|---:|\n")
    for mode, _, label in modes:
        r = ctrl_results[mode]
        f.write(
            f"| {label} | {r['mean_rms']:.4f} | {r['peak_tau_max']:.1f} | "
            f"{r['mean_power']:.1f} | {100*r['sat_frac']:.0f} |\n"
        )
    f.write("\n## Per-joint envelope\n\n")
    f.write("| joint | peak τ | RMS τ | limit | suggest cont | suggest peak |\n|---|---:|---:|---:|---:|---:|\n")
    for name, row in sizing["joints"].items():
        f.write(
            f"| {name} | {row['peak_tau_Nm']:.1f} | {row['rms_tau_Nm']:.1f} | "
            f"{row['limit_Nm']:.0f} | {row['suggest_cont_Nm']:.1f} | {row['suggest_peak_Nm']:.1f} |\n"
        )
    f.write("\n## Contact compliance\n\n")
    for name, c in compliance.items():
        f.write(f"- **{name}** (Kp={c['kp']}): est k ≈ {c['k_est']:.1f} N/m, peak |F|={c['f'].max():.1f} N\n")
    f.write("\n## IK (teleport FK)\n\n")
    for r in ik_rows:
        f.write(f"- target {np.round(r['target'],3)} → |err|={r['err_m']*1000:.1f} mm\n")

print(f"\nSaved {report_path}\nSaved {md_path}")

fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
fig.suptitle("Impedance full pack (gravity on, τ clipped to UR limits)", fontsize=12)
colors = {"baseline": "#7f7f7f", "ff": "#1f77b4", "ff_g": "#2ca02c"}

ax = axes[0, 0]
for mode, _, label in modes:
    r = ctrl_results[mode]
    ax.plot(r["t"], np.linalg.norm(r["err"], axis=1), color=colors[mode], label=label)
ax.set_title("Tracking error ‖e‖₂")
ax.set_ylabel("rad")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
kps = list(KP_GRID)
ax.plot(kps, [sweep[k]["mean_rms"] for k in kps], "o-", label="mean RMS")
ax.set_xlabel("Kp")
ax.set_ylabel("rad")
ax.set_title("Gain sweep")
ax.grid(True, alpha=0.3)
ax2 = ax.twinx()
ax2.plot(kps, [sweep[k]["peak_tau_max"] for k in kps], "s--", color="#d62728")
ax2.set_ylabel("peak |τ| (N·m)", color="#d62728")

ax = axes[1, 0]
r = ctrl_results["ff_g"]
for j in range(6):
    ax.plot(r["t"], r["tau"][:, j], lw=0.9, label=JOINTS[j])
ax.set_title("Motor τ clipped (FF+gravity)")
ax.set_ylabel("N·m")
ax.legend(fontsize=6, ncol=3)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
x = np.arange(6)
w = 0.35
ax.bar(x - w / 2, best["peak_tau"], w, label="peak")
ax.bar(x + w / 2, best["rms_tau"], w, label="RMS")
ax.plot(x, MAX_TAU, "k--", lw=1.2, label="limit")
ax.set_xticks(x)
ax.set_xticklabels(JOINTS)
ax.set_title(f"Motor envelope (Kp={best_kp:.0f})")
ax.set_ylabel("N·m")
ax.legend(fontsize=7)
ax.grid(True, axis="y", alpha=0.3)

ax = axes[2, 0]
for name, c in compliance.items():
    ax.plot(c["defl"], c["f"], ".", ms=3, label=f"{name} (k≈{c['k_est']:.0f})")
ax.set_xlabel("deflection (m)")
ax.set_ylabel("|F| (N)")
ax.set_title("Contact compliance")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[2, 1]
errs = [r["err_m"] * 1000 for r in ik_rows]
ax.bar(np.arange(len(errs)), errs, color="#9467bd")
ax.set_xticks(np.arange(len(errs)))
ax.set_xticklabels([f"T{i+1}" for i in range(len(errs))])
ax.set_ylabel("mm")
ax.set_title("IK position error (teleport FK)")
ax.grid(True, axis="y", alpha=0.3)

plot_path = os.path.join(args.out_dir, "impedance_full_pack_analysis.png")
fig.savefig(plot_path, dpi=160)
print(f"Saved plot: {plot_path}")

npz_path = os.path.join(args.out_dir, "impedance_full_pack_analysis.npz")
np.savez_compressed(
    npz_path,
    max_tau=MAX_TAU,
    **{f"ctrl_{m}_{k}": v for m, d in ctrl_results.items() for k, v in d.items() if isinstance(v, np.ndarray)},
)
print(f"Saved npz: {npz_path}")
print("\nDONE impedance full pack")
world.stop()
simulation_app.close()
