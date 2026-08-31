#!/usr/bin/env python3
"""Custom OpenArm motor-kit comparison in MuJoCo (no Isaac GPU needed).

Primary sizing metric: inverse-dynamics (ID) torque envelope along a soft
sine — required τ for perfect tracking, no PD amplification.

Also runs a soft closed-loop PD+g demo clipped to each motor kit.

Writes docs/openarm_motor_comparison.{png,npz,md}.

Usage
-----
    .venv/bin/python -u analyze_openarm_motors.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from openarm_motor_common import (
    AMP,
    JOINT_NAMES,
    MOTOR_KITS,
    N_ARM,
    Q_HOME,
    pd_torque,
    qd_ref_at,
    qdd_ref_at,
    q_ref_at,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_MJCF = Path.home() / "openarm_mujoco" / "v1" / "openarm.xml"


def id_envelope(model: mujoco.MjModel, seconds: float, hz: float) -> dict:
    """Required τ along q_ref via MuJoCo inverse dynamics (perfect tracking)."""
    data = mujoco.MjData(model)
    n_steps = int(seconds * hz)
    t = np.zeros(n_steps)
    tau_id = np.zeros((n_steps, N_ARM))

    for i in range(n_steps):
        ti = i / hz
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.qpos[:N_ARM] = q_ref_at(ti)
        data.qvel[:N_ARM] = qd_ref_at(ti)
        data.qacc[:N_ARM] = qdd_ref_at(ti)
        mujoco.mj_inverse(model, data)
        t[i] = ti
        tau_id[i] = data.qfrc_inverse[:N_ARM]

    peak = np.max(np.abs(tau_id), axis=0)
    rms = np.sqrt(np.mean(tau_id**2, axis=0))
    return {"t": t, "tau_id": tau_id, "peak_tau": peak, "rms_tau": rms}


def run_closed_loop(model: mujoco.MjModel, kit: dict, seconds: float, hz: float) -> dict:
    data = mujoco.MjData(model)
    n_steps = int(seconds * hz)
    sim_substeps = max(1, int(round(1.0 / (hz * model.opt.timestep))))
    assert model.nu >= N_ARM

    data.qpos[:N_ARM] = Q_HOME
    mujoco.mj_forward(model, data)
    for _ in range(int(1.5 * hz)):
        tau = pd_torque(
            data.qpos[:N_ARM], data.qvel[:N_ARM], Q_HOME, data.qfrc_bias[:N_ARM].copy()
        )
        tau = np.clip(tau, -kit["tau_lim"], kit["tau_lim"])
        data.ctrl[:N_ARM] = tau
        data.ctrl[N_ARM:] = 0.0
        for _ in range(sim_substeps):
            mujoco.mj_step(model, data)

    t = np.zeros(n_steps)
    err_log = np.zeros((n_steps, N_ARM))
    tau_log = np.zeros((n_steps, N_ARM))
    sat_log = np.zeros((n_steps, N_ARM), dtype=bool)

    for i in range(n_steps):
        ti = i / hz
        q_des = q_ref_at(ti)
        tau_cmd = pd_torque(
            data.qpos[:N_ARM], data.qvel[:N_ARM], q_des, data.qfrc_bias[:N_ARM].copy()
        )
        tau = np.clip(tau_cmd, -kit["tau_lim"], kit["tau_lim"])
        sat = np.abs(tau_cmd) > kit["tau_lim"] + 1e-9
        data.ctrl[:N_ARM] = tau
        data.ctrl[N_ARM:] = 0.0
        for _ in range(sim_substeps):
            mujoco.mj_step(model, data)
        t[i] = ti
        err_log[i] = q_des - data.qpos[:N_ARM]
        tau_log[i] = tau
        sat_log[i] = sat

    rms_err = np.sqrt(np.mean(err_log**2, axis=0))
    return {
        "rms_err": rms_err,
        "peak_tau": np.max(np.abs(tau_log), axis=0),
        "sat_pct": 100.0 * np.mean(sat_log, axis=0),
        "mean_rms": float(np.mean(rms_err)),
        "mean_sat": float(np.mean(100.0 * np.mean(sat_log, axis=0))),
        "tau_lim": kit["tau_lim"].copy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--hz", type=float, default=100.0)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs")
    args = parser.parse_args()

    if not args.mjcf.is_file():
        raise SystemExit(f"OpenArm MJCF not found: {args.mjcf}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    print(f"Loaded {args.mjcf}  nq={model.nq} nu={model.nu} dt={model.opt.timestep}")

    print("\n=== Inverse-dynamics torque envelope (sizing) ===")
    id_res = id_envelope(model, args.seconds, args.hz)
    print(f"  peak |τ_id| (N·m): {np.array2string(id_res['peak_tau'], precision=2)}")
    print(f"  RMS  |τ_id| (N·m): {np.array2string(id_res['rms_tau'], precision=2)}")

    # Headroom vs each kit: peak_id / limit
    print("\n=== ID peak vs kit limits (headroom = limit/peak) ===")
    for name, kit in MOTOR_KITS.items():
        peak = id_res["peak_tau"]
        lim = kit["tau_lim"]
        ratio = lim / np.maximum(peak, 1e-6)
        ok = peak <= lim
        print(
            f"  {kit['label']}: all_ok={bool(np.all(ok))}  "
            f"min_headroom={float(np.min(ratio)):.2f}x  "
            f"wrist_ok={bool(np.all(ok[4:]))}"
        )

    results = {}
    for name, kit in MOTOR_KITS.items():
        print(f"\n=== Closed-loop PD '{kit['label']}' ===")
        out = run_closed_loop(model, kit, args.seconds, args.hz)
        results[name] = out
        print(f"  mean RMS err: {out['mean_rms']:.4f}")
        print(f"  sat %:        {np.array2string(out['sat_pct'], precision=1)}")

    # Plot: ID peak vs limits + closed-loop sat/err
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax_id, ax_head, ax_err, ax_sat = axes.ravel()
    x = np.arange(N_ARM)
    width = 0.18

    ax_id.bar(x, id_res["peak_tau"], color="#444444", alpha=0.85, label="ID peak |τ|")
    for i, (name, kit) in enumerate(MOTOR_KITS.items()):
        ax_id.plot(x + (i - 1.5) * 0.05, kit["tau_lim"], "_", color=kit["color"], markersize=14, mew=2)
    ax_id.set_title("ID peak |τ| (sizing) · ticks = kit limits")
    ax_id.set_ylabel("N·m")
    ax_id.set_xticks(x)
    ax_id.set_xticklabels(JOINT_NAMES)
    ax_id.grid(True, axis="y", alpha=0.3)
    ax_id.legend(fontsize=8)

    # Headroom for Damiao
    dam = MOTOR_KITS["damiao_v2"]["tau_lim"]
    head = dam / np.maximum(id_res["peak_tau"], 1e-6)
    colors = ["#2ca02c" if h >= 1.0 else "#d62728" for h in head]
    ax_head.bar(x, head, color=colors, alpha=0.85)
    ax_head.axhline(1.0, color="k", ls="--", lw=1)
    ax_head.set_title("Damiao headroom = limit / ID peak (≥1 OK)")
    ax_head.set_ylabel("×")
    ax_head.set_xticks(x)
    ax_head.set_xticklabels(JOINT_NAMES)
    ax_head.grid(True, axis="y", alpha=0.3)

    for i, (name, kit) in enumerate(MOTOR_KITS.items()):
        r = results[name]
        off = (i - 1.5) * width
        ax_err.bar(x + off, r["rms_err"], width, label=kit["label"], color=kit["color"])
        ax_sat.bar(x + off, r["sat_pct"], width, color=kit["color"], alpha=0.85)

    ax_err.set_title("Closed-loop RMS tracking error (soft PD)")
    ax_err.set_ylabel("rad")
    ax_err.legend(fontsize=7, loc="upper right")
    ax_sat.set_title("Closed-loop saturation %")
    ax_sat.set_ylabel("%")
    for ax in (ax_err, ax_sat):
        ax.set_xticks(x)
        ax.set_xticklabels(JOINT_NAMES)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"OpenArm motor kits — ID sizing + soft PD (amp={np.array2string(AMP, precision=2)})",
        fontsize=12,
    )
    plot_path = args.out_dir / "openarm_motor_comparison.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    print(f"\nWrote {plot_path}")

    np.savez_compressed(
        args.out_dir / "openarm_motor_comparison.npz",
        id_peak=id_res["peak_tau"],
        id_rms=id_res["rms_tau"],
        id_tau=id_res["tau_id"],
        t=id_res["t"],
        **{f"{k}_sat": results[k]["sat_pct"] for k in results},
        **{f"{k}_rms": results[k]["rms_err"] for k in results},
    )

    md_path = args.out_dir / "openarm_motor_comparison.md"
    lines = [
        "# OpenArm custom motor kit comparison",
        "",
        f"Model: `{args.mjcf}`. Soft sine amp={np.array2string(AMP, precision=2)}.",
        "",
        "**Primary metric = inverse dynamics (ID)** along the reference — required τ "
        "for perfect tracking. Closed-loop PD is demo-only (soft gains).",
        "",
        "## ID peak τ (sizing)",
        "",
        "| joint | ID peak τ | Damiao limit | headroom |",
        "|---|---:|---:|---:|",
    ]
    for i, jn in enumerate(JOINT_NAMES):
        h = dam[i] / max(id_res["peak_tau"][i], 1e-6)
        lines.append(
            f"| {jn} | {id_res['peak_tau'][i]:.2f} | {dam[i]:.1f} | {h:.2f}× |"
        )

    lines += ["", "## Kit fit (ID peak ≤ limit on every joint?)", "", "| kit | fits? | min headroom |", "|---|---|---:|"]
    for name, kit in MOTOR_KITS.items():
        peak = id_res["peak_tau"]
        lim = kit["tau_lim"]
        ok = bool(np.all(peak <= lim))
        ratio = float(np.min(lim / np.maximum(peak, 1e-6)))
        lines.append(f"| {kit['label']} | {'yes' if ok else 'no'} | {ratio:.2f}× |")

    lines += ["", "## Closed-loop soft PD (secondary)", "", "| kit | mean RMS | mean sat % |", "|---|---:|---:|"]
    for name, kit in MOTOR_KITS.items():
        r = results[name]
        lines.append(f"| {kit['label']} | {r['mean_rms']:.4f} | {r['mean_sat']:.1f} |")

    lines += ["", f"Plot: `{plot_path.name}`", ""]
    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path}")

    summary = {
        "id_peak_Nm": id_res["peak_tau"].tolist(),
        "id_rms_Nm": id_res["rms_tau"].tolist(),
        "damiao_headroom": (dam / np.maximum(id_res["peak_tau"], 1e-6)).tolist(),
        "closed_loop": {
            name: {"mean_rms": results[name]["mean_rms"], "mean_sat": results[name]["mean_sat"],
                   "sat_pct": results[name]["sat_pct"].tolist()}
            for name in MOTOR_KITS
        },
    }
    (args.out_dir / "openarm_motor_comparison.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
