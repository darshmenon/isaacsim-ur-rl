#!/usr/bin/env python3
"""Live OpenArm motor viewer — MuJoCo 3D + live graphs.

Virtual spring-damper impedance:
  τ = K_spring*(q_des−q) − K_damp*qd + g   (clipped to motor kit limits)

Controls (3D window focus)
--------------------------
  Tab / n       next motor kit
  b / p         previous motor kit
  [ / ]         softer / stiffer spring (×0.8 / ×1.25)
  - / =         less / more damping ratio
  1/2/3/4       spring presets soft/medium/stiff/very_stiff
  r             reset pose to home
  Space         pause / resume
  Esc / q       quit

Usage
-----
    .venv/bin/python -u view_openarm_live.py
    .venv/bin/python -u view_openarm_live.py --spring stiff --kit damiao_v2
    .venv/bin/python -u view_openarm_live.py --config configs/openarm_motor.yaml
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import yaml

from openarm_motor_common import (
    JOINT_NAMES,
    MOTOR_KITS,
    N_ARM,
    Q_HOME,
    SPRING_PRESETS,
    make_spring,
    pd_torque,
    q_ref_at,
    set_kit_ctrlrange,
)

DEFAULT_MJCF = Path.home() / "openarm_mujoco" / "v1" / "openarm.xml"
KIT_ORDER = list(MOTOR_KITS.keys())
PRESET_ORDER = list(SPRING_PRESETS.keys())
HIST = 400


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Live OpenArm spring/motor viewer")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--config", type=Path, default=None, help="YAML (configs/openarm_motor.yaml)")
    parser.add_argument("--kit", default=None, choices=KIT_ORDER)
    parser.add_argument("--spring", default=None, choices=PRESET_ORDER, help="spring preset")
    parser.add_argument("--scale", type=float, default=None, help="extra spring scale")
    parser.add_argument("--damp-ratio", type=float, default=None, dest="damp_ratio")
    parser.add_argument("--kp", type=float, default=None, help="uniform spring override (N·m/rad)")
    parser.add_argument("--kd", type=float, default=None, help="uniform damper override")
    parser.add_argument("--hz", type=float, default=50.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    spring_cfg = cfg.get("spring", {})
    preset = args.spring or spring_cfg.get("preset", "medium")
    scale = args.scale if args.scale is not None else float(spring_cfg.get("scale", 1.0))
    damp_ratio = (
        args.damp_ratio
        if args.damp_ratio is not None
        else float(spring_cfg.get("damp_ratio", 1.0))
    )
    kit_name = args.kit or cfg.get("motor_kit", "damiao_v2")

    if not args.mjcf.is_file():
        raise SystemExit(f"MJCF not found: {args.mjcf}")

    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)
    data.qpos[:N_ARM] = Q_HOME
    mujoco.mj_forward(model, data)

    kit_i = KIT_ORDER.index(kit_name)
    preset_i = PRESET_ORDER.index(preset) if preset in PRESET_ORDER else 1
    paused = False
    t_sim = 0.0

    set_kit_ctrlrange(model, MOTOR_KITS[KIT_ORDER[kit_i]]["tau_lim"])

    def refresh_spring():
        nonlocal kp, kd
        kp, kd = make_spring(
            preset=PRESET_ORDER[preset_i],
            scale=scale,
            kp=args.kp,
            kd=args.kd,
            damp_ratio=damp_ratio,
        )

    kp = kd = None  # set by refresh_spring
    refresh_spring()

    t_hist: deque[float] = deque(maxlen=HIST)
    q_hist = [deque(maxlen=HIST) for _ in range(N_ARM)]
    qd_hist = [deque(maxlen=HIST) for _ in range(N_ARM)]
    tau_hist = [deque(maxlen=HIST) for _ in range(N_ARM)]
    err_hist = [deque(maxlen=HIST) for _ in range(N_ARM)]
    sat_hist = [deque(maxlen=HIST) for _ in range(N_ARM)]

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    fig.canvas.manager.set_window_title("OpenArm live motor graphs")
    ax_q, ax_tau, ax_err, ax_sat = axes.ravel()

    q_lines, qdes_lines, tau_lines, lim_lines, err_lines = [], [], [], [], []
    for j in range(N_ARM):
        (ln_q,) = ax_q.plot([], [], lw=1.2, label=JOINT_NAMES[j])
        (ln_d,) = ax_q.plot([], [], lw=0.8, ls="--", color=ln_q.get_color(), alpha=0.6)
        q_lines.append(ln_q)
        qdes_lines.append(ln_d)
        (ln_t,) = ax_tau.plot([], [], lw=1.2, label=JOINT_NAMES[j])
        (ln_l,) = ax_tau.plot([], [], lw=0.8, ls=":", color=ln_t.get_color(), alpha=0.7)
        tau_lines.append(ln_t)
        lim_lines.append(ln_l)
        (ln_e,) = ax_err.plot([], [], lw=1.1, label=JOINT_NAMES[j])
        err_lines.append(ln_e)

    sat_bars = ax_sat.bar(np.arange(N_ARM), np.zeros(N_ARM), color="#1f77b4", alpha=0.85)
    ax_sat.set_xticks(np.arange(N_ARM))
    ax_sat.set_xticklabels(JOINT_NAMES)
    ax_sat.set_ylim(0, 100)
    ax_sat.set_ylabel("sat % (rolling)")
    ax_sat.set_title("Saturation")
    ax_sat.grid(True, axis="y", alpha=0.3)

    ax_q.set_title("q (solid) vs q_des (dashed)")
    ax_q.set_ylabel("rad")
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(fontsize=7, ncol=2, loc="upper right")
    ax_tau.set_title("|τ| solid · kit limit dotted")
    ax_tau.set_ylabel("N·m")
    ax_tau.grid(True, alpha=0.3)
    ax_err.set_title("Tracking error")
    ax_err.set_ylabel("rad")
    ax_err.set_xlabel("t (s)")
    ax_err.grid(True, alpha=0.3)

    status = fig.suptitle("", fontsize=11)
    fig.show()
    fig.canvas.flush_events()

    ctrl_dt = 1.0 / args.hz
    last_plot = 0.0

    print("Live OpenArm viewer — virtual spring-damper motors")
    print("  Tab=kit  [/]=spring  -/==damp  1-4=presets  Space=pause  r=reset  q=quit")
    print(f"  kit={MOTOR_KITS[kit_name]['label']}  spring={PRESET_ORDER[preset_i]}  "
          f"scale={scale:.2f}  ζ={damp_ratio:.2f}")
    print(f"  Kp={np.array2string(kp, precision=1)}  Kd={np.array2string(kd, precision=1)}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.8
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135

        def on_key(keycode: int) -> None:
            nonlocal kit_i, preset_i, scale, damp_ratio, paused, t_sim
            if keycode in (258, ord("N"), ord("n")):
                kit_i = (kit_i + 1) % len(KIT_ORDER)
                set_kit_ctrlrange(model, MOTOR_KITS[KIT_ORDER[kit_i]]["tau_lim"])
                print("kit ->", MOTOR_KITS[KIT_ORDER[kit_i]]["label"])
            elif keycode in (ord("B"), ord("b"), ord("P"), ord("p")):
                kit_i = (kit_i - 1) % len(KIT_ORDER)
                set_kit_ctrlrange(model, MOTOR_KITS[KIT_ORDER[kit_i]]["tau_lim"])
                print("kit ->", MOTOR_KITS[KIT_ORDER[kit_i]]["label"])
            elif keycode == ord("["):
                scale = max(0.05, scale * 0.8)
                refresh_spring()
                print(f"spring softer  scale={scale:.2f}  mean Kp={float(np.mean(kp)):.1f}")
            elif keycode == ord("]"):
                scale = min(20.0, scale * 1.25)
                refresh_spring()
                print(f"spring stiffer scale={scale:.2f}  mean Kp={float(np.mean(kp)):.1f}")
            elif keycode == ord("-"):
                damp_ratio = max(0.1, damp_ratio * 0.8)
                refresh_spring()
                print(f"damping ζ={damp_ratio:.2f}")
            elif keycode in (ord("="), ord("+")):
                damp_ratio = min(5.0, damp_ratio * 1.25)
                refresh_spring()
                print(f"damping ζ={damp_ratio:.2f}")
            elif keycode in (ord("1"), ord("2"), ord("3"), ord("4")):
                preset_i = int(chr(keycode)) - 1
                scale = 1.0
                refresh_spring()
                print(f"preset -> {PRESET_ORDER[preset_i]}  mean Kp={float(np.mean(kp)):.1f}")
            elif keycode == 32:
                paused = not paused
                print("paused" if paused else "running")
            elif keycode in (ord("R"), ord("r")):
                data.qpos[:N_ARM] = Q_HOME
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                t_sim = 0.0
                print("reset")
            elif keycode in (256, ord("Q"), ord("q")):
                viewer.close()

        viewer.user_key_callback = on_key

        while viewer.is_running():
            step_start = time.time()
            kit = MOTOR_KITS[KIT_ORDER[kit_i]]
            lim = kit["tau_lim"]

            if not paused:
                # Recompute the PD+gravity torque every physics substep (not
                # just once per control tick) — holding a stale qfrc_bias/
                # velocity-feedback torque across ~10 substeps at 500 Hz is
                # what caused the runaway saturation/instability, especially
                # on the low-inertia wrist joints.
                n_sub = max(1, int(round(ctrl_dt / model.opt.timestep)))
                sub_dt = model.opt.timestep
                for k in range(n_sub):
                    qdes = q_ref_at(t_sim + k * sub_dt)
                    tau_cmd = pd_torque(
                        data.qpos[:N_ARM],
                        data.qvel[:N_ARM],
                        qdes,
                        data.qfrc_bias[:N_ARM].copy(),
                        kp=kp,
                        kd=kd,
                    )
                    tau = np.clip(tau_cmd, -lim, lim)
                    sat = np.abs(tau_cmd) > lim + 1e-9
                    data.ctrl[:N_ARM] = tau
                    data.ctrl[N_ARM:] = 0.0
                    mujoco.mj_step(model, data)
                t_sim += ctrl_dt

                q = data.qpos[:N_ARM].copy()
                t_hist.append(t_sim)
                for j in range(N_ARM):
                    q_hist[j].append(q[j])
                    qd_hist[j].append(qdes[j])
                    tau_hist[j].append(float(tau[j]))
                    err_hist[j].append(float(qdes[j] - q[j]))
                    sat_hist[j].append(1.0 if sat[j] else 0.0)

            viewer.sync()

            now = time.time()
            if now - last_plot >= 0.05 and len(t_hist) > 2:
                last_plot = now
                tt = np.asarray(t_hist)
                for j in range(N_ARM):
                    q_lines[j].set_data(tt, np.asarray(q_hist[j]))
                    qdes_lines[j].set_data(tt, np.asarray(qd_hist[j]))
                    tau_lines[j].set_data(tt, np.abs(np.asarray(tau_hist[j])))
                    lim_lines[j].set_data(tt, np.full_like(tt, lim[j]))
                    err_lines[j].set_data(tt, np.asarray(err_hist[j]))
                    sat_pct = 100.0 * float(np.mean(sat_hist[j])) if sat_hist[j] else 0.0
                    sat_bars[j].set_height(sat_pct)
                    sat_bars[j].set_color(kit["color"])

                for ax in (ax_q, ax_tau, ax_err):
                    ax.relim()
                    ax.autoscale_view()
                    if len(tt):
                        ax.set_xlim(max(0.0, tt[-1] - HIST / args.hz), max(tt[-1], 1.0))

                mean_sat = float(np.mean([np.mean(s) for s in sat_hist if len(s)])) * 100
                mean_err = float(
                    np.mean([np.sqrt(np.mean(np.square(e))) for e in err_hist if len(e)])
                )
                status.set_text(
                    f"{kit['label']}  |  spring={PRESET_ORDER[preset_i]}×{scale:.2f}  ζ={damp_ratio:.2f}  "
                    f"Kp̄={float(np.mean(kp)):.0f}  |  t={t_sim:.1f}s  RMS={mean_err:.3f}  "
                    f"sat={mean_sat:.0f}%  |  {'PAUSED' if paused else 'LIVE'}"
                )
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            elapsed = time.time() - step_start
            if elapsed < ctrl_dt:
                time.sleep(ctrl_dt - elapsed)

    plt.ioff()
    plt.close(fig)
    print("closed")


if __name__ == "__main__":
    main()
