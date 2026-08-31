"""Shared OpenArm motor-analysis motion / gains / spring impedance.

Gentle sine for closed-loop demos, plus analytic qd/qdd for inverse-dynamics
(ID) torque envelopes — the fair motor-sizing metric (no PD amplification).

Joint impedance is a virtual spring-damper:
  τ = K_spring * (q_des - q) - K_damp * qd + g
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

N_ARM = 7
JOINT_NAMES = [f"J{i}" for i in range(1, N_ARM + 1)]

Q_HOME = np.array([0.0, -0.5, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

# Soft sizing motion — small amp, slow freq (esp. wrist)
AMP = np.array([0.10, 0.08, 0.08, 0.10, 0.05, 0.04, 0.05], dtype=np.float64)
FREQ = np.array([0.18, 0.20, 0.18, 0.22, 0.20, 0.18, 0.20], dtype=np.float64)

# Default soft closed-loop spring (demo only — not for motor peak sizing)
KP = np.array([25.0, 25.0, 18.0, 18.0, 10.0, 10.0, 10.0], dtype=np.float64)

# Diagonal of the joint-space mass matrix at Q_HOME (measured via MuJoCo
# mj_fullM on ~/openarm_mujoco/v1/openarm.xml). Wrist links (J5-J7) are as
# light as ~0.001-0.005 kg*m^2, ~2 orders of magnitude below J1/J2. Unit-mass
# critical damping (kd=2*sqrt(kp)) is wildly oversized there, which combined
# with a coarse control-rate zero-order hold on qfrc_bias/velocity feedback
# makes the loop numerically unstable (see run_openarm loops: control must be
# recomputed every physics substep, not held across substeps).
INERTIA_EST = np.array([0.136, 0.300, 0.064, 0.088, 0.00105, 0.00478, 0.00429], dtype=np.float64)

KD = 2.0 * np.sqrt(KP * INERTIA_EST)

# Named spring profiles: scale on the default per-joint spring vector
SPRING_PRESETS = {
    "soft": 0.5,      # compliant
    "medium": 1.0,    # default
    "stiff": 2.5,     # stiff virtual spring
    "very_stiff": 5.0,
}

MOTOR_KITS = {
    "damiao_v2": {
        "label": "Damiao v2 (DM8009/4340/4310)",
        "tau_lim": np.array([40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 7.0]),
        "color": "#1f77b4",
    },
    "equal_10nm": {
        "label": "Equal ±10 N·m (v1 default)",
        "tau_lim": np.full(N_ARM, 10.0),
        "color": "#2ca02c",
    },
    "weak_hobby": {
        "label": "Weak hobby (±5/3/1.5)",
        "tau_lim": np.array([5.0, 5.0, 3.0, 3.0, 1.5, 1.5, 1.5]),
        "color": "#d62728",
    },
    "strong_up": {
        "label": "Strong upgrade (±60/40/15)",
        "tau_lim": np.array([60.0, 60.0, 40.0, 40.0, 15.0, 15.0, 15.0]),
        "color": "#9467bd",
    },
}


def set_kit_ctrlrange(model, tau_lim: np.ndarray) -> None:
    """Sync MuJoCo actuator ctrlrange to the selected kit's torque limits.

    The MJCF bakes in a fixed +-10/12 N.m ctrlrange; without this, MuJoCo
    silently clamps data.ctrl there regardless of which kit is selected,
    making every kit comparison identical and starving the PD+g loop of
    authority once tracking error grows.
    """
    model.actuator_ctrlrange[:N_ARM, 0] = -tau_lim
    model.actuator_ctrlrange[:N_ARM, 1] = tau_lim


def q_ref_at(t: float) -> np.ndarray:
    return Q_HOME + AMP * np.sin(2.0 * np.pi * FREQ * t)


def qd_ref_at(t: float) -> np.ndarray:
    w = 2.0 * np.pi * FREQ
    return AMP * w * np.cos(w * t)


def qdd_ref_at(t: float) -> np.ndarray:
    w = 2.0 * np.pi * FREQ
    return -AMP * (w**2) * np.sin(w * t)


def make_spring(
    preset: str = "medium",
    scale: float | None = None,
    kp: Sequence[float] | float | None = None,
    kd: Sequence[float] | float | None = None,
    damp_ratio: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-joint spring (Kp) and damper (Kd).

    Parameters
    ----------
    preset : soft | medium | stiff | very_stiff
        Multiplies the default per-joint spring vector.
    scale : optional extra multiplier on top of preset.
    kp : override spring (scalar or length-7).
    kd : override damper (scalar or length-7). If None, Kd = 2*ζ*√Kp.
    damp_ratio : ζ for critical-damping-style Kd when kd is None.
    """
    if preset not in SPRING_PRESETS:
        raise ValueError(f"unknown spring preset {preset!r}; choose {list(SPRING_PRESETS)}")
    if kp is None:
        k = KP * SPRING_PRESETS[preset]
    else:
        k = np.broadcast_to(np.asarray(kp, dtype=np.float64), (N_ARM,)).copy()
    if scale is not None:
        k = k * float(scale)
    if kd is None:
        d = 2.0 * float(damp_ratio) * np.sqrt(np.maximum(k, 1e-9) * INERTIA_EST)
    else:
        d = np.broadcast_to(np.asarray(kd, dtype=np.float64), (N_ARM,)).copy()
    return k, d


def pd_torque(
    q: np.ndarray,
    qd: np.ndarray,
    q_des: np.ndarray,
    g: np.ndarray,
    kp: np.ndarray | None = None,
    kd: np.ndarray | None = None,
) -> np.ndarray:
    """Virtual spring-damper + gravity/bias feedforward: τ = K(q*−q) − D q̇ + g."""
    k = KP if kp is None else kp
    d = KD if kd is None else kd
    return k * (q_des - q) - d * qd + g
