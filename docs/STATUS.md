# Project status — done vs left

## Accuracy (updated)

| Result | Trust? | Notes |
|---|---|---|
| Drake gravity / ID / IK | **Yes** | `analyze_dynamics_drake.py` — g≈90 N·m @ home, τ within UR limits, IK ≈0 mm |
| Isaac gravity + clipped τ | **Much better** | Gravity on (`|g|≈126 N·m`); τ clipped to `[330,330,150,56,56,56]`; FF+g best tracking |
| Isaac IK (teleport FK) | **Yes** | ~1–3 mm |
| Contact stiffness in Isaac | **Still broken** | Force still 0 (pad/filter); keep as known gap |
| Wrist J4–J6 at 56 N·m sat | **Expected** | Hitting asset effort limits under this motion |

**Drake** is installed in `.venv` (`pydrake`). Do **not** install Pinocchio into the Isaac venv (broke NumPy); Isaac needs `numpy<2`.

---

## What we did
- Impedance motor + full-pack analysis (Isaac)
- Drake reference dynamics on UR10 URDF (`assets/ur10_drake_nomesh.urdf`)
- README images: Isaac premiere, impedance plots, Drake plot
- SAC / force-collect / ACT short train demos earlier
- `docs/STATUS.md`, `docs/motor_sizing_report.md`, `docs/drake_dynamics_report.json`
- **OpenArm custom motor kits** (MuJoCo): cloned `~/openarm_mujoco` + `~/openarm_description`;
  `analyze_openarm_motors.py` compares Damiao / flat-10 / weak / strong packs →
  `docs/openarm_motor_comparison.{md,png,json}`
- **OpenArm Isaac check** (PhysX): `analyze_openarm_isaac.py` imports
  `assets/openarm_v1_isaac.urdf`, effort-mode PD+g on left arm →
  `docs/openarm_isaac_motor_comparison.{md,png,json}`
- **OpenArm fixes**: soft motion + per-joint PD in `openarm_motor_common.py`;
  MuJoCo now sizes via **inverse-dynamics peak τ** (Damiao fits with ≥6.5× headroom
  on soft sine; weak hobby fails J2). Closed-loop sat was PD windup — not true motor need.
  Isaac gravity still ~0 after USD/view enable (known gap).

## What’s left
1. Fix Isaac contact force readout (pad collision filter / contact API)
2. ACT closed-loop eval (py3.10↔3.12)
3. ROS 2 / URDF bridge (optional)
4. Cartesian / OSC impedance (optional)
5. Premiere MP4 export
6. Optional: Isaac Lab OpenArm bringup (beyond MuJoCo motor kits)
7. OpenArm Isaac gravity still near-zero after enable — dig into body flags

## Commands
```bash
.venv/bin/python -u analyze_dynamics_drake.py
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python -u analyze_impedance_full_pack.py --headless
.venv/bin/python -u analyze_openarm_motors.py   # custom Damiao motor kits, no GPU
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python -u analyze_openarm_isaac.py --headless
```
