# Project status — done vs left

## Accuracy (updated)

| Result | Trust? | Notes |
|---|---|---|
| Drake gravity / ID / IK | **Yes** | `analyze_dynamics_drake.py` — g≈90 N·m @ home, τ within UR limits, IK ≈0 mm |
| Isaac gravity + clipped τ | **Yes** | `|g|≈89.8 N·m` (now matches Drake closely); τ clipped to `[330,330,150,56,56,56]`; FF+g best tracking |
| Isaac IK (teleport FK) | **Yes** | ~2–3 mm |
| Contact stiffness in Isaac | **Fixed (2026-08-31)** | `stiff` trial now reads real contact: peak F=1.3 N, est k≈103 N/m. `soft` trial still 0 (likely drifts off the pad footprint before reaching contact height — smaller residual gap, not the frozen-robot bug) |
| Wrist J4–J6 at 56 N·m sat | **Expected** | Hitting asset effort limits under this motion |

**2026-08-31 correction — earlier numbers above (`|g|≈126 N·m` and everything in "What we did") were measured on a frozen robot.** Root cause: the UR10's raw all-joints-near-zero USD default pose is a near-singular/near-self-collision configuration (EE ends up ~5cm off the ground). Once gravity is enabled, driving out of that pose via open-loop effort-mode PD (Kp/Kd computed in Python, zero PD at the physics level) deadlocks the PhysX solver — commanding hundreds of N·m produced **zero net motion** (mean mechanical power ~0 W, 100% saturation) through the entire controller-comparison and gain-sweep sections in every prior run. `analyze_impedance_full_pack.py` now teleports the robot to a safe home pose (`robot.set_joint_positions(...)`, still in position-drive mode) immediately after enabling gravity and before switching to effort mode, which sidesteps the deadlock. Re-run numbers: controller comparison mean|P| now 23–67 W (previously ~0 W), meanRMS 0.24–0.43 rad (previously ~0.95, i.e. not tracking at all), gain-sweep recommended Kp=40/Kd≈12.6. The contact-compliance probe additionally had two independent bugs on top of the frozen-robot issue: the press direction (`q_reach[1] - 0.01*k`) was moving the wrist away from the pad instead of into it (fixed: `+`), and the pad's hardcoded position (`x=0.55`) didn't match `q_reach`'s actual forward kinematics (wrist settles at `x≈0.84`) — fixed by repositioning the pad under the wrist's actual settled position instead of assuming FK matches a fixed coordinate.

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
  on soft sine; weak hobby fails J2). Closed-loop sat was mostly a zero-order-hold
  bug (torque held stale across physics substeps) + unit-mass damping gains ignoring
  the wrist links' tiny inertia — fixed in MuJoCo and Isaac, sat now 0% for Damiao.
  Isaac gravity-forces *readout* (`get_generalized_gravity_forces`) still reads ~0
  even though gravity is verified real in the physics (zero-torque free-fall moved
  the arm 2.1 rad in 1s) — a tensor-API bug, not a "gravity disabled" issue.

## What’s left
1. ~~Fix Isaac contact force readout~~ **Done 2026-08-31** — was a frozen-robot deadlock (see correction above), not a filter/API issue. `soft` trial still reads 0 (minor residual: likely drifts horizontally off the pad before reaching contact height at the softer gain) — optional follow-up if the soft-gain compliance number is needed.
2. ACT closed-loop eval (py3.10↔3.12)
3. ROS 2 / URDF bridge (optional)
4. Cartesian / OSC impedance (optional)
5. Premiere MP4 export
6. Optional: Isaac Lab OpenArm bringup (beyond MuJoCo motor kits)
7. OpenArm Isaac gravity-forces readout still near-zero — root cause narrowed
   to a tensor-API bug (physics itself confirmed correct via free-torque
   free-fall test); needs Isaac Sim source/issue-tracker digging to fix

## Commands
```bash
.venv/bin/python -u analyze_dynamics_drake.py
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python -u analyze_impedance_full_pack.py --headless
.venv/bin/python -u analyze_openarm_motors.py   # custom Damiao motor kits, no GPU
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python -u analyze_openarm_isaac.py --headless
```
