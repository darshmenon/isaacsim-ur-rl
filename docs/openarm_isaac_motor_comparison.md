# OpenArm motor kits — Isaac Sim check

URDF: `/home/asimov/isaacsim-ur-rl/assets/openarm_v1_isaac.urdf` (paths rewritten from `openarm_description`).
Controller: PD + gravity FF (soft motion, per-joint Kp), 50 Hz, 4.0 s.
Articulation: `/openarm/root_joint` left arm only driven.
Gravity sample |g|=0.13 N·m after USD/view enable.

## Summary

| kit | mean RMS (rad) | mean sat % |
|---|---:|---:|
| Damiao v2 (DM8009/4340/4310) | 0.5887 | 71.1 |
| Equal ±10 N·m (v1 default) | 0.8192 | 81.6 |
| Weak hobby (±5/3/1.5) | 0.6204 | 95.0 |
| Strong upgrade (±60/40/15) | 0.5703 | 56.1 |

## Damiao v2 per-joint (Isaac)

| joint | peak τ | limit | sat % |
|---|---:|---:|---:|
| J1 | 27.97 | 40.0 | 0.0 |
| J2 | 40.00 | 40.0 | 21.5 |
| J3 | 27.00 | 27.0 | 76.5 |
| J4 | 27.00 | 27.0 | 100.0 |
| J5 | 7.00 | 7.0 | 100.0 |
| J6 | 7.00 | 7.0 | 100.0 |
| J7 | 7.00 | 7.0 | 100.0 |

Plot: `openarm_isaac_motor_comparison.png`

Compare with MuJoCo: `docs/openarm_motor_comparison.md`.
