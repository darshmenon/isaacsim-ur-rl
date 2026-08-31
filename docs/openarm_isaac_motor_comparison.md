# OpenArm motor kits — Isaac Sim check

URDF: `/home/asimov/isaacsim-ur-rl/assets/openarm_v1_isaac.urdf` (paths rewritten from `openarm_description`).
Controller: PD + gravity FF (soft motion, per-joint Kp), 50 Hz, 4.0 s.
Articulation: `/openarm` left arm only driven.
Gravity sample |g|=0.13 N·m after USD/view enable.

## Summary

| kit | mean RMS (rad) | mean sat % |
|---|---:|---:|
| Damiao v2 (DM8009/4340/4310) | 0.1105 | 0.0 |
| Equal ±10 N·m (v1 default) | 0.1111 | 28.6 |
| Weak hobby (±5/3/1.5) | 0.1177 | 49.6 |
| Strong upgrade (±60/40/15) | 0.0874 | 0.0 |

## Damiao v2 per-joint (Isaac)

| joint | peak τ | limit | sat % |
|---|---:|---:|---:|
| J1 | 1.04 | 40.0 | 0.0 |
| J2 | 20.01 | 40.0 | 0.0 |
| J3 | 19.18 | 27.0 | 0.0 |
| J4 | 4.65 | 27.0 | 0.0 |
| J5 | 5.51 | 7.0 | 0.0 |
| J6 | 5.95 | 7.0 | 0.0 |
| J7 | 2.29 | 7.0 | 0.0 |

Plot: `openarm_isaac_motor_comparison.png`

Compare with MuJoCo: `docs/openarm_motor_comparison.md`.
