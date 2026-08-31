# OpenArm motor kits — Isaac Sim check

URDF: `/home/asimov/isaacsim-ur-rl/assets/openarm_v1_isaac.urdf` (paths rewritten from `openarm_description`).
Controller: PD + gravity FF (soft motion, per-joint Kp), 50 Hz, 1.0 s.
Articulation: `/openarm` left arm only driven.
Gravity sample |g|=0.13 N·m after USD/view enable.

## Summary

| kit | mean RMS (rad) | mean sat % |
|---|---:|---:|
| Damiao v2 (DM8009/4340/4310) | 0.1174 | 18.9 |
| Equal ±10 N·m (v1 default) | 0.1306 | 8.0 |
| Weak hobby (±5/3/1.5) | 0.1875 | 66.9 |
| Strong upgrade (±60/40/15) | 0.1483 | 0.0 |

## Damiao v2 per-joint (Isaac)

| joint | peak τ | limit | sat % |
|---|---:|---:|---:|
| J1 | 11.54 | 40.0 | 0.0 |
| J2 | 16.80 | 40.0 | 0.0 |
| J3 | 11.94 | 27.0 | 0.0 |
| J4 | 20.46 | 27.0 | 0.0 |
| J5 | 6.32 | 7.0 | 0.0 |
| J6 | 7.00 | 7.0 | 50.0 |
| J7 | 7.00 | 7.0 | 82.0 |

Plot: `openarm_isaac_motor_comparison.png`

Compare with MuJoCo: `docs/openarm_motor_comparison.md`.
