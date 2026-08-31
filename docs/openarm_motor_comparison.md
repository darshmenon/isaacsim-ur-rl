# OpenArm custom motor kit comparison

Model: `/home/asimov/openarm_mujoco/v1/openarm.xml`. Soft sine amp=[0.1  0.08 0.08 0.1  0.05 0.04 0.05].

**Primary metric = inverse dynamics (ID)** along the reference — required τ for perfect tracking. Closed-loop PD is demo-only (soft gains).

## ID peak τ (sizing)

| joint | ID peak τ | Damiao limit | headroom |
|---|---:|---:|---:|
| J1 | 0.07 | 40.0 | 568.85× |
| J2 | 6.11 | 40.0 | 6.54× |
| J3 | 1.81 | 27.0 | 14.89× |
| J4 | 3.35 | 27.0 | 8.05× |
| J5 | 0.12 | 7.0 | 60.42× |
| J6 | 0.31 | 7.0 | 22.78× |
| J7 | 0.39 | 7.0 | 17.87× |

## Kit fit (ID peak ≤ limit on every joint?)

| kit | fits? | min headroom |
|---|---|---:|
| Damiao v2 (DM8009/4340/4310) | yes | 6.54× |
| Equal ±10 N·m (v1 default) | yes | 1.64× |
| Weak hobby (±5/3/1.5) | no | 0.82× |
| Strong upgrade (±60/40/15) | yes | 9.81× |

## Closed-loop soft PD (secondary)

| kit | mean RMS | mean sat % |
|---|---:|---:|
| Damiao v2 (DM8009/4340/4310) | 0.8341 | 78.6 |
| Equal ±10 N·m (v1 default) | 0.8072 | 87.0 |
| Weak hobby (±5/3/1.5) | 0.4141 | 81.9 |
| Strong upgrade (±60/40/15) | 0.8731 | 75.0 |

Plot: `openarm_motor_comparison.png`
