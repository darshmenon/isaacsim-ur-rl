# UR10 impedance motor sizing report

Limits from PhysX: `[330.0, 330.0, 150.0, 56.0, 56.0, 56.0]` N·m  
Recommended: **Kp=40**, Kd≈12.6 (ff+gravity, low saturation)

## Controller comparison

| mode | mean RMS | peak |τ| | mean |P| | sat % |
|---|---:|---:|---:|---:|
| PD (no ff, no g) | 0.2479 | 56.0 | 25.9 | 70 |
| PD + velocity FF | 0.4271 | 56.0 | 23.0 | 94 |
| PD + FF + gravity | 0.2378 | 101.2 | 67.4 | 96 |

## Per-joint envelope

| joint | peak τ | RMS τ | limit | suggest cont | suggest peak |
|---|---:|---:|---:|---:|---:|
| J1 | 4.4 | 3.0 | 330 | 4.5 | 5.3 |
| J2 | 97.1 | 88.3 | 330 | 132.5 | 116.5 |
| J3 | 40.1 | 34.3 | 150 | 51.5 | 48.1 |
| J4 | 56.0 | 42.5 | 56 | 56.0 | 56.0 |
| J5 | 54.0 | 42.3 | 56 | 56.0 | 56.0 |
| J6 | 56.0 | 41.9 | 56 | 56.0 | 56.0 |

## Contact compliance

- **soft** (Kp=35.0): est k ≈ nan N/m, peak |F|=0.0 N
- **stiff** (Kp=120.0): est k ≈ 103.1 N/m, peak |F|=1.3 N

## IK (teleport FK)

- target [0.45 0.1  0.35] → |err|=1.9 mm
- target [ 0.5  -0.15  0.4 ] → |err|=2.4 mm
- target [0.4 0.2 0.5] → |err|=2.8 mm
