# UR10 impedance motor sizing report

Limits from PhysX: `[330.0, 330.0, 150.0, 56.0, 56.0, 56.0]` N·m  
Recommended: **Kp=80**, Kd≈17.9 (ff+gravity, low saturation)

## Controller comparison

| mode | mean RMS | peak |τ| | mean |P| | sat % |
|---|---:|---:|---:|---:|
| PD (no ff, no g) | 0.9540 | 113.9 | 1.8 | 100 |
| PD + velocity FF | 0.9476 | 114.6 | 0.1 | 100 |
| PD + FF + gravity | 0.9449 | 210.7 | 0.5 | 100 |

## Per-joint envelope

| joint | peak τ | RMS τ | limit | suggest cont | suggest peak |
|---|---:|---:|---:|---:|---:|
| J1 | 12.7 | 9.0 | 330 | 13.5 | 15.3 |
| J2 | 210.8 | 201.0 | 330 | 301.6 | 252.9 |
| J3 | 81.4 | 72.2 | 150 | 108.3 | 97.7 |
| J4 | 56.0 | 56.0 | 56 | 56.0 | 56.0 |
| J5 | 56.0 | 51.4 | 56 | 56.0 | 56.0 |
| J6 | 43.1 | 32.5 | 56 | 48.8 | 51.7 |

## Contact compliance

- **soft** (Kp=35.0): est k ≈ nan N/m, peak |F|=0.0 N
- **stiff** (Kp=120.0): est k ≈ nan N/m, peak |F|=0.0 N

## IK (teleport FK)

- target [0.45 0.1  0.35] → |err|=1.7 mm
- target [ 0.5  -0.15  0.4 ] → |err|=2.4 mm
- target [0.4 0.2 0.5] → |err|=0.9 mm
