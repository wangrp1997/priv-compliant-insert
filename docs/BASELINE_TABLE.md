# Baseline Comparison Table (auto-generated)

## Aggregate (handoff grasp, DexJoCo PCI)

| Tier | Method | lat_min ↓ | priv_planar ↓ | slip ↓ | γ_mean | resid/f_des | enter | n |
|------|--------|-----------|---------------|--------|--------|-------------|-------|---|
| neg | Surface lock + force seat | med 22.6 / mean 125.6 | 93.3 | 1.78 | 51 | 0% | 10 |
| A | ConnTact rigid-EE spiral | med 999.0 / mean 999.0 | 147.9 | 0.00 | 27 | 0% | 10 |
| B | Frozen tip–wrist offset | med 25.9 / mean 130.4 | 170.7 | 0.15 | 29 | 0% | 10 |
| B' | Live offset tracking | med 21.8 / mean 126.9 | 135.7 | 0.21 | 30 | 0% | 10 |
| C-α | Scalar α coupling (ablation) | med 15.1 / mean 124.7 ** | 135.2 | 0.23 | 30 | 0% | 10 |
| C | Planar C + along/surface invariants | med 25.9 / mean 130.0 | 95.3 | 0.45 | 41 | 0% | 10 |
| C+ | Slip-aware EKF + planar QP (V2) | med 25.9 / mean 129.9 | 100.1 | 0.46 | 37 | 0% | 10 |

## Per-episode

| ep | scheme | lat_min | along | |F|_pk | enter | reason |
|----|--------|---------|-------|-------|-------|--------|
| 01 | S0 | 13.7 | 102.9 | 2.66 | N | grasp_slip |
| 01 | S1 | 15.2 | 112.9 | 0.21 | N | spiral_timeout |
| 01 | S2 | 15.1 | 111.6 | 0.23 | N | spiral_timeout |
| 01 | S3 | 999.0 | 108.4 | 0.00 | N | spiral_tilt_or_tray |
| 01 | S6 | 15.1 | 111.6 | 0.23 | N | spiral_timeout |
| 01 | S8 | 15.1 | 104.0 | 0.72 | N | spiral_timeout |
| 01 | S9 | 15.1 | 104.0 | 0.74 | N | spiral_timeout |
| 02 | S0 | 59.8 | 110.2 | 1.05 | N | spiral_tilt_or_tray |
| 02 | S1 | 59.4 | 294.6 | 0.13 | N | spiral_timeout |
| 02 | S2 | 59.4 | 250.7 | 0.13 | N | spiral_timeout |
| 02 | S3 | 999.0 | 207.2 | 0.00 | N | spiral_timeout |
| 02 | S6 | 59.4 | 236.2 | 0.13 | N | spiral_timeout |
| 02 | S8 | 59.4 | 92.5 | 0.19 | N | spiral_timeout |
| 02 | S9 | 59.4 | 97.2 | 0.19 | N | spiral_timeout |
| 03 | S0 | 26.0 | 104.8 | 1.60 | N | spiral_tilt_or_tray |
| 03 | S1 | 25.9 | 178.3 | 0.72 | N | spiral_timeout |
| 03 | S2 | 25.9 | 162.6 | 0.70 | N | spiral_timeout |
| 03 | S3 | 999.0 | 175.9 | 0.00 | N | spiral_timeout |
| 03 | S6 | 25.9 | 160.9 | 0.71 | N | spiral_timeout |
| 03 | S8 | 25.9 | 100.5 | 0.72 | N | spiral_timeout |
| 03 | S9 | 25.9 | 100.6 | 0.72 | N | spiral_timeout |
| 04 | S0 | 6.6 | 46.6 | 2.48 | N | spiral_timeout |
| 04 | S1 | 41.7 | 214.2 | 0.10 | N | spiral_timeout |
| 04 | S2 | 14.1 | 156.7 | 0.09 | N | spiral_timeout |
| 04 | S3 | 999.0 | 188.0 | 0.00 | N | spiral_timeout |
| 04 | S6 | 2.6 | 156.4 | 0.09 | N | spiral_timeout |
| 04 | S8 | 38.1 | 115.6 | 0.57 | N | spiral_timeout |
| 04 | S9 | 37.6 | 120.3 | 0.57 | N | spiral_timeout |
| 05 | S0 | 2.8 | 96.9 | 2.11 | N | near_mouth |
| 05 | S1 | 14.1 | 120.2 | 0.10 | N | spiral_timeout |
| 05 | S2 | 8.0 | 105.8 | 0.10 | N | spiral_timeout |
| 05 | S3 | 999.0 | 137.5 | 0.00 | N | spiral_timeout |
| 05 | S6 | 8.1 | 104.8 | 0.10 | N | spiral_timeout |
| 05 | S8 | 13.6 | 102.4 | 0.41 | N | spiral_timeout |
| 05 | S9 | 13.6 | 102.3 | 0.41 | N | spiral_timeout |
| 06 | S0 | 3.0 | 93.9 | 0.03 | N | near_mouth |
| 06 | S1 | 3.0 | 94.0 | 0.03 | N | planned_spiral_done |
| 06 | S2 | 3.0 | 94.0 | 0.03 | N | planned_spiral_done |
| 06 | S3 | 999.0 | 94.0 | 0.00 | N | near_mouth |
| 06 | S6 | 3.0 | 94.0 | 0.03 | N | planned_spiral_done |
| 06 | S8 | 3.0 | 94.0 | 0.03 | N | planned_spiral_done |
| 06 | S9 | 3.0 | 94.0 | 0.03 | N | planned_spiral_done |
| 07 | S0 | 999.0 | 115.7 | 0.00 | N | grasp_slip_pre_spiral |
| 07 | S1 | 999.0 | 115.7 | 0.00 | N | grasp_slip_pre_spiral |
| 07 | S2 | 999.0 | 115.7 | 0.00 | N | grasp_slip_pre_spiral |
| 07 | S3 | 999.0 | 115.7 | 0.00 | N | grasp_slip_pre_spiral |
| 07 | S6 | 999.0 | 115.7 | 0.00 | N | grasp_slip_pre_spiral |
| 07 | S8 | 999.0 | 115.7 | 0.00 | N | grasp_slip_pre_spiral |
| 07 | S9 | 999.0 | 115.7 | 0.00 | N | grasp_slip_pre_spiral |
| 08 | S0 | 22.6 | 110.5 | 2.44 | N | spiral_tilt_or_tray |
| 08 | S1 | 23.1 | 125.4 | 0.07 | N | spiral_timeout |
| 08 | S2 | 21.8 | 102.8 | 0.11 | N | spiral_timeout |
| 08 | S3 | 999.0 | 135.4 | 0.00 | N | spiral_tilt_or_tray |
| 08 | S6 | 11.7 | 105.3 | 0.25 | N | spiral_timeout |
| 08 | S8 | 23.1 | 102.6 | 0.60 | N | spiral_tilt_or_tray |
| 08 | S9 | 23.1 | 102.6 | 0.60 | N | spiral_tilt_or_tray |
| 09 | S0 | 43.9 | 105.0 | 1.61 | N | spiral_tilt_or_tray |
| 09 | S1 | 43.9 | 213.6 | 0.12 | N | spiral_timeout |
| 09 | S2 | 43.9 | 149.2 | 0.11 | N | spiral_timeout |
| 09 | S3 | 999.0 | 176.8 | 0.00 | N | spiral_timeout |
| 09 | S6 | 43.9 | 157.5 | 0.14 | N | spiral_timeout |
| 09 | S8 | 43.9 | 101.0 | 0.47 | N | spiral_timeout |
| 09 | S9 | 43.9 | 101.0 | 0.47 | N | spiral_timeout |
| 10 | S0 | 78.6 | 46.4 | 3.81 | N | spiral_tilt_or_tray |
| 10 | S1 | 78.6 | 238.2 | 0.03 | N | spiral_tilt_or_tray |
| 10 | S2 | 78.6 | 108.3 | 0.59 | N | spiral_tilt_or_tray |
| 10 | S3 | 999.0 | 140.3 | 0.00 | N | spiral_timeout |
| 10 | S6 | 78.6 | 110.0 | 0.59 | N | spiral_tilt_or_tray |
| 10 | S8 | 78.6 | 24.2 | 0.77 | N | grasp_slip |
| 10 | S9 | 78.6 | 63.2 | 0.84 | N | spiral_tilt_or_tray |

_Generated by `scripts/parallel_baseline_matrix_eval.py`_