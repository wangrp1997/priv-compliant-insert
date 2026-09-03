# Track-B FASR — REJECTED (engineering patch)

> **Status:** Rejected as primary under user hard correction (2026-09-03).  
> Superseded by principled **OIGS** (`docs/TRACK_B_OIGS.md`).

## Why rejected

FASR stacked wrist known-path + tangential-force **threshold** bias + EMA spiral-center shift.  
No new estimator / object-impedance / constrained-MPC equation — cite was stretch (Tang spiral).

## Recorded smoke (do not expand)

`outputs/scheme_l3/S2_trackB_fasr_smoke` eps **3,9,10**:

| ep | planar_min | mouth | tip_err_max | enter |
|----|------------|-------|-------------|-------|
| 03 | 5.06 | T | 88.6 | 0 |
| 09 | 13.56 | F | 121.5 | 0 |
| 10 | 15.69 | F | 110.2 | 0 |

**RATE 0/3.** tip_spiral_err inflated vs Type-A; not a priv-metric win.
