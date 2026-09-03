# Problem lock: known tip–mouth path → seat + upright + force enter

## User lock (2026-09-03)

- **Known:** hole mouth / tip spiral (or tip→mouth path) — from planning + pre-spiral vision.  
- **Study:** under non-rigid grasp, keep **tip on face**, **axis ≲ 25°**, and achieve **real force-controlled hole enter**.  
- **Not the focus:** blind discovery of where the hole is; not `tip_gt+noise` fake sensing.

## Evidence already in hand

| Result | Meaning under this lock |
|--------|-------------------------|
| ConnTact / Franka / PSFT **0/10** | Known-ish hole + wrist spiral ≠ tip seat/upright force enter |
| Oracle-v2 **9/10** | If tip↔mouth known and tip servo+seat+press: **control ceiling** |
| Type-A **5/10** | Deployable planar tip track along planned path; still loses on slip / mouth press / upright |

## Deployable algorithm stack (aligned)

1. Frozen mouth / spiral center from vision (known path)  
2. Tip-Hybrid: SEAT → UPRIGHT → follow path with planar_C / Type-A recover  
3. Near mouth: force probe enter (F/T), reject float/along/axis fails  

Oracle remains appendix upper bound only.
