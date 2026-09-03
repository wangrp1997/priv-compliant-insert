# From privileged quantities → real paths (no GT+noise fake)

> User lock: **禁止** `tip_gt + N(0,σ)` 冒充可观测。  
> Oracle-v2 **9/10** = 控制律诊断上界（特权 GT），可写附录证明上限。  
> 主表只有下面 Path A / B。

## Two legitimate paths

### Path A — Estimate what Oracle used (same controller)

Recover tip / extrinsic contact / seat / slip with **real estimators** from eligible sensors, then feed the **same** Tip-Hybrid + tip→mouth / Type-A law.

| Quantity | Estimator direction (literature) | Eligible inputs |
|----------|----------------------------------|-----------------|
| Tip / contact on tray | Kim TEC / tactile extrinsic FG (`refs/Tactile-Estimator-Controller`) | Fingertip tactile + kinematics |
| Mouth XY | Frozen vision coarse hole (pre-spiral) | RGB / known tray TF |
| Seat | Force residual / tactile normal | Wrist F/T, fingertips |
| Axis | Extrinsic contact orientation | Tactile + FK |
| Slip | Shear / innov / finger lag | Tactile + proprio |

Sim milestone: wire a **tactile/contact estimator** (or PCI port of TEC factors with sim tactile), output `tip_hat` **without reading peg xpos for control**. GT tip only for priv diag.

**v0 implemented (Track A):** `ContactTipEstimator` + config `S2_trackA_est_oracle_ctrl.yaml` + `docs/TRACK_A_TIP_ESTIMATOR.md`. Uses MuJoCo peg↔tray contact geometry (disclose residual privilege); same Oracle-v2 hooks.

### Path B — Never need those quantities

Already the deployable line: wrist F/T + proprio + planner spiral + `planar_C` / Type-A / force-hole — **no tip GT, no hole GT in loop**.  
Rate today: Type-A **5/10**. Improve control/estimation of **C and slip**, not hole attractor.

## What Oracle is for

- Prove **control ceiling** when tip↔mouth known → Oracle-v2 9/10.
- Does **not** prove deployable success.
- Gap Oracle − Path B = “need better sensing (Path A) and/or better non-priv control (Path B)”.

## Invalid (do not claim)

- `tip_hat = tip_gt + noise` as “sensor observation”
- Reporting Oracle rate in deployable table
