# Track-A privileged diagnosis (Phase 0) — 2026-09-03

> Gate: numbers **before** any v4 fix. Sources: `diag_spiral_priv.py` + `ep*_summary.json` meta.  
> SUCCESS_STANDARD locked; Oracle-v2 **9/10** diagnostic only.

## Runs compared

| run | out | RATE (ep1–4) |
|-----|-----|--------------|
| Track-A **v0** | `S2_trackA_est_oracle_smoke` | **2/4** (ok 1,2; fail 3,4) |
| Track-A v3 | `..._smoke_v3` | **1/4** (ok 2 only) |
| Track-A v3b | `..._smoke_v3b` | **0/4** |
| Oracle-v2 | `S2_oracle_tip_v2_ep1_10` ep01–04 | **3/4** (fail 4 only) |

## Metric meanings (do not conflate)

| field | meaning |
|-------|---------|
| `spiral_contact_frac` / `cok` | wrist residual ≥ `seat_contact_n` (≈0.08 N) during spiral |
| `tip_hybrid_seat_frames` | Tip-Hybrid FSM spent in mode `seat` |
| `tip_est_seated_frames` | ContactTipEstimator saw peg↔tray `fsum ≥ seat_force_n` (0.015 N) |
| `tip_est_err_peak_mm` | planar ‖tip_hat − tip_GT‖ peak (log only; not control) |
| `tip_est_last_source` | final estimator source string |

## Claim check: “long spiral seat frames ≈0 on peg↔tray”

**VERIFIED for tip_est peg↔tray seat; NOT for wrist contact.**

| run/ep | n_spiral (diag) | cfrac (wrist) | cok/cfr | hyb_seat | **est_seat** | tip_est_err_peak_mm | last_source |
|--------|-----------------|---------------|---------|----------|--------------|---------------------|-------------|
| v0/ep02 | 5183 | 0.9996 | 5181/5183 | **3862** | *(not logged)* | **71.3** | wrist_hold |
| v0/ep03 | 2323 | **1.0** | 2323/2323 | **0** | *(not logged)* | **46.4** | wrist_hold |
| v0/ep04 | 314 | **0.003** | 1/314 | 299 | *(not logged)* | **68.1** | wrist_hold |
| v3/ep02 | 7159 | 0.942 | 6742/7159 | 2242 | **0** | 46.8 | lost_freeze |
| v3/ep03 | 8988 | 0.982 | 8830/8988 | **0** | **0** | **407.5** | lost_freeze |
| v3b/ep03 | 8988 | 0.996 | 8952/8988 | **0** | 31 | **609.7** | lost_freeze |
| oracle/ep03 | 135 | 1.0 | 135/135 | **130** | n/a | n/a | n/a (tip GT) |

- Wrist force can be “fully seated” (`cfrac≈1`) while **`tip_est_seated_frames=0`** for thousands of steps (v3 ep02/03).
- ep03 also never enters Tip-Hybrid `seat` mode (`hyb_seat=0`) on Track-A; Oracle ep03 spends **130** frames in seat.
- ep04 is different: wrist contact almost absent (`cfrac≈0.003`) — float / surface_press path, not “long spiral zero peg contact.”

## Fail ep tip_hat vs GT + axis

| run/ep | tip_est_err_peak_mm | last_source | final_axis_err_deg | mouth_enter | diag note |
|--------|---------------------|-------------|--------------------|-------------|-----------|
| v0/03 | **46.4** | wrist_hold | **73.4** | no | `priv_planar_min≈4.5` mm, `priv_mouth=True`, `bl_tip_err_mean≈1.3` mm; gate axis fail |
| v0/04 | **68.1** | wrist_hold | **44.7** | no | cfrac≈0; surface_press |
| v3/03 | **407.5** | lost_freeze | **9.0** | no | planar timeout (`priv_planar_min≈14.9` mm); freeze-at-bad-init |
| v3/01 | 20.5 | lost_freeze | **56.8** | no | axis fail (was ok under v0) |
| oracle/03 | — | tip GT | **4.3** | **yes** | short spiral n=135 |
| oracle/04 | — | tip GT | **42.8** | no | same ep04 axis ceiling |

Contrast: Oracle tip spiral err mean ≈5 mm; Track-A v0 tip spiral err mean ≈29–53 mm on fails.

## Root cause (from numbers, not guess)

1. **Observation starvation:** peg↔tray force-seat updates ≈0 on long spirals even when wrist residual says seated → tip_hat lives in holdover (`wrist_hold` / `lost_freeze`).
2. **freeze_planar pathology:** any early accepted measure (or weak init) sets freeze anchor → if that anchor is wrong, tip_hat stuck hundreds of mm from GT (v3 ep03 peak **407** mm) while wrist force still looks fine.
3. **v0 wrist_hold** drifts less catastrophically (peak 46–71 mm) → keeps **2/4**, but fails ep03 upright/force gate at mouth (axis **73°**).

## ONE fix chosen for v4 (justified by above)

**Gated freeze + first-strong-contact snap** (single change family):

- Only set freeze anchor (`_have_seated_tip`) when peg↔tray `fsum ≥ seat_force_n` (true tip_est seat), **not** on weak/any contact.
- Until first strong seat: planar bridge = `wrist + offset` (no freeze-at-init).
- On first strong seat: hard-accept tip measure (bypass one-shot outlier gate) so tip_hat can leave bad finger/seed init.

Why this and not lower force alone / re-enable wrist_delta / PCA: numbers show freeze-of-bad-init is the v3 killer (407 mm); wrist_delta already **0/4**; PCA previously nuked rate. Gating freeze to real seat restores v0-safe bridge when peg↔tray seat is absent, and allows recovery when a strong contact finally appears.

## Next

Out-root: `outputs/scheme_l3/S2_trackA_est_oracle_smoke_v4` ep1–4 + re-diag. Gate: must not regress vs v0 **2/4**; ≥3/4 may expand ep1–10; else STOP.

## v4 result (gated freeze + first-strong-seat snap) — STOP

Fix landed in `tip_contact_estimator.py` (`freeze_require_seat=true`, first strong seat hard-snap) + config flag `surface_tip_est_freeze_require_seat`.

| ep | mouth_enter | fail_reason | tip_est_err_peak_mm | est_seat | last_source | axis_err_deg |
|----|-------------|-------------|---------------------|----------|-------------|--------------|
| 01 | no | surface_press | 20.5 | 136 | lost_freeze | **56.8** |
| 02 | no | surface_tilt_or_tray | **177.8** | 11 | lost_freeze | 9.0 (planar timeout lat_min≈7 mm) |
| 03 | *(no summary)* | worker crash `is_fasr_mode` NameError mid-run (infra; unrelated to est) | — | — | — | — |
| 04 | no | surface_press | 68.9 | 0 | wrist_hold | **44.6** |

- **RATE 0/4** (vs v0 **2/4**) → **gate FAIL** → **do not expand** ep1–10.
- ep01/02 already lose vs v0 wins → even re-running ep03 cannot meet ≥2/4 keep.
- Honest Track-A baseline remains **v0 smoke 2/4**. Code keeps gated-freeze switch for next authorized round (contact reliability / force proxy still open).

## Continuation (post-v4 STOP) — contact starvation → estimation problem

> Hard rule (user 2026-09-03): **no** threshold / freeze / geom-id / per-ep patches.  
> Patch line `smoke_v5` **abandoned** before RATE; not progress.

### Deeper numbers (v0 `surface_meta.force_trace`, planned_spiral)

| ep | spiral N | priv_peg_tray ncon>0 | depth_mm mean (ncon=0) | wrist cfrac | tip_est last | RATE role |
|----|----------|----------------------|------------------------|-------------|--------------|-----------|
| 01 | 110 | **93 (84.5%)** | −2.8 | 0.53 | wrist_hold | **ok** |
| 02 | 5183 | **7 (0.14%)** | −78.7 | ≈1.0 | wrist_hold | **ok** (lucky holdover) |
| 03 | 2323 | **11 (0.47%)** | −42.7 | 1.0 | wrist_hold | **fail** axis≈73° |
| 04 | 314 | **0** | −7.4 | ≈0 | wrist_hold | **fail** surface_press |
| Oracle/03 | 135 | **0** | −2.4 | 1.0 | n/a (tip GT) | **ok** |

- When tip truly rides the face (ep01 depth≈−2 mm), peg↔tray contact observation **works** (84%).
- Long “wrist-seated” spirals (ep02/03) still have **≪1%** peg↔tray contact frames → tip_hat lives in holdover; Oracle succeeds on ep03 with **tip GT** despite same zero priv_peg_tray.
- Conclusion: starvation is **real** (not just a force gate), and **wrist residual ≠ extrinsic tip seat**. Holdover / freeze cannot invent missing tip planar state.

### Priv gap → estimation problem statement

**Scientific gap (general, not ep-hack):**  
Under non-rigid grasp, Oracle-v2 needs tip planar pose on the tray. Eligible / residual-privileged measurements (wrist wrench, finger proprio, intermittent extrinsic contact points) arrive asynchronously and often **omit** peg↔tray contact for thousands of steps while wrist F/T still looks seated. Recover \(\hat{t}_\parallel\) (and optionally contact normal / grasp coupling \(C\)) with a **contact-state observer**, not a holdover flag.

**Problem (Track-A):**

> Estimate extrinsic tip planar state \(t_\parallel\) online from wrench + proprio (+ intermittent contact geometry), under process model that **does not** assume \(t = w + o\) (rigid offset), then feed the **same** Oracle-v2 control law.

**Literature basis (cites):**

| Cite | Role for Track-A |
|------|------------------|
| Kim et al., ICRA 2023 — Simultaneous Tactile Estimation and Control; `refs/Tactile-Estimator-Controller` | Factor-graph extrinsic contact + ContactMotion (control at contact ≠ rigid wrist) |
| Bronars et al., ICRA 2024 TEXterity (arXiv:2403.00049) | Discrete+continuous extrinsic pose estimator–controller |
| Kim & Rodriguez, ICRA 2022 Active Extrinsic Contact (arXiv:2110.03555) | Contact-line / wrench geometry without full tip GT |
| Doshi-style wrist F/T lever arm (in-tree CLEP) | Deployable soft measurement \(z_w\) when tactile FG absent |
| Pfanne / in-tree `CouplingEKF` | Slip-aware \(C\) process; inflate \(P_C\) on slip — not tip←wrist |

**Design doc:** `docs/TRACK_A_THEORY_ESTIMATOR.md` · skeleton `src/pci/tip_theory_estimator.py`.  
**Forbidden as “progress”:** lowering `seat_force_n`, freeze flags, geom-name toggles, outlier-gate widening, per-ep thresholds, reporting patch `smoke_v5` RATE.
