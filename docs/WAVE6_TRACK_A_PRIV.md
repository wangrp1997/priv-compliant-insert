# Wave-6 Track-A — Phase 0 priv eval (mandatory)

> Baseline: GMHQP **6/10** (`S2_trackB_gmhqp_ep1_10`).  
> Prior Track-A tip_obs full **4/10** STOP (won ep05; destroyed 4/8/9).  
> Tool: `scripts/diag_spiral_priv.py`. SUCCESS_STANDARD locked.

## 1. Numbers

### 1.1 GMHQP residual fails {2,3,5,7}

| ep | planar_mm | mouth | slip_mm | tip_spiral_max | bl_tip_err | reason | Class |
|----|-----------|-------|---------|----------------|------------|--------|-------|
| 02 | **11.19** | F | 24.5 | 19.7 | 3.46 | timeout | F1 + Type-A regress |
| 03 | **11.50** | F | 16.9 | 18.2 | 2.71 | timeout | F1 pure |
| 05 | **8.92** | F | 164.5 | 13.3 | 5.92 | timeout | F1/F2 mix + Type-A regress |
| 07 | **11.65** | F | 186.8 | 34.1 | 5.08 | timeout | F2→F1 residual |

Axis OK-ish; bottleneck = planar never mouth (same as `GMHQP_RESIDUAL_PRIV_RX.md`).

### 1.2 GMHQP wins destroyed by tip_obs {4,8,9} (GMHQP run)

| ep | planar_mm | mouth | slip_mm | tip_spiral_max | bl_tip_err | reason |
|----|-----------|-------|---------|----------------|------------|--------|
| 04 | **4.49** | **T** | 19.0 | 15.7 | 3.81 | hole_detected |
| 08 | **4.49** | **T** | 168.7 | 17.6 | 7.24 | hole_detected |
| 09 | **4.41** | **T** | 12.1 | 15.6 | 2.96 | hole_detected |

Geom-proxy tip task already closes planar→mouth on these eps.

### 1.3 tip_obs full on same eps (`S2_trackA_gmhqp_tip_obs_ep1_10`)

| ep | planar_mm | mouth | slip_mm | tip_spiral_max | bl_tip_err | vs GMHQP |
|----|-----------|-------|---------|----------------|------------|----------|
| 02 | **15.69** | F | 9.4 | **62.6** | 5.88 | F1 worse |
| 03 | **12.05** | F | 16.3 | **61.7** | 1.95 | F1 worse |
| **04** | **15.46** | F | **122.8** | **125.5** | **90.8** | **destroyed win** (ˆt drift) |
| **05** | **4.42** | **T** | 14.9 | 13.3 | 66.7 | **enter-win** |
| 07 | **15.67** | F | 33.2 | 15.8 | 8.79 | still F1 |
| **08** | **14.92** | F | 162.3 | 30.3 | 16.5 | **destroyed win** |
| **09** | **10.16** | F | 13.8 | **70.3** | 7.31 | **destroyed win** |

Pattern: hard `tip_obj←ˆt` lets ˆt free-run when extrinsic contact is weak → planar lag / slip / destroy geom wins. ep05 enters despite large `bl_tip_err` (ˆt≠GT) — opportunistic, not observability fixed.

## 2. Math estimation problem (one sentence)

**Recover an observable extrinsic tip \(\hat t\) that stays consistent with in-hand object tip kinematics (TEC PoseDiff / TacGraph) while absorbing intermittent extrinsic contact, so GMHQP tip-task \(e=\Pi(\hat t-p^*)\) does not free-run away from the geom-proxy solution that already wins {4,8,9}.**

## 3. Search → REUSE | INVENT

| Candidate | Source | Fits? | 1–2 wk? | Verdict |
|-----------|--------|-------|---------|---------|
| Kim TEC PoseDiff / DispDiff | `refs/Tactile-Estimator-Controller/FACTORS.md` | Yes — links gripper↔object tip consistency | Yes — soft meas already in EKF API | **REUSE** |
| TacGraph (MERL 2026) in-hand pose + extrinsic contact FG | arXiv:2512.23856 | Yes — joint pose+contact; kinematic constraints | Partial (no full FG); take soft in-hand tip prior | **REUSE structure** |
| Ma relative-motion extrinsic (ICRA 2021) | arXiv:2103.08108 | Object motion ↔ contact locus | Cite | Ancestor |
| TEXterity continuous pose | arXiv:2403.00049 | Full iSAM+GelSlim | No (heavy) | Cite only |
| Always hard-replace tip_obj←ˆt (wave-5) | TRACK_A_GMHQP_TIP_OBS | Wrong when \(P\) large / contact rare | — | **Failed**; caused 4/8/9 destroy |
| Variance / freeze / ep-if soups | — | Forbidden | — | Reject |
| New parallel acronym | — | — | — | Forbidden |

**Decision: REUSE** — add TEC **PoseDiff-style soft geom-tip measurement** into existing TEC-slim EKF; control still GMHQP with \(\hat t\), but \(\hat t\) is information-fused (geom prior ≪ contact when seated). Wiring name `track_a_gmhqp_tip_fuse` (**not** a new method). **Invent:** none.

## 4. What we will not tune

- `k_tip` / `c_ridge` / `mouth_r` / seat force thresholds  
- EKF variance grid search / freeze flags / ep switches  
- tip_gt / tip_gt+noise / peg xpos in control  

## 5. Smoke plan

Eps **2,3,4,5,7,8,9** (GMHQP fails + destroyed wins + tip_obs win).  
Gate: **enter-win ≥1 on {2,3,5,7}** AND **no loss on keep-set {4,8,9}** (must enter).  
Else STOP / fail-subset-only expand rules per METHOD_GATE.

## 6. Smoke result (2026-09-03) — tip_fuse

Out: `outputs/scheme_l3/S2_trackA_gmhqp_tip_fuse_smoke` · **REUSE** PoseDiff soft geom prior.

| ep | enter | planar / mouth / slip / tip_spiral_max | vs GMHQP |
|----|-------|----------------------------------------|----------|
| 02 | 0 | 15.7 / F / 9.4 / 70.6 | still F1 |
| 03 | 0 | 13.0 / F / 16.3 / 26.8 | still F1 |
| **04** | **0** | 15.5 / F / **565** / **438** | **lost keep** (worse than tip_obs) |
| **05** | **1** | **4.16 / T** / 23.7 / 13.3 | enter-win (GMHQP fail) |
| 07 | 0 | **7.51 / T** / 116 / 30.2 | planar↓+mouth (no enter) |
| **08** | **0** | 12.6 / F / 163 / 18.2 | **lost keep** |
| **09** | **0** | 6.09 / T / 12.8 / 68.2 | **lost keep** (mouth but no enter) |

**RATE 1/7.** Enter-win on fails: ep05 only (≥1 OK). Keep-set {4,8,9}: **0/3 keep → FAIL.**  
**METHOD_GATE STOP** — no variance search, no ep1–10. Deployable still **GMHQP 6/10**.  
Honest read: soft geom prior reduced free-run vs tip_obs on some fails (ep07 mouth) but did **not** restore geom-proxy wins; ep04 still ˆt/slip catastrophe.
