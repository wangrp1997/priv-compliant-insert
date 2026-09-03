# Track-A wave-6: TEC tip_fuse → GMHQP (PoseDiff geom prior)

> Gate: `docs/METHOD_GATE_NO_PATCH.md`（检索优先 → 没有再创新）  
> Phase-0: `docs/WAVE6_TRACK_A_PRIV.md`  
> Prior fail: tip_obs full **4/10** (won 5; destroyed 4/8/9)  
> Baseline deployable: GMHQP **6/10**. SUCCESS_STANDARD locked.

## 1. Priv gap → math object

Hard `tip_obj←ˆt` lets ˆt free-run (ep04 tip_spiral_max **125** mm, bl_tip_err **91** mm) and destroys GMHQP wins {4,8,9} while residual fails {2,3,7} stay F1.

**Math:** estimate \(\hat t\) consistent with in-hand object tip kinematics **and** intermittent extrinsic contact, then drive unchanged GMHQP \(e=\Pi(\hat t-p^*)\).

## 2. Search → REUSE | INVENT

| Candidate | Fits math? | Verdict |
|-----------|------------|---------|
| Kim TEC **PoseDiff / DispDiff** (`refs/.../FACTORS.md`) | gripper↔object tip consistency | **REUSE** |
| TacGraph in-hand pose + extrinsic contact FG (arXiv:2512.23856) | joint pose+contact priors | **REUSE structure** |
| TEXterity full iSAM | too heavy | cite |
| wave-5 hard replace | failed | reject |
| variance/freeze soups | METHOD_GATE | forbidden |

**Decision: REUSE** — soft geom-tip measurement in TEC-slim EKF (`meas_geom_tip_var ≫ meas_contact_var`). Wiring name `track_a_gmhqp_tip_fuse` only. **Invent: none.**

## 3. Equations (reuse)

\[
\hat t \leftarrow \mathrm{EKF}\big(
  \underbrace{t + C\Delta w}_{\text{ContactMotion}},\;
  \underbrace{z_{\mathrm{geom}}}_{\text{PoseDiff soft}},\;
  \underbrace{z_c}_{\text{extrinsic}},\;
  \underbrace{z_w}_{\text{wrench}}
\big)
\]

\[
e=\Pi(p^*-\hat t),\quad \Delta w_\parallel = C^{+}_{\mathrm{GMHQP}} K_p e
\]

Fixed info weights (not success thresholds): `meas_contact_var=2e-7`, `meas_geom_tip_var=1.5e-5`, `meas_wrench_var=4e-5`.

## 4. Not tuned

`k_tip`, `c_ridge`, `mouth_r`, seat force, freeze flags, EKF variance grid, ep-if.

## 5. Disclosure

| Signal | Status |
|--------|--------|
| Soft \(z_{\mathrm{geom}}\) = in-hand peg tip geom | same GMHQP tactile/object-pose surrogate |
| Intermittent peg↔tray `contact.pos` | residual sim tactile |
| Wrist F/T | eligible |
| tip / peg xpos GT in control | **no** |

## 6. Files

| Item | Path |
|------|------|
| Phase-0 | `docs/WAVE6_TRACK_A_PRIV.md` |
| This design | `docs/TRACK_A_GMHQP_TIP_FUSE.md` |
| Observer | `src/pci/tip_theory_estimator.py` (`geom_tip_prior`) |
| Config | `configs/scheme_l3/S2_trackA_gmhqp_tip_fuse.yaml` |
| Unit | `scripts/smoke_track_a_gmhqp_tip_fuse.py` |

## 7. Smoke gate

Eps **2,3,4,5,7,8,9**.  
Need: enter-win ≥1 on GMHQP fails {2,3,5,7} **and** keep {4,8,9}.  
Else STOP — no retune.

## 8. Smoke result (2026-09-03)

Out: `S2_trackA_gmhqp_tip_fuse_smoke` · **RATE 1/7** (only ep05).

- Enter-win on GMHQP fails: **ep05** (planar 4.16 + mouth) — fail-subset half-pass.
- Keep-set {4,8,9}: **all lost** (ep04 slip 565 / tip_spiral 438 mm; ep08/09 no enter).
- ep07: planar 11.7→7.5 + mouth (priv improve, no enter) — structure partially helps F1, not enough for gate.

**Verdict: keep-set FAIL → METHOD_GATE STOP.** No variance soup. No ep1–10.  
Deployable baseline remains Track-B GMHQP **6/10**. Oracle 9/10 diagnostic only.
