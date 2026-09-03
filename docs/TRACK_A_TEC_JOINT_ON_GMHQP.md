# Track-A: TEC joint est–ctrl on GMHQP (general)

> Mandate: `docs/GENERAL_ALGO_MANDATE.md` — 失败 ep 只是探针；**禁止**按 ep / keep-set 设计律。  
> Gate: `docs/METHOD_GATE_NO_PATCH.md`（检索优先 → 没有再创新）  
> Base control structure: GMHQP (`docs/TRACK_B_GRASP_MAP_HQP.md`)  
> Wave context: `docs/WAVE7_ON_GMHQP.md`  
> SUCCESS_STANDARD locked; Oracle RATE diagnostic-only.

---

## 1. General estimation problem (no episode indices)

Under non-rigid grasp, known planar path \(p^*(\theta)\):

\[
\dot t_\parallel = C(q)\, \dot w_\parallel,\qquad C(q)\neq I
\]

**State to recover** (for any grasp / slip realization):

\[
x = \big(t_\parallel,\; C,\; c_{\mathrm{ext}}\big)
\]

- \(t_\parallel\): extrinsic tip on tray plane  
- \(C\): wrist→tip planar coupling  
- \(c_{\mathrm{ext}}\): extrinsic contact mode / locus (point / line / patch)

**Math object:** joint estimation–control of extrinsic contact so that tip-referenced tracking

\[
e=\Pi_n(p^*-\hat t)
\]

is driven through grasp-map HQP **only when** \(\hat t\) is information-consistent with contact + grasp; otherwise fall back to in-hand object tip surrogate (same class as GMHQP geom tip).

Control ring: **no** tip GT / tip_gt+noise / peg xpos.

---

## 2. Search → REUSE

| Candidate | Fits general object? | Verdict |
|-----------|----------------------|---------|
| **Kim TEC** ICRA 2023 — simultaneous tactile est–ctrl (factor graph: ContactMotion, PoseDiff, Wrench, energy) | Yes — estimate contact+grasp while planning estimable motions | **REUSE** |
| Active Extrinsic (Kim ICRA 2022) | Wrench / contact geometry cues | **REUSE** (meas) |
| TEXterity full iSAM | Continuous extrinsic pose; heavier | Cite ancestor |
| Free-run EKF tip_hat → tip task (prior tip_obs / tip_fuse) | Broke observability: open-loop \(\hat t\) into law | Reject structure |
| Ep / keep-set switches | Forbidden by GENERAL_ALGO_MANDATE | Forbidden |

**Decision: REUSE** Kim TEC joint est–ctrl (slim factor / EKF port) + **unchanged GMHQP** tip-referenced Escande+\(C^{+}\).  
Wiring name `track_a_tec_joint_gmhqp` only — **not** a new acronym invention.

---

## 3. General equations (TEC slim + GMHQP)

### 3.1 Estimation (past horizon spirit → recursive)

**ContactMotion predict:**

\[
\hat t \leftarrow \hat t + \hat C\,\Delta w_\parallel
\]

**Measurements (information form):**

| Factor | Measurement | Role |
|--------|-------------|------|
| Extrinsic contact | intermittent \(z_c\) (sim: peg↔tray contact pos surrogate) | hard tip |
| PoseDiff / DispDiff | in-hand geom tip \(z_g\) | grasp consistency prior |
| Wrench | wrist F/T lever \(z_w\) | Active Extrinsic soft |

Kalman / NIS consistency (Bar-Shalom):

\[
\nu = z_c - H\hat x,\quad
\mathrm{NIS}=\nu^\top S^{-1}\nu,\quad
S=HPH^\top+R
\]

Accept contact update when \(\mathrm{NIS}\le \chi^2_{0.95,2}=5.991\).

### 3.2 Consistency gate → tip argument of GMHQP (**theory**, not ep lists)

\[
p_{\mathrm{tip}}^{\mathrm{task}} =
\begin{cases}
\hat t & \text{if contact mode estimable (contact meas + NIS OK}\\
& \quad+\ \|\Pi(\hat t-z_g)\|\le 3\sigma(R_g)\ +\ \mathrm{uncert}\le 3\sigma(R_g)\\
& \quad+\ \text{no slip inflate)}\\
z_g & \text{otherwise (hold in-hand tip = GMHQP default)}
\end{cases}
\]

This is the TEC principle: **use estimate in control only when the contact mode is estimable**; otherwise regularize to grasp-consistent tip. Not an episode patch.

### 3.3 Control (unchanged GMHQP)

\[
e=\Pi_n(p^*-p_{\mathrm{tip}}^{\mathrm{task}}),\quad
v_{\mathrm{tip}}^*=K_p e,\quad
\Delta w_\parallel=\hat C^{+}\,v_{\mathrm{tip}}^*
\]

Escande: SEAT ≻ UPRIGHT ≻ TIP_PATH. Fingers: GraspQP pose-hold (existing).

### 3.4 Joint aspect (Active Extrinsic / TEC)

Control Δw enters ContactMotion predict next step → estimation and control share the same \(C,\hat t\) state. Estimability failure → \(p_{\mathrm{tip}}^{\mathrm{task}}=z_g\) preserves the general GMHQP law without free-running \(\hat t\).

---

## 4. Why prior Track-A failed (structure, not ep IDs)

Free-run \(\hat t\) into tip task ignores TEC’s “control only when estimable” rule → observer drift becomes command → tip-referenced HQP tracks a wrong object. Soft PoseDiff prior alone (tip_fuse) still **always** fed \(\hat t\) to the law. Fix is **gated residual into an existing tip-referenced HQP**, not freeze flags or per-ep YAML.

---

## 5. Anti-patch

| Refuse | Why |
|--------|-----|
| `if ep in {…}` / keep-set structure | GENERAL_ALGO_MANDATE |
| Retune `k_tip`, mouth_r, seat force, NIS χ² grid for RATE | METHOD_GATE knob soup |
| tip_gt / peg xpos in control | Track-A lock |
| Claim win from fail-subset-only smoke | primary metric = full RATE |

Allowed fixed theory params: EKF \(Q,R\), χ² 95% 2-DoF, 3σ from \(R_g\), GMHQP \(K_p,C_{\mathrm{ridge}}\) (same as baseline).

---

## 6. Disclosure (residual privilege)

| Signal | Status |
|--------|--------|
| Intermittent peg↔tray `contact.pos` → \(z_c\) | residual sim tactile (TEC GelSlim stand-in) — **named** |
| Soft \(z_g\) = in-hand peg tip geom | same GMHQP object-pose surrogate |
| Wrist F/T | eligible |
| tip / peg xpos GT in control | **no** |

---

## 7. Files

| Item | Path |
|------|------|
| This design | `docs/TRACK_A_TEC_JOINT_ON_GMHQP.md` |
| Joint gate | `src/pci/track_a_tec_joint.py` |
| Observer | `src/pci/tip_theory_estimator.py` (`ExtrinsicTipStateEKF` + NIS) |
| Law | `src/pci/track_b_gmhqp.py` (unchanged equations) |
| Config | `configs/scheme_l3/S2_trackA_tec_joint_gmhqp.yaml` |
| Unit | `scripts/smoke_track_a_tec_joint_gmhqp.py` |
| Eval out | `outputs/scheme_l3/S2_trackA_tec_joint_gmhqp_ep1_10` |
| Ref | `refs/Tactile-Estimator-Controller` |

---

## 8. Evaluation (primary = full distribution)

```bash
MUJOCO_GL=egl PYTHONPATH=src:/home/wangrenpeng/dexjoco:/home/wangrenpeng/dexjoco/dexjoco:/home/wangrenpeng/dexjoco/embodied_grasp_insertion \
  /home/wangrenpeng/miniconda3/envs/dexjoco/bin/python \
  scripts/parallel_force_enter_ep10.py \
  --config configs/scheme_l3/S2_trackA_tec_joint_gmhqp.yaml \
  --out-root outputs/scheme_l3/S2_trackA_tec_joint_gmhqp_ep1_10 \
  --eps 1,2,3,4,5,6,7,8,9,10 --workers 2
```

**Primary metric:** enter RATE on ep1–10 vs GMHQP **6/10** and Oracle ceiling **9/10** (diag).  
Few-ep smoke allowed for **debug only**, not as success claim.  
If RATE ≤ 6/10: **STOP retune**; revise algorithm structure, not thresholds.  
Diag: `scripts/diag_spiral_priv.py` on run summaries (mode-class evidence, not ep patches).

---

## 9. Result (full ep1–10, 2026-09-03)

Out: `outputs/scheme_l3/S2_trackA_tec_joint_gmhqp_ep1_10/` · `summary_enter10.json`  
Diag dump: `outputs/scheme_l3/S2_trackA_tec_joint_gmhqp_ep1_10/diag_fails.txt`

| Metric | Value |
|--------|-------|
| **RATE** | **6/10 = 60%** |
| ok | **1,4,6,8,9,10** |
| fail | **2,3,5,7** |
| vs GMHQP **6/10** | **tie** (identical ok/fail set) |
| vs Type-A **5/10** | +1 (same delta as GMHQP) |
| vs prior tip_obs **4/10** | no free-run regression on keep-set |
| `tec_joint_gated_on_frac` | **0.0 on all eps** — tip_hat never entered tip task |
| `tec_joint_last_reason` | geom_disagree / no_contact / nis_reject (never gated_on) |
| Oracle-v2 | 9/10 diagnostic only — **not** claimed |

Fail `diag_spiral_priv` (mode-class; numbers **≡** `S2_trackB_gmhqp_ep1_10`):

| ep | priv_planar_min | mouth | slip_mm | tip_err_mean | class |
|----|-----------------|-------|---------|--------------|-------|
| 02 | 11.19 | F | 24.5 | 12.4 | F1 / surface_tilt |
| 03 | 11.50 | F | 16.9 | 12.1 | F1 |
| 05 | 8.92 | F | 164.5 | 6.7 | F1(+slip) |
| 07 | 11.65 | F | 186.8 | 10.7 | F1(+slip) |

**Honest diagnosis:** hard NIS+contact+PoseDiff conjunction never passed → pipeline degenerated to pure GMHQP geom tip `z_g` (hence identical RATE/diag). Structure correctly avoided free-run damage; did **not** deliver usable extrinsic tip into control.

**Verdict:** METHOD_GATE / GENERAL_ALGO_MANDATE → **STOP retune** (do not loosen χ² / 3σ for RATE).  
**Paper table:** do **not** update — deployable remains **GMHQP 6/10**.  
Next **algorithm** revise: Active Extrinsic estimability motions + softer information fusion (tip_hat residual weighted by `P`, not hard α∈{0,1} that stays 0) inside the same TEC joint graph — still general, no ep patches.
