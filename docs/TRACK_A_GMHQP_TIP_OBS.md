# Track-A: tip observer → GMHQP residual (wave-5)

> Gate: `docs/METHOD_GATE_NO_PATCH.md`（**检索优先 → 没有再创新**）  
> RX: `docs/GMHQP_RESIDUAL_PRIV_RX.md` § Track A  
> Baseline: GMHQP **6/10** fails ep **2,3,5,7** (planar 8.9–11.7 mm, mouth F).  
> Oracle tip GT enters all four. Type-A wins ep2/5 that GMHQP lost.  
> SUCCESS_STANDARD locked; Oracle RATE stays diagnostic-only.

## 1. Priv gap → scientific problem

| Evidence | Number |
|----------|--------|
| GMHQP ep2/3/5/7 | `priv_planar_min` 8.9–11.7 mm, mouth F, spiral_timeout |
| Same eps Oracle | planar~4.5, mouth T, enter |
| Same eps Type-A | enter on **2,5**; GMHQP regresses those wins |
| Diag | `bl_tip_err` small (2–6 mm) while `tip_spiral_err` 12–34 mm |

**Math object:** under non-rigid grasp, recover extrinsic tip \(\hat t\) (not wrist TCP, not geom proxy) and drive tip-task error \(e=\Pi(\hat t-p^*)\) through existing grasp-map \(C^{+}\) / Escande HQP — without tip GT.

## 2. Search → reuse | invent

> METHOD_GATE §3–4: WebSearch + `refs/` first; invent only if no adequate published method.

| Candidate | Source | Fits math object? | Portable 1–2 wk? | Verdict |
|-----------|--------|-------------------|------------------|---------|
| **Kim TEC** (ICRA 2023) ContactMotion / extrinsic contact locus | `refs/Tactile-Estimator-Controller` (FACTORS.md `ContactMotion`); arXiv:2303.03385 | Yes — estimates extrinsic contact state from tactile+proprio | Yes — **already ported** as TEC-slim EKF `ExtrinsicTipStateEKF` | **REUSE** |
| **TEXterity** (ICRA 2024) continuous extrinsic pose | Bronars et al. arXiv:2403.00049; cite-only in REF_REUSE_MAP | Yes (pose+contact); heavier iSAM+GelSlim | Partial (no GelSlim; full FG >1–2 wk) | Cite ancestor; use tip-only subset already in TEC-slim |
| **Active Extrinsic** (ICRA 2022) | Kim & Rodriguez arXiv:2110.03555 | Wrench/contact geometry cues | Yes — wrench meas already in TEC-slim | **REUSE** (measurement) |
| **GMHQP** tip task \(e=\Pi(p_{\mathrm{tip}}-p^*)\!\to\!C^{+}\) | `docs/TRACK_B_GRASP_MAP_HQP.md`; Montana/Pfanne/Escande | Control half yes; fails when \(p_{\mathrm{tip}}\)=geom proxy | Already in tree | **REUSE** law; replace tip input only |
| CLEP / PHIG / FASR / OIGS / tip+axis-as-Oracle | prior Track-A/B | Wrong object (wrist path, Oracle PD, or failed gate) | — | Reject for this residual |
| New parallel acronym | — | — | — | **Forbidden** (search hit TEC+GMHQP) |

**Decision: REUSE** — wire existing **Kim TEC-slim tip observer** \(\hat t\) into existing **GMHQP** tip task.  
Config/mode label `track_a_gmhqp_tip_obs` = wiring name only (**not** a new method).  
**Invent:** none this wave.

## 3. Why not a patch

| Forbidden | Why |
|-----------|-----|
| Retune `k_tip` / `c_ridge` / `mouth_r` / `f_seat` | knob search on failed residual |
| tip_gt / tip_gt+noise / peg xpos in control | residual GT — not Track-A |
| Force Oracle-v2 mouth PD on these eps | changes law; RX asks GMHQP law |
| Ep-specific if ep02… | METHOD_GATE |
| Variance soup on EKF process/meas | smoke validates structure once |
| Rebrand as new acronym | search already named TEC + GMHQP |

## 4. Reused equations (cite only)

**Observer (TEC ContactMotion → TEC-slim EKF, already in tree):**  
\(t\leftarrow t+C\Delta w_\parallel\); intermittent \(z_c\) (peg↔tray contact surrogate); soft wrench tip; slip inflates \(P_C\).  
Detail: `docs/TRACK_A_THEORY_ESTIMATOR.md` / `tip_theory_estimator.py`.

**Control (unchanged GMHQP):** tip input = \(\hat t\) not geom \(p_{\mathrm{tip}}\):

\[
e=\Pi_n(p^*(\theta)-\hat t),\quad
v_{\mathrm{tip}}^*=K_p e,\quad
\Delta w_\parallel=C_{\mathrm{GMHQP}}^{+}v_{\mathrm{tip}}^*
\]

Escande seat ≻ upright ≻ tip path. Online \(C_{\mathrm{GMHQP}}\) from \(\Delta\hat t\approx C\Delta w\).  
Detail: `docs/TRACK_B_GRASP_MAP_HQP.md` §1.4.

**Not in control:** `feat.tip_pos`, peg xpos, tip_gt+noise, Oracle mouth PD.

## 5. Disclosure (residual privilege)

| Signal | Status |
|--------|--------|
| Intermittent peg↔tray `contact.pos` → \(z_c\) | residual sim tactile surrogate (TEC GelSlim stand-in) |
| Wrist F/T wrench tip | eligible (Active Extrinsic) |
| Priv grasp slip (pose-hold QP) | residual, disclosed |
| Geom tip / peg xpos | **audit / tip_est_err only** — not control |

## 6. Files

| Item | Path |
|------|------|
| This design | `docs/TRACK_A_GMHQP_TIP_OBS.md` |
| Observer (reuse) | `src/pci/tip_theory_estimator.py` (`ExtrinsicTipStateEKF`) |
| Law (reuse) | `src/pci/track_b_gmhqp.py` |
| Wire | `src/pci/sim_runner.py` — tip_hat → GMHQP `tip_obj` |
| Config | `configs/scheme_l3/S2_trackA_gmhqp_tip_obs.yaml` |
| Unit | `scripts/smoke_track_a_gmhqp_tip_obs.py` |
| Smoke out | `outputs/scheme_l3/S2_trackA_gmhqp_tip_obs_smoke` |
| Full out | `outputs/scheme_l3/S2_trackA_gmhqp_tip_obs_ep1_10` |
| Ref | `refs/Tactile-Estimator-Controller` |

## 7. Eval gate

```bash
MUJOCO_GL=egl PYTHONPATH=src:/home/wangrenpeng/dexjoco:/home/wangrenpeng/dexjoco/dexjoco:/home/wangrenpeng/dexjoco/embodied_grasp_insertion \
  /home/wangrenpeng/miniconda3/envs/dexjoco/bin/python \
  scripts/parallel_force_enter_ep10.py \
  --config configs/scheme_l3/S2_trackA_gmhqp_tip_obs.yaml \
  --out-root outputs/scheme_l3/S2_trackA_gmhqp_tip_obs_smoke \
  --eps 2,3,5,7 --workers 2
```

Gate vs GMHQP on {2,3,5,7}: **enter-win ≥1** OR **`priv_planar_min`<5 mm + mouth**.  
Else **STOP** — no knob search, no ep1–10 expand.

## 8. Smoke result (2026-09-03)

Out: `outputs/scheme_l3/S2_trackA_gmhqp_tip_obs_smoke` · **REUSE** TEC-slim → GMHQP (no new acronym).

| ep | enter | priv_planar_min / mouth / slip / tip_err_max | vs GMHQP fail |
|----|-------|-----------------------------------------------|---------------|
| 02 | 0 | 15.7 / F / 9.4 / 62.6 | still fail (planar worse) |
| 03 | 0 | 12.0 / F / 16.3 / 61.7 | still fail |
| **05** | **1** | **4.42 / T** / 14.9 / 13.3 | **enter-win** (GMHQP planar 8.9 F) |
| 07 | 0 | 15.7 / F / 33.2 / 15.8 | still fail |

**RATE 1/4.** Gate **PASS** (enter-win ep05; also planar&lt;5+mouth).  
Meta: `tip_baseline_mode=track_a_gmhqp_tip_obs`, `tip_oracle_enable=false`, residual privilege includes `gmhqp_tip_task_uses_tip_hat`.  
`tip_est_err` still large on fails — estimate quality open; **METHOD_GATE: no variance knob search**.

## 9. Full ep1–10 (2026-09-03)

Out: `outputs/scheme_l3/S2_trackA_gmhqp_tip_obs_ep1_10` · diag: `diag_priv.txt`.

```bash
MUJOCO_GL=egl PYTHONPATH=src:/home/wangrenpeng/dexjoco:/home/wangrenpeng/dexjoco/dexjoco:/home/wangrenpeng/dexjoco/embodied_grasp_insertion \
  /home/wangrenpeng/miniconda3/envs/dexjoco/bin/python \
  scripts/parallel_force_enter_ep10.py \
  --config configs/scheme_l3/S2_trackA_gmhqp_tip_obs.yaml \
  --out-root outputs/scheme_l3/S2_trackA_gmhqp_tip_obs_ep1_10 \
  --eps 1-10 --workers 3 --video
```

| Metric | Track-A tip_obs | Track-B GMHQP | Type-A |
|--------|-----------------|---------------|--------|
| RATE | **4/10** | **6/10** | **5/10** |
| ok eps | 1,5,6,10 | 1,4,6,8,9,10 | 1,2,4,5,6 |
| fail eps | 2,3,4,7,8,9 | 2,3,5,7 | 3,7,8,9,10 |

vs GMHQP: **+ep05 enter**, but **destroys wins 4,8,9** (kept 1,6,10). Net **−2**.  
Fail-subset remount: ep05 still enter; ep02/03/07 still F1 (planar ~12–16 mm, mouth F).  
Destroys: ep04/08 large slip (123/162 mm) + tip_err; ep09 planar 10.2 mouth F.

**Verdict: RATE ≤6/10 and no clear net gain → METHOD_GATE STOP — no retune / no EKF variance search.**  
Deployable baseline remains **Track-B GMHQP 6/10**. Paper table: **no update** (tip_obs not better).
