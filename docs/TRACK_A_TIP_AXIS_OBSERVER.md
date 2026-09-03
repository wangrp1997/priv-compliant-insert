# Track-A tip+axis observer (F4 / F3) — TEC/TEXterity style

> Gate: `docs/METHOD_GATE_NO_PATCH.md`  
> RX: `docs/DUAL_TRACK_PRIV_RX.md` § Track A  
> Priv numbers: `docs/TRACK_A_PRIV_DIAG.md`  
> SUCCESS_STANDARD locked; Oracle-v2 **9/10** diagnostic only.  
> Honest Track-A RATE baseline: v0 **2/4** until this beats it.

## 1. Priv gap → scientific problem

| Evidence | Number |
|----------|--------|
| Track-A v0 ep03 | tip near mouth (`priv_planar_min≈4.5`, mouth T) but **axis 73°** → no enter |
| Theory EKF (tip-only) | **0/4**; tip_est_err ~90 mm without contact |
| Oracle-v2 ep03 | tip GT → axis **4.3°** + enter |

**Problem (estimation):** under intermittent peg↔tray contact and non-rigid grasp, recover both extrinsic tip planar state \(\hat t_\parallel\) **and** peg axis \(\hat a_{\mathrm{peg}}\), then feed the **same** Oracle-v2 law (tip PD + seat + upright + mouth press). Tip-only observers leave F3/F4 upright failure intact.

## 2. Why not a patch

| Forbidden | Why |
|-----------|-----|
| freeze / `seat_force_n` retune | Holdover invents no axis; v3/v4/theory tip-only already failed gate |
| Variance grid search | Smoke validates theory structure, not knob search |
| `tip_gt+noise` / peg xpos | Residual GT in control — not Track-A |
| Wrist-only tip EKF again | Already **0/4**; missing orientation state |

## 3. Cites

| Paper | Role |
|-------|------|
| Kim et al. ICRA 2023 TEC; `refs/Tactile-Estimator-Controller` FACTORS.md | **ContactMotion**: control at contact ≠ rigid wrist Δ |
| Bronars et al. ICRA 2024 TEXterity (arXiv:2403.00049) | Continuous **extrinsic pose** (position + orientation) with intermittent tactile |
| Kim & Rodriguez ICRA 2022 Active Extrinsic (arXiv:2110.03555) | Contact / wrench geometry without full tip GT |
| Doshi / in-tree CLEP | Soft tip measurement \(z_w\) from wrist F/T |

## 4. State

Tray tangent \(\{t_1,t_2\}\) from known plane / hole normal \(n\).

\[
x =
\begin{bmatrix}
t_x & t_y & a_1 & a_2 & c_{11} & c_{12} & c_{21} & c_{22}
\end{bmatrix}^\top
\in \mathbb{R}^8
\]

\[
\hat a = \mathrm{normalize}\big(a_n\,n + a_1 t_1 + a_2 t_2\big),\quad
a_n=\sqrt{\max(\varepsilon,1-\|a_\parallel\|^2)}\ge 0
\]

- \((t_x,t_y)\): tip planar extrinsic locus.  
- \((a_1,a_2)\): peg axis in tangent coords (TEXterity orientation, hemisphere toward \(+n\)).  
- \(C\in\mathbb{R}^{2\times2}\): grasp coupling (wrist planar Δ → tip planar Δ).

## 5. Process (ContactMotion + orientation)

Wrist planar step \(\Delta w_\parallel=\Pi_n(w_k-w_{k-1})\); wrist rotation \(\Delta R\):

\[
\begin{aligned}
t_{k|k-1} &= t_{k-1} + C_{k-1}\,\Delta w_\parallel + w_t,\\
a_{k|k-1} &= \Pi_{\mathrm{tan}}\big(\Delta R\,a_{k-1}\big) + w_a,\\
C_{k|k-1} &= C_{k-1} + w_C.
\end{aligned}
\]

**Slip:** inflate \(P_C,P_a\); reset \(C\leftarrow I\); **do not** snap \(t\leftarrow w+o\) or \(a\leftarrow\) peg xpos.

## 6. Measurements

| \(z\) | Model | Privilege |
|-------|-------|-----------|
| \(z_c\) intermittent peg↔tray contact point | \(H_t=[I_2\,|\,0]\), small \(R_c\) | residual: sim `contact.pos` |
| \(z_w\) wrench lever tip | Active Extrinsic / Doshi; large \(R_w\) | eligible |
| \(z_a^{\mathrm{finger}}=\mathrm{normalize}(\hat t - p_{\mathrm{grasp}})\) | finger tip FK centroid → tip | eligible proprio |
| \(z_a^{\mathrm{wrist}}\) | wrist approach axis soft prior | eligible |
| seated contact normal / \(+n\) soft pull | orientation cue when extrinsic seat exists | residual contact frame |

Absence of \(z_c\) is **not** a negative seat (starve pathology).

## 7. Output → Oracle-v2 (same law)

| Oracle input | Source |
|--------------|--------|
| tip planar | \(\hat t\) |
| peg axis / upright | \(\hat a\) (**not** `feat.peg_axis` GT) |
| mouth XY | known path / socket |
| seat / float | wrench + optional \(z_c\) presence |
| slip reseat | residual priv grasp slip (disclosed) |

**Forbidden in control:** `feat.tip_pos`, peg xpos, `tip_gt+noise`, privileged `feat.peg_axis`.

## 8. Files

| Item | Path |
|------|------|
| This design | `docs/TRACK_A_TIP_AXIS_OBSERVER.md` |
| Module | `src/pci/tip_theory_estimator.py` → `ExtrinsicTipAxisEKF` |
| Config | `configs/scheme_l3/S2_trackA_tip_axis.yaml` |
| Unit | `scripts/smoke_track_a_tip_axis.py` |
| Smoke out | `outputs/scheme_l3/S2_trackA_tip_axis_smoke` |

## 9. Eval gate

```bash
MUJOCO_GL=egl PYTHONPATH=src:/home/wangrenpeng/dexjoco:/home/wangrenpeng/dexjoco/dexjoco:/home/wangrenpeng/dexjoco/embodied_grasp_insertion \
  /home/wangrenpeng/miniconda3/envs/dexjoco/bin/python \
  scripts/parallel_force_enter_ep10.py \
  --config configs/scheme_l3/S2_trackA_tip_axis.yaml \
  --out-root outputs/scheme_l3/S2_trackA_tip_axis_smoke \
  --eps 1-4 --workers 2
```

Gate vs v0 **2/4**; ≥3/4 may expand ep1–10; else **STOP**, no variance knob search.

## 10. Smoke result (2026-09-03)

Out: `outputs/scheme_l3/S2_trackA_tip_axis_smoke`

| | Oracle-v2 ep1–4 | Track-A v0 | tip-only theory | **tip+axis EKF** |
|--|-----------------|------------|-----------------|------------------|
| RATE | **3/4** | **2/4** | **0/4** | **2/4** |
| ok eps | 1,2,3 | 1,2 | — | 1,2 |

- ep03: mouth T, planar~4.5 mm, final axis≈**77°**, `tip_est_axis_err_peak≈70°`, last `wrench_meas+finger_axis` — axis_hat 未撑起 upright；与 v0 同败型。
- ep04: surface_press / axis≈43°（与 Oracle 同子集天花板类）。
- Gate: **= v0 2/4，未 ≥3/4 → STOP 不扩**；**不**搜 process/meas 方差。
- Residual privilege disclosed in meta (`tip_est_residual_privilege`).
