# Track-A theory estimator — extrinsic tip state observer

> Path A of `docs/DUAL_TRACK_TO_ORACLE.md`.  
> Priv numbers: `docs/TRACK_A_PRIV_DIAG.md` (post-v4 continuation).  
> SUCCESS_STANDARD locked; Oracle-v2 **9/10** diagnostic only.  
> Honest Track-A RATE baseline remains v0 **2/4** until this observer beats it.

## 1. Why this is not a patch

| Patch (forbidden) | Why it fails the scientific bar |
|-------------------|----------------------------------|
| Lower `seat_force_n` / widen outlier gate | Ep-agnostic thresholds; does not recover tip when contacts are **absent** (ep03: 11/2323) |
| `freeze_planar` / `freeze_require_seat` | Holdover invents no information; v3/v4 **regressed** vs v0 |
| Geom-name / force-gate toggles | Engineering around one matcher; Oracle ep03 succeeds with tip GT at **0** priv_peg_tray |
| `tip = wrist + const offset` / `wrist_delta` | Rigid-grasp assumption; smoke_v2 **0/4**; false under slip |

**Allowed:** a published contact-state observer (TEC / TEXterity / Active Extrinsic spirit) with explicit process + measurement models and disclosed residual privilege.

## 2. Scientific problem (general)

Given known tray plane / mouth path (problem lock), recover extrinsic tip planar state under non-rigid grasp when peg↔tray contact updates **starve**, while wrist F/T may still look seated.

\[
\hat{t}_\parallel \;\leftarrow\;
\mathrm{Observer}\big(F,\tau,\;w,\;q_{\mathrm{finger}},\;z_c^{\mathrm{intermittent}}\big)
\quad\text{s.t.}\quad
t \neq w + o
\]

Feed \(\hat{t}\) into the **same** Oracle-v2 law (tip PD + seat + mouth press + slip reseat).

## 3. Cites (theory basis)

| Paper / ref | Equation / factor used here |
|-------------|----------------------------|
| Kim et al. ICRA 2023 TEC; `refs/Tactile-Estimator-Controller` FACTORS.md | **ContactMotion**: local motion at estimated contact equals control; not rigid wrist Δ |
| Bronars et al. ICRA 2024 TEXterity | Continuous extrinsic pose estimation with intermittent tactile |
| Kim & Rodriguez ICRA 2022 Active Extrinsic (arXiv:2110.03555) | Contact geometry from interaction without full tip GT |
| Doshi / in-tree CLEP | Soft measurement \(z_w\) from wrist wrench lever arm |
| Pfanne + `pci.coupling_ekf` | Slip-aware planar coupling \(C\); inflate \(P_C\) on slip |

Full GTSAM/iSAM stack is **not** runtime-ported (cite-only). This module is a **TEC-slim EKF** with the same state / measurement roles.

## 4. State

Tray tangent frame \(\{t_1,t_2\}\) from known plane normal \(n\) (hole / spiral axis).

\[
x =
\begin{bmatrix} t_x & t_y & c_{11} & c_{12} & c_{21} & c_{22} \end{bmatrix}^\top
\in \mathbb{R}^6
\]

- \((t_x,t_y)\): tip planar coordinates (extrinsic contact locus on tray).  
- \(C\in\mathbb{R}^{2\times2}\): grasp coupling (wrist planar Δ → tip planar Δ).  
- Tip world: \( \hat{t} = o_n + t_x\,t_1 + t_y\,t_2 \) with along-normal from last seated height or wrist bridge **only for Z bookkeeping**, never as planar truth.

Optional later: contact normal as state (TEXterity continuous pose); v1 keeps \(n\) known-path.

## 5. Process model (non-rigid)

Wrist planar step \(\Delta w_\parallel = \Pi_n(w_k - w_{k-1})\). Predict:

\[
\begin{aligned}
t_{k|k-1} &= t_{k-1} + C_{k-1}\,\Delta w_\parallel + w_t,\\
C_{k|k-1} &= C_{k-1} + w_C,
\end{aligned}
\qquad
Q = \mathrm{diag}(q_t,q_t,q_C,\ldots)
\]

**Kim ContactMotion spirit:** commanded motion is applied **at the contact**, through \(C\), not as \(t \leftarrow t + \Delta w\).

**Slip (bgf / TEC):** if wrench shear high or \(\|C\Delta w\|\) disagrees with latest contact innov, set \(C\leftarrow I\) and inflate \(P_C\) — do **not** snap \(t\leftarrow w+o\).

When no measurement for many steps: predict-only; \(\mathrm{tr}(P_t)\) grows → meta `tip_est_uncert_m` for diagnostics (controller may soft-reseat; still not a freeze patch).

## 6. Measurement models

### 6.1 Intermittent extrinsic contact point \(z_c\) (residual privilege in sim)

Sim stand-in: force-weighted peg↔tray `contact.pos` projected to plane (same residual privilege as Track-A v0; disclose).  
Hardware target: TEC tactile extrinsic factor output.

\[
z_c = H_t\,x + v_c,\quad H_t = [I_2\;|\;0],\quad R_c = r_c I_2
\]

Update only when a contact point exists; **absence is not a negative seat measurement** (that was the starve pathology).

### 6.2 Wrist-wrench contact (Active Extrinsic / Doshi) — always eligible

\[
r_\perp = \frac{n\times\tau}{n\cdot F+\varepsilon},\quad
z_w = \Pi_n(w + r_\perp)
\]

\[
z_w = H_t\,x + v_w,\quad R_w = r_w I_2 \gg R_c
\]

Gate weakly on \(\|F\|\) / \(|n\cdot F|\) only as **measurement validity** (information form), not as a success-standard seat threshold hack.

### 6.3 Finger proprio

Finger-force centroid may initialize \(C\approx I\) near grasp, **never** as tip planar measurement (v0/v3 failure mode).

## 7. Output to Oracle-v2

| Oracle input | Theory-est source |
|--------------|-------------------|
| tip planar XY | \(\hat{t}\) from EKF mean |
| mouth XY | known path / socket (unchanged) |
| seat / float | wrench \(n\cdot F\) + optional \(z_c\) presence (disclose) |
| slip reseat trip | residual priv grasp COM (unchanged disclosure) until shear innov replaces it |

**Forbidden in control:** `feat.tip_pos`, peg xpos, `tip_gt+noise`.

## 8. Files

| Item | Path |
|------|------|
| Priv → problem map | `docs/TRACK_A_PRIV_DIAG.md` § continuation |
| This design | `docs/TRACK_A_THEORY_ESTIMATOR.md` |
| Module | `src/pci/tip_theory_estimator.py` |
| Config | `configs/scheme_l3/S2_trackA_theory_est.yaml` |
| Unit check | `scripts/smoke_track_a_theory_est.py` (filter equations) |
| Wire | `sim_runner` mode `track_a_theory_est` / `theory_est_oracle` |
| Smoke out | `outputs/scheme_l3/S2_trackA_theory_est_smoke` |

## 9. Eval gate / smoke RATE (2026-09-03)

```bash
MUJOCO_GL=egl python scripts/parallel_force_enter_ep10.py \
  --config configs/scheme_l3/S2_trackA_theory_est.yaml \
  --out-root outputs/scheme_l3/S2_trackA_theory_est_smoke \
  --eps 1-4 --workers 2

PYTHONPATH=src python scripts/diag_spiral_priv.py \
  outputs/scheme_l3/S2_trackA_theory_est_smoke/ep*/ep*_summary.json
```

| | Oracle-v2 ep1–4 | Track-A v0 | **Theory EKF** |
|--|-----------------|------------|----------------|
| RATE | **3/4** | **2/4** | **0/4** |
| ok eps | 1,2,3 | 1,2 | — |

- Control path: `tip_baseline_mode=track_a_theory_est`, `tip_est_backend=theory_ekf`, `obs_noise=false`, `privileged_oracle_tip_servo=false`.
- Residual privilege disclosed: intermittent peg↔tray `contact.pos` + wrench eligible + priv grasp slip.
- `tip_est_err_mean_mm`: ep01≈18；ep02≈94 / ep04≈92（`tip_est_seated_frames=0`，末源 `wrench_meas`）；ep03≈38。
- Gate: **< v0 2/4 → STOP**；不扩 ep1–10；**不**把 process/meas 方差当搜索旋钮重调。
- Honest Track-A RATE baseline remains v0 **2/4**.

