# Paper baseline matrix (SUCCESS_STANDARD, ep1–10)

**Metric (locked):** force/ConnTact hole enter + tip seat on tray during spiral + `axis_err ≤ 25°`  
**Source:** `docs/SUCCESS_STANDARD.md` (same protocol as ConnTact faithful 0/10)  
**Do not loosen.** Oracle is privileged diagnostic only — never in the deployable main table.  
**Live mirror:** `outputs/scheme_l3/PAPER_BASELINE_MATRIX_SUCCESS.md`

Workers=3, `--video`, `MUJOCO_GL=egl`. Fresh ablation runs under `outputs/scheme_l3/paper_*` unless marked **cite-only**.

---

## Table 1 — Main deployable comparison (SUCCESS_STANDARD)

| Method | Config | Out root | RATE | Role | Notes |
|--------|--------|----------|------|------|-------|
| ConnTact faithful | `S3_conntact_faithful.yaml` | `S3_conntact_faithful_ep1_10` | **0/10** | published baseline | cite-only; rigid-EE port |
| Franka faithful | `S3_franka_faithful.yaml` | `S3_franka_faithful_ep1_10` | **0/10** | published baseline | cite-only |
| PSFT faithful | `S3_psft_faithful.yaml` | `S3_psft_faithful_ep1_10` | **0/10** | published baseline | cite-only |
| Tip-Hybrid + planar_C (v10) | `S2_force_insert.yaml` | `S2_force_enter_ep1_10v10` | **4/10** | prior ours | cite-only; ok 1,4,5,6 |
| Type-A | `S2_typeA.yaml` | `S2_typeA_ep1_10` | **5/10** | ours (recover stack) | ok 1,2,4,5,6 |
| Track-B GMHQP | `S2_trackB_gmhqp.yaml` | `S2_trackB_gmhqp_ep1_10` | **6/10** | ours (prior best) | ok 1,4,6,8,9,10; see † |
| **Track-B GMHQP-AC** | `S2_trackB_gmhqp_ac.yaml` | `S2_trackB_gmhqp_ac_ep1_10` | **7/10** | **ours (best deployable)** | ok 1,4,6,7,8,9,10; see †‡ |

† **GMHQP / AC disclosure:** in-hand peg-tip geom is an **object-localization surrogate** (same class as OIGS/Pfanne), not tip GT in an Oracle loop and **not** claimed tip-free deployable. Residual privilege named in `docs/PRIVILEGED_SENSING_DISCLOSURE.md` and `docs/TRACK_B_GRASP_MAP_HQP.md`.  
‡ **AC:** online \(\hat C\) + uncertainty-weighted \(C^{+}\) on GMHQP (`docs/TRACK_B_ADAPTIVE_C_GMHQP.md`); same tip-geom disclosure.

### Ablation / internal (same metric; not the “ours” headline)

| ID | Config | Out root | RATE | Label |
|----|--------|----------|------|-------|
| S0 | `S0_baseline_surface_lock.yaml` | `paper_S0_ep1_10` | **0/10** | ablation / internal |
| S1 | `S1_tip_wrist_ff.yaml` | `paper_S1_ep1_10` | **0/10** | ablation / internal |
| S2live | `S2_tip_track_live.yaml` | `paper_S2live_ep1_10` | **0/10** | ablation / internal |
| S6 | `S6_priv_coupling.yaml` | `paper_S6_ep1_10` | **0/10** | ablation / internal |
| S8 planar_C alone | `S8_planar_C.yaml` | `paper_S8_ep1_10` | **0/10** | planar_C only (not Tip-Hybrid stack) |

### Smoke-only (not full ep1–10; do not promote to main RATE)

| Method | Out | RATE | Notes |
|--------|-----|------|-------|
| Track-A tip_hat v0 | `S2_trackA_est_oracle_smoke` | **2/4** | smoke only; not expanded to ep1–10 |

---

## Appendix A — Diagnostic ceiling (NOT deployable)

| Method | Config | Out root | RATE | Label |
|--------|--------|----------|------|-------|
| Oracle tip servo v1 | `S2_oracle_tip.yaml` | `S2_oracle_tip_ep1_10` | **7/10** | privileged tip→mouth (v1) |
| **Oracle-v2** | `S2_oracle_tip_v2.yaml` | `S2_oracle_tip_v2_ep1_10` | **9/10** | privileged ceiling (v2) |

**Rule:** do **not** put Oracle RATE in Table 1 or any deployable comparison until a non-GT Path A/B matches it without tip/hole GT in the loop.

Oracle-v2 success eps: **1, 2, 3, 5, 6, 7, 8, 9, 10** (fail ep04).  
Oracle-v1 success eps: **1, 2, 4, 5, 6, 9, 10**.

---

## Table 2 — Per-ep enter: Type-A / GMHQP / GMHQP-AC

| ep | Type-A | GMHQP | AC | vs GMHQP |
|----|--------|-------|----|----------|
| 01 | ✓ | ✓ | ✓ | keep ok |
| 02 | ✓ | ✗ | ✗ | keep fail |
| 03 | ✗ | ✗ | ✗ | keep fail |
| 04 | ✓ | ✓ | ✓ | keep ok |
| 05 | ✓ | ✗ | ✗ | keep fail |
| 06 | ✓ | ✓ | ✓ | keep ok |
| 07 | ✗ | ✗ | ✓ | **win** |
| 08 | ✗ | ✓ | ✓ | keep ok |
| 09 | ✗ | ✓ | ✓ | keep ok |
| 10 | ✗ | ✓ | ✓ | keep ok |

**Net:** AC **7/10** vs GMHQP **6/10** (**+1**): win **ep07**; no keep-set regression. Residual fails **2,3,5** (F1 planar).

Cite-only Tip-Hybrid (v10) success eps: **1, 4, 5, 6**.

---

## Honesty

- Published priors under SUCCESS_STANDARD: ConnTact / Franka / PSFT = **0/10**.
- Progression (deployable): Tip-Hybrid **4/10** → Type-A **5/10** → GMHQP **6/10** → **GMHQP-AC 7/10**.
- **S8 planar_C alone = 0/10**; headline “ours” for planar_C-only is not Tip-Hybrid / Type-A / GMHQP / AC.
- Oracle-v2 **9/10** is diagnostic only (AC matches Oracle-v1 **7/10** rate, different eps — not the same method).
- Track-A tip_hat **2/4** is smoke-only.
- GMHQP/AC tip-geom surrogate must stay disclosed (†‡).

## Related cite-only (not in primary headline)

| ID | Out | Rate | Label |
|----|-----|------|-------|
| S9 EKF-QP | `S9_ekf_qp_ep1_10v1` | 1/10 | internal / experimental |
