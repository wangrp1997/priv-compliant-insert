# Privileged failure taxonomy (fail-driven wave)

> **合规特权诊断:** Type-A **5/10**；Oracle-v2 **9/10**（诊断上界，不入可部署主表）；Track-A tip_hat smoke **2/4**；Track-B JSE **2/6** BORDERLINE。  
> **科学问题:** 已知 tip–口平面轨迹下，非刚性抓取如何保贴面 + 轴≤25° + 真力入孔（`SUCCESS_STANDARD` 不放宽）。  
> 诊断命令: `PYTHONPATH=src python scripts/diag_spiral_priv.py <run>/ep*/ep*_summary.json`

Cited runs (2026-09-03):

| Run | RATE | Fail eps |
|-----|------|----------|
| `S2_typeA_ep1_10` | **5/10** | 3,7,8,9,10 |
| `S2_oracle_tip_v2_ep1_10` | **9/10** (diag) | 4 |
| `S2_trackA_est_oracle_smoke` | **2/4** | 3,4 |
| `S2_trackB_joint_smoke` | **2/6** | 3,7,8,10 |

---

## 1. Failure-type table (priv metrics)

| ID | Type | Episodes (priv) | Signature (numbers) | Oracle-v2 component that fixes when tip known |
|----|------|-----------------|---------------------|-----------------------------------------------|
| **F1** | Planar saturate, never mouth | TypeA **3,9**; Joint **3,8,10** | `priv_planar_min` **6–12 mm**, `priv_mouth=False` (TypeA3: 8.4 mm; Joint3: 8.5; Joint10: 12.2). TypeA9: min 7.5 mm / mouth True once then timeout + `priv_tip_spiral_gate_fail` | tip→mouth PD (closes planar gap); mouth press secondary |
| **F2** | Massive grasp slip + tip lag | TypeA **7,10**; Joint **7** (slip↓ but still fail) | slip **600–1032 mm** (TypeA7/10); tip_spiral_err max **79–167 mm**; `priv_mouth=False`; Joint10 slip improved 1032→198 mm still planar 12 mm | `slip_reseat` + SEAT reseat + grasp tighten; then tip→mouth resume |
| **F3** | At mouth, axis / tilt / press | Oracle **ep04**; TrackA **ep04**; TypeA **ep08** | mouth-ish (`priv_planar_min≈4.5 mm` or TypeA8 5.9 mm + mouth True) but `priv_axis_ok=False` or `surface_tilt_or_tray` / `surface_press`; Oracle04: `near_mouth`, no force-hole | `hold_enter_seat` (no stop if axis>25°) + upright hold before press; tip PD alone insufficient on ep04 |
| **F4** | tip_hat bad upright (Track A) | TrackA **ep03** | `priv_mouth=True`, `near_mouth`, but axis fail (`priv_axis_ok=False`); tip_hat drives Oracle law with wrong upright cue (report ≈73°) | same Oracle upright/hold_enter — **needs better tip_hat / axis estimate**, not tip_gt+noise |

---

## 2. Ranked algorithm proposals (dual track only)

| Rank | Method | Track | Attacks | Why not tip_gt+noise | 1–2 wk |
|------|--------|-------|---------|----------------------|--------|
| **1 (primary)** | **HardHQP** — Escande hard hierarchy seat≻upright≻path | **B** | F1, F3 (path must not eat seat/axis) | Wrist + F/T + FK axis on known spiral; no tip pose in law | **High** — skeleton landed; soft QP already in tree |
| 2 | Kim TEC-slim tip state + Oracle-v2 ctrl | **A** | F4, residual F3 | Estimates tip_hat from contact/F/T; discloses estimator | Medium — FG heavy; slim factor ~2 wk (`refs/Tactile-Estimator-Controller`) |
| 3 | Grasp impedance / Pfanne object impedance + reseat | **B** | F2 | Proprio slip + GraspQP; no tip GT | High — GraspQP in tree; Pfanne cite + `coupling_ekf` |
| 4 | Active extrinsic contact-line (no full tip) | **B** | F1 near mouth | Contact Jacobian / line, not tip XY GT | Medium — Kim ICRA 2022; needs contact Jac |
| 5 | PHIG (already primary lit Track-B) | **B** | F1–F3 soft | Known path impedance + grasp co-opt | Wire in parallel (other agent); not this fail-driven pick |
| — | JSE Type-A+B+mouth restack | **B** | tried | — | **Reject primary** — already **2/6 BORDERLINE** |

Cites: Escande et al. IJRR 2014; Kim TEC ICRA 2023 / TEXterity ICRA 2024; GraspQP `refs/graspqp`; Active extrinsic Kim ICRA 2022; Pfanne EKF object impedance (cite + in-tree EKF).

---

## 3. Primary next method

**HardHQP (Track B)** — see `docs/FAIL_DRIVEN_HARD_HQP.md`.

Rationale: F1 dominates deployable fails (soft seat/path tradeoff saturates planar away from mouth). Oracle-v2 shows tip known + seat/press/reseat → 9/10; HardHQP enforces the same **priority** without tip GT by freezing path until seat/upright hard levels clear.
