# Track-B CLEP: Contact-Line Extrinsic Pivot (no tip GT)

> **合规特权诊断:** Type-A F1 fails 饱和 `priv_planar_min` 6–12 mm；Oracle-v2 同集到口 `priv_planar_min≈4.5 mm` + `priv_mouth`。  
> **科学问题:** 已知螺旋上，非刚性抓取腕≠尖时，如何让 tip 贴面走到口再力入。  
> **Track:** B — 控制环**不用** tip GT / tip_gt+noise。  
> **SUCCESS_STANDARD** 不放宽；Oracle 9/10 仅诊断。

## 1. Why this method (vs failed Track-B)

| Tried | Failure mode | CLEP difference |
|-------|--------------|-----------------|
| PHIG | Wrist→path impedance; tip lag ignored → never mouth | Drive **estimated contact**→path, not wrist→path |
| HardHQP | Hard seat≻upright≻path still wrist residual | Same contact-proxy tip track; soft seat only |
| JSE | Type-A+B+mouth restack | New sensing law (F/T contact line) |

F1 root: `wrist≠tip` (C, slip). When tip is seated, extrinsic contact ≈ tip on plane.  
Kim Active Extrinsic Contact (ICRA 2022) estimates contact line without full tip GT; Doshi-style F/T point-contact gives a sim-deployable substitute (no GelSlim).

## 2. Ranked options this wave (cites)

| Rank | Option | Attack | Cite |
|------|--------|--------|------|
| **1 CLEP (pick)** | F/T contact-line / point → pivot tip along known spiral | **F1** | Kim & Rodriguez ICRA 2022 arXiv:2110.03555; Doshi F/T contact ctrl |
| 2 | Grasp impedance stiffen C≈I (Pfanne + GraspQP) | F2 | Pfanne object impedance; `refs/graspqp` — PHIG already tried soft grasp, **0/6** |
| 3 | TEC-slim joint est–ctrl (Kim ICRA 2023 / TEXterity) | F1/F4 | `refs/Tactile-Estimator-Controller`; borderline Track-A if tip_hat enters PD |

## 3. Equations (eligible sensors only)

Signals: frozen mouth / spiral `p*(θ)`, wrist pose `w`, wrist wrench `(F, τ)`, FK peg axis, proprio slip.  
**Forbidden:** peg tip xpos, hole GT beyond frozen center.

Contact estimate (planar support, seated):

\[
r_\perp = \frac{n \times \tau}{n\cdot F + \varepsilon},\quad
\hat{c} = \Pi_n\!\big(w + r_\perp\big)
\]

Fallback point-contact when \(|n\cdot F|\) small:

\[
r = \frac{F\times\tau}{\|F\|^2+\varepsilon},\quad \hat{c}=\Pi_n(w+r)
\]

EMA: \(\hat{c}^+ = \alpha\hat{c} + (1-\alpha)\hat{c}_{\mathrm{prev}}\).

Wrist command (pivot / track contact toward path):

\[
\Delta w_\parallel = \mathrm{clip}\big(K_c\,\Pi_n(p^*-\hat{c}),\;\Delta_{\max}\big)
\]

\[
\Delta w_\perp = k_n\,(f_{\mathrm{des}}-f_n)\,n
\]

Near mouth (`r_cmd` + F/T): axial press. Low \(f_n\): SEAT only (path scale↓).

## 4. Files

| Item | Path |
|------|------|
| Design | `docs/TRACK_B_CLEP.md` (this) |
| Module | `src/pci/track_b_clep.py` |
| Config | `configs/scheme_l3/S2_trackB_clep.yaml` |
| Unit smoke | `scripts/smoke_track_b_clep.py` |
| Hook | `baseline_hold_r(..., mode="track_b_clep")` + `sim_runner` |

## 5. Fail-subset gate

Smoke eps **3,8,9,10** (worst planar / Type-A F1 joint).  
Gate: enter-win ≥1 vs Type-A on subset **or** clear `priv_planar_min`→mouth (&lt;5 mm + `priv_mouth`) before expand.  
Else **stop**; do not run ep1–10.

## 6. Smoke RATE (2026-09-03)

| Run | eps | RATE | vs Type-A | Notes |
|-----|-----|------|-----------|-------|
| `S2_trackB_clep_smoke` | 3,8,9,10 | **0/4** | Type-A **0/4** | no enter-win; lat_min 11–16 mm worse than Type-A |

Per-ep CLEP: ep03/08 `priv_tip_spiral_gate_fail`; ep09 `surface_press`; ep10 `surface_tilt_or_tray`.  
Gate **FAIL** → **do not expand** ep1–10. Next only if user authorizes estimator fix / different Track-B line.
