# Track-B OIGS — Object Impedance Grasp Stiffening (principled)

> **合规特权诊断:** 主导残差 **F1** — 非刚性下腕≠tip（\(C\neq I\)），Type-A ep3 `priv_planar_min=8.4` mm 且 slip 仅 18.6 mm；Oracle 同集 4.47 mm。  
> 见 `docs/TRACK_B_PRIV_DIAG.md`。  
> **Track-B:** 无 tip GT / tip_gt+noise。  
> **SUCCESS_STANDARD** 不放宽；Oracle 9/10 仅诊断。

---

## 0. Honest: FASR was a patch (rejected as primary)

| Check | FASR |
|-------|------|
| New estimator / controller equation? | **No** — wrist path + \(f_t\) threshold bias + EMA center shift |
| Paper-derived law? | Stretch cite only (Tang spiral); no object impedance / grasp QP |
| Smoke (record) | `S2_trackB_fasr_smoke` eps 3,9,10 → **RATE 0/3**; planar 5.1 / 13.6 / 15.7 mm; tip_err max 89–121 mm |
| Status | **Rejected as engineering patch** under user hard correction |

---

## 1. Problem → theory → priv metric

### 1.1 Scientific gap (F1)

Known spiral \(p^*(\theta)\) is tracked at the **wrist**. Under non-rigid grasp:

\[
\dot x_{\mathrm{tip}} = C(q)\,\dot x_{\mathrm{wrist}},\qquad C(q)\neq I
\]

so tip never reaches mouth → saturates **`priv_planar_min`** (and often `tip_spiral_err`), even when `grasp_slip` is small (Type-A ep03).

Oracle closes the gap by servoing **tip** directly. Track-B must instead make \(C\approx I\) **without** tip GT.

### 1.2 Theory cites

1. **Pfanne et al., RA-L 2020** — *Object-Level Impedance Control for Dexterous In-Hand Manipulation* ([doi:10.1109/LRA.2020.2974702](https://doi.org/10.1109/LRA.2020.2974702)):  
   object wrench from pose error; QP internal forces under friction cones; maintain desired grasp → prevent slip / keep object pose in hand.

2. **GraspQP** (`refs/graspqp`, leggedrobotics): grasp map \(G\), cone-bounded \(\lambda\) → fingertip forces.

In-tree: `src/pci/compliant/priv_grasp_opt.py` already implements Pfanne-style \(w_{\mathrm{des}}\) + GraspQP \(\lambda\) (sim geom disclosed as tactile/object-localization surrogate).

### 1.3 Equations (OIGS)

**Object impedance (Pfanne):** latch freeze defines \(x_0\) (peg-in-hand pose). Error \(e_x = x - x_0\):

\[
w_{\mathrm{des}} = -K_x e_x - D_x \dot e_x
\]

**Grasp force QP (GraspQP spirit):**

\[
\min_{\lambda\ge 0}\; \|G\lambda - w_{\mathrm{des}}\|^2 + \lambda_{\mathrm{reg}}\|\lambda-\lambda_0\|^2
\quad\mathrm{s.t.}\quad f_{\min}\le \|f_i(\lambda)\|\le f_{\max}
\]

Finger admittance: \(\Delta q \leftarrow\) map \(f_{\mathrm{des}}(\lambda) - f_{\mathrm{meas}}\).

**Wrist known-path (only when grasp makes \(C\approx I\) valid):**

\[
\alpha = \mathrm{clip}\!\Big(1 - \frac{\|e_x\|}{e_{\mathrm{rigid}}},\,0,\,1\Big)
\]

\[
\Delta w = \alpha\, K_p\,\Pi_n\big(p^*(\theta)-w\big)
+ (1-\alpha)\,0
\quad+\quad \text{seat axial if } f_n < f_{\mathrm{seat}}
\]

When \(\|e_x\|\) large → freeze planar path, let grasp QP reseat (theory: do not command tip motion while \(C\) invalid).  
When \(\|e_x\|\) small → wrist spiral ≈ tip spiral → attack **`priv_planar_min`**.

Secondary: sustained grasp → lower **`grasp_slip_mm`** / **`tip_spiral_err`** (F2 coupling).

### 1.4 What this is *not*

Not PHIG soft wrist impedance with occasional grasp_scale bump.  
Not CLEP noisy contact XY.  
Not HardHQP seat≻path without continuous object impedance.  
Not FASR \(f_t\) threshold recenter.

---

## 2. Anti-patch (refuse to tune)

| Refuse | Why |
|--------|-----|
| Per-ep gain retunes / ep-specific YAML | Violates general scientific gap |
| Stack Type-A LOCAL_RECOVER + STAR sat frames | Restack of failed FSM soup |
| `mouth_press_scale`, `f_edge_n`, EMA center patches | FASR-class engineering |
| tip_gt / tip_gt+noise / peg xpos in law | Track-B lock |
| Loosen SUCCESS_STANDARD axis/along/float | Locked |
| Expand ep1–10 without gate | enter-win≥1 or planar&lt;5 mm + mouth |

Allowed knobs = **equation parameters** from Pfanne/GraspQP (\(K_x, D_x, f_{\mathrm{squeeze}}, \lambda_{\mathrm{reg}}, e_{\mathrm{rigid}}\)) fixed for the smoke; not per-ep.

---

## 3. Implementation map

| Piece | Path |
|-------|------|
| Design | `docs/TRACK_B_OIGS.md` (this) |
| Wrist + \(\alpha(e_x)\) | `src/pci/track_b_oigs.py` |
| Object impedance QP | `src/pci/compliant/priv_grasp_opt.py` (armed spiral-wide) |
| Config | `configs/scheme_l3/S2_trackB_oigs.yaml` |
| Unit smoke | `scripts/smoke_track_b_oigs.py` |
| Hook | `baseline_hold_r(..., mode="track_b_oigs")` + `surface_tip_search_pose_hold` |

Disclosure: sim uses privileged peg/tray geom inside GraspQP as **surrogate** for Pfanne in-hand localization / tactile (same as existing `priv_grasp_opt`); **not** tip XY in wrist PD.

---

## 4. Smoke

Eps **3, 9, 10** (F1-hardest planar from priv diag).  
Out: `outputs/scheme_l3/S2_trackB_oigs_smoke`.  
Gate: enter-win ≥1 **or** `priv_planar_min`<5 mm + `priv_mouth` vs Type-A same eps → else STOP, no ep1–10.

### Result (2026-09-03)

| ep | OIGS planar / mouth / slip | Type-A | enter |
|----|----------------------------|--------|-------|
| 03 | **5.46** / T / 36 | 8.39 / F / 19 | 0 |
| 09 | 11.26 / F / 328 | 7.47 / T / 125 | 0 |
| 10 | **9.75** / F / **15** | 12.27 / F / 1032 | 0 |

**RATE 0/3.** ep03 planar↓ + mouth; ep10 slip 1032→15 mm; **no** planar&lt;5 mm + mouth sustained enter.  
Gate **FAIL** → **do not expand** ep1–10.
