# Track-B Grasp-Map Operational Space HQP (GMHQP)

> **合规特权诊断:** Track-B GMHQP **6/10**（`S2_trackB_gmhqp_ep1_10`）；Type-A **5/10**；Oracle-v2 **9/10**（诊断上界）。  
> **科学问题:** 已知 tip→口平面轨迹下，非刚性抓取如何保贴面 + 轴≤25° + 真力入孔。  
> **病 F1:** \(C\neq I\)（腕≠tip）；Type-A ep3 slip 仅 **18.6** mm 仍 `priv_planar_min=8.4` mm mouth F；Oracle → planar **4.47** + mouth（tip PD）。  
> **OIGS 教训:** planar 8.4→5.5 + mouth，但 `tip_spiral_err→77` mm — 任务写在 **腕 TCP**，不是 tip/物体坐标。  
> **Track-B:** 无 Oracle `tip_gt` 模式；物体 tip 来自 in-hand 定位代理（与 Pfanne/OIGS 同披露 geom）。  
> **SUCCESS_STANDARD** 不放宽。门禁: `docs/METHOD_GATE_NO_PATCH.md`。

---

## 1. Problem → theory → priv metric

### 1.1 Scientific gap (F1)

已知螺旋 \(p^*(\theta)\) 在非刚性下：

\[
\dot x_{\mathrm{tip}} = C(q)\,\dot x_{\mathrm{wrist}},\qquad C(q)\neq I
\]

腕跟路径 ≠ tip 跟路径 → **`priv_planar_min`** 饱和（ep3: 8.4 mm）。

OIGS（Pfanne α 门控腕路径）仍解

\[
\Delta w \propto \alpha\,\Pi(p^*-w)
\]

故 tip 可漂到 `tip_spiral_err` 数十 mm。

### 1.2 One-sentence math object

**控制问题:** 在不把逐步 tip GT 当 Oracle 伺服的前提下，把 tip 任务误差 \(e=\Pi(p_{\mathrm{tip}}-p^*)\) 经 grasp map / 耦合 \(C\) 映射到腕+指，并以 Escande HQP 硬优先座面/轴。

### 1.3 Theory cites

1. **Montana** — grasp map \(G\): contact wrenches ↔ object wrench / twist.  
2. **Pfanne et al., RA-L 2020** — object-level impedance; in-hand object pose → \(w_{\mathrm{des}}\).  
3. **GraspQP** (`refs/graspqp`) — \(G\lambda\approx w_{\mathrm{des}}\) under friction cones → fingertip forces.  
4. **Escande, Mansard, Wieber IJRR 2014** — hierarchical QP: hard levels before soft path.

### 1.4 Equations (GMHQP)

**Object / tip task (not wrist TCP):** tip from in-hand object localization (sim: peg tip geom = tactile/object-pose surrogate):

\[
e = \Pi_n\big(p^*(\theta) - p_{\mathrm{tip}}\big)
\]

**Desired tip velocity:** \(v_{\mathrm{tip}}^* = K_p\, e\).

**Grasp / coupling map (Montana spirit, planar):** online \(C\in\mathbb{R}^{2\times2}\) from finite differences

\[
\Delta t_\parallel \approx C\,\Delta w_\parallel
\quad\Rightarrow\quad
\Delta w_\parallel = C^{+}\,v_{\mathrm{tip}}^*
\]

(ridge toward \(I\) when \(C\) ill-conditioned).

**Escande hard hierarchy:**

| Level | Condition | Action |
|-------|-----------|--------|
| 0 SEAT | \(f_n < f_{\mathrm{seat}}\) | freeze planar; axial press |
| 1 UPRIGHT | axis \(>\) soft | tip-pivot proxy; path frozen |
| 2 TIP_PATH | else | \(\Delta w = C^{+} K_p e\) (+ mouth press if \(r\le r_{\mathrm{mouth}}\)) |

**Fingers (Pfanne + GraspQP):** existing `priv_grasp_opt` pose-hold QP armed spiral-wide (same as OIGS) — \(w_{\mathrm{des}}\) from in-hand error; \(\lambda\) via \(G\).

### 1.5 Distinct from OIGS / Oracle / HardHQP

| Method | Path error | Map |
|--------|------------|-----|
| OIGS | \(\Pi(p^*-w)\) × α | wrist TCP |
| HardHQP | \(\Pi(p^*-w)\) | wrist + hard seat/upright |
| Oracle | \(\Pi(p_{\mathrm{mouth}}-p_{\mathrm{tip}})\) | assume \(C=I\) tip PD |
| **GMHQP** | \(\Pi(p^*-p_{\mathrm{tip}})\) | \(C^{+}\) + Escande + GraspQP |

Primary priv metric: **`priv_planar_min`** → &lt;5 mm + `priv_mouth`（或 enter-win≥1）。  
Secondary: `tip_spiral_err` 不得像 OIGS 那样爆炸。

---

## 2. Anti-patch (refuse to tune)

| Refuse | Why |
|--------|-----|
| FASR / mouth_press_scale / \(f_t\) bias | engineering soup |
| PHIG / CLEP wrist-path substitutes | already failed planar |
| JSE restack Type-A FSM | no new equation |
| tip_gt+noise / Oracle mode flag | Track-B lock |
| Per-ep YAML / ep03 if | METHOD_GATE |
| Loosen SUCCESS_STANDARD | locked |

Allowed = equation params \(K_p, C_{\mathrm{ridge}}, f_{\mathrm{seat}}, \ldots\) fixed for smoke.

---

## 3. Implementation map

| Piece | Path |
|-------|------|
| Design | `docs/TRACK_B_GRASP_MAP_HQP.md` (this) |
| Controller | `src/pci/track_b_gmhqp.py` |
| Hook | `baseline_hold_r(..., mode="track_b_gmhqp")` |
| Config | `configs/scheme_l3/S2_trackB_gmhqp.yaml` |
| Unit | `scripts/smoke_track_b_gmhqp.py` |
| Pose-hold QP | `surface_tip_search_pose_hold: true` |

Disclosure: peg tip geom = object localization surrogate (same class as OIGS/Pfanne); **not** claimed deployable tip-free; residual privilege named.

---

## 4. Smoke gate

Eps **3, 9, 10**. Out: `outputs/scheme_l3/S2_trackB_gmhqp_smoke`.  
Gate: enter-win ≥1 **or** `priv_planar_min`<5 mm + `priv_mouth` → else **STOP**, no ep1–10.

### Result (2026-09-03)

| ep | GMHQP planar / mouth / slip / tip_err_max | Type-A | OIGS | enter |
|----|------------------------------------------|--------|------|-------|
| 03 | 11.50 / F / 16.9 / **18.2** | 8.39 / F / 19 | 5.46 / T / 36 / **77** | 0 |
| 09 | **4.41** / T / 12.1 / 15.6 | 7.47 / T / 125 | 11.26 / F / 328 | **1** |
| 10 | **4.31** / T / **11.4** / 16.1 | 12.27 / F / 1032 | 9.75 / F / 15 | **1** |

**RATE 2/3.** Gate **PASS** (enter-win=2; ep09/10 also planar&lt;5 mm + mouth).  
ep03 still F1 planar lag (worse than Type-A) — tip_err stays ~18 mm (not OIGS 77 mm).

---

## 5. Full ep1–10 (2026-09-03, authorized)

Out: `outputs/scheme_l3/S2_trackB_gmhqp_ep1_10` · config `S2_trackB_gmhqp.yaml` · `diag_spiral_priv.py` on all eps.

| | RATE | ok | fail |
|--|------|----|------|
| **GMHQP** | **6/10** | 1,4,6,8,9,10 | 2,3,5,7 |
| Type-A | 5/10 | 1,2,4,5,6 | 3,7,8,9,10 |
| Oracle-v2 | 9/10 | diagnostic only | — |

vs Type-A: **win** ep08/09/10；**lose** ep02/05；**keep fail** ep03/07；**keep ok** ep01/04/06. Net **+1**.

| ep | enter | priv_planar_min / mouth / slip / tip_err_max | reason |
|----|-------|-----------------------------------------------|--------|
| 01 | 1 | 4.44 / T / 68.6 / 15.2 | hole_detected |
| 02 | 0 | 11.19 / F / 24.5 / 19.7 | surface_tilt_or_tray |
| 03 | 0 | 11.50 / F / 16.9 / 18.2 | priv_tip_spiral_gate_fail |
| 04 | 1 | 4.49 / T / 19.0 / 15.7 | hole_detected |
| 05 | 0 | 8.92 / F / 164.5 / 13.3 | priv_tip_spiral_gate_fail |
| 06 | 1 | 3.03 / T / 0.1 / 4.5 | hole_detected |
| 07 | 0 | 11.65 / F / 186.8 / 34.1 | priv_tip_spiral_gate_fail |
| 08 | 1 | 4.49 / T / 168.7 / 17.6 | hole_detected |
| 09 | 1 | 4.41 / T / 12.1 / 15.6 | hole_detected |
| 10 | 1 | 4.31 / T / 11.4 / 16.1 | hole_detected |

Fails still F1 planar 8.9–11.7 mm (no mouth). tip_err_max stays ≤34 mm (not OIGS 77).  
**METHOD_GATE:** report + STOP — **no retune**. SUCCESS_STANDARD locked. Oracle 9/10 stays diagnostic-only.
