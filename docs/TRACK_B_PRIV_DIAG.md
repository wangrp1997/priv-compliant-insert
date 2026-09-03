# Track-B privileged diagnosis (Phase 0)

> **合规特权诊断:** Type-A **5/10**；Oracle-v2 **9/10**（诊断上界）；PHIG/HardHQP/CLEP enter-win=0；JSE **2/6** BORDERLINE。  
> **科学问题:** 已知 tip–口平面轨迹下，非刚性抓取如何保贴面 + 轴≤25° + 真力入孔（`SUCCESS_STANDARD` 不放宽）。  
> **Track-B:** 控制环禁用 tip GT / tip_gt+noise。  
> 命令: `PYTHONPATH=src python scripts/diag_spiral_priv.py <run>/ep*/ep*_summary.json`

Cited runs (2026-09-03):

| Run | RATE | Notes |
|-----|------|-------|
| `S2_typeA_ep1_10` | **5/10** | fails 3,7,8,9,10 |
| `S2_oracle_tip_v2_ep1_10` | **9/10** diag | fail only ep04 |
| `S2_trackB_phig_smoke` | **0/6** | planar worse |
| `S2_failDriven_hardHQP_smoke` | **0/3** | slip↓, no enter-win |
| `S2_trackB_clep_smoke` | **0/4** | planar worse |
| `S2_trackB_joint_smoke` | **2/6** | BORDERLINE |

---

## 1. Per-ep fail taxonomy table (Type-A + Oracle contrast)

| ep | TypeA F | planar_min | mouth | slip_mm | tip_err_max | resid_μ | axis_peak° | Oracle same ep | Oracle does differently |
|----|---------|------------|-------|---------|-------------|---------|------------|----------------|-------------------------|
| **03** | **F1** | **8.39** | F | 18.6 | 16.5 | 0.63 | 12.6 | ok; planar **4.47**; mouth T; slip **11.4** | tip→口 PD closes planar; slip already small — **not F2** |
| **07** | **F2** | **10.19** | F | **672.5** | **79.3** | 0.85 | **90** | ok; planar 4.46; slip **4.6**; tip_err 15.6 | slip_reseat + tip PD; axis held ~24° |
| **08** | **F3** | **5.89** | T | 86.3 | 40.3 | 0.18 | 21.1 | ok; planar **1.76**; slip 37; tip_err 15.6 | mouth press + upright; TypeA already mouth-ish |
| **09** | **F1** | **7.47** | T† | 124.6 | 19.3 | 0.31 | 23.9 | ok; planar 4.44; slip **2.5** | tip PD + reseat; mouth once then TypeA timeout |
| **10** | **F2** | **12.27** | F | **1031.9** | **167.5** | 3.28 | **90** | ok; planar **1.01**; slip **6.0** | reseat + tip PD; residual force collapses |

† TypeA ep09: `priv_mouth=True` once then `spiral_timeout` / `priv_tip_spiral_gate_fail`.

Oracle **ep04** (only Oracle fail): planar 4.47, mouth T, slip 10.5, reason `near_mouth` / `surface_press`, axis_final **42.8°** → F3 upright/press residual (diagnostic ceiling).

---

## 2. Failed Track-B methods — which metric got worse (numbers)

| Method | ep | planar TypeA→M | mouth | slip TypeA→M | tip_err_max TypeA→M | Verdict |
|--------|----|----------------|-------|--------------|---------------------|---------|
| **PHIG** | 3 | 8.4→**5.71** | F→T | 19→65 | 16→**89** | planar slightly↓ but tip_err **爆炸**; no enter |
| PHIG | 7 | 10.2→**15.7** | F | 673→228 | 79→**114** | planar **worse** |
| PHIG | 8 | 5.9→**15.4** | T→F | 86→113 | 40→**131** | planar **worse**; lost mouth |
| PHIG | 9 | 7.5→**11.7** | T→F | 125→138 | 19→**138** | planar **worse** |
| PHIG | 10 | 12.3→**13.5** | F | 1032→186 | 168→114 | slip↓; planar still bad |
| **HardHQP** | 3 | 8.4→**6.41** | F→T | 19→65 | 16→**89** | mouth flag; still timeout |
| HardHQP | 7 | 10.2→**4.50** | F→T | 673→**108** | 79→28 | **best F2 metric**; force_hole but `force_hole_along_reject` |
| HardHQP | 10 | 12.3→**14.7** | F | 1032→**33** | 168→89 | slip **大降**; planar **worse** → F1 residual after slip fix |
| **CLEP** | 3 | 8.4→**11.4** | F | 19→17 | 16→**64** | planar **worse** |
| CLEP | 8 | 5.9→**15.7** | T→F | 86→134 | 40→**104** | planar **worse** |
| CLEP | 9 | 7.5→**12.7** | T→F | 125→12 | 19→44 | slip↓; planar **worse** |
| CLEP | 10 | 12.3→**15.7** | F | 1032→27 | 168→49 | slip↓; planar **worse** |
| **JSE** | 3 | 8.4→8.47 | F | ≈ | ≈ | no change |
| JSE | 9 | 7.5→**4.37** | T | 125→57 | ≈ | **enter-win** (only) |
| JSE | 10 | 12.3→12.2 | F | 1032→**198** | 168→31 | slip↓; planar stuck |

**Summary:** F1 path substitutes (PHIG / CLEP) systematically **inflate** `priv_planar_min` and `tip_spiral_err`. Slip-first (HardHQP / JSE) can cut `grasp_slip_mm` by 5–30× but **does not guarantee** `priv_planar_min`<5 mm (ep10 HardHQP 14.7 mm).

---

## 3. Dominant residual gap

| Candidate | Evidence | Residual after failed methods |
|-----------|----------|-------------------------------|
| **F1 planar never mouth** | TypeA ep3 slip only **18.6 mm** yet planar **8.4**; Oracle → **4.47** | **Dominant.** CLEP/PHIG made planar worse; HardHQP best 6.4 mm still no enter |
| F2 slip | TypeA ep7/10 slip 673/1032 | Partially solved by HardHQP (→108/33) without enter-win; ep10 becomes F1 |
| F3 mouth axis/press | TypeA ep08; Oracle ep04 | Secondary for Track-B now |

**Target metric for next method:** `priv_planar_min` → **&lt;5 mm** with `priv_mouth=True` (and enter if possible), **without tip GT**.

Smoke eps (hardest for this gap): **3, 9, 10**  
- ep3: clean F1 (low slip)  
- ep9: F1 with fleeting mouth  
- ep10: F1 residual after slip can be controlled  

---

## 4. Next method (diagnosis → one pick)

### 4.1 Rejected mid-flight: FASR (engineering patch)

| Item | FASR |
|------|------|
| Nature | Wrist path + \(f_t\) threshold bias + EMA center — **no new estimator/controller equation** |
| Smoke | `S2_trackB_fasr_smoke` eps 3,9,10 → **RATE 0/3**; planar 5.1 / 13.6 / 15.7 mm |
| Status | **Rejected** under user hard correction (anti-patch) |

### 4.2 Primary: **OIGS** — Object Impedance Grasp Stiffening (Track B)

| Item | Choice |
|------|--------|
| Attacks | **F1** mechanism \(C\neq I\) (wrist≠tip) under non-rigid known path |
| Priv metric | **`priv_planar_min`** (primary); secondary `grasp_slip` / `tip_spiral_err` |
| Theory | Pfanne RA-L 2020 object impedance + GraspQP \(\lambda\); wrist path scaled by \(\alpha(\|e_x\|)\) |
| Why not tip GT | Object pose error from in-hand latch (sim geom = disclosed surrogate); wrist uses known \(p^*\) only |
| Anti-patch | See `docs/TRACK_B_OIGS.md` §2 — no per-ep retunes / FSM soup / mouth_press_scale |

Design + code: `docs/TRACK_B_OIGS.md`.
