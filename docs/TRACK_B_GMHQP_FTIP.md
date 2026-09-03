# Track-B GMHQP-FTIP (REUSE compose, wave-6)

> **合规特权诊断:** GMHQP **6/10**；余败 ep2/3/5/7 全 F1；`bl_tip_err`~3 mm ≪ `priv_planar`~9–12 mm（代理 tip 假闭合）。  
> Phase-0: `docs/WAVE6_TRACK_B_PRIV.md` · 门禁: `docs/METHOD_GATE_NO_PATCH.md`。  
> **科学问题:** 已知 tip→口平面轨迹下，非刚性抓取如何保贴面 + 轴≤25° + 真力入孔。  
> Track-B：无 Oracle `tip_gt`。SUCCESS_STANDARD 不放宽。

---

## 0. Search → reuse | invent

### 0.1 Math object

Escande seat≻upright≻path 下，tip 任务 \(e=\Pi(p_{\mathrm{tip}}-p^*)\!\to\!C^{+}\) 中的 \(p_{\mathrm{tip}}\) 若取 in-hand geom，会与外禀接触 tip 不一致而假闭合；需可部署 \(p_{\mathrm{tip}}=\hat c\)（腕 F/T）。

### 0.2 Candidates

| Candidate | Fit | Status |
|-----------|-----|--------|
| Hybrid GMHQP⊕planar_C | 任务切换 | wave-5 **0/4**（proxy err 不饱和） |
| CLEP alone | FT \(\hat c\)→腕 PD | **0/4**（无 \(C^{+}\)/Escande） |
| tip_obs TEC→GMHQP | Track-A | full **4/10** STOP |
| **Doshi / Kim Active Extrinsic / Contact Occupancy FT tip** | \(\hat c=w+(n\times\tau)/(n\cdot F)\) | **REUSE as tip_obj** |
| FASR/OIGS/PHIG | — | 禁 |

### 0.3 Decision

| | |
|--|--|
| **Verdict** | **REUSE compose** |
| **What** | CLEP/Doshi FT \(\hat c\) → **same GMHQP** Escande+\(C^{+}\) law |
| **Cite** | Doshi F/T contact; Kim Active Extrinsic ICRA 2022; Contact Occupancy spirit; Montana+Escande GMHQP |
| **Not** | invent 发表缩写；CLEP 腕 PD；FASR/OIGS/PHIG；tip_gt+noise |

---

## 1. Equations

\[
r_\perp=\frac{n\times\tau}{n\cdot F+\varepsilon},\quad
\hat c=\Pi_n(w+r_\perp)
\]

EMA \(\hat c\)（同 CLEP）。Tip 任务：

\[
e=\Pi(p^*-\hat c),\quad
\Delta w_\parallel=C^{+}K_t e
\]

Escande 硬层不变：seat ≻ upright ≻ tip_path。  
FT 无效 → geom tip fallback（披露）。

Primary metric: enter-win≥1 on {2,3,5,7} **or** planar&lt;5+mouth on ≥2 eps.

---

## 2. Anti-patch

| Refuse | Why |
|--------|-----|
| FASR / mouth_press soup | METHOD_GATE |
| Hybrid tip_err 再调阈值 | 已证 proxy 假闭合 |
| tip_gt+noise | Track-B |
| Per-ep YAML | METHOD_GATE |
| Loosen SUCCESS_STANDARD | locked |

---

## 3. Implementation

| Piece | Path |
|-------|------|
| Design | this file |
| Controller | `src/pci/track_b_gmhqp_ftip.py` |
| Hook | `baseline_hold_r(..., mode="track_b_gmhqp_ftip")` |
| Config | `configs/scheme_l3/S2_trackB_gmhqp_ftip.yaml` |
| Unit | `scripts/smoke_track_b_gmhqp_ftip.py` |

Disclosure: FT contact tip = extrinsic contact surrogate；geom fallback named。

---

## 4. Smoke gate

Eps **2,3,5,7** · Out `outputs/scheme_l3/S2_trackB_gmhqp_ftip_smoke`.  
PASS → may ep1–10；FAIL → **STOP** 不调参。

### Result (2026-09-03)

**RATE 0/4** · Gate **FAIL**.

| ep | FTIP planar / mouth / tip_err_max | GMHQP same | enter |
|----|-----------------------------------|------------|-------|
| 02 | **4.40** / **T** / 15.7（`force_hole_along_reject`） | 11.2 / F | 0 |
| 03 | 10.8 / F / **62.8** | 11.5 / F | 0 |
| 05 | **14.2** / F / **67.9** | 8.9 / F | 0 |
| 07 | 11.7 / F / **80.6** | 11.7 / F | 0 |

- enter-win = **0**（vs GMHQP 同集 0）  
- planar&lt;5+mouth：仅 **ep02**（需 ≥2）→ gate 未过  
- ep03/05/07：`tip_spiral_err` 爆炸（FT \(\hat c\) 噪声）劣于 GMHQP  

**METHOD_GATE STOP** — 不搜参；不扩 ep1–10。Deployable 仍 **GMHQP 6/10**。