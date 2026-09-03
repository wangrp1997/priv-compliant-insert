# Track-B Escande GMHQP ⊕ planar_C (reuse, not invent)

> **合规特权诊断:** GMHQP **6/10**；余败 ep2/3/5/7 全为 F1（planar 8.9–11.7 mm，mouth F）；ep2/5 相对 Type-A **丢分**；Oracle 同集全进。  
> RX: `docs/GMHQP_RESIDUAL_PRIV_RX.md` · 门禁: `docs/METHOD_GATE_NO_PATCH.md`（**检索优先 → 没有再创新**）。  
> **科学问题:** 已知 tip→口平面轨迹下，非刚性抓取如何保贴面 + 轴≤25° + 真力入孔。  
> Track-B：无 Oracle `tip_gt` 模式。SUCCESS_STANDARD 不放宽。  
> Code alias: `track_b_hybrid_gmhqp`（RX 称呼 Hybrid；**不是**新发表缩写）。

---

## 0. Search → reuse | invent（编码前必填）

### 0.1 Math object (from priv gap)

**控制问题:** 在 Escande 硬优先 seat ≻ upright ≻ path 下，Level-2 tip-task  
\(e=\Pi(p_{\mathrm{tip}}-p^*)\!\to\!C^{+}\)  
因 tip 代理 / \(C\) 与真 tip 路径不一致而**饱和**（planar 停在 9–12 mm）时，如何选一个**仍沿已知螺旋、无 tip GT** 的可容许路径任务，以挽回 Type-A 已用 planar_C 拿下的进孔。

### 0.2 WebSearch + `refs/` + prior docs

| Candidate | Source | Fits math object? | Portable ≤2 wk? | Tried / status |
|-----------|--------|-------------------|-----------------|----------------|
| **Escande HQP** | IJRR 2014; `tip_tracking_qp` / HardHQP / GMHQP | 饱和约束 → 降级任务 | **Yes（已在树）** | Reuse hierarchy |
| **S8 planar_C** | `PlanarCouplingEstimator`; Type-A baseline | 已知路径 + 窗口 \(C^{-1}e_{\mathrm{tip}}\)；ep2/5 enter | **Yes（已在树）** | **Reuse as fallback task** |
| **CouplingEKF / ContactMotion** | `coupling_ekf.py`; bgf + Pfanne | 在线 \(C\)，非冻结 geom tip 映射 | Yes | Reuse C update |
| Pfanne object impedance | RA-L 2020; OIGS | 物体阻抗腕路径 | Yes | OIGS smoke **0/3** — 不作主任务重跑 |
| Montana grasp map | cite; GMHQP | tip↔腕 \(C\) | Yes | Keep as **healthy** tip-task |
| Kim TEC FG | `refs/Tactile-Estimator-Controller` | tip 估计（Track A） | Full FG >2 wk | Track-A 线；非本 Track-B 控制 |
| Kim Active Extrinsic | CLEP | 接触线无 tip GT | Tried | CLEP **0/4** — 不重堆 |
| FASR / Tip-STAR+JSE restack | — | Forbidden | — | METHOD_GATE 禁 |

Web 补充：Escande 对 *saturated* 约束用 active-set / 层级可行域；未见独立「tip-proxy saturate → planar_C」新论文需移植。Pfanne 是物体阻抗 + 摩擦锥 QP（已用 GraspQP），不是本缺口的新控制律。

### 0.3 Decision

| | |
|--|--|
| **Verdict** | **REUSE / compose** — 不是 invent 平行缩写 |
| **What** | Escande 层级 = seat ≻ upright ≻ {**GMHQP tip-task** \| **S8 planar_C**} |
| **When switch** | tip 残差平台（通用 F1 签名），非 per-ep |
| **Cite** | Escande IJRR 2014；Montana；Pfanne/ContactMotion；S8/Type-A planar_C |
| **Not** | 新造「HybridGMHQP」当发表方法；FASR/OIGS 再叠；tip_gt+noise；阈值汤 |

Novelty（若有）仅在于：**把已验证的两套律放进 Escande Level-2 任务切换**；方程与估计器全部来自树上已有模块。

---

## 1. Problem → theory → priv metric

### 1.1 Residual gap (F1 under GMHQP)

| ep | GMHQP planar / mouth | Type-A | Oracle |
|----|----------------------|--------|--------|
| 02 | **11.2** / F（丢 Type-A 进） | enter 4.4 | enter |
| 03 | **11.5** / F | fail 8.4 | enter |
| 05 | **8.9** / F（丢 Type-A 进） | enter 4.5 | enter |
| 07 | **11.7** / F（slip↓ 仍卡平面） | fail 10.2 | enter |

### 1.2 Composed equations (reuse)

**Escande hard levels:**

| Level | Condition | Action |
|-------|-----------|--------|
| 0 SEAT | \(f_n < f_{\mathrm{seat}}\) | freeze planar; axial press |
| 1 UPRIGHT | axis \(>\) soft | tip-pivot; path frozen |
| 2a tip_task | healthy | GMHQP: \(\Delta w = C_{\mathrm{fd}}^{+} K_t \Pi(p^*-p_{\mathrm{tip}})\) |
| 2b planar_c | tip residual plateau | S8: \(\Delta w = C_{\mathrm{cm}}^{-1} K_p \Pi(p^*-p_{\mathrm{tip}})\) |

**Saturation (general F1):** \(\|e_{\mathrm{tip}}\| > \tau\) and progress over window \(W\) &lt; \(\varepsilon\) for \(N\) frames → Level 2b.  
**Hysteresis:** return to 2a when \(\|e_{\mathrm{tip}}\| < \tau_{\mathrm{rec}}\).

\(C_{\mathrm{cm}}\) = 窗口 LS ContactMotion（同 `PlanarCouplingEstimator` / Type-A）。

### 1.3 Distinct from forbidden restacks

| Method | Relation |
|--------|----------|
| GMHQP alone | 仅 2a — 本组合保留为健康支路 |
| Type-A | planar_C + Tip-STAR；本处 **只复用 planar_C 律**，不叠 Tip-STAR FSM |
| OIGS / FASR / CLEP | 已 FAIL 或不相关 — 不重堆 |
| HardHQP | 腕路径硬优先；本处健康时仍 tip-task |

Primary metric: enter-win≥1 on {2,3,5,7} **or** `priv_planar_min`<5 mm + mouth.

---

## 2. Anti-patch (refuse to tune)

| Refuse | Why |
|--------|-----|
| FASR / mouth_press / \(f_t\) soup | METHOD_GATE |
| Tip-STAR / JSE / OIGS restack as “new” | 无新方程；禁 |
| tip_gt+noise | Track-B lock |
| Per-ep YAML | METHOD_GATE |
| Loosen SUCCESS_STANDARD | locked |
| Claiming invent when composing | honest REUSE |

Allowed = 固定复用模块默认方程参数（同 GMHQP + S8 量级）。

---

## 3. Implementation map

| Piece | Path |
|-------|------|
| Design | `docs/TRACK_B_HYBRID_GMHQP.md` (this) |
| Controller | `src/pci/track_b_hybrid_gmhqp.py`（组合 GMHQP + planar_C） |
| Hook | `baseline_hold_r(..., mode="track_b_hybrid_gmhqp")` |
| Config | `configs/scheme_l3/S2_trackB_hybrid_gmhqp.yaml` |
| Unit | `scripts/smoke_track_b_hybrid_gmhqp.py` |
| Pose-hold QP | `surface_tip_search_pose_hold: true` |

Disclosure: peg tip geom = object localization surrogate（同 GMHQP）；残差特权已命名。

---

## 4. Smoke gate

Eps **2, 3, 5, 7**. Out: `outputs/scheme_l3/S2_trackB_hybrid_gmhqp_smoke`.  
Gate: enter-win ≥1 **or** `priv_planar_min`<5 mm + `priv_mouth` → else **STOP**.

### Result (2026-09-03)

Out: `outputs/scheme_l3/S2_trackB_hybrid_gmhqp_smoke` · **RATE 0/4**.

| ep | Hybrid planar / mouth / slip / tip_err_max | GMHQP same | Type-A | enter |
|----|---------------------------------------------|------------|--------|-------|
| 02 | 11.2 / F / 24.5 / 19.7 | 11.2 / F | **enter** | 0 |
| 03 | 11.5 / F / 16.9 / 18.2 | 11.5 / F | fail 8.4 | 0 |
| 05 | **5.6** / **T** / 159 / 17.7 | 8.9 / F | **enter** | 0 |
| 07 | 11.7 / F / 98 / 52.9 | 11.7 / F | fail 10.2 | 0 |

**Gate FAIL**（enter-win=0；无 planar&lt;5+mouth；ep05 口到了但 planar 5.6≥5）.  
vs GMHQP：ep05 planar 8.9→5.6 + mouth（priv 改善，未过门）；其余基本同构.  
**METHOD_GATE STOP** — 不搜参；不扩 ep1–10。诚实 REUSE 组合未收回 Type-A 进孔。
