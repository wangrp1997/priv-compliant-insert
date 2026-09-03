# Track-B Contact-Opt MPC on GMHQP (wave-7)

> **Mandate:** `docs/GENERAL_ALGO_MANDATE.md` — 失败 ep 只是探针；**禁止** fail-subset / keep-set 进控制器。  
> **Base law:** GMHQP（Montana \(C^{+}\) + Escande + GraspQP）。本文件在其上叠**通用**接触短时域优化。  
> **门禁:** `docs/METHOD_GATE_NO_PATCH.md` · `docs/WAVE7_ON_GMHQP.md`  
> SUCCESS_STANDARD 不放宽。控制环禁用 tip GT。

---

## 1. General optimal control problem (no episode IDs)

已知平面路径 \(p^*(\theta)\)、口邻域 \(p_{\mathrm{mouth}}\)。非刚性抓取下 \(\dot x_{\mathrm{tip}}=C(q)\,\dot x_{\mathrm{wrist}}\)，\(C\neq I\)。

\[
\begin{aligned}
\min_{u_{0:H-1}} \;
&\; J_{\mathrm{track}} + J_{\mathrm{contact}} + J_{\mathrm{seat}} + J_{\mathrm{upright}} + J_u \\
J_{\mathrm{track}} &= \sum_k w_t\,\big\|\Pi\big(p^*_k - t_k\big)\big\|^2 \\
J_{\mathrm{contact}} &= \sum_k w_c\,I_{\mathrm{valid}}\,\big\|\Pi\big(p_{\mathrm{mouth}} - \hat c_k\big)\big\|^2 \\
J_{\mathrm{seat}} &= \sum_k w_f\,(f_{n,k}-f^*)^2 \\
J_{\mathrm{upright}} &\quad\text{Escande hard: axis soft-cone before path} \\
J_u &= \sum_k w_u\|u_k\|^2
\end{aligned}
\]

约束（部署成功标准）: 贴面、轴≤25°、真力入孔。  
决策: 平面 tip 增量 \(u\)，经 \(\Delta w_\parallel=C^{+}u\) 下发；指侧 GraspQP 共栈。  
观测: 腕 F/T → \(\hat c\)（Active Extrinsic / Doshi）；in-hand tip surrogate \(t\)（与 GMHQP 同披露）；**不用 tip GT**。

**这是对任意 rollout 的一般律**，不是某几个失败编号的补丁。

---

## 2. Diagnostic probes only (not design targets)

GMHQP 6/10 上，部分 rollout 出现 **F1**：代理 tip 路径残差小、真平面误差仍大（`bl_tip_err`≪`priv_planar`）。  
失败编号仅作**模式探针**（说明假闭合可发生），**不**写入控制器分支、不作为优化目标集。

| Probe class | Evidence (examples only) | Implication for **general** law |
|-------------|--------------------------|----------------------------------|
| F1 false tip closure | residual fails under GMHQP | Need contact residual in cost, not tip_obj swap |
| FTIP regress | tip_obj:=ĉ → tip_err blow-up | \(\hat c\) in **cost only** |
| Hybrid miss | switch on proxy tip_err | Horizon multi-objective, not ep switch |

---

## 3. Search → REUSE (upgrade GMHQP as a general law)

| Candidate | Role | Verdict |
|-----------|------|---------|
| GMHQP Escande+\(C^{+}\) | base hierarchy + tip task | **keep** |
| Kim Active Extrinsic ICRA’22 | contact line → mouth | **REUSE** \(e_c\) |
| LeTac-MPC / TACTIC | path + contact force over horizon | **REUSE** \(J\) structure |
| GraspQP pose-hold | finger co-opt | **keep** |
| FTIP / Hybrid / FASR | tip swap / fail-gated switch | **拒** |

**Verdict: REUSE compose** — 接触线残差 ⊕ 短时域路径/力代价 ⊕ 既有 GMHQP。工程标签 `track_b_gmhqp_compc`；不 invent 平行发表缩写。

---

## 4. Algorithm (general law on GMHQP)

**tip_obj 始终 = in-hand tip**（永不 `tip_obj:=\hat c`）。

Escande L0 SEAT / L1 UPRIGHT：同 GMHQP。

L2 — 常值开环 \(u\)、地平线 \(H\)，再取第一步（receding）:

\[
\begin{aligned}
e_t &= \Pi(p^*-t),\qquad
e_c = \Pi(p_{\mathrm{mouth}}-\hat c)\\
t_k &= t_0 + k\,u,\quad
\hat c_k = \hat c_0 + k\,u\\
\alpha(e_t) &= \frac{\sigma^2}{\sigma^2+\|e_t\|^2}
\quad\text{（残差尺度：tip 大 → 纯 tip 任务；tip 假小 → 接触权重升）}\\
J(u) &= \sum_{k=0}^{H-1}\Big(
  w_t\|e_t-ku\|^2 + w_c I_{\mathrm{valid}}\alpha\|e_c-ku\|^2
\Big) + H w_u\|u\|^2
\end{aligned}
\]

闭式（各向同性）:

\[
u^\star =
\frac{S_1(w_t e_t + w_c I_{\mathrm{valid}}\alpha\, e_c)}
{S_2(w_t + w_c I_{\mathrm{valid}}\alpha)+H w_u}
\]

\[
\Delta w_\parallel = C^{+}(K_p u^\star),\qquad
a_x \mathrel{+}= k_f(f^*-f_n)
\quad\text{（软座面力；硬 seat 仍 Escande）}
\]

\(\alpha\) 是**状态函数**，不是 episode 开关；无 keep-set / fail-set 逻辑。

---

## 5. Anti-patch

| Refuse | Why |
|--------|-----|
| `if ep in …` / keep-set branches | GENERAL_ALGO_MANDATE |
| tip_obj := FT ĉ | FTIP 毁 tip 任务一致性 |
| 仅优化 fail-subset 再宣称解决 | 探针≠目标 |
| 阈值搜参过 RATE | 改方程，不旋钮 |

---

## 6. Implementation map

| Piece | Path |
|-------|------|
| Design | this file |
| Controller | `src/pci/track_b_gmhqp_compc.py` |
| Hook | `baseline_hold_r(..., mode="track_b_gmhqp_compc")` |
| Config | `configs/scheme_l3/S2_trackB_gmhqp_compc.yaml` |
| Unit | `scripts/smoke_track_b_gmhqp_compc.py` |

Disclosure: geom tip = object-localization surrogate（同 GMHQP）；\(\hat c\) = 可部署腕 F/T 外禀接触线索。

---

## 7. Evaluation (reporting only)

| Eval | Role |
|------|------|
| **Primary:** full ep1–10 RATE | vs GMHQP **6/10**；主结论 |
| Fail/keep subsets | **诊断/对照 only**；不进设计、不进控制器 |
| RATE ≤ 6/10 | **STOP retune**；改方程后再评 |

Out: `outputs/scheme_l3/S2_trackB_gmhqp_compc_ep1_10`.

---

## 8. Results

| Run | RATE | vs GMHQP 6/10 | Notes |
|-----|------|---------------|-------|
| full ep1–10 | **4/10** | **−2** | primary; ok **1,6,8,10**; fail **2,3,4,5,7,9** |
| vs GMHQP keep | — | lost **4,9**; new wins **∅** | kept 1,6,8,10; residual fails 2,3,5,7 still F1-ish |

Out: `outputs/scheme_l3/S2_trackB_gmhqp_compc_ep1_10` · log `..._ep1_10_run.log` · `summary_enter10.json`.

### Fail probes (`diag_spiral_priv.py`; not design targets)

| ep | priv_planar_min | bl_tip_err_mean | tip_spiral_err_mean | slip | reason | note |
|----|-----------------|-----------------|--------------------|------|--------|------|
| 02 | 8.7 | 5.9 | 10.5 | 22 | tilt/tray | F1; contact residual did not close mouth |
| 03 | 11.6 | 7.5 | 7.3 | 16 | priv gate | F1; along drift +118 mm |
| 04 | 15.4 | 6.5 | 16.9 | 19 | tilt/tray | **was GMHQP win** (planar~4.5+mouth); COMPC regress |
| 05 | 10.6 | 8.6 | 0.9 | 91 | surface_press | F1+slip; tip_err mean small, planar still >10 |
| 07 | 15.7 | 8.0 | 6.8 | 59 | priv gate | F1+slip |
| 09 | 9.2 | 6.2 | 8.4 | 27 | tilt/tray | **was GMHQP win** (planar~4.4+mouth); COMPC regress |

**Verdict:** RATE **4/10 ≤ GMHQP 6/10** → **STOP retune** (GENERAL_ALGO_MANDATE). Contact cost on GMHQP **regressed** keep-set {4,9}; no residual enter-win. Revise **equation / law**, not thresholds. **Do not** paper-table update; deployable remains **GMHQP 6/10**.
