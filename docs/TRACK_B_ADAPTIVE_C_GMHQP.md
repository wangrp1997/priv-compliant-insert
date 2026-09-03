# Track-B Adaptive-C on GMHQP (AC-GMHQP)

> **Mandate:** `docs/GENERAL_ALGO_MANDATE.md` — 通用律，禁止 ep 补丁。  
> **Base:** GMHQP（Montana \(C^{+}\) + Escande + GraspQP）。  
> **正交:** COMPC 改 L2 **接触短时域代价**；本文件改 **grasp-map \(C\) 的在线估计与不确定度映射**。  
> SUCCESS_STANDARD 不放宽。控制环禁用 tip GT。

---

## 1. Scientific problem (general)

已知平面路径 \(p^*(\theta)\)。非刚性抓取：

\[
\dot x_{\mathrm{tip}} = C(q)\,\dot x_{\mathrm{wrist}},\qquad C\neq I
\]

GMHQP 已用有限差分 EMA 估 \(C\)。病态/滑移时 \(C\) 漂移 → \(C^{+}\) 把 tip 任务映错腕增量 → `priv_planar_min` 饱和。  
**一般控制升级:** 用可观测的 tip↔腕增量，在线滤波估计 \(C\)，并以协方差调制 Tikhonov，使映射在不确定时退回 \(I\)。

---

## 2. Search → REUSE (not invent parallel brand)

| Candidate | Role | Verdict |
|-----------|------|---------|
| GMHQP Escande + tip task | keep hierarchy + \(e=\Pi(p^*-t)\) | **keep** |
| `CouplingEKF` (`src/pci/coupling_ekf.py`) | tip+\(C\) 联立 EKF + slip reset | **REUSE** |
| Pfanne IROS’17 / RA-L’20 | grasp state from proprio; object impedance | **cite** |
| Reactive slip / online \(G\) update | slip → reinflate model | **spirit** |
| COMPC contact MPC | contact residual in cost | **orthogonal — do not duplicate** |
| FTIP / Hybrid / FASR | tip swap / ep switch | **拒** |

**Verdict: REUSE compose** — CouplingEKF 在线 \(\hat C\) ⊕ 不确定度加权 \(C^{+}\) ⊕ 既有 Escande。  
标签 `track_b_gmhqp_ac`（Adaptive-\(C\)）。COMPC 并行评测；本线不叠接触 MPC。

---

## 3. Equations (general law)

**State (planar tangent frame):**

\[
x = \big[t_\parallel^\top,\; c_{11},c_{12},c_{21},c_{22}\big]^\top
\]

**Predict (Montana kinematics proxy):**

\[
t_\parallel \leftarrow t_\parallel + C\,\Delta w_\parallel
\]

过程噪声对角：\(Q_t\) 于 tip，\(Q_C\) 于 \(C\)。

**Update:** in-hand tip surrogate \(z=t_{\mathrm{obj},\parallel}\)（与 GMHQP 同披露；**非 tip GT 伺服**）。

**Slip reset (bgf / Kim spirit):** 若 \(\|\Delta t_{\mathrm{obs}}\| < \rho\,\|C\Delta w\|\)，则 \(C\leftarrow I\)，放大 \(P_C\)。

**Escande (unchanged):** L0 SEAT → L1 UPRIGHT → L2 tip path.

**L2 Adaptive map (vs GMHQP EMA):** tip 任务仍用几何 tip

\[
e = \Pi(p^* - t_{\mathrm{obj}}),\qquad v^* = K_p e
\]

不确定度加权 Tikhonov（\(P_C=\mathrm{Cov}(\mathrm{vec}\,C)\)，innov = tip 更新残差）:

\[
\begin{aligned}
\lambda &= \lambda_0 + \lambda_P\,\mathrm{Tr}(P_C) + \lambda_i\,\|y_{\mathrm{tip}}\| \\
\Delta w_\parallel &= \big(\hat C^\top\hat C + \lambda I\big)^{-1}\hat C^\top v^*
\end{aligned}
\]

\(\lambda\) 大 → 映射退向 \(I\)（安全）；滑移后 \(P_C\) 大 → 自动保守。无 episode 开关。

### vs COMPC / base GMHQP

| | GMHQP | COMPC | **AC (this)** |
|--|-------|-------|---------------|
| Tip task \(e\) | \(\Pi(p^*-t)\) | same | same |
| L2 velocity | \(v^*=K_p e\) | contact-horizon \(u^\star\) | \(v^*=K_p e\) |
| \(C\) source | FD+EMA | FD+EMA | **CouplingEKF + slip** |
| \(C^{+}\) | fixed ridge \(\lambda_0\) | fixed \(\lambda_0\) | **\(\lambda(P_C,\mathrm{innov})\)** |

---

## 4. Anti-patch

| Refuse | Why |
|--------|-----|
| `if ep in …` / keep-set | GENERAL_ALGO_MANDATE |
| tip_obj := FT ĉ / tip GT | Track-B lock |
| 叠 COMPC 再搜参冲 RATE | 正交线分开评 |
| 仅修 fail-subset | 探针≠目标 |

---

## 5. Implementation map

| Piece | Path |
|-------|------|
| Design | this file |
| Controller | `src/pci/track_b_gmhqp_ac.py` |
| EKF | `src/pci/coupling_ekf.py` (REUSE; expose \(P_C\)) |
| Hook | `baseline_hold_r(..., mode="track_b_gmhqp_ac")` |
| Config | `configs/scheme_l3/S2_trackB_gmhqp_ac.yaml` |
| Unit | `scripts/smoke_track_b_gmhqp_ac.py` |

Disclosure: peg tip geom = object localization surrogate（同 GMHQP/OIGS）。

---

## 6. Eval plan

**Unit smoke only first**（GPU 忙时禁止抢满量）:

```bash
PYTHONPATH=src python scripts/smoke_track_b_gmhqp_ac.py
```

**Full ep1–10（GPU 空闲后再跑；勿与 COMPC/TEC 并行抢）:**

```bash
cd /home/wangrenpeng/priv_compliant_insert
MUJOCO_GL=egl PYTHONPATH=src:/home/wangrenpeng/dexjoco:/home/wangrenpeng/dexjoco/dexjoco:/home/wangrenpeng/dexjoco/embodied_grasp_insertion \
  /home/wangrenpeng/miniconda3/envs/dexjoco/bin/python scripts/parallel_force_enter_ep10.py \
  --config configs/scheme_l3/S2_trackB_gmhqp_ac.yaml \
  --out-root outputs/scheme_l3/S2_trackB_gmhqp_ac_ep1_10 \
  --eps 1-10 --workers 2 --video 2>&1 | tee outputs/scheme_l3/S2_trackB_gmhqp_ac_ep1_10_run.log
# then:
PYTHONPATH=src python scripts/diag_spiral_priv.py outputs/scheme_l3/S2_trackB_gmhqp_ac_ep1_10/ep*/ep*_summary.json
```

对比基线: GMHQP **6/10**；COMPC 另表。RATE≤基线 → STOP 旋钮，改方程。

---

## 7. Full ep1–10 result (2026-09-03)

- Out: `outputs/scheme_l3/S2_trackB_gmhqp_ac_ep1_10`
- **RATE 7/10**（ok **1,4,6,7,8,9,10**；fail **2,3,5**）
- vs GMHQP **6/10**：keep 1,4,6,8,9,10；**new win ep07**；无 keep 回归；净 **+1**
- Fail probes: planar F1（ep2/3 ~11.4 mm；ep5 ~5.9 mm）仍无 mouth
- **METHOD_GATE:** > baseline → 更新论文主表；**STOP 调参**（不旋钮搜参冲更高）
- Deployable headline → **GMHQP-AC 7/10**（同 † tip-geom disclosure）
