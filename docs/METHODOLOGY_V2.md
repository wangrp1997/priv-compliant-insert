# Methodology V2 — Slip-Aware Coupling Estimation + Constrained Tip Tracking

> 在 planar C 一步反演失败后的**理论升级路径**（非调参清单）。  
> Sim：privileged tip 仅作 **estimator oracle**；部署：触觉 + 腕 F/T + FK。

---

## 1. 科学问题（不变）

非刚性 grasp 下 $\Delta \mathbf{p}_{tip} \not\approx \Delta \mathbf{x}_{wrist}$。  
目标：tip 接触平面上的螺旋跟踪 + 在线 $\mathbf{C(q)}$。

---

## 2. 三层架构（V2）

```
┌─────────────────────────────────────────────────────────┐
│ L3  Grasp stabilization (已有 GraspQP / priv_grasp_opt) │
├─────────────────────────────────────────────────────────┤
│ L2  Constrained tip tracking QP (新)                    │
│     min ||p_tip - p*||² + λ_F(F_n-F_d)²                  │
│     s.t. n^T(p_tip-p0)=0, F_min≤F_n≤F_max, |Δq|≤δ     │
├─────────────────────────────────────────────────────────┤
│ L1  Recursive coupling + tip state estimation (新)      │
│     x = [p_tip^∥, vec(C), o_wt]                         │
│     predict: random walk / slip event reset             │
│     update: (wrist, tactile/proprio)                  │
└─────────────────────────────────────────────────────────┘
```

### L1 — 状态估计（替代滑窗 LS）

| 选项 | 文献 | 适用 |
|------|------|------|
| **EKF / InEKF** | Pfanne IROS'17; Piga diff-EKF sliding; Hartley contact-InEKF | 递推、slip 后协方差膨胀 |
| **因子图 iSAM** | Kim ICRA'23; Yu ICRA'18; TacGraph | 多模态、contact factor |
| **粒子滤波** | MidasTouch CoRL'22 | 多模态 slip belief |

**推荐（sim→真机路径）：** 轻量 **EKF**  
- 状态：tip 平面位姿 $[x,y]$ + $\mathbf{C}$ 四参数（或 $\alpha$, $\beta$ 低秩）  
- 观测：$\Delta \mathbf{p}_{tip}$（sim priv / 真机 tactile shear）  
- slip：$|\Delta \mathbf{p}_{tip}| / |\Delta \mathbf{x}_{w}| < \tau$ → reset $\mathbf{P}$, $\mathbf{C}\leftarrow \mathbf{I}$

**教科书：** Craig Ch.11 hybrid force/position；Modern Robotics Ch.12 接触约束。

### L2 — 约束 QP（替代裸 C⁻¹ 一步）

每步解（Escande HQP 单层即可）：

$$
\min_{\Delta \mathbf{q}_w} \ \|\mathbf{p}_t + \hat{\mathbf{C}}\Delta\mathbf{q}_w - \mathbf{p}^*\|^2 + \lambda_F (F_n - F_d)^2
$$
$$
\text{s.t.}\ \mathbf{n}^\top (\mathbf{p}_t - \mathbf{p}_0) = 0,\quad F_{\min} \le F_n \le F_{\max},\quad \|\Delta \mathbf{q}_w\| \le \delta
$$

**螺旋期手指锁死时的力门控（V2.1，已实现）：**  
冻结 grasp 下腕部开环追 spiral 会把 peg 怼滑。`spiral_force_gate` 在 `baseline_hold_r` 内：当 wrist residual $|F_r| > F_{gate}$ 或 peg-in-hand slip $> \tau_{slip}$ 时，将 planar track/step 缩至 `overload_planar_scale`，并沿法向 unload——与非 baseline `planned_spiral` 过载逻辑对齐（ConnTact compliance 思路）。

- 贴面：Lynch & Park Ch.12 不可穿透 / 滚动-滑动约束的简化版（平面保持）  
- 导纳/阻抗：Hogan 1985；Spong Ch.10 — 法向 compliance 与平面 QP 分离  
- **不是** full MPC；对比文献：one-step MPC (Zometa'23), SECMPC (Toussaint'22)

### L3 — 抓握（已有）

GraspQP / `priv_grasp_opt.py` — 与 spiral 层正交，继续并行。

---

## 3. 与 V1（当前代码）对比

| | V1 planar C | V2 |
|---|-------------|-----|
| 估计 | 滑窗 LS | EKF + slip reset |
| 控制 | $\mathbf{C}^{-1}\mathbf{e}$ 一步 | 约束 QP |
| 贴面 | 投影 heuristic | 硬等式约束 |
| 理论引用 | 弱 | KF + contact QP + Hogan |

---

## 4. Sim privileged 写法（文献共识）

- **主表**：S0–S6 可部署 sensing（腕 F/T + proprio）  
- **Diagnostic 行**：privileged tip oracle 估 $\mathbf{C}$ 误差上界  
- **不分**：proposed 与 ConnTact 用不同 tip 观测  

模板见 `docs/PRIVILEGED_SENSING_DISCLOSURE.md`。

---

## 5. 实现顺序（一次做完，不扫参）

1. `CouplingEKF` in `tip_tracking_baselines.py`（替换 PlanarCouplingEstimator LS）
2. `tip_tracking_qp.py` — 2D QP with plane + force slack
3. Config `S9_ekf_qp.yaml` — 唯一 proposed V2
4. 跑 ep01–10 一次 vs S2/S3/S6

---

## 6. 关键参考文献（方法论）

- Kim et al., ICRA 2023 — extrinsic contact FG  
- Choi et al., RA-L 2021 — kinesthetic + spiral PiH  
- Pfanne & Chalon, IROS 2017 — EKF from joint torque  
- Piga et al., Frontiers 2021 — diff-EKF under sliding  
- Escande et al., IJRR 2014 — HQP  
- Hogan, ASME 1985 — impedance/admittance  
- Lynch & Park, Modern Robotics Ch.12 — contact kinematics  
- Park et al., RA-L 2020 — PSFT spiral baseline  
