# Multi-agent brief: tip spiral under non-rigid grasp

> **Gate:** 任何方案代理必须先读本文件 §1 特权科学问题，再动手。  
> **成功标准:** `docs/SUCCESS_STANDARD.md`（贴面 + 轴≤25° + 力入孔）— **不放宽**  
> **方法门禁:** `docs/METHOD_GATE_NO_PATCH.md` + `docs/GENERAL_ALGO_MANDATE.md` — 通用科学问题→通用算法；失败 ep 仅探针，禁止按 ep 补丁。  
> **检索优先:** 诊断缺口若已有可落地文献/refs 方案 → **复用**；没有再 **创新** 方程方案。  
> **问题重钉（2026-09-03）:** tip/孔口轨迹可视为**已知**（视觉粗定位 / 规划螺旋）；研究重点是非刚性抓取下 **保贴面、不太歪、真力控进孔**。  
> **诊断上界:** Oracle-v2（已知 tip↔口）**9/10**（不可部署主表）  
> **可部署当前最好:** Track-B GMHQP-AC **7/10**（`S2_trackB_gmhqp_ac_ep1_10`；ok 1,4,6,7,8,9,10；曾 GMHQP **6/10** / Type-A **5/10**）

---

## 1. 科学问题（锁定）

### 1.1 设定

- **已知:** 孔口邻域 / tip 应走的平面螺旋（或 tip→口路径）可由规划 + 螺旋前视觉给出。  
- **未知/难:** 非刚性 grasp 下 tip 是否贴面、轴是否直立、力是否真进孔；腕命令 ≠ tip 运动（\(C(q)\)、滑移）。

### 1.2 一句话

> **在非刚性 dexterous grasp 下，给定已知的 tip→口平面轨迹，如何在全程保持 tip 贴面、peg 大致直立，并用可部署力觉完成真力控入孔？**

不是：再发明「盲搜孔在哪」。  
是：**已知轨迹上的接触约束力控**（seat + upright + force-enter），外加必要的腕→tip 耦合处理。

### 1.3 与旧瓶颈的关系

旧 v10 诊断「tip 到不了口」在**轨迹未知/跟踪失败**时成立。  
若口/轨迹已知（Oracle / 视觉初值），剩余失败主要是：

| 型 | 代表 | 含义 |
|----|------|------|
| 到口无力入 | Oracle-v1 ep03；Type-A ep08/09 | 缺 mouth 加压 / 力门 |
| 到口轴歪 | Oracle-v2 ep04；Type-A ep07/10 | 缺入孔前 upright hold |
| 滑移丢贴面 | Type-A/B ep7/9/10 | 需 grasp restore + SEAT |

### 1.4 基线含义（已发表 0/10）

ConnTact / Franka / PSFT：即使孔位大致已知，也是 **腕 TCP 螺旋 + 高度/力伪入孔**，无 tip 贴面/直立约束 → 非刚性下假 enter，被 SUCCESS_STANDARD 拒掉。  
这支持论文 gap：**已知搜孔轨迹 ≠ 非刚性下真力控进孔**。

### 1.5 特权工具

```bash
PYTHONPATH=src python scripts/diag_spiral_priv.py <run>/ep*/ep*_summary.json
# 汇报: priv_planar_min / priv_mouth / slip / along / axis；主表仍走 SUCCESS_STANDARD
```

Oracle = 已知 tip+口的控制上界；主表方法可用「已知螺旋中心/口」但不许用逐步 tip GT（除非标成估计器 Path A）。

---

## 2. 多代理分工

### Agent-R：已有算法参考升级（reuse）

**任务:** 在 `refs/` 与文献中找可升级路径，落到可跑 config/代码，**不发明全新范式**。

优先参考（已在树内）：

| Ref | 路径 | 可升级点 |
|-----|------|----------|
| Kim TECterity FG | `refs/Tactile-Estimator-Controller` | tip+C 联立估计；sim 用 priv tip 代触觉 |
| Active extrinsic | `refs/` / arXiv:2110.03555 | tip 枢轴保接触 + 接触线估计 |
| BGF / LeTac-MPC | `refs/bgf`, `refs/LeTac-MPC` | slip reset、MPC 恢复 |
| Pfanne EKF | cite + `src/pci/coupling_ekf.py` | **跑通 S9_ekf_qp** 对比 S8 |
| ConnTact | `refs/ConnTact` | 仅作刚性 baseline，勿当 tip 解 |
| Escande HQP | `tip_tracking_qp.py` | 平面+力硬约束 QP |

**交付:**

1. 选 1 条主升级线（推荐：**S9 EKF+QP 完整跑 ep1–10** 或 Kim 式 tip 状态因子简化版）  
2. 对照特权表：fail ep 的 `priv_planar_min` 是否下降  
3. 更新 `docs/REF_REUSE_MAP.md`  
4. 禁止放宽 SUCCESS_STANDARD

### Agent-I：诊断驱动创新（invent）

**已落地(未超基线):** Tip-MSAR — `docs/TIP_TRACK_INNOVATION_MSAR.md` → 仍 4/10（ep04 回退）  
**下一版:** Tip-STAR — `docs/TIP_TRACK_THEORY_NEXT.md`  
**代码:** `tip_star_*` + `sim_runner` hooks + `configs/scheme_l3/S2_star.yaml`  
**评测门:** 先 fail-subset smoke ep2,3,4,8,9（无 ep04 回退）再 ep1–10  
**禁止:** 特权 hole attractor 作为可部署宣称；成功率未 >4/10 不算进步

### Agent-D：纯诊断 /  oracle 上界（optional）

**任务:** 特权 tip 直接当控制反馈（oracle upper bound）：若 tip PD 到孔口仍失败 → 问题在物理/抓取非跟踪；若成功 → 跟踪/估计是瓶颈。

**协议:** `docs/ORACLE_TIP_SERVO.md` · config `configs/scheme_l3/S2_oracle_tip.yaml` (v1) / `S2_oracle_tip_v2.yaml` (v2) · mode `priv_oracle_tip_servo`

交付：`priv_oracle_tip_servo` ep1–10 率 + 与 v10 对比（主表外分列）。v2 诊断上界 **9/10**（v1=7/10）；**禁止**写入可部署主表。

---

## 3. 共享约束

1. 先读 §1；汇报开头写：`合规特权诊断: … / 科学问题: …`  
2. 成功标准不放宽（贴面 + 轴 + 力入孔）  
3. 主表与 oracle 分列（`PRIVILEGED_SENSING_DISCLOSURE.md`）  
4. 改 `sim_runner` / `tip_tracking_*` 保持 Tip-Hybrid 接口可开关  
5. 评测：`scripts/parallel_force_enter_ep10.py --out-root outputs/scheme_l3/<tag>`

---

## 4. 当前代码锚点

- Hybrid FSM: `src/pci/tip_tracking_baselines.py` `TipHybridState` / `tip_hybrid_select` / `tip_pivot_upright`  
- planar_C: `baseline_hold_r` + `PlanarCouplingEstimator`  
- EKF+QP: `src/pci/coupling_ekf.py`, `tip_tracking_qp.py`, `configs/scheme_l3/S9_ekf_qp.yaml`  
- 螺旋环: `src/pci/sim_runner.py` planned_spiral  
- 方法草稿: `docs/TIP_HYBRID.md`, `docs/METHODOLOGY_V2.md`, `docs/DIAG_FORCE_GATE.md`
