# Track-B: Joint Seat–Enter (known-path force enter)

> Dual-track Track B (`docs/DUAL_TRACK_TO_ORACLE.md`): **no peg-tip GT** in the loop.  
> Target ceiling: Oracle-v2 **9/10** (diagnostic only; never in deployable table until matched without tip/hole GT).  
> SUCCESS_STANDARD locked (`docs/SUCCESS_STANDARD.md`).

---

## 0. Scientific lock

**已知:** 孔口邻域 / tip 平面螺旋（规划 + 视觉粗定位）。  
**难:** 非刚性 grasp 下保贴面、轴锥 ≤25°、真力控入孔；腕命令 ≠ tip（\(C(q)\)、滑移）。

一句话：在已知 tip→口轨迹上，**联立优化腕平面跟踪、抓取恢复、口区加压**，逼近 Oracle-v2 的 seat+upright+force-enter，而不用逐步 tip GT。

---

## 1. Method: Joint Seat–Enter (JSE)

把「贴面 / 直立 / 进孔」写成**分层约束下的联立控制**（Escande HQP 精神；实现上用 Tip-Hybrid FSM + 门控层，非完整 QP）。

### 1.1 Decision variables

| 变量 | 符号 | 含义 | 传感器 |
|------|------|------|--------|
| 腕平面步 | \(\Delta x_w \in \mathbb{R}^2\) | tip 螺旋跟踪（经 `planar_C`） | 规划口心 + tip 估计（sim: priv tip **仅诊断**；控制用 FK/耦合估计） |
| 抓取尺度 | \(s_g \ge 1\) | 左手 hold 收紧 | proprio slip peak；**sim 指尖接触力可作 tactile-surrogate（须披露）** |
| 耦合 | \(C\)（可选） | 腕→tip 平面映射；滑移时 \(C\leftarrow I\) | EMA / EKF slip reset |

不引入孔位特权吸引（禁 MSAR `e_mouth` PD）；口心 = **已知规划螺旋中心**。

### 1.2 Constraints (priority)

硬审计与软控制阈值：

1. **平面接触 (SEAT):** tip float / seat resid 在容差内；丢接触 → 冻螺旋、加压 reseat。  
2. **轴锥:** 入孔审计 `axis_err ≤ 25°`；控制软阈 ~16° + confirm（`hold_enter_until_upright`）。  
3. **力入孔残差:** 近口时轴向加压；仅真 jam（\(F \gtrsim 1.2\,\mathrm{N}\)）卸载，禁止把 soft seat ~0.7 N 当 unload。

优先级（高→低）——相对 Tip-Hybrid 的扩展：

```
GRASP_RESTORE (Type-B) 
  → SEAT (Tip-Hybrid)
  → UPRIGHT (Tip-Hybrid / Type-A before recover)
  → LOCAL_RECOVER (Type-A inward)
  → SPIRAL track (planar_C on known path)
  → MOUTH_PRESS / force-hole (deploy mouth enter)
```

Tip-Hybrid 仍拥有 SEAT→UPRIGHT→SPIRAL 基 FSM；JSE **不替换** Hybrid，只在其上叠 recover / tighten / mouth。

### 1.3 Known path usage

规划：关于已知口心 \(c\) 的内向 Archimedean 螺旋 \(p^\*(r_\mathrm{cmd},\theta)\)。

跟踪误差（平面投影）：

\[
e = \Pi_n\!\big(p_\mathrm{tip} - p^\*(r_\mathrm{cmd})\big),\quad
\Delta x_w \approx \mathrm{clip}\big(C^\dagger e\big)
\]

饱和（Type-A，内向）：

\[
\mathrm{sat} = \big(\|e\| > \tau_e\big) \land \big(\rho_\mathrm{tip} > r_\mathrm{cmd}+\delta\big) \land \mathrm{seat\_ok}
\]

→ `ṙ←0`，局部径向棘轮 \(r_\mathrm{tgt}=\min(r_\mathrm{cmd},\max(r_\mathrm{mouth},\rho-\Delta))\)，**不用**孔 GT 吸引。

近口健康（\(|r_\mathrm{cmd}-\rho|\lesssim\delta\)）→ STAR idle → **保 ep04**。

### 1.4 Mouth press (Oracle residual class)

当 tip 已在规划口环（\(\rho_\mathrm{tip}\) / \(r_\mathrm{cmd}\) 门，非 priv lat  alone）：

- 停螺旋进尺 → SEAT + mouth_probe  
- XY hold（planner center，不追特权 hole axis）  
- `seat_press_scale` 加压；`jam_n ≳ 1.2` 才 unload  
- 高度下落入孔须 seat + along 门（拒 float-along）

### 1.5 Slip / grasp co-opt (Type-B)

slip peak 边沿闩锁 → 冻螺旋 → SEAT → `bypass_freeze` 真收紧 → \(C\leftarrow I\) → 恢复。  
与 LOCAL_RECOVER 共存：restore 结束后允许 Type-A recover 继续收环。

### 1.6 vs Tip-Hybrid / Oracle / Track-A

| | Tip-Hybrid alone | **JSE (Track-B)** | Oracle-v2 |
|--|------------------|-------------------|-----------|
| tip GT | 否 | **否** | 是（诊断） |
| path | 规划螺旋 | 同左 + Type-A recover | tip→口 PD |
| slip | 弱 | Type-B tighten | oracle reseat |
| mouth | 弱 probe | deploy mouth press | oracle mouth press |
| 目标率 | 4–5/10 | → 逼近 9/10 | 9/10 上界 |

Track-A = 估计 tip 后套 Oracle 律；Track-B = **不估全 tip 状态**，用已知路径 + 约束联立。

### 1.7 Disclosure (tactile-surrogate)

- 控制：**禁止** peg tip xpos / hole GT 逐步反馈。  
- Sim 可用：腕 F/T、proprio slip、指尖接触力作 **触觉代理**（部署换真触觉）。  
- 主表只报 SUCCESS_STANDARD；Oracle 率不上可部署表。

---

## 2. Landing

| Piece | Path |
|-------|------|
| This doc | `docs/TRACK_B_JOINT_SEAT_ENTER.md` |
| Config | `configs/scheme_l3/S2_trackB_joint.yaml`（Type-A 基 + Type-B slip + mouth） |
| Code | 已有 `sim_runner` / `tip_tracking_baselines`（无新范式代码） |

### Config merge knobs

- **Type-A:** `surface_tip_star_*` LOCAL_RECOVER + upright-before-recover + hold_enter  
- **Type-B:** `surface_tip_slip_*` reseat / grasp_scale / bypass_freeze / retrip latch  
- **Mouth:** `surface_deploy_mouth_enter` + jam_n / seat_press_scale（yield spiral 默认 OFF，防假 yield）

---

## 3. Smoke gate

```bash
cd /home/wangrenpeng/priv_compliant_insert
PYTHONPATH=src:/home/wangrenpeng/dexjoco \
  MUJOCO_GL=egl \
  /home/wangrenpeng/miniconda3/envs/dexjoco/bin/python \
  scripts/parallel_force_enter_ep10.py \
  --config configs/scheme_l3/S2_trackB_joint.yaml \
  --eps 3,4,7,8,9,10 \
  --out-root outputs/scheme_l3/S2_trackB_joint_smoke \
  --workers 3
```

相对 Type-A 全量：ep03/08/09 平面饱和；ep07/10 大滑移；ep04 已成功（须保持）。  
Oracle-v2 在该集上仅 **ep04 失败**（轴歪）；JSE 应 **保 ep04**，并争取在 {3,7,8,9,10} 上赢 ≥2 或清晰 priv 改善。

**Gate → full ep1–10:** 保 ep04；且 win ≥2 of {3,7,8,9,10} **或** 明确 `priv_planar` / slip 改善。

### Smoke `S2_trackB_joint_smoke` (2026-09-03) — **2/6；gate BORDERLINE**

| Ep | enter | vs Type-A | note |
|----|-------|-----------|------|
| 4 | **1** | kept | STAR idle / no mouth attractor |
| 9 | **1** | **win** | was Type-A fail; mouth+slip → lat≈3.7 mm |
| 3 | 0 | ≈ | planar~8.5 mm saturate |
| 7 | 0 | slip↓ modest | still planar~10 mm / axis~40° |
| 8 | 0 | priv_mouth | lat~5.8 mm；到口无力入未关 |
| 10 | 0 | **slip 1032→198 mm** | planar 仍~12 mm |

- 保 ep04：**是**；fail-set enter-win：**1/5**（仅 ep09）。  
- 明确 priv/slip 改善：ep10 slip 大降；ep08 `priv_mouth=True`。  
- **未开全量**（enter-win <2）；接近 Oracle 成功集仍差 ep03/07/08/10 平面/轴。  
- 相对 Oracle-v2：该子集 Oracle 仅败 ep04；JSE **保 ep04** 且赢 ep09，但未齐 Oracle 的 3/7/8/10。
