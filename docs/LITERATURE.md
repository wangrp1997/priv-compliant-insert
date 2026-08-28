# Phase B 文献参考（双臂柔顺插孔）

> **不用 echo_insert**（本地实现有问题）。Phase B 按下列经典/近期文献设计。

---

## 推荐流水线（与 Phase A hybrid ALIGN 衔接）

```
ALIGN 完成（孔口 standoff）
    → B1 恒力贴面 + 平面螺旋搜索（找孔）
    → B2 轴向 admittance/阻抗插入（限 F_axial, F_lat）
    → B3 坐底检测 + 右臂松手（insert_ok）
```

左臂：高刚度持 tray（位控为主，吸收反力）。  
右臂：任务坐标系 admittance（孔轴 Z / 横向 XY / 姿态）。

---

## 核心文献

| 文献 | 要点 | PCI 对应 |
|------|------|----------|
| [Park et al. TIE 2017](https://doi.org/10.1109/tie.2017.2682002) Compliance-Based Peg-in-Hole Without Force Feedback | 程序化柔顺 + 螺旋力轨迹 SFT | 无 FT 时的 fallback（sim 仍有 FT） |
| [PSFT, RA-L 2020](https://doi.org/10.1109/lra.2020.3000428) Partial Spiral Force Trajectory + tilted peg | 部分螺旋、减方差 | **B1 搜索**首选 |
| [Lee et al. RA-L 2022](https://doi.org/10.1109/lra.2022.3187497) Dual-Arm Peg-in-Hole | 双臂 + 灵巧手 insert | **双臂分工**主参考 |
| [Intelligent Service Robotics 2020](https://doi.org/10.1007/s41315-020-00136-1) UR3 dual-arm | linear + **spiral search** + **impedance** | **B1→B2 三段**直接模板 |
| [TUM bottle screw](https://mediatum.ub.tum.de/doc/1703739/) Task-Based Compliance | 对准=位控/视觉，插入=柔顺力控 | 与 PCI Phase A/B 分界一致 |
| [IEEE TASE 2024](https://doi.org/10.1109/tase.2024.3487988) Wrist FT narrow clearance | 触觉找孔、窄间隙 | F/T 门控与接触状态 |
| [Contact-state + spiral, 2014](https://orbilu.uni.lu/handle/10993/30478) | wrench 判 peg-on-hole | 可选：接触状态机 |
| [Insert-One IROS 2024](https://rl.cs.rutgers.edu/publications/HaonanIROS2024A.pdf) | 视觉对准 → **impedance 插入** | 两阶段范式（我们用 privileged 替视觉） |
| [Dual-arm hybrid CAC 2024](https://doi.org/10.1109/cac63892.2024.10865403) | 视觉伺服 + **力探索** + fuzzy impedance | 力探索≈B1 螺旋 |

---

## B1 螺旋搜索（力引导）

- 恒压 `F_push` 贴住孔所在平面（沿 −hole_axis）
- XY 平面 **Archimedean / 部分螺旋 PSFT**（参考 I-RIM 2020、PSFT 2020）
- **入孔判据**：`F_z` 骤降或 contact 变化（Lee 2022 / contact-state 文献）
- 双臂：左臂 frozen/slow，右臂执行搜索

---

## B2 轴向插入（admittance / impedance）

- 任务空间：**Z 轴低刚度**（柔顺推进），**XY 高刚度**（限横漂）
- 参考 Insert-One、UR3 论文：`F_axial` 上限、`F_lat` 超限则停/退
- **Jam**：位移停滞 + 力升 → 小退 + 回 B1（TASE 2024 思路）
- 不用 echo 的 energy optimizer；用 **固定增益 admittance** 即可 v1

```text
Δx = K_a · (F_des - F_meas)   # 孔轴方向 F_des = F_insert
τ   = 阻尼项 + 限幅
```

---

## B3 完成

- `AssemblyContactLabeler.insert_ok`（dexjoco 已有，**仅 eval / release 触发**）
- 右臂 gradual open；左臂维持 tray（hybrid RELEASE 可参考阈值，但不走 snap）

---

## 手指柔顺（PCI 创新）

检索结论：

| 来源 | 内容 | PCI 用法 |
|------|------|----------|
| FSMJIC tendon hand (IEEE ROBIO 2016) | 单指关节阻抗 / 柔顺抓取 | 高载指放松思路 |
| DLR Hand II (IROS 2003) | 多指关节阻抗 | 平衡各指力 |
| multi-finger admittance 综述 | 任务空间 admittance | 与腕部 B2 并列 |

**未见** Allegro 双臂 insert 四指平衡柔顺开源 → `src/pci/compliant/fingers.py`：
- 输入：`FingerForceLabeler` 四指 12d 接触力
- 输出：16d hand delta（高载放松 + 四指力平衡）
- 与 B1/B2 并行，release 阶段由 `insert.py` 统一松手

---

## Phase B 传感器边界

- **允许**：腕 6D F/T、四指指尖力、Phase A 结束时冻结的 tool 系（`task_frame.py`）
- **禁止**：tip/socket/hole 真值参与控制环


1. **PSFT / 部分螺旋** + wrist F/T 入孔检测（B1）
2. **任务空间 admittance** 轴向插入（B2）
3. 可选：contact-state 简化版（peg-on-hole 检测）
4. 不做：echo ECHO optimizer、snap、神经网络搜索（2207.07524 作长期可选）

---

## v2 路线：真导纳 + 指力优化（2026-08 调研）

**结论**：单臂刚性 peg 上，腕导纳 + PSFT 螺旋足够；我们是 **双臂 + Allegro 持 peg + 仅 F/T**，需在经典律上补 **内外力分解 + 四指力再分配**。

### 理论对不对？

对。文献控制律是「**力驱动、轴选让位**」：

```text
Δx = K_a (F_des - F_meas) - B_a v     # 腕任务空间导纳
comply_axes: Z soft, XY stiff         # 插入期
Δq_i = K_f (f_des,i - f_i) - B_f q̇_i  # 指关节导纳
```

ConnTact / Ott ICRA2010 / Park PSFT 2020 / Zhang TIE 2018 均验证：柔顺+优化能把 peg 推进去。

### PCI v1 差在哪？

v1 是「力读数调步长的开环位姿」，不是导纳；无 `comply_axes`、无指关节阻尼项。

### 创新缺口（必须自研）

1. 腕 wrench = 孔接触 + 四指抓持 + tray 耦合 → 需 **内外力分解**（文献多假设刚性夹爪）
2. 四指力 → QP/简化平衡再分配（Pfanne RA-L 2020 物体级阻抗；无物体位姿时简化）
3. jam 恢复：抬 z + 回螺旋（franka scripted），非单步 retreat

### 开源对照

| 源 | 用途 |
|----|------|
| `refs/ConnTact` | 恒力 + 轴选柔顺 + 螺旋找孔 |
| `refs/irl_control` | OSC 任务空间导纳 `u ∝ F_ext` |
| GraspQP / GeoDEx | 指力 QP / 触觉 admittance（参考，不必整库） |

### 实现优先级

1. `task_frame.admit_step` + 轴选刚度  
2. `insert.py` / `search.py` 改真导纳  
3. `fingers.py` 关节导纳 + 力平衡  
4. jam → 抬 z 回搜  
5. 可选：轻量接触态 / QP 指力
