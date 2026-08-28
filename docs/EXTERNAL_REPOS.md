# 外部开源仓库（网络检索）

> 本地 dexjoco/echo 不可靠；Phase A/B 优先参考下列 **GitHub 仓库**。

---

## Phase A — 粗对准 / PBVS

| 仓库 | 链接 | 可抄什么 | 备注 |
|------|------|----------|------|
| **peg-in-hole-visual-servoing** | https://github.com/RasmusHaugaard/peg-in-hole-visual-servoing | **双臂**：`peg_robot` + `aux_robots` 协同 VS；demo 标定孔位 | CORL 2020；需配 model 仓库 |
| peg-in-hole-visual-servoing-model | https://github.com/RasmusHaugaard/peg-in-hole-visual-servoing-model | 训练/推理 VS 模型 | 上者依赖 |
| **VisualServoing** | https://github.com/paulrupprecht/VisualServoing | **PBVS/IBVS** P 控制器、marker 位姿误差 | UR5+ROS；教学向 |
| visual_servo_course | https://github.com/littlefiveRobot/visual_servo_course | MuJoCo **PBVS/IBVS** 仿真 | UR5e；无双臂 |
| geometric-peg-in-hole | https://github.com/geometricpeginhole/geometric-peg-in-hole | **双臂 PyBullet** 几何驱动装配 IL | 偏学习，非解析 PBVS |
| CFVS (论文) | https://arxiv.org/abs/2209.08864 | coarse OAKN → fine OPN 两阶段 | **无官方代码** |

**PCI 映射**：特权几何 ≈ PBVS；可直接对标 CORL 2020 双臂 VS 框架（我们用 sim 真值替视觉网络）。

---

## Phase B — 螺旋搜索 + 柔顺插入

| 仓库 | 链接 | 可抄什么 | 备注 |
|------|------|----------|------|
| **swri-robotics/ConnTact** | https://github.com/swri-robotics/ConnTact | **完整状态机**：找面 → **SpiralSearch** → 全轴 compliant 插入 | 工业级；ROS；PCI B1/B2 首选模板 |
| **franka-peg-in-hole** | https://github.com/avinash246813579/franka-peg-in-hole | SCAN→APPROACH→DESCEND→INSERT；**力反馈螺旋** | PyBullet；单臂但逻辑清晰 |
| **panda_impedance_control** | https://github.com/fida-121/panda_impedance_control | **Cartesian impedance** 1kHz、刚度/阻尼 | libfranka 真机；B2 admittance 参考 |
| peg-in-hole-gym | https://github.com/guodashun/peg-in-hole-gym | PyBullet peg 环境、多臂 | 需自写力控 |
| Isaac_Lab_UR5e_Peg_in_Hole | https://github.com/JonasFano/Isaac_Lab_UR5e_Peg_in_Hole | **Cartesian impedance** + PPO | Isaac Lab；单臂 |
| isl-org/0shot-object-insertion | https://github.com/isl-org/0shot-object-insertion | 触觉插入 RL + **ROS operational space** | ICRA 2023；接触丰富插入 |
| benjaminalt/dpse | https://github.com/benjaminalt/dpse | 螺旋/probe **搜索策略优化** | IROS 2022；长期调参用 |

**经典论文配套（无代码或弱代码）**：Insert-One IROS 2024、Lee RA-L 2022 双臂 dexterous — **未放官方 repo**。

---

## 双臂 + MuJoCo + OSC/Admittance

| 仓库 | 链接 | 可抄什么 | 备注 |
|------|------|----------|------|
| **ir-lab/irl_control** | https://github.com/ir-lab/irl_control | **双臂 UR5** OSC + **admittance**；NIST 插入适配器 | MuJoCo；`examples/admit_test` |
| dual-arm-peg-in-hole-mujoco-sim | https://github.com/RPM-lab-UMN/dual-arm-peg-in-hole-mujoco-sim | cap-the-bottle **双臂 peg** MuJoCo | dm_control OSC |
| Manipulator-Mujoco | https://github.com/jzw0025/Manipulator-Mujoco | 通用 **OperationalSpaceController** | 可挂双臂 env |
| Baxter-Peg-in-Hole | https://github.com/matsniklasson/Baxter-Peg-in-Hole | **双臂 hybrid 速度/力控** | 老项目；论文式参考 |
| ASEN-5254-Project | https://github.com/AllenDevaraj/ASEN-5254-Project | **双臂 Franka**：一臂持 receptacle、一臂 insert | TAMP+笛卡尔；分工像 PCI |

---

## PCI 推荐 clone 顺序

1. **ConnTact** — B1 螺旋 + B2 compliant 状态机（改接到 dexjoco wrist action）
2. **peg-in-hole-visual-servoing** — 双臂 coarse VS 框架（特权几何替网络）
3. **irl_control** — 双臂 admittance/OSC 在 MuJoCo 的实现
4. **franka-peg-in-hole** — 单臂螺旋+插入状态机（逻辑最短）

## 明确不用作主参考

- 本地 `echo_insert`、`priv_snap_insert`
- CFVS / Insert-One / Lee RA-L 2022 — 无可用官方实现
