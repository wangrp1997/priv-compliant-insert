# Literature Baselines & Comparison Map

> 网络/教材检索汇总（2026-09-02）。用于 Related Work 与 **额外 citation baseline**（不必全部实现）。

---

## A. 必须实现的 sim 基线（已有）

| ID | 方法 | 文献 |
|----|------|------|
| S3 | ConnTact rigid-EE spiral | SwRI ConnTact; Watson et al. AR 2020 |
| S0 | Blind force seat | — (negative) |
| S1/S2 | Frozen / live offset | tip servo 消融 |
| S6 | Scalar α | isotropic C ablation |
| S8 | Planar C + invariants | proposed V1 |

---

## B. 应用层 spiral / compliance 对照（cite，可不跑）

| 方法 | 作者/年份 | 与我们的关系 |
|------|-----------|--------------|
| Archimedean spiral search | Chhatpar & Branicky IROS 2001 | 经典几何 baseline |
| Whitney quasi-static assembly | ASME 1982 | RCC / 四阶段 |
| PSFT | Park et al. RA-L 2020 | 柔顺 spiral，刚性 EE |
| Choi kinesthetic + spiral | RA-L 2021 | **最接近任务**：多指 + spiral 前 kinesthetic |
| Van Wyk force-based dex hand PiH | TRO 2018 | 多指力控 PiH |
| ∂PSE search optimization | Alt IROS 2022 | 搜孔策略优化，非 coupling |

---

## C. 估计层对照（方法论引用 / 未来 baseline）

| 方法 | 类型 | 文献 | 可对比点 |
|------|------|------|----------|
| Sim-Tact extrinsic FG | iSAM | Kim ICRA 2023 | 外接触估计 |
| TEXterity | FG+Viterbi | Bronars ICRA 2024 | in-hand + extrinsic |
| Active extrinsic PiH | iSAM | Kim ICRA 2022 | pivot + insertion |
| TacGraph | iSAM2 | Van der Merwe RA-L 2026 | in-hand + extrinsic |
| EKF joint torque | EKF | Pfanne IROS 2017 | **无触觉** kinesthetic |
| diff-EKF sliding | EKF | Piga Frontiers 2021 | slip 下跟踪 |
| MidasTouch | PF | Suresh CoRL 2022 | 滑动触觉 |
| GelSLAM session reset | pose graph | Huang arXiv 2025 | slip → reset |
| Tactile-RL insertion | RL | Dong ICRA 2021 | 触觉 vs 腕 F/T ablation |

---

## D. 控制层对照（cite / 可选 sim baseline）

| 方法 | 类型 | 文献 |
|------|------|------|
| Hybrid position/force | 经典 | Raibert & Craig 1981 |
| Hogan impedance | 经典 | Hogan ASME 1985 |
| HQP task priority | QP | Escande IJRR 2014 |
| Contact-implicit MPC | MPC | Kim IROS 2024; IMPACT arXiv 2026 |
| STOCS contact-implicit TO | TO | RSS 2023 |
| LeTac-MPC | tactile MPC | Xu TRO 2024 |
| bgf trajectory modulation | MPC+ACTP | Nazari Nature MI 2025 |

---

## E. 教材（方法论脚注）

| 主题 | 书目 | 章节 |
|------|------|------|
| 接触运动学 | Lynch & Park, *Modern Robotics* | Ch.12 |
| 力/阻抗控制 | Craig, *Introduction to Robotics* | Ch.11 |
| 阻抗算子 | Spong, Hutchinson & Vidyasagar | Ch.10 |
| 非线性控制背景 | Slotine & Li | Ch.6–9（无导纳专章） |
| 力控专著 | Siciliano & Villani, *Robot Force Control* | 全书 |

---

## F. Privileged sim 披露（写法参考）

| 论文 | 做法 |
|------|------|
| Learning by Cheating (CoRL 2019) | GT 仅 teacher |
| RMA (CoRL 2022) | oracle stage → deployable stage |
| PTLD / SBLR (2026) | X_priv vs X_sensor 分表 |
| CoRMA (2026) | sim-only Z，部署移除 |
| Tactile-RL (ICRA 2021) | 同协议 ablate 传感 |

详见 `docs/PRIVILEGED_SENSING_DISCLOSURE.md`。

---

## G. 我们 vs 文献 gap（写 Introduction）

| 文献主流 | 我们 |
|----------|------|
| 刚性 EE spiral | tip-referenced + C(q) |
| 已知环境 regrasp | **盲** tray spiral search |
| 全 FG/MPC 重 | 轻量 EKF + 1-step QP（V2 目标） |
| 平行夹爪 | Allegro handoff |
