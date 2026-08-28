# PCI 方案：特权 PBVS 接孔 + 双臂柔顺插入

> **Priv Compliant Insert** — 两阶段解析控制，替代 IGFP 学习型 attention 完成 insert 段。

---

## 1. 动机

| 路线 | handoff sim | 问题 |
|------|-------------|------|
| IGFP A0/A1/A2+ | 0/10 | 小 BC + gaze/fusion，从未 contact |
| π0.5 insert_ft | 10–12/30（172 ep FT） | 大 VLA，非力觉可解释；数据与 mix854 不同 |
| **PCI（本方案）** | 目标 ≥ hybrid PBVS 单段成功率 | 特权几何 + 力柔顺，可复现、可调试 |

**假设**：handoff 后失败主要在 **最后 1–2 cm 对准 + 接触插入**。  
Phase A 用 **hybrid_insert 粗 PBVS**（双臂相对、无 snap）；Phase B 用力柔顺精插。

**非目标（v1）**：
- ❌ `priv_snap_insert`（snap / O2H / pin / 手指粘物体）
- ❌ 学习型 policy / IGFP gaze
- ❌ 全任务从 grasp 开始

---

## 2. 总体流水线

```
zarr demo replay ──→ peg_lift_end（handoff 帧）
        │
        ▼
┌───────────────────────────────────────┐
│ Phase A：Hybrid PBVS 粗对准（ALIGN）    │
│ 复用：hybrid_insert.controller         │
│ 双臂相对 PBVS → standoff，无 snap/pin  │
│ 左 tray servo + 右 wrist PBVS          │
└───────────────────────────────────────┘
        │  gate: hybrid ALIGN 完成条件
        │  (pos_tol, angle_tol, standoff)
        ▼
┌───────────────────────────────────────┐
│ Phase B：PCI 双臂柔顺插入（新写）       │
│ 不用 hybrid INSERT 硬 PBVS             │
│ B1 力反馈横向搜索 → B2 轴向 compliant   │
│ B3 insert_ok + 松手                     │
└───────────────────────────────────────┘
        ▼
   insert_ok / 失败原因记录
```

---

## 3. Phase A — PBVS 粗对准（standoff，**禁止进孔**）

### 3.1 做法

- handoff 后 hybrid **ALIGN 模式**（`target_along = pbvs_standoff_m`），双臂相对 PBVS 收到孔前 standoff
- hybrid 一切到 **INSERT** 或 standoff gate 满足 → **立刻 deactivate**，不走 hybrid 硬插入
- 切换 Phase B 前：孔平面 **高斯噪声**（仅 setup，4mm 级 lateral）故意偏孔

### 3.2 退出条件 → Phase B

- `tip_dist ∈ [standoff ± tol]` 且 lat/axis 粗合格
- 或 hybrid 即将 INSERT（blocked）
- **不用** insert_ok / 进孔作为 Phase A 成功

### 3.3 Phase B 入口（传感器）

1. **贴平面**：恒力导纳纯轴向推，直到 `|Fz| ≥ contact_min`（无螺旋）
2. **螺旋搜孔**：PSFT + 力导纳 XY
3. **柔顺插入**：Fz 降判入孔 → admittance insert

---

## 4. Phase B — 双臂柔顺插入（文献方案）

> 详见 **`docs/LITERATURE.md`**。不用 echo_insert。

### 4.1 经典三段（UR3 dual-arm / Lee RA-L 2022）

1. **B1 螺旋搜索**：恒力贴面 + XY 部分螺旋（PSFT）；`F_z` 降 = 入孔
2. **B2 Admittance 插入**：孔轴低刚度推进，`F_axial`/`F_lat` 限幅；jam 则退
3. **B3 完成**：`insert_ok` → 松右手

### 4.2 双臂分工

| 臂 | Phase A | Phase B |
|----|---------|---------|
| 左 | tray PBVS 协同 | 高刚度持 tray |
| 右 | wrist PBVS 对准 | 螺旋搜索 + admittance 插入 |

### 4.3 力信号

- **Phase A**：特权几何（hybrid PBVS）
- **Phase B 控制**：腕 6D F/T + 四指指尖力 + 冻结 tool 系；**不用** tip/socket 真值
- **评测 only**：`AssemblyContactLabeler.insert_ok`

---

## 5. 软件结构

```
priv_compliant_insert/
├── docs/PLAN.md          ← 本文件
├── docs/REUSE.md
├── configs/
│   └── default.yaml      # 增益、阈值、超时
├── scripts/
│   ├── smoke_ep0.py      # P0：单 ep handoff→A→B
│   └── eval_batch.py     # P2：10/30 ep 批量 + summary.json
└── src/pci/
    ├── __init__.py
    ├── features.py       # 特权几何封装（调 hybrid_insert）
    ├── approach.py       # Phase A PBVSApproachController
    ├── compliant/
    │   ├── __init__.py
    │   ├── search.py     # B1 螺旋/微扰
    │   ├── insert.py     # B2/B3 力控推进 + release
    │   ├── fingers.py    # 四指力平衡柔顺（PCI 创新）
    │   └── task_frame.py # 冻结 tool 系（无 sim 真值）
    ├── pipeline.py       # A→B 状态机 + 诊断
    └── sim_runner.py     # 接 dexjoco handoff env
```

---

## 6. 实施阶段

### P0 — 脚手架 + 单 ep smoke ✅

- [x] P0-1：`configs/default.yaml` 默认增益
- [x] P0-2：`features.py` 封装 tip/socket/lat/ang/along
- [x] P0-3：`sim_runner.py` hybrid ALIGN-only + force handoff
- [x] P0-4：`scripts/smoke_ep0.py` — ep0 insert_ok ✅

### P1 — Phase B 柔顺插入 🔄

- [x] P1-1：`compliant/search.py` — ConnTact 螺旋 + `insert_along_hole_delta`
- [x] P1-2：`compliant/insert.py` — 轴向 compliant + jam retreat
- [x] P1-3：`pipeline.py` 串联 A→B
- [x] P1-4：`compliant/fingers.py` — 四指 admittance（文献无现成 Allegro insert 开源）
- [x] P1-5：Phase B 改 sensor-only（`task_frame` + `FingerForceLabeler`）
- [ ] **验收**：ep0–3 至少 1/4 insert_ok（ep0 ✅，ep1–3 待调参）

### P1/v2 — 真导纳 + jam 回搜 🔄

- [ ] 腕/指子控制器：真导纳律 + `prev_delta_xyz`/`dt` 阻尼项（并行 agent）
- [x] `pipeline.py`：insert jam（`retreat`）→ 抬脱后回 `COMPLIANT_SEARCH` 螺旋，非直接 DONE
- [x] `pipeline.py`：全程并行手指柔顺；release 时 finger delta 清零
- [x] `pipeline.py` → search/insert 可选传 `prev_delta_xyz`/`dt`（签名兼容，旧控制器不受影响）
- **v2 定义**：真导纳 + 指导纳 + jam 回搜；**验收仍 `insert_ok`**，Phase B **不用特权几何控**

### P2 — 批量 eval + 对比 ⬜

- [ ] P2-1：`eval_batch.py` 同 IGFP 10 ep（0–9）+ 协议字段对齐
- [ ] P2-2：输出 `eval_summary.json`（success、failure_reason、ever_contact）
- [ ] P2-3：与 IGFP 0/10、π0.5 handoff 30ep 表对比写进 README
- [ ] **验收**：成功率 > 0；失败分布可解释

### P3 — 鲁棒性 ⬜

- [ ] P3-1：peg_lost / tray_lost watchdog（复用 hybrid 阈值）
- [ ] P3-2：参数与 `HybridInsertConfig` 对齐文档

### P4 —（可选）退化到 public ⬜

- [ ] 用 ego + 粗略 hole/tip 估计替代真值（接 gaze 或 ARUco），仅 approach 段
- [ ] insert 段仍力控（不依赖特权）

---

## 7. 评测协议（与 IGFP 对齐）

| 项 | 值 |
|----|-----|
| 任务 | demo replay handoff → policy/control 段 |
| episodes | 先 0–9（10），再扩展 30 ep seed |
| 超时 | 50 s / 1500 policy steps |
| 成功 | `insert_ok` 稳定 |
| 记录 | peg_lost, tray_lost, alignment, ever_insert_contact |

---

## 8. 风险与对策

| 风险 | 对策 |
|------|------|
| Phase A 仍 peg_lost | 复用 priv_snap regrasp；减小 tip_step |
| Phase B jam | echo_insert stall + 小退 + 重搜索 |
| 仅 sim 特权 | P4 再谈 public；v1 先证 sim insert 可行 |
| 与 π0.5 比不公平 | 定位不同：PCI = 解析 last-mm；π0.5 = 端到端 learning |

---

## 9. 与 IGFP 关系

- IGFP **暂停** 新 fusion 训练；mix854 数据仍可给 PCI 录 force 日志
- PCI 若 P2 成功 → 证明 insert 段瓶颈在 **控制/柔顺**，不在 attention 架构
- 长期：PCI Phase B 力控轨迹可作 IGFP / π0.5 的 **residual Teacher**（可选，非 v1）

---

## 10. 立即下一步

1. 实现 `approach.py` 薄封装 `HybridInsertController` ALIGN
2. `smoke_ep0.py` 接 handoff env（参考 `eval_lerobot_demo_handoff_insert.py`）
3. ep0 跑通 ALIGN gate，再开 Phase B 柔顺
