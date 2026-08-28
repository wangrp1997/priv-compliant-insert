# 参考仓库阅读笔记（refs/）

Clone 命令：`nros-proxy-on && git clone --depth 1 ... refs/<name>`

## ConnTact (`swri-robotics/ConnTact`)

- **状态机**：FindSurface → SpiralToFindHole → FindSurfaceFullCompliant → Exit
- **B1 螺旋**（`spiral_search.py`）：恒力 `seeking_force=[0,0,-7]` + 时间参数化 Archimedean `x=amp*cos(2πft), y=amp*sin(2πft)`
- **入孔判据**：`z <= surface_height - 0.4mm`（ConnTact 用位姿，PCI 改 F/T + along）
- **B2 插入**：`FindSurfaceFullCompliant`，`comply_axes=[1,1,1]` 全轴柔顺 + 向下力

## franka-peg-in-hole (`avinash246813579/franka-peg-in-hole`)

- **状态机**：SCAN → APPROACH → DESCEND → INSERT
- **螺旋**：`r = R_BASE + R_GROWTH * spiral_step`，`theta = OMEGA * step`；jam 时 `F_z > 2N` 且高于孔顶 → 抬 z + 继续螺旋
- **慢速插入**：`INSERT_Z_RAMP_PER_STEP = 0.5mm/step`

## irl_control (`ir-lab/irl_control`)

- **双臂 admittance**（`osc.py`）：`u_all -= J.T @ Mx @ (u_task + ext_f)`，`ext_f` 来自腕力
- **分工**：左臂 target 位控 + 外力反弹；PCI 左臂 frozen，仅右臂 admittance

## peg-in-hole-visual-servoing (CORL 2020)

- **双臂 coarse**：`peg_robot` + `aux_robots` 持 receptacle；PCI Phase A 用 hybrid 双臂 PBVS 替代 VS 网络

## Phase B 传感器边界

- **允许**：腕部 6D F/T、四指指尖接触力（12d）、Phase A 结束时冻结的 **腕/tool 坐标系**
- **禁止**：tip/socket/hole 真值、along/lat 特权几何参与控制
- **评测 only**：`insert_ok` 接触标签仅写 summary，release 也可由 Fz 坐底触发

## 手指柔顺（PCI 创新）

文献有 FSMJIC / DLR Hand 单指关节阻抗，**未见 Allegro 双臂 insert 四指平衡柔顺开源** → `compliant/fingers.py`：
- 高载指放松、低载指略收紧，插入期保持 peg 在握持中


| 参考 | PCI 模块 |
|------|----------|
| ConnTact SpiralToFindHole | `compliant/search.py` |
| ConnTact FindSurfaceFullCompliant + irl admittance | `compliant/insert.py` |
| hybrid_insert ALIGN | `sim_runner.py` Phase A |
| franka jam/lift | `insert.py` stall + retreat |
