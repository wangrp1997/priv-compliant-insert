# Force Gate 诊断：ep05 / ep08 × tip_track_live / planar_C

> 数据来源：`outputs/scheme_l3/` 下四份 `*_summary.json` 的 `force_trace`，仅取 `phase == planned_spiral` 段。  
> 生成时间：2026-09-02

---

## 1. 指标汇总（planned_spiral 段）

| ep | 方法 | 终止原因 | 帧数 | \|resid\|/f_des 均值 | \|resid\|/f_des 峰值 | grasp_slip (mm) | priv_planar_min (mm) | tip_spiral_err 均值 (mm) | tip_spiral_err 峰值 (mm) | force_scale 均值 | 门控激活占比 |
|----|------|----------|------|----------------------|----------------------|-----------------|----------------------|-------------------------|-------------------------|------------------|-------------|
| 05 | tip_track_live | spiral_timeout | 1147 | **1.17** | 4.09 | **13.5** | **0.95** | 7.65 | 13.1 | **0.92** | **79%** |
| 05 | planar_C | near_mouth | 263 | **55.0** | 87.2 | 31.9 | 1.47 | 8.47 | 12.8 | 1.00 | 0%† |
| 08 | tip_track_live | near_mouth | 1426 | 2.30 | 17.1 | **67.4** | 1.42 | 14.33 | **40.5** | 1.00 | 0%† |
| 08 | planar_C | spiral_tilt_or_tray | 403 | **80.2** | 97.6 | 46.5 | 1.51 | 12.27 | 22.4 | 1.00 | 0%† |

† planar_C 与 ep08 tip_track_live 的 summary 未写入逐帧 `bl_force_scale`（旧 run）；\|resid\|/f_des 仍可直接反映腕部过载。

补充对照：

| ep | 方法 | lat_min (mm) | lat_min − priv_planar_min (mm) | tip_xy 峰值 (mm) | priv 到口 | priv 入孔 |
|----|------|-------------|--------------------------------|-----------------|-----------|-----------|
| 05 | tip_track_live | 7.97 | +7.0 | 25.7 | ✓ | ✗ |
| 05 | planar_C | 1.47 | +0.01 | 7.1 | ✓ | ✓ |
| 08 | tip_track_live | 2.97 | +1.5 | **62.5** | ✓ | ✗ |
| 08 | planar_C | 22.67 | **+21.2** | 18.3 | ✗ | ✗ |

配置背景（`S2_tip_track_live.yaml` / `S8_planar_C.yaml`）：`handoff_left_locked=True`（交接后左抓冻结），`f_des≈0.025–0.04 N`，`planar_gate_n=0.048 N`，`priv_gate_grasp_slip_m=0.1 m`（100 mm，诊断用宽松阈值）。

---

## 2. 根因链：冻结 grasp → 腕部高力 → slip

```text
交接 grasp 冻结 (left FROZEN, scale≈1.25)
        ↓
planned_spiral：腕部开环追 spiral 目标，手指指令不再收紧
        ↓
planar_C：C⁻¹ 放大平面误差 → |resid_r| ≫ f_des（55–80×）
tip_track_live（无门控时）：追 tip 误差，长时 spiral 仍积累 slip
        ↓
腕部推力经非刚性 grasp 传到 peg → peg-in-hand 相对 wrist 位移
        ↓
grasp_slip_peak ↑（ep08 达 67 mm）
        ↓
tip 与 wrist 解耦 → tip_spiral_err ↑（ep08 live 峰值 40 mm，tip_xy 62 mm）
        ↓
lat 无法贴近 priv_planar_min（ep08 planar_C 差 21 mm）→ 终止 near_mouth / tilt / timeout
```

### 2.1 ep05：同场景对比

- **tip_track_live（力门控生效）**：\|resid\|/f_des ≈ 1.2，79% 帧 `force_scale < 1`（最低 0.0096），spiral 内 slip 仅 **13.5 mm**；腕力被压住，但 spiral 跑满 1147 帧仍 **timeout**（tip_xy 25.7 mm，未入孔）。
- **planar_C（无门控记录、高 resid）**：\|resid\|/f_des ≈ **55×**，峰值 2.18 N；263 帧即 **near_mouth**，slip **31.9 mm**（≈2.4× live）。C 矩阵一步反演把腕部“顶死”，Compliance 无 unload。

→ 同 ep 下，**冻结 grasp + 高 resid/f_des 与 slip 单调对应**；力门控将 slip 减半量级，但未解决 timeout。

### 2.2 ep08：几何更难 + slip 雪崩

- 初始 lateral ≈ 23 mm（ep05 为 ≈14 mm），spiral 半径更大，搜索更长（1426 帧）。
- **tip_track_live**：\|resid\|/f_des 均值仅 2.3，但 **无 force_scale 衰减**（该 run 旧日志），slip 达 **67 mm**，tip_spiral_err 峰值 **40 mm**——典型“腕力不算极端、但 grasp 已滑脱、tip 追不上 spiral”。
- **planar_C**：\|resid\|/f_des ≈ **80×**，resid 峰值 2.44 N；lat_min 22.7 mm 而 priv_planar_min 1.5 mm（**gap 21 mm**），说明 C 估计/反演在 slip 后完全失效，**spiral_tilt_or_tray** 终止。

### 2.3 与 METHODOLOGY_V2 §2 一致

`spiral_force_gate` 设计意图：冻结 grasp 下，当 \|F_r\| > F_gate 或 slip > τ 时，缩小 planar step 并法向 unload。  
ep05 tip_track_live 已验证该路径（force_scale 均值 0.92，slip 13.5 mm）；planar_C 与 ep08 旧 run 仍表现为 **高 resid/f_des + 高 slip**，符合“未有效门控 → 怼滑”假设。

---

## 3. 失败模式分型

| 类型 | 代表 | 特征 |
|------|------|------|
| A. 腕部过载型 | ep05/08 planar_C | resid/f_des ≫ 1（50–80×），短 spiral，slip 30–47 mm |
| B. 长时 slip 累积型 | ep08 tip_track_live | resid/f_des 中等（2×），spiral 极长，slip 67 mm，tip_xy 62 mm |
| C. 门控有效但仍失败 | ep05 tip_track_live | resid/f_des ≈ 1，slip 13.5 mm，spiral_timeout（搜索未完成） |

---

## 4. 论文主表推荐指标

建议在 Baseline 表（或 Diagnostic 子表）增加以下三列，直接对应本根因链：

### 4.1 `grasp_slip_mm`

- **定义**：`surface_meta.grasp_slip_peak_m × 1000`（spiral 前 relatch 后累积 peg-in-hand 滑移）。
- **理由**：失败主因是 grasp 内 slip，而非 purely 平面误差；ep08 live 67 mm vs ep05 live 13.5 mm 区分度强。
- **报告**：median ↓ 优于 mean（ep08 离群大）；可附 p90。

### 4.2 `force_scale`

- **定义**：planned_spiral 段 `bl_force_scale` 均值（或 gated 帧占比）。
- **理由**：力门控是否生效的一标量；ep05 live 均值 0.92 / 79% gated vs planar_C 1.0 直接解释 slip 差异。
- **报告**：mean ± 可附 `frac(gated)`；与 \|resid\|/f_des 列互证。

### 4.3 `priv_planar_min`

- **定义**：planned_spiral 段 privileged tip 平面距离下界（mm）。
- **理由**：oracle 可达最优 lateral；`lat_min − priv_planar_min` 量化“非刚性 + slip 导致的不可达 gap”（ep08 planar_C：**21 mm**）。
- **报告**：median ↓；与 enter rate 联读。

可选辅助列（Diagnostic，不进主表）：

- `resid_fdes_ratio_peak`：腕部过载瞬时程度；
- `tip_spiral_err_max_mm`：spiral 跟踪质量（slip 后恶化）。

---

## 5. 结论与下一步

1. **根因确认**：冻结 grasp 下腕部追 spiral → resid 远超 f_des（planar_C 尤甚）→ peg-in-hand slip → tip 解耦 → 失败。
2. **力门控有效但未充分**：ep05 tip_track_live 已将 slip 压至 13.5 mm，需在所有 baseline（含 planar_C）统一记录 `force_scale` 并确保 gate 触发。
3. **planar_C 需约束 QP + gate**：裸 C⁻¹ 在 slip 后 lat−priv gap 爆炸（ep08: 21 mm），应配合 EKF-QP（S9）或强制 overload unload。
4. **priv_gate 注记**：当前 `_priv_tip_spiral_gate` 的 force 统计不含 `planned_spiral` phase，导致 `force_ok=False (nan)`；与物理失败无关，评估脚本应改用 planned_spiral 段或修复 phase 过滤。

---

## 6. 复现命令

```bash
python3 scripts/diag_spiral_priv.py \
  outputs/scheme_l3/ep05/tip_track_live/ep05_summary.json \
  outputs/scheme_l3/ep05/planar_C/ep05_summary.json \
  outputs/scheme_l3/ep08/tip_track_live/ep08_summary.json \
  outputs/scheme_l3/ep08/planar_C/ep08_summary.json
```

力门控实现：`src/pci/tip_tracking_baselines.py` → `spiral_force_gate()` / `baseline_hold_r()`。
