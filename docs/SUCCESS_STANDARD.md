# Force-enter success standard (locked)

Success requires **all**:

1. Deployable force (or ConnTact height) hole enter
2. **Tip on tray/hole face during spiral** — **true seat search**, not float-over-mouth
3. Peg axis roughly upright at enter (`axis_err ≤ 25°`)

## Hard seat (user lock 2026-09-04)

浮空搜孔 **一律不算成功**。螺旋段必须真正贴面：

| Gate | Default | Meaning |
|------|---------|---------|
| `min_contact_frac` | **0.80** | 全程螺旋腕残差 ≥ `seat_contact_n` 的帧占比 |
| `min_early_contact_frac` | **0.85** | 前 `early_frames`（默认 60）贴面占比 |
| `min_spiral_frames` | **30** | 拒绝 DEGENERATE（n≈1 假成功） |
| `max_float_m` | 0.004 | tip 相对接触面抬起峰值 |
| `max_tip_rise_m` | 0.006 | along 抬升 |
| `max_along_m` | 0.100 | 入孔时 along 仍过高 → 拒 |
| `max_axis_err_deg` | 25 | 侧躺/歪轴拒 |

Reject explicitly:

- Float search / START_FLOAT（early_cfrac≈0，along0 离盘）
- Sideways / fallen peg（`force_hole_axis_reject`）
- Floating tip over hole with lat_min only（`force_hole_along_reject`）
- Degenerate spiral（`force_hole_degenerate_reject`）

Hard audit knobs (`configs/scheme_l3/S2_force_insert.yaml` + code defaults):

- `surface_planned_priv_enter_max_axis_err_deg: 25`
- `surface_planned_priv_enter_max_along_m: 0.100`
- `surface_planned_priv_enter_max_float_m: 0.004`
- `surface_planned_priv_enter_max_tip_rise_m: 0.006`
- `surface_planned_priv_enter_min_contact_frac: 0.80`
- `surface_planned_priv_enter_min_early_contact_frac: 0.85`
- `surface_planned_priv_enter_early_frames: 60`
- `surface_planned_priv_enter_min_spiral_frames: 30`

Control (not just report): `surface_pre_spiral_seat_enable` + `surface_spiral_require_seat` — 未贴面时只 reseat，禁止缩半径搜孔。

论文旁注可保留旧 `min_contact_frac: 0` 的 legacy RATE；**主表 / 演示 / 诚实 Rate 必须走本标准。**
