# Hard seat lock (2026-09-04)

用户硬性条件：**禁止浮空搜孔；必须真正贴面搜孔。**

## 已落地

1. `docs/SUCCESS_STANDARD.md`：`min_contact_frac=0.80`、`early≥0.85`、`n≥30` 为成功硬门禁。
2. `sim_runner.py` 成功判定默认同上；浮空 → `force_hole_float_reject`；短螺旋 → `force_hole_degenerate_reject`。
3. 控制：
   - `path_reverse` 后强制 reseat
   - `pre-spiral SEAT`（平面冻结，只加压）
   - spiral 中 `|r|<seat` → 禁缩半径 + 加压 reseat
4. scheme_l3 主 config 已写上对应开关。

## Smoke（进行中）

- Track-B: `outputs/scheme_l3/seat_hard_smoke_B_ac_ep07/`
- Track-A/Type-A: `outputs/scheme_l3/seat_hard_smoke_A_typeA_ep02/`

过 early/cfrac 后再扩 ep1–10；旧论文 7/10 不得当贴面成功率。
