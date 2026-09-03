# Deployable Type A: track saturate (no privileged mouth attractor)

> Agent-I · deployable invent for v10 fails **ep2,3,8** (`priv_planar` 8–14 mm).  
> Does **not** loosen `SUCCESS_STANDARD.md`.  
> Companion theory: `TIP_TRACK_THEORY_NEXT.md` (Tip-STAR).

---

## 0. Compliance header

- **特权诊断:** Type A = tip 平面跟踪饱和；`priv_planar_min` 停在 8–14 mm，`priv_mouth=False`，`lat≡priv`，轴多数已 <25°。
- **科学问题:** 非刚性抓取下，使 tip 在接触平面上跟踪到孔口邻域（≲4.5 mm），同时贴面 + 轴直立，再力入孔。

诚实基线：Tip-Hybrid+planar_C **4/10**（`S2_force_enter_ep1_10v10`）。  
先验烟雾：`S2_star_smoke_23489` / `S2_ours_v11_smoke_23489` 均为 **1/5**（仅保 ep04）。

---

## 1. Why track saturates (inward spiral)

S2 螺旋是 **inward**（`r_cmd` 从大缩到口）。

| 量 | 健康跟踪 | Type A 饱和 |
|----|----------|-------------|
| `r_cmd` | ≈ `ρ_tip` | 规划已缩到口附近 |
| `ρ_tip` | 跟着 `r_cmd` 内收 | **卡在 8–14 mm** |
| `e_tip` | 小 | 大（追不上当前环） |

旧 STAR 门（按 outward 写的）：

```
sat = (‖e_tip‖ > τ) ∧ (r_cmd > ρ_tip + δ)   # 仅“命令跑在 tip 外面”
```

内向时 tip 在外、`r_cmd` 在内 → `r_cmd > ρ_tip` **恒假** → **`LOCAL_RECOVER` 帧数 = 0**（STAR/v11 smoke 实测）。

次要坑：

1. **过早 `yield_mouth`:** 只看 `r_cmd ≤ mouth`，tip 仍在 8–14 mm 就冻螺旋、让位 probe → 永远收不进。
2. **sticky `slip_peak` → 永久 `GRASP_RESTORE`:** ep02 上 5k+ restore、0 recover（Type B 闩锁另线；Type A 需恢复后能进 LOCAL_RECOVER）。
3. **平面 recover 无 upright 优先:** STAR smoke ep02 口到了但轴 ~50° → axis reject。

禁止可部署路径：特权 `e_mouth` PD（MSAR）；那会搞掉 ep04。

---

## 2. Deployable recover (no hole attractor)

### 2.1 Gate（方向感知）

```
radial_lag =
  outward: r_cmd > ρ_tip + δ
  inward:  ρ_tip > r_cmd + δ

sat = (‖e_tip‖ > τ_e) ∧ radial_lag ∧ seat_ok ∧ mode==SPIRAL
track_fail ← sat streak ≥ N_sat
```

近口健康：`|r_cmd − ρ_tip| ≲ δ` → STAR idle → **保 ep04**（反 MSAR 回归）。

### 2.2 LOCAL_RECOVER

```
ṙ ← 0   # freeze planned index / radius advance

# inward (S2): tip-azimuth radial pull + ratchet ρ toward planner mouth radius
r_tgt = min(r_cmd, max(mouth, ρ_tip − Δshrink))
p_local = c + r_tgt · unit(Π(tip − c))

# outward: pin radius to tip ring, keep spiral phase
p_local = c + ρ_tip · unit(Π(p* − c))
```

只用 **planner center / r_cmd / tip / mouth 半径常数**，不用孔位特权吸引。  
棘轮防止 `ṙ=0` 时 tip 卡在外环（烟雾 ep08：1.6k recover 帧仍 ~10 mm）。

### 2.3 Upright priority

```
if track_fail ∧ axis > soft:
  mode ← UPRIGHT; star ← yield_upright; ṙ ← 0
  # 禁止此时 planar local_recover
elif track_fail:
  star ← local_recover; target ← p_local; full tip-pivot upright scale
```

入孔：`hold_enter_until_upright` 用 **soft 轴阈值（默认 16°）+ confirm streak**，严于审计 25°（防 ep02 口到了再倾倒）。

### 2.4 Yield mouth（修正）

仅当 **tip 已近口**（`ρ_tip ≤ 1.5·mouth`）且 **`‖e_tip‖` 小** 才 yield；`r_cmd` 单独缩到口不够（防 ep03 假 yield）。

---

## 3. Expected map vs v10 Type A

| Ep | v10 | 期望 |
|----|-----|------|
| 2,3,8 | planar 8–14 mm timeout/tilt | LOCAL_RECOVER 真正开火 → planar↓ → 可能 mouth+enter |
| 4 | success | STAR idle / 不口吸引 → **保持** |

Gate（本轨烟雾 `eps 2,3,4,8`）：**ep04 仍成功** 且 **{2,3,8} 至少一个成功** 才全量 1–10。

### Smoke `S2_typeA_smoke` (2026-09-03) — **2/4；gate PASS**

| Ep | enter | vs v10 | note |
|----|-------|--------|------|
| 4 | **1** | kept | STAR idle / no mouth attractor |
| 2 | **1** | **win** | was Type A fail; upright hold 防轴拒 |
| 3 | 0 | still | planar~8.4 mm |
| 8 | 0 | partial | planar 13.7→5.9 mm；tilt abort |

→ 允许开全量 `ep1–10`（本笔记不自动宣称 >4/10）。

---

## 4. Landing / file touches (minimize agent conflict)

| File | Touch |
|------|-------|
| `docs/DEPLOY_TYPE_A_TRACK.md` | 本笔记 |
| `src/pci/tip_tracking_baselines.py` | `tip_star_radial_lag` / saturate / local_target / yield_mouth（方向感知） |
| `src/pci/sim_runner.py` | 传入 `spiral_dir`；`yield_upright` → `mode=upright` |
| `configs/scheme_l3/S2_deploy_typeA.yaml` | **本轨专用 config**（基：`S2_ours_v11`） |

不改 SUCCESS_STANDARD；不启 MSAR hole attractor；Type B 闩锁与其它代理共享 `tip_slip_*` 时只读+方向参数。
