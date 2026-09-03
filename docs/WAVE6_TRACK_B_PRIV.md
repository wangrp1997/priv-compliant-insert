# Wave-6 Track-B Phase-0 priv (GMHQP residual)

> **合规特权诊断:** Deployable GMHQP **6/10**；余败 **2,3,5,7**；Oracle-v2 同集全进；Type-A ep2/5 曾进。  
> **科学问题:** 已知 tip→口平面轨迹下，非刚性抓取如何保贴面 + 轴≤25° + 真力入孔。  
> 命令: `PYTHONPATH=src python scripts/diag_spiral_priv.py …`  
> 门禁: `docs/METHOD_GATE_NO_PATCH.md` · RX: `docs/GMHQP_RESIDUAL_PRIV_RX.md`

## 1. Numbers (2026-09-03 re-run)

| ep | GMHQP planar / mouth / slip / tip_err_max / **bl_tip_err** | Type-A | Oracle | Class |
|----|-----------------------------------------------------------|--------|--------|-------|
| **02** | **11.2** / F / 24.5 / 19.7 / **3.5** | **enter** 4.4 | enter 4.5 | F1 + regress |
| **03** | **11.5** / F / 16.9 / 18.2 / **2.7** | fail 8.4 | enter 4.5 | F1 pure |
| **05** | **8.9** / F / 164 / 13.3 / **5.9** | **enter** 4.5 | enter 4.4 | F1/F2 + regress |
| **07** | **11.7** / F / 187 / 34.1 / **5.1** | fail 10.2 | enter 4.5 | F2→F1 residual |

全 `spiral_timeout`、`priv_mouth=False`。轴未成 F3 主因。

**关键签名:** `bl_tip_err`（控制 tip 任务误差）**远小于** `priv_planar_min` / `tip_spiral_err` → **代理 tip 已闭合路径，真 tip 未到口**。

Wave-5 Hybrid 用 proxy tip_err 做饱和切换 → 几乎不触发（err~3 mm ≪ sat 8 mm）→ smoke **0/4**。

## 2. One-sentence control problem

**控制问题:** 在 Escande seat≻upright≻path 与 Montana \(C^{+}\) tip 任务下，当 in-hand geom tip 与外禀接触 tip 不一致导致 \(e=\Pi(p_{\mathrm{proxy}}-p^*)\) 假闭合时，如何用可部署腕 F/T 估计外禀接触 tip 替换任务 tip（无 tip GT），使真平面误差进孔。

## 3. Search → REUSE | INVENT

| Candidate | Fit? | Status |
|-----------|------|--------|
| Escande⊕planar_C Hybrid | 饱和切换 | wave-5 **0/4 STOP** |
| CLEP alone (FT→腕 PD) | 接触估计对 | **0/4**（缺 \(C^{+}\)/Escande） |
| Kim TEC tip_obs→GMHQP | Track-A | full **4/10** ≤GMHQP STOP |
| CouplingEKF / ContactMotion C | 在线 \(C\) | Hybrid 已用；缺口在 tip 坐标非仅 \(C\) |
| **Doshi / Kim Active Extrinsic / Contact Occupancy FT tip** | \(r=(n\times\tau)/(n\cdot F)\) → \(\hat c\) | **REUSE as tip_obj into GMHQP** |
| FASR / OIGS / PHIG restack | — | METHOD_GATE 禁 |

**Verdict: REUSE compose** — CLEP/Doshi FT 接触 tip + GMHQP 律（`track_b_gmhqp_ftip`）。不 invent 平行发表缩写；不叠 FASR/OIGS/PHIG。

## 4. Gate

Fail-subset smoke **2,3,5,7**：enter-win≥1 vs GMHQP **或** planar&lt;5+mouth on ≥2 eps → else **STOP** 不调参。

### Wave-6 result

Out `S2_trackB_gmhqp_ftip_smoke`：**RATE 0/4**；gate **FAIL**（仅 ep02 达 planar&lt;5+mouth；enter-win=0）。  
见 `docs/TRACK_B_GMHQP_FTIP.md` §4。