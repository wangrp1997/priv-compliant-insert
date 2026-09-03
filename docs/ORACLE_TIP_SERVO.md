# Privileged tip→mouth servo oracle (Agent-D)

> **NOT DEPLOYABLE.** Diagnostic upper bound only.  
> Disclosure: `docs/PRIVILEGED_SENSING_DISCLOSURE.md`  
> Success standard: `docs/SUCCESS_STANDARD.md` (do not loosen)

## Scientific question

§1 bottleneck: tip never reaches mouth (`priv_planar_min` 8–14 mm on fails).

Oracle asks: **if tip planar position and hole are known, and we PD tip→mouth (assume local wrist↔tip ≈ I), what enter rate remains under SUCCESS_STANDARD?**

| Oracle result | Implication for deployable work |
|---------------|----------------------------------|
| Rate ≫ baseline (4/10) | Tracking/estimation to mouth is the bottleneck |
| Rate ≈ baseline / still fails | Physics, grasp slip, or force-enter beyond planar tip reach |

## Protocol (v1)

- Config: `configs/scheme_l3/S2_oracle_tip.yaml`
- Mode: `surface_tip_baseline: priv_oracle_tip_servo`
- Target: live privileged hole-axis mouth (`_hole_axis_point_at_tip` + `feat.hole_axis`), same geometry as `priv_planar` / `feat.lateral_m` (not frozen `spiral_center`)
- Mapping: identity tip PD `Δw = clip(k · P_n(p_mouth − p_tip))` (no planar_C / EKF)
- Tip-Hybrid SEAT→UPRIGHT→SPIRAL kept; oracle mouth PD active in upright+spiral
- `r_cmd` for force-hole radius gate = tip planar lat (not frozen spiral index)
- Meta flags: `tip_oracle_enable`, `privileged_oracle_tip_servo`

## Protocol (v2) — mouth press + slip reseat

Config: `configs/scheme_l3/S2_oracle_tip_v2.yaml` (still **NOT deployable**).

Same tip→mouth PD as v1, plus flags `surface_tip_oracle_*` only:

| Flag | Role |
|------|------|
| `surface_tip_oracle_mouth_press` | When tip lat ≲ mouth, stronger axial press / mouth-probe into hole (ep03) |
| `surface_tip_oracle_mouth_hard_unload_n` | Raise unload bar so soft seat resid does not retract at mouth |
| `surface_tip_oracle_slip_reseat` | On large *new* slip: freeze planar, SEAT reseat + tighten grasp, then resume tip→mouth (ep07/08) |
| `surface_tip_oracle_hold_enter_seat` | Do not stop on force-hole cue if along float or axis>25° (SUCCESS_STANDARD) |

Tip-Hybrid SEAT/UPRIGHT still owns priority over mouth PD during reseat.

```bash
cd /home/wangrenpeng/priv_compliant_insert
PYTHONPATH=src:/home/wangrenpeng/dexjoco \
  /home/wangrenpeng/miniconda3/envs/dexjoco/bin/python \
  scripts/parallel_force_enter_ep10.py \
  --config configs/scheme_l3/S2_oracle_tip_v2.yaml \
  --out-root outputs/scheme_l3/S2_oracle_tip_v2_ep1_10 \
  --eps 1-10 --workers 3

PYTHONPATH=src python scripts/diag_spiral_priv.py \
  outputs/scheme_l3/S2_oracle_tip_v2_ep1_10/ep*/ep*_summary.json
```

## Eval result (Agent-D, 2026-09-02) — v1

Out: `outputs/scheme_l3/S2_oracle_tip_ep1_10/`

| | Deployable baseline Tip-Hybrid+planar_C (v10) | **Privileged oracle tip→mouth (v1)** |
|--|--|--|
| RATE | **4/10** | **7/10** (diagnostic only) |
| Success eps | 1,4,5,6 | 1,2,4,5,6,9,10 |
| Fail eps | 2,3,7,8,9,10 | 3,7,8 |

Fail notes (SUCCESS_STANDARD not loosened):
- ep03: mouth reachable (`priv_planar_min≈4.4mm`) but no force-hole
- ep07: force-hole but `force_hole_along_reject` (along float)
- ep08: tip saturates `priv_planar_min≈5.7mm`, tilt/tray

**Implication:** solving tip planar→mouth (privileged) lifts rate 4→7/10; remaining fails are force-seat / along / residual tip lag under slip — deployable must still solve tip tracking **and** mouth force-enter under seat+axis.

## Eval result (Agent-D) — v2

Out: `outputs/scheme_l3/S2_oracle_tip_v2_ep1_10/`

| | Deployable Tip-Hybrid+planar_C (v10) | Oracle v1 | **Oracle v2 (diagnostic)** |
|--|--|--|--|
| RATE | **4/10** | **7/10** | **9/10** (NOT deployable) |
| Success eps | 1,4,5,6 | 1,2,4,5,6,9,10 | 1,2,3,5,6,7,8,9,10 |
| Fail eps | 2,3,7,8,9,10 | 3,7,8 | **4** |

v1→v2 fail-subset: **ep03/07/08 fixed**; new fail **ep04** (axis≈43° at mouth, no force-hole / surface_press).

**Implication:** privileged tip→mouth + mouth press + slip reseat + seat/axis hold shows **>7/10 is possible** under SUCCESS_STANDARD (diagnostic 9/10). Remaining ep04 is upright-at-mouth under slip — deployable still needs tip tracking **and** seat+axis force-enter; do **not** put 9/10 in the deployable table.
