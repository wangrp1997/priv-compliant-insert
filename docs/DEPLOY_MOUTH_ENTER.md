# Deployable mouth force-enter (Agent-I)

> Closes oracle residual class after tip reaches mouth.  
> Success standard: `docs/SUCCESS_STANDARD.md` (do **not** loosen).  
> Base: Tip-Hybrid + planar_C (`S2_force_insert.yaml`).

## Scientific residual (oracle / MSAR)

Privileged tip→mouth lifts RATE 4→7/10 (`docs/ORACLE_TIP_SERVO.md`). Remaining fails:

| ep | Class | Notes |
|----|-------|-------|
| 03 | At mouth, **no force-hole** | `priv_planar_min≈4.4mm`, mouth-probe 320f, `|r|≈0.69` stuck |
| 07 | Force-hole + **along reject** | Height-drop “enter” with low seat contact → float |
| MSAR 03 | Same as 03 | Mouth reached, probe never seats into bore |

Root cause (deployable): mouth-probe treated soft seat residual (`|r|~0.7N`) as hard unload → **retracted** instead of pressing into bore; height-drop accepted float along.

## Deployable cues (no privileged hole pose)

Yield spiral when:

```
yield = (r_cmd ≤ ρ_mouth·1.15 ∧ ρ_tip ≤ ρ_mouth·1.5)
      ∨ (ρ_tip ≤ ρ_mouth ∧ ‖e_tip‖ < τ_near)
      ∨ (near_ring ∧ peak_resid − resid ≥ δ_dip ∧ peak ≥ seat)
```

- `ρ_tip` = planar ‖tip − spiral_center‖ (planner center, same Archimedean origin)
- `e_tip` = planar tip−waypoint tracking error
- Force cue = residual dip after seated peak (F/T), not hole GT

On yield: stop spiral advance → Tip-Hybrid **SEAT** + existing **mouth_probe** / force-hole detect.

## Mouth-probe ownership (strengthened)

When `surface_deploy_mouth_enter: true`:

1. Gate on planner `ρ_tip` / `r_cmd` (not priv lat alone).
2. XY hold toward **planner spiral_center** (not `_hole_axis_point_at_tip`).
3. At mouth + seat contact: **press into bore** (`seat_press_scale`); unload only on true jam (`jam_n ≳ 1.2N`).
4. Height-drop enter requires seat contact and `along ≤ max_along` (reject float-along / oracle ep07 class).

## Config / code

| Piece | Path |
|-------|------|
| Doc | `docs/DEPLOY_MOUTH_ENTER.md` |
| Cue | `deploy_mouth_yield_cue` in `src/pci/tip_tracking_baselines.py` |
| Hooks | `sim_runner` spiral yield + mouth_probe |
| Config | `configs/scheme_l3/S2_deploy_mouth.yaml` |

## Eval

```bash
cd /home/wangrenpeng/priv_compliant_insert
PYTHONPATH=src:/home/wangrenpeng/dexjoco \
  /home/wangrenpeng/miniconda3/envs/dexjoco/bin/python \
  scripts/parallel_force_enter_ep10.py \
  --config configs/scheme_l3/S2_deploy_mouth.yaml \
  --out-root outputs/scheme_l3/S2_mouth_smoke \
  --eps 1,3,4,5,6,8,9 --workers 4

PYTHONPATH=src python scripts/diag_spiral_priv.py \
  outputs/scheme_l3/S2_mouth_smoke/ep*/ep*_summary.json
```

Gate: keep v10 successes among `{1,4,5,6}` if tested; improve enter on mouth-reached fails (esp. ep03-class).

### Smoke result (Agent-I, 2026-09-03)

Out: `outputs/scheme_l3/S2_mouth_smoke/` → **RATE 3/7** (ep04,05,06).

- Keep among {1,4,5,6}: **4,5,6**; ep01 fail (plain `S2_force_insert` also fails ep01 here — shared tip-track regression).
- ep03/08/09: still `priv_planar_min` 8–14 mm (tip never at mouth) → mouth-probe ownership not exercised.
- Closing oracle ep03/07 residual still needs tip→mouth first.

## Spiral yield (default OFF)

`surface_deploy_mouth_yield_spiral: false` by default. Early yield on planner `ρ_tip`
fired while tip was still 10–40 mm from the true mouth (frozen spiral_center drift)
and regressed v10 successes. Mouth-probe ownership still runs on the existing
`near_mouth` / lat gate; SEAT press + jam unload + upright/seat height-drop gates remain.

## Explicit non-goals

- No privileged tip/hole in control loop.
- Do not loosen SUCCESS_STANDARD.
- Prefer enabling/tuning existing mouth_probe / force_hole paths over new FSM soup.
