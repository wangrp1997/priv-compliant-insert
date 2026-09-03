# ConnTact faithful reproduction protocol

> Paper baseline only. Not an invented PCI spiral.

## Source of truth

| Item | Path |
|------|------|
| Code | `refs/ConnTact/src/conntact/spiral_search.py` `SpiralToFindHole` |
| Params | `refs/ConnTact/config/connfig_peg_10mm.yaml` `task.spiral_params` |
| License | Apache-2.0 |
| Our port | `src/pci/conntact_repro.py` |
| Config | `configs/scheme_l3/S3_conntact_faithful.yaml` |
| Eval | `scripts/parallel_force_enter_ep10.py --config .../S3_conntact_faithful.yaml` |

## Locked to match reference

1. **Time Archimedean spiral**  
   `amp = min_amplitude + safe_clearance * mod(2π f t, max_cycles)`  
   `x = amp cos(2π f t)`, `y = amp sin(2π f t)`  
   Defaults: `f=0.15 Hz`, `min_amplitude=0.002 m`, `max_cycles=62.83`, `safe_clearance=0.02/100 m` (as in ConnTact `__init__`).

2. **Center** = known hole XY (ConnTact TF `target_hole_position`).  
   Sim: socket axis ∩ tip plane (`surface_tip_spiral_center: socket`).

3. **Pose command** = spiral XY + **current Z** (seeking force owns normal).  
   Sim: `conntact_tcp` baseline tracks **wrist/TCP** to planar target (not tip-error servo).

4. **Hole found** = `z ≤ surface_height − 0.0004 m` (`exit_conditions`).  
   Sim: tip world-Z vs Z at spiral start; sets `force_planned_hole_entered`.

5. **Seeking** = constant into-surface wrench (ConnTact `[0,0,-7]` N).  
   Sim: position-admittance seeking around `surface_planned_spiral_f_des_n` (force **scale** differs; see adaptations).

## Explicit adaptations (must disclose in paper)

| ConnTact (UR / NIST board) | Our DexJoCo handoff |
|----------------------------|---------------------|
| Rigid peg–TCP | Allegro **non-rigid** grasp (this is the stress test) |
| Seeking wrench −7 N | Soft sim contact → ~0.025 N residual band (same *role*, different scale) |
| Absolute world XY | Tray-plane tangent basis `(t1,t2)` if tray tilted |
| Full FSM FindSurface → Insert | Phase-A PBVS already seated; we reproduce **SpiralToFindHole only** |
| Success = height drop | Paper table still applies `SUCCESS_STANDARD` (height/force enter + seat + axis≤25°) |

## Not this baseline

- Old `S3_conntact_spiral.yaml` with `surface_tip_search_mode: spiral` + tip-centered **inward** geometric spiral → **not** faithful; keep only as legacy ablation.
- Tip-Hybrid / MSAR / STAR / planar_C / S9 → **ours / invent**, not ConnTact.

## Numeric lock

```bash
PYTHONPATH=src python -c "from pci.conntact_repro import assert_matches_ref_sample; assert_matches_ref_sample(); print('OK')"
```

## Eval command

```bash
PYTHONPATH=src:$DEXJOCO... python scripts/parallel_force_enter_ep10.py \
  --config configs/scheme_l3/S3_conntact_faithful.yaml \
  --eps 1-10 --workers 4 \
  --out-root outputs/scheme_l3/S3_conntact_faithful_ep1_10
```

Report: SUCCESS_STANDARD rate + per-ep `spiral_reason` / lat_min / axis / priv_planar (diag only).

## Eval result (SUCCESS_STANDARD)

- RATE: **0/10** on `outputs/scheme_l3/S3_conntact_faithful_ep1_10/`
- Many `force_hole=1` from height-drop are **false enters** (along still 100–170 mm) → rejected by seat/along/axis audit.
- Paper claim: faithful ConnTact under non-rigid handoff grasp does not achieve tip-on-surface upright force-enter.
