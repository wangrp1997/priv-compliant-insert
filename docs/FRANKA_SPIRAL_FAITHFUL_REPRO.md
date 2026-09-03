# Franka jam-spiral faithful reproduction protocol

> Paper baseline only. Not an invented PCI spiral.

## Source of truth

| Item | Path |
|------|------|
| Code | `refs/franka-peg-in-hole/scripted_expert_peg_v1.py` `ScriptedExpertPegV1` |
| Events | `refs/franka-peg-in-hole/peg_hole_events.py` (hole geometry context) |
| README | `refs/franka-peg-in-hole/README.md` |
| License | MIT |
| Our port | `src/pci/franka_spiral_repro.py` |
| Config | `configs/scheme_l3/S3_franka_spiral_faithful.yaml` |
| Eval | `scripts/parallel_force_enter_ep10.py --config .../S3_franka_spiral_faithful.yaml` |

## Locked to match reference

1. **Jam predicate**  
   `jammed = (upward_force > FORCE_THRESHOLD) AND (peg still above hole top)`  
   Ref: `FORCE_THRESHOLD_N = 2.0` (sim scale adapted; see below).

2. **Spiral growth (only while jammed; never resets mid-search)**  
   `spiral_step += 1` iff jammed  
   `r = min(R_BASE + R_GROWTH * spiral_step, R_MAX)`  
   `theta = OMEGA * spiral_step`  
   Defaults: `R_BASE=R_GROWTH=0.0002 m`, `R_MAX=0.015 m`, `OMEGA=0.4 rad/step`.

3. **Apply XY once ever jammed** (`spiral_step > 0`)  
   `target_xy = hole_xy + r (cos θ, sin θ)` even after force clears (preserves search progress).

4. **Lift-and-search while jammed**  
   Ref: `target_z = hole_top + 0.040`  
   Sim: unload along contact normal by `surface_franka_jam_lift_step_m` (default 3 mm/step) while jammed — absolute +40 mm would immediately float tip off tray.

5. **Center** = known hole XY (privileged socket ∩ tip plane).  
   `surface_tip_spiral_center: socket`.

6. **Pose command** = spiral XY + seeking/lift owns normal.  
   Sim: `franka_tcp` baseline tracks **wrist/TCP** to planar target (not tip-error servo).

## Explicit adaptations (must disclose in paper)

| Franka (Isaac / FixedJoint peg) | Our DexJoCo handoff |
|---------------------------------|---------------------|
| Rigid `FixedJoint` peg–TCP | Allegro **non-rigid** grasp (this is the stress test) |
| `FORCE_THRESHOLD_N = 2.0` | Soft sim residual → `0.08 N` jam band (overload role; seeking ~0.025 N) |
| Absolute world XY + peg-center Z | Tray-plane tangent basis `(t1,t2)` + normal lift steps |
| Full SCAN→APPROACH→DESCEND→INSERT | Phase-A PBVS already seated; we reproduce **INSERT jam spiral only** |
| Success = insert depth below hole top | Paper table still applies `SUCCESS_STANDARD` (height/force enter + seat + axis≤25°) |
| Absolute jam lift +40 mm | Repeated normal unload steps (keeps contact-plane search meaningful) |

## Not this baseline

- Tip-Hybrid / MSAR / STAR / planar_C / S9 → **ours / invent**, not Franka.
- ConnTact `conntact_time_spiral` → separate faithful baseline (`S3_conntact_faithful`).
- Old geometric `spiral` / `planned_spiral` tip-servo → not Franka.

## Numeric lock

```bash
PYTHONPATH=src python -c "from pci.franka_spiral_repro import assert_matches_ref_sample; assert_matches_ref_sample(); print('OK')"
```

## Eval command

```bash
PYTHONPATH=src:$DEXJOCO... python scripts/parallel_force_enter_ep10.py \
  --config configs/scheme_l3/S3_franka_spiral_faithful.yaml \
  --eps 1-10 --workers 3 \
  --out-root outputs/scheme_l3/S3_franka_faithful_ep1_10 \
  --video
```

Report: SUCCESS_STANDARD rate + per-ep `spiral_reason` / lat_min / axis / along at enter / priv_planar (diag only).

## Eval result (SUCCESS_STANDARD)

- RATE: **0/10** on `outputs/scheme_l3/S3_franka_faithful_ep1_10/`
- Per-ep (enter=SUCCESS_STANDARD; lat_min / along / axis at end; jam_step):

| ep | enter | force_hole_raw | along_mm | lat_min_mm | axis_deg | jam_step | reject / reason |
|----|-------|----------------|----------|------------|----------|----------|-----------------|
| 1 | 0 | 1 | 100.8 | 15.0 | 11.9 | 1 | force_hole_along_reject |
| 2 | 0 | 1 | 162.6 | 60.5 | 22.1 | 4 | force_hole_along_reject |
| 3 | 0 | 0 | 143.4 | 4.5 | 3.7 | 70 | priv_tip_spiral_gate / near_mouth |
| 4 | 0 | 0 | 313.2 | 41.7 | 31.6 | 1316 | surface_press / timeout float |
| 5 | 0 | 1 | 102.2 | 13.8 | 7.5 | 1 | force_hole_along_reject |
| 6 | 0 | 0 | 94.0 | 3.0 | 9.2 | 0 | priv_tip_spiral_gate / near_mouth |
| 7 | 0 | 0 | 115.7 | — | 21.0 | — | grasp_slip_pre_spiral |
| 8 | 0 | 1 | 102.4 | 22.7 | 12.2 | 0 | force_hole_along_reject |
| 9 | 0 | 0 | 140.1 | 4.3 | 13.6 | 25 | surface_press / near_mouth |
| 10 | 0 | 1 | 106.5 | 56.1 | 17.6 | 0 | force_hole_along_reject |

- Height-drop `force_hole=1` on ep1,2,5,8,10 are **false enters** (along still ≈100–163 mm) → rejected by seat/along audit.
- Paper claim: faithful Franka jam spiral under non-rigid handoff grasp does not achieve tip-on-surface upright force-enter.

## Limitations found (brief)

1. **TCP≠tip**：腕部追孔心/螺旋，非刚性抓取下 tip 平面 lat 常停在 14–60 mm。
2. **假入孔**：tip Z 掉 0.4 mm 就标 height-enter，但 along 仍 ~100 mm+。
3. **Jam 抬升**：软接触下 overload jam 会把 tip 抬离贴面（ep04 float≈95 mm）。
4. **与 ConnTact faithful 同级**：同为刚性 EE 假设 baseline，SUCCESS_STANDARD 下均为 0/10。
