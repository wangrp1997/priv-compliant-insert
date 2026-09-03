# Deployable Type B — slip explode → GRASP_RESTORE

> Agent-I · deployable only (proprio / slip / force).  
> Does **not** loosen `SUCCESS_STANDARD.md`.  
> Scientific problem (`MULTI_AGENT_BRIEF` §1): under non-rigid grasp, tip must reach mouth (`priv_planar ≲ 4.5 mm`) while seat + axis≤25°.

## 1. Failure signature (v10)

Type B eps **7, 9, 10** on `S2_force_enter_ep1_10v10`:

| Ep | `grasp_slip` | `priv_planar_min` | Note |
|----|--------------|-------------------|------|
| 7 | ~205 mm | ~14 mm | slip explode → C broken |
| 9 | ~172 mm | ~9 mm | + along drift |
| 10 | ~255 mm | ~9 mm | tip_err peaks high |

## 2. Protocol (deployable)

On slip trip (proprio slip peak / lag gate — **no hole pose**):

1. **Freeze** spiral radius advance (`ṙ ← 0`)
2. **Reseat SEAT** — Tip-Hybrid forced to `seat` for `N_r` frames
3. **Real finger tighten** — scale left hold, **`bypass_freeze`** under `surface_freeze_left_hand`
4. **C ← I** — planar coupling / EKF slip reset
5. **Resume** spiral after `N_r` (latch so sticky slip peak cannot re-arm every frame)

### Lesson (STAR)

Claimed grasp tighten was a **no-op** while `surface_freeze_left_hand` blocked `_tighten_left_grasp`.  
Type B must set `surface_tip_slip_grasp_bypass_freeze: true` and audit `tip_slip_grasp_tighten_ok` / `mean|q|` before→after.

### Latch

`grasp_slip_peak` is monotonic. Without edge latch, `star_reseat_left` refreshes every frame and spiral never resumes.  
Use `tip_slip_can_enter_restore` + `star_slip_armed_m` + `retrip_delta_m`.

## 3. Config knobs

| Knob | Role |
|------|------|
| `surface_tip_slip_enable` | Type-B path alone (STAR off) |
| `surface_tip_star_enable` | also enables Type B under full STAR |
| `surface_tip_slip_tau_m` | slip peak trip |
| `surface_tip_slip_reseat_frames` | freeze+SEAT duration |
| `surface_tip_slip_grasp_scale` | left finger scale |
| `surface_tip_slip_grasp_bypass_freeze` | **required** for real tighten |
| `surface_tip_slip_force_seat` | reseat Tip-Hybrid SEAT |
| `surface_tip_slip_retrip_delta_m` | re-arm only after +Δ slip |

Aliases: `surface_tip_star_slip_*` / `surface_tip_star_grasp_*` fall back when slip_* unset.

## 4. Code

- Gates: `src/pci/tip_tracking_baselines.py` — `tip_star_slip_trip`, `tip_slip_can_enter_restore`
- Hooks: `src/pci/sim_runner.py` — GRASP_RESTORE + `_tighten_left_grasp(..., bypass_freeze=True)`
- Config: `configs/scheme_l3/S2_deploy_typeB.yaml`

## 5. Smoke gate

```bash
MUJOCO_GL=egl python scripts/parallel_force_enter_ep10.py \
  --config configs/scheme_l3/S2_deploy_typeB.yaml \
  --eps 4,7,9,10 \
  --out-root outputs/scheme_l3/S2_typeB_smoke
```

Pass:

- ep04 success **kept**
- on 7/9/10: slip peak **down** OR `priv_planar` **improves** vs v10
- ideally ≥1 extra enter among 7,9,10
- meta/log: `tip_slip_grasp_tighten_ok=true`, `mean|q|` increases, `Type-B GRASP_RESTORE trip` printed

Forbidden: privileged hole attractor as deployable claim.
