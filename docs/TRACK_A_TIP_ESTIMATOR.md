# Track A — Tip estimator for Oracle-v2 control law

> Path A of `docs/OBSERVABILITY_BRIDGE.md` / `docs/DUAL_TRACK_TO_ORACLE.md`.  
> Goal: feed **same** Oracle-v2 law (`priv_oracle_tip_servo` + mouth press + slip reseat) with **`tip_hat`**, not tip GT and **not** `tip_gt+noise`.  
> Success standard: `docs/SUCCESS_STANDARD.md` (do not loosen).  
> Disclosure: `docs/PRIVILEGED_SENSING_DISCLOSURE.md`.

## Scientific framing

Oracle-v2 proves: if tip↔mouth planar pose is known and tip PD + seat + mouth press + slip reseat run, enter rate can reach **9/10**.  
Track A asks: **can we estimate those Oracle inputs from eligible / sensor-like signals and keep the same controller?**

Problem lock (`docs/PROBLEM_LOCK_KNOWN_PATH.md`): mouth / spiral path may stay **known** (frozen vision / socket coarse). Study seat + upright + force enter under non-rigid grasp.

## What Oracle-v2 consumes

| Quantity | Oracle-v2 source | Track-A estimate |
|----------|------------------|------------------|
| Tip planar XY on tray | peg tip GT (`feat.tip_pos`) | `ContactTipEstimator` → `tip_hat` |
| Mouth XY | live hole-axis point at tip height (known path OK) | same socket / hole TF (known path; freeze OK) |
| Seat / float | tip depth + wrist residual | peg↔tray contact force + wrist F/T |
| Axis / upright | peg axis GT (hybrid UPRIGHT) | still Tip-Hybrid + feat axis for audit; control upright from hybrid |
| Slip | priv grasp COM in wrist | **residual privilege** for reseat trip (disclose); tip lag uses `tip_hat` lat |

## Eligible inputs (sim)

| Signal | Module | Role |
|--------|--------|------|
| Wrist F/T (R/L) | `pci.sensors.read_wrist_wrench_*` | seat / overload / force-hole |
| Allegro fingertip forces | `FingerForceLabeler` / `cfrc_ext` on tip bodies | grasp quality; fallback centroid |
| Finger tip body xpos | proprio / FK | wrist–grasp geometry |
| Proprio wrist site | `actual_action44_from_sites` | holdover `tip ≈ wrist + offset` |
| MuJoCo peg↔tray `contact.pos` + `mj_contactForce` | `tip_contact_estimator.py` | **primary tip locus** (see residual privilege) |

**Not used in tip_hat:** `data.xpos[peg]`, `peg_insert_end_pos`, `feat.tip_pos` (GT only for logging / SUCCESS_STANDARD audit).

## Estimator v0 design

Code: `src/pci/tip_contact_estimator.py` · mode `track_a_est_oracle`.

1. **Seated measure:** force-weighted / far-from-grasp cluster of peg↔tray contacts → project to tray plane through previous `tip_hat` along hole/spiral normal → EMA.  
2. **Lost contact (float / in-hole):** **non-rigid holdover** (config `surface_tip_est_holdover_mode`):
   - `freeze_planar` (current default): after any accepted peg↔tray measure, freeze planar tip at last contact tip (TEC / extrinsic: tip is a contact observation, not wrist FK child). If **never** measured, bridge with `wrist+offset` (avoid freeze-at-bad-init).  
   - `wrist_delta` (legacy): `tip_hat ← tip_hat + planar(wristΔ)` — treats grasp as rigid; caused smoke_v2 **0/4**.  
   - `decay` / `offset_only`: softer variants. Offset refresh gated by `surface_tip_est_seat_only_offset`.  
3. **Init:** prefer current peg↔tray contacts at spiral start; if none and `surface_tip_est_allow_gt_seed=false`, finger-force centroid fallback (weak).  
4. **Plug-in:** `sim_runner` sets `tip_ctrl = tip_hat`, keeps Oracle-v2 mouth PD / mouth press / slip reseat (`surface_tip_oracle_*`).
5. **PCA free-end:** gated off (`surface_tip_est_use_pca: false`) — prior on-path nuked rate.

Mouth stays “known”: Oracle mouth PD still uses socket + hole axis coplanar with **control tip** (`_hole_axis_point_at_tip(tip_hat, socket, hole_u)`). Per problem lock this is allowed as frozen vision / known path — **not** tip GT.

## Residual privilege (ruthless)

| Item | Status |
|------|--------|
| Peg body xpos / tip GT in **control** | **No** |
| `tip_gt + N(0,σ)` | **Rejected** (invalid bridge) |
| MuJoCo peg↔tray `contact.pos` | **Yes — residual.** Sim stand-in for extrinsic / tray-tactile contact localization (TEC spirit). Named body geoms. Must disclose; not hardware-ready until tactile/extrinsic module replaces it. |
| Priv grasp slip (`peg_in_right_wrist`) for reseat trip | **Yes — residual.** Tip lag gate uses `tip_hat` lat; slip magnitude still priv. Replace later with shear/innov. |
| Socket / hole axis for mouth | Allowed as **known path** (vision coarse); still sim object TF until real vision freeze. |
| SUCCESS_STANDARD audit tip/axis | Privileged metrics only — not control. |

## Config / eval

```bash
cd /home/wangrenpeng/priv_compliant_insert
MUJOCO_GL=egl PYTHONPATH=src:/home/wangrenpeng/dexjoco \
  /home/wangrenpeng/miniconda3/envs/dexjoco/bin/python \
  scripts/parallel_force_enter_ep10.py \
  --config configs/scheme_l3/S2_trackA_est_oracle_ctrl.yaml \
  --out-root outputs/scheme_l3/S2_trackA_est_oracle_smoke \
  --eps 1-4 --workers 2

PYTHONPATH=src python scripts/diag_spiral_priv.py \
  outputs/scheme_l3/S2_trackA_est_oracle_smoke/ep*/ep*_summary.json
```

Meta flags: `track_a_tip_est`, `tip_est_err_peak_mm`, `tip_est_last_source`, `tip_est_residual_privilege`.

## Relation to TEC (Kim)

`refs/Tactile-Estimator-Controller`: factor-graph extrinsic contact from fingertip tactile + kinematics.  
v0 does **not** port the NN/GTSAM stack; it implements the **contact-geometry observation** TEC would output (`tip` / contact point on object–environment), using sim contacts as the observation model. Next: replace peg↔tray contact read with fingertip tactile extrinsic factors when available.

## Reporting

| Table | Rate |
|-------|------|
| Oracle-v2 (priv tip GT) | 9/10 diagnostic only |
| Track A (this) | estimator + same ctrl — disclose residual privilege |
| Deployable Type-A | 5/10 — no tip GT |

## Smoke (2026-09-03)

Out: `outputs/scheme_l3/S2_trackA_est_oracle_smoke/` (v0 baseline).

| | Oracle-v2 on ep1–4 | Track-A **v0** ep1–4 |
|--|--|--|
| RATE | **3/4** (fail ep04) | **2/4** (fail ep03, ep04) |
| Success | 1,2,3 | 1,2 |
| Fail notes | ep04 axis≈43° | ep03 axis≈73° gate; ep04 axis≈45° surface_press |

Control path confirmed: `tip_baseline_mode=track_a_est_oracle`, `obs_noise=false`, `privileged_oracle_tip_servo=false`, `track_a_tip_est=true`.

### tip_hat vs GT (v0 fail eps)

| ep | tip_est_err_peak_mm | last_source | axis_err_deg | note |
|----|---------------------|-------------|--------------|------|
| 03 | **46.4** | wrist_hold | **73.4** | near mouth; gate force fail; tip_hat lag |
| 04 | **68.1** | wrist_hold | **44.7** | cfrac≈0; surface_press |

Holdover pathology: `_apply_holdover` + `wrist_delta` pushes tip_hat with wrist as if grasp rigid → non-rigid slip makes tip_hat drift (confirmed smoke_v2 **0/4**).

### Holdover fix attempts (not better than v0 — **STOP**, no ep1–10)

| run | out | RATE | note |
|-----|-----|------|------|
| v2 | `..._smoke_v2` | **0/4** | far-grasp / wrist_delta compound |
| v3 | `..._smoke_v3` | **1/4** | pure `freeze_planar`; ok only ep02; ep01 axis≈57°; ep03 axis≈9° but planar 15 mm timeout (tip_hat freeze-at-init → err peak **407** mm) |
| v3b | `..._smoke_v3b` | **0/4** | freeze after any contact + never-seat offset bridge; still worse |

**Gate:** ≤2/4 and worse than v0 → **do not expand** ep1–10. Keep v0 as Track-A baseline RATE.  
**Next (needs new auth):** contact-observation reliability (why peg↔tray seat frames=0 on long spirals) + gated freeze only while seated force high; do not re-enable PCA.

### Priv diag + v4 (2026-09-03) — STOP

Phase-0 numbers: `docs/TRACK_A_PRIV_DIAG.md`.

- Claim **verified**: long spiral can have **`tip_est_seated_frames=0`** while wrist `spiral_contact_frac≈1` (v3 ep02/03); ep03 also `tip_hybrid_seat_frames=0` vs Oracle hyb_seat=130.
- ONE fix tried: **gated freeze** (`freeze_require_seat`) + first strong seat hard-snap → out `S2_trackA_est_oracle_smoke_v4`.
- **RATE 0/4** (ep01 axis≈57°; ep02 planar timeout tip_est peak≈178 mm; ep03 infra crash; ep04 axis≈45°) vs v0 **2/4** → **STOP, no ep1–10**.
- Honest RATE still v0 **2/4**; config flag retained.

### Theory line (post-v4) — not a patch

- User lock: no threshold / freeze / geom-id / per-ep patches; **abandon** `smoke_v5` patch RATE.
- Priv gap → estimation problem: `docs/TRACK_A_PRIV_DIAG.md` § continuation + `docs/TRACK_A_THEORY_ESTIMATOR.md`.
- Skeleton: `src/pci/tip_theory_estimator.py` (`ExtrinsicTipStateEKF`) + `configs/scheme_l3/S2_trackA_theory_est.yaml`.
- Unit equations only: `scripts/smoke_track_a_theory_est.py` (no ep RATE until sim_runner wire).
