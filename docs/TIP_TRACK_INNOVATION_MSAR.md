# Tip-MSAR: Mouth-Seek with Along-Retain / Slip-Aware Reset

> Agent-I invent. Driven by §1 A/B/C on `S2_force_enter_ep1_10v10`.  
> Does **not** loosen `SUCCESS_STANDARD.md`. Axis/UPRIGHT stays Tip-Hybrid; this invents **planar tip→mouth** only.

## 0. Privileged diagnosis (locked)

| Fail type | ep | Signature | Implication |
|-----------|----|-----------|-------------|
| A track saturate | 2,3,8 | `priv_planar_min` stuck 8.5–14 mm; tip never at mouth | Spiral FF + C-gate keeps chasing unreachable waypoints |
| B slip explode | 7,9,10 | slip 170–255 mm; tip_err peak 60–169 mm | C obsolete; must freeze advance + reseat + re-ID C |
| C along drift | 9 | along +35 mm | Lost face press during long spiral |

Success: `priv_planar_min` 3–4.5 mm, `priv_mouth=true`. Axis already OK on most fails → **no attitude-first invent**.

Scientific question: under non-rigid grasp, make extrinsic tip track to mouth neighborhood (`priv_planar ≲ 4.5 mm`) while retaining seat + upright.

---

## 1. Name & core idea

**Tip-MSAR** = Mouth-Seek · Along-Retain · Slip-Aware Reset

Core: **stop expanding spiral when tip cannot follow; pull tip toward hole axis in-plane; if slip or lose contact, freeze radius, reseat, reset C, then resume.**

Three mechanisms map 1:1 to A/B/C:

1. **Mouth attractor (A):** when estimated tip lateral `ρ > ρ_mouth`, blend spiral FF → tip→hole planar PD.
2. **Slip-aware freeze + reseat (B):** when slip rate / tip lag exceeds gate, freeze Archimedean advance, short GraspQP burst, `C ← I`, re-warm LS.
3. **Along-retain (C):** if along rises or contact drops, hard SEAT (force on −n) before any planar chase.

Dual timescale: slow layer updates `C`; fast layer servos tip residual with short clipped steps.

---

## 2. Notation

- `p_t ∈ ℝ³` tip position (deploy: tactile/extrinsic estimate; sim oracle OK for C-ID only)
- `h_xy` hole axis ∩ tray plane
- `ρ = ‖Π(p_t − h)‖` planar radial distance (Π = tray-plane projector)
- `ρ_mouth ≈ 4.5 mm` (same mouth latch as SUCCESS path; **not** relaxed)
- `w` wrist site; `Δw` commanded wrist step
- `C ∈ ℝ²ˣ²` planar coupling: `Π Δp_t ≈ C Π Δw`
- `s` grasp slip (peg tip vs finger frame lag)
- `a` along (tip height along hole axis; seat metric)
- `r, φ` Archimedean spiral state (radius / angle)
- `e_tip = Π(p*_spiral − p_t)` spiral track error
- `e_mouth = Π(h − p_t)` mouth attractor error

---

## 3. FSM (extends Tip-Hybrid; no new attitude mode)

```
SEAT ──(float/along OK)──► UPRIGHT ──(axis OK)──► SPIRAL_TRACK
                                                      │
                         ┌── slip > τ_s or ‖e_tip‖ > τ_e ──► SLIP_RESEAT
                         │                                         │
                         │                    (grasp burst + C←I + seat OK)
                         │                                         ▼
                         │                                  SPIRAL_TRACK
                         │
                         ├── ρ > ρ_mouth ──► MOUTH_SEEK  (same plane press)
                         │                         │
                         │              (ρ ≤ ρ_mouth) ──► SPIRAL_TRACK / enter
                         │
                         └── Δa > τ_a or contact↓ ──► SEAT (Along-Retain)
```

Priority (highest first):

1. `SEAT` if float or along-drift (type C)
2. `UPRIGHT` if axis soft exceed (reuse Tip-Hybrid; do not invent new orient)
3. `SLIP_RESEAT` if slip/tip-lag gate (type B)
4. `MOUTH_SEEK` if `ρ > ρ_mouth` (type A)
5. else `SPIRAL_TRACK` (planar_C spiral)

`SEAT` / `UPRIGHT` / `SLIP_RESEAT` **freeze** `ṙ` (spiral radius advance).

---

## 4. Equations

### 4.1 Slow layer — coupling

On each planar step with contact OK and not in slip:

```
Δt = Π(p_t − p_t⁻),  Δw_xy = Π(w − w⁻)
C ← argmin_C ‖Δt − C Δw_xy‖² + λ‖C − I‖²     (LS, same spirit as PlanarCoupling)
```

On slip gate trip:

```
C ← I,  clear LS buffer,  freeze ṙ for N_freeze steps
```

### 4.2 Fast layer — wrist step (type A fix)

Target blend:

```
α = clip( (ρ − ρ_mouth) / ρ_blend , 0, 1 )     # α→1 far from mouth
e = (1−α) e_tip + α e_mouth
Δw_xy* = C⁺ e                                 # pseudo-inverse / ridge LS
Δw_xy  = clip( g_f · Δw_xy* , Δ_max )         # short step + force gate g_f
```

Force gate (keep face):

```
g_f = 1  if |f_n − f_des| < τ_f else  β ∈ (0,1]
```

Along-retain (type C):

```
Δw += −k_a · n · max(0, a − a_seat)           # press back to seat band
if a − a₀ > τ_a:  mode ← SEAT,  ṙ ← 0
```

### 4.3 Slip-aware reseat (type B)

Detect:

```
slip_trip = (s > τ_s) ∨ (‖Δw_xy‖ > ε ∧ ‖Δt‖ < γ‖Δw_xy‖) ∨ (‖e_tip‖ > τ_e)
```

Action for `N_r` steps:

```
ṙ = 0
Δw_xy = 0
run GraspQP burst (finger tighten / re-stabilize)
Δw_n = −k_seat n                              # keep tip on tray
then C ← I, resume SPIRAL_TRACK / MOUTH_SEEK
```

### 4.4 Spiral advance gate

```
allow_ṙ = (mode == SPIRAL_TRACK ∨ mode == MOUTH_SEEK)
          ∧ (‖e‖ < τ_track) ∧ (not slip_trip) ∧ seat_ok
ṙ = v_r · allow_ṙ
```

Type A intuition: if tip stuck at 9–14 mm, `α→1` pulls toward hole instead of expanding `r` further; `allow_ṙ` stays false until tip error shrinks.

---

## 5. Landing files

| Piece | File | Change |
|-------|------|--------|
| FSM + modes `mouth_seek` / `slip_reseat` | `src/pci/tip_tracking_baselines.py` | extend `TipHybridState`; add `tip_msar_select`, `mouth_attract_step`, `slip_reseat_step` |
| Blend / C⁺ step | same | reuse `PlanarCouplingEstimator` + ridge solve; do not replace Tip-Hybrid upright |
| Spiral freeze / α blend hook | `src/pci/sim_runner.py` planned_spiral loop | after `tip_hybrid_select`, if MSAR enable → override planar Δw and `hyb_allow_adv` |
| Config | `configs/scheme_l3/S2_msar.yaml` (or flag on S2) | `surface_tip_msar_enable`, `ρ_mouth`, `τ_s`, `τ_e`, `τ_a`, `N_freeze`, `ρ_blend` |
| Grasp burst | `src/pci/compliant/priv_grasp_opt.py` | short call from SLIP_RESEAT only |

Keep Tip-Hybrid switchable: MSAR is an option **inside** spiral branch, not a replace of seat/upright.

---

## 6. Privileged verification (mandatory)

Do **not** change SUCCESS_STANDARD knobs.

```bash
# baseline already:
# outputs/scheme_l3/S2_force_enter_ep1_10v10/  → 4/10

PYTHONPATH=src python scripts/parallel_force_enter_ep10.py \
  --out-root outputs/scheme_l3/S2_force_enter_ep1_10_msar

PYTHONPATH=src python scripts/diag_spiral_priv.py \
  outputs/scheme_l3/S2_force_enter_ep1_10_msar/ep*/ep*_summary.json
```

Report table must include per-ep:

- `priv_planar_min_mm` (goal: fails → ≲ 4.5)
- `priv_mouth`
- `tip_spiral_err_{mean,max}_mm`
- `grasp_slip_mm`
- `along` drift / end
- SUCCESS_STANDARD enter (force + seat + axis≤25°) — main table
- Oracle tip (if used for C-ID only) listed separately per `PRIVILEGED_SENSING_DISCLOSURE.md`

Pass criteria for this invent (honest):

1. Fail-subset ep2,3,7,8,9,10: `priv_planar_min` drops vs v10; ideally `priv_mouth=true`
2. Success ep1,4,5,6: no regression on SUCCESS_STANDARD
3. Axis invent not the story — do not tune upright to fake enter

---

## 7. Why not attitude invent

v10 fails already have `axis_err` mostly <25°; bottleneck is **planar tip never at mouth**. MSAR allocates DOF: n=force seat, plane=mouth/spiral, R=maintain Tip-Hybrid only.
