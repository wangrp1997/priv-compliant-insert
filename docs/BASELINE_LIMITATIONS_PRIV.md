# Baseline → privileged failure → our fix map

> Agent-I deliverable. Does **not** loosen `SUCCESS_STANDARD.md`.  
> Scientific problem (`MULTI_AGENT_BRIEF` §1): under non-rigid grasp, make extrinsic tip reach mouth (`priv_planar ≲ 4.5 mm`) while seat + axis≤25°, then force-enter.  
> Honest ours baseline: Tip-Hybrid + planar_C = **4/10** on `S2_force_enter_ep1_10v10`.

---

## 0. Locked success (unchanged)

All required:

1. Deployable force (or audited height) hole enter  
2. Tip on tray/hole face during spiral (reject float / along reject)  
3. Peg axis upright at enter (`axis_err ≤ 25°`)

---

## 1. ConnTact faithful (`S3_conntact_faithful_ep1_10`) — **0/10**

| Ep | SUCCESS fail | Privileged / audit signature | Root limitation |
|----|--------------|------------------------------|-----------------|
| 1 | `spiral_tilt_or_tray` | no stable spiral; float peak ~10 mm | Rigid TCP seeking ≠ non-rigid tip seat |
| 2,5,8,10 | `force_hole_along_reject` | height/force “enter” while `along` 100–160 mm; lat_min often 8–65 mm | **Z-drop / residual FP**: wrist/TCP height≠tip-in-bore |
| 3 | `force_hole_float_reject` | lat_min~5 mm but float peak~4 mm | Tip not seated on face |
| 4 | `force_hole_axis_reject` | axis~33° at “enter” | No tip-pivot upright before accept |
| 6 | `priv_tip_spiral_gate_fail` / near_mouth | lat_min~3 mm, force_hole=0 | Wrist spiral near mouth without force enter |
| 7 | `grasp_slip_pre_spiral` | slip~101 mm, no spiral | Non-rigid slip before search |
| 9 | `surface_press` / float | float~6 mm | Seeking without tip seat FSM |

**What ours must fix (tip-referenced planar_C / Tip-Hybrid / Tip-STAR-v11):**

| ConnTact gap | Our mechanism |
|--------------|---------------|
| Wrist≠tip track | `planar_C` tip error on planner spiral (not TCP Archimedean alone) |
| Height-drop FP | Force hole + SUCCESS seat/along/axis audit (no Z-only success) |
| No upright gate | Tip-Hybrid `UPRIGHT` + tip-pivot; v11 **hold enter until axis≤soft** |
| No seat FSM | Tip-Hybrid `SEAT` before spiral advance |
| Slip / non-rigid | Type-B `GRASP_RESTORE` + real left grasp tighten (bypass handoff freeze) |

Paper claim: faithful ConnTact under dex handoff does **not** achieve tip-on-surface upright force-enter.

---

## 2. Tip-Hybrid + planar_C v10 (`S2_force_enter_ep1_10v10`) — **4/10**

Success: ep **1,4,5,6**. Fail: **2,3,7,8,9,10**.

| Type | Ep | Privileged signature | Limitation |
|------|-----|----------------------|------------|
| A track saturate | 2,3,8 | `priv_planar_min` **8.5–14 mm**, `priv_mouth=False`, `lat≡priv`, tip_err lag | Spiral radius outruns tip; C-gated steps never reach mouth |
| B slip explode | 7,9,10 (+2) | `grasp_slip` **170–255 mm**, tip_err peaks high | Slip breaks wrist→tip map; freeze/`C←I` without real grasp restore |
| C along drift | 9 | along drift **+35 mm** class | Long spiral loses seat/normal press |
| Axis (mostly OK) | fails | axis often already **&lt;25°** | Bottleneck is **planar mouth reach**, not tilt |

**What Tip-STAR / ours v11 must fix:**

| Gap | Knob / mechanism | Deployable? |
|-----|------------------|-------------|
| A | Track-saturate gate → `LOCAL_RECOVER` (pin ρ to tip ring, freeze `ṙ`) | Yes — planner state only |
| A/ep04 | Gate only when `r_cmd > ρ_tip + δ` (no always-on mouth blend) | Yes — anti-MSAR regression |
| B | Slip trip → freeze + `C←I` + **real grasp tighten** | Yes — proprio/tactile slip |
| B | Must bypass `surface_freeze_left_hand` lock | Yes — STAR-only override |
| C | Tip-Hybrid SEAT priority never overridden by STAR | Yes |
| Enter tipped (STAR smoke ep02) | **Upright hold**: do not `force_stop_on_hole` while axis &gt; soft; full upright during planar recover | Yes — F/T + tip-pivot |

Forbidden in deployable path: privileged hole attractor `e_mouth` (MSAR).

---

## 3. Tip-STAR smoke (`S2_star_smoke_23489`) — lesson only

| Ep | Outcome vs v10 | Lesson for v11 |
|----|----------------|----------------|
| 4 | **kept** success | Track-saturate idle near mouth works |
| 2 | lat_min~2.6 mm but **axis~50°** → axis reject | Mouth reach without upright hold = false win |
| 3,8,9 | still fail; local_recover often 0 | Need sat gate + upright/grasp fixes, not priv attractor |
| — | `freeze_left_hand` → `_tighten_left_grasp` no-op | Type B “real tighten” was dead |

---

## 4. Franka faithful (`S3_franka_faithful_ep1_10`) — **0/10**

Rigid-EE jam+spiral port under non-rigid Allegro handoff. SUCCESS_STANDARD rate **0/10**.

| Fail cluster | Ep | Signature | Limitation |
|--------------|-----|-----------|------------|
| Along reject after “force hole” | 1,2,5,8,10 | force_hole=1, along still **100–160 mm**, lat_min often **14–60 mm** | Jam/force cue on wrist ≠ tip-in-bore (same family as ConnTact height FP) |
| Near mouth, no force enter | 3,6 | lat_min **3–4.5 mm**, force_hole=0 | No tip-referenced seek / mouth probe ownership |
| Surface / tilt | 4,9 | surface_press; ep04 axis~32°, along~313 mm | No Tip-Hybrid SEAT→UPRIGHT before spiral accept |
| Pre-spiral slip | 7 | slip~101 mm, lat=999 | Non-rigid grasp; no Type-B restore |

**What ours must fix:** tip-referenced `planar_C` + Tip-Hybrid seat/upright + STAR track-saturate + grasp tighten + SUCCESS along/axis audit (already rejects Franka FPs).

---

## 5. PSFT paper reimpl (`S3_psft_faithful_ep1_10`) — **0/10**

Park-style compliance spiral under non-rigid handoff. SUCCESS_STANDARD **0/10**.

| Fail cluster | Typical signature | Limitation |
|--------------|-------------------|------------|
| Force/along FP | force_hole with along still high / lat far | Soft spiral cue ≠ tip-in-bore seat |
| Gate / timeout | priv_tip_spiral_gate_fail; planar not at mouth | No tip-referenced `planar_C` track-saturate |
| Slip / surface | slip or surface_press early | Rigid-EE assumption; need Type-B + Tip-Hybrid SEAT |

**What ours must fix:** same map as ConnTact/Franka — tip track + seat/upright + STAR recover + SUCCESS audit.

*Per-ep table: see `outputs/scheme_l3/S3_psft_faithful_ep1_10/summary_enter10.json`.*

---

## 6. Ablation / matrix rows (`paper_S*_ep1_10`) — fill as ready

| Tag | Rate (so far) | Limitation vs tip-ref ours |
|-----|---------------|----------------------------|
| S0 surface lock | **0/10** | Blind force seat; no tip spiral → mouth never by design |
| S1 tip/wrist FF | pending | Offset feedforward; no C adapt under slip |
| S2live / S6 / S8 | pending | S8≈v10 path; still Type A/B without STAR-v11 |

---

## 7. Ours claim checklist (v11)

Must address **each** prior limitation above without relaxing SUCCESS_STANDARD:

1. Tip track (not TCP) → planar_C + track-saturate recover  
2. No Z-drop success → force + seat + along + axis audit  
3. Upright before accept enter → hybrid UPRIGHT + **hold stop until axis OK**  
4. Slip → real grasp tighten (bypass freeze) + C reset  
5. No privileged mouth PD in deployable STAR path  

Progress bar: rate **&gt; 4/10**, keep ep04, no ep1/5/6 regression. Oracle tip-servo (7/10) stays **off** main table.

### Smoke `S2_ours_v11_smoke_23489` (eps 2,3,4,8,9) — **1/5**; **no full launch**

| Ep | enter | priv_mouth | priv_planar_min | vs v10 |
|----|-------|------------|-----------------|--------|
| 4 | **1** | True | ~3.8 mm | kept |
| 2 | 0 | **True** | 5.4 mm (was 12.0) | axis 0.1° (was STAR 50° reject); grasp_boost on |
| 3 | 0 | **True** | 6.0 mm (was 8.5) | mouth yes, force enter no |
| 8 | 0 | **True** | 4.5 mm (was 13.7) | mouth yes; along drift +36 mm |
| 9 | 0 | — | 8.6 mm | still Type A/B |

Gate: ep04 kept **yes**; extra success vs v10 fail set **no** → **do not** run `S2_ours_v11_ep1_10` yet.  
Next bottleneck: mouth reached but along/force enter not converting (yield→probe), not upright FP.
