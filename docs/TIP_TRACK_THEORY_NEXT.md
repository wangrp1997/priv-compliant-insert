# Tip-STAR: Saturating Track-Aware Recovery

> Agent-I next invent after Tip-MSAR.  
> **Must beat** Tip-Hybrid 4/10; MSAR tied 4/10 is **not** progress.  
> Does **not** loosen `SUCCESS_STANDARD.md`.

---

## 0. Why MSAR cannot claim progress

| Fact | Value |
|------|-------|
| Tip-Hybrid+planar_C baseline | **4/10** (ep1,4,5,6) on `S2_force_enter_ep1_10v10` |
| Tip-MSAR | **4/10** (ep1,5,6,8) on `S2_force_enter_ep1_10_msar` |
| Rate | tied → **no claim** |
| ep04 | success → fail (`priv_planar` 4.5→12.1 mm) → **regression** |

Mouth geometry improved on ep03/08/09, but:

1. **Privileged mouth attractor** (`e_mouth = Π(h − tip)`) is not a deployable claim (hole pose in loop).
2. Blend fires whenever `ρ > ρ_mouth`, including healthy tracks that briefly leave the latch band → ep04 lost.
3. `tip_err > 12 mm` slip trip fires on successes too (ep01/04/05 tip_err_max 14–16 mm) → freeze + `C←I` without real grasp restore.
4. Type B (ep2/7/10) still planar ~12–13 mm, slip ~190–200 mm — **SLIP_RESEAT incomplete**.
5. ep03/09 reach mouth then fail ENTER — need existing mouth-probe / SEAT, not more planar pull.

Scientific problem (unchanged): under non-rigid grasp, make tip reach mouth neighborhood while keeping seat + upright, then force-enter under SUCCESS_STANDARD.

---

## 1. Name & one coherent story

**Tip-STAR** = **S**aturating **T**rack-**A**ware **R**ecovery

Core rule:

> Intervene **only** when tip tracking has **saturated behind** the spiral radius.  
> Never pull tip to privileged hole.  
> On slip: **real grasp tighten** + freeze `ṙ` + `C←I`.  
> Near mouth / force cue: **yield** to Tip-Hybrid SEAT + mouth-probe (do not fight).

This replaces MSAR’s always-on mouth blend with a **track-failure gate** and a **deployable local recover**.

---

## 2. Deployable vs privileged

| Signal | Deployable? | Role in STAR |
|--------|-------------|--------------|
| Spiral waypoint `p*`, `r_cmd`, phase | Yes (planner state) | Track error `e_tip = Π(p* − tip)` |
| Tip estimate / extrinsic tip | Yes (tactile path; sim tip OK for C-ID) | Residual + tip radius about **planner center** |
| Wrist F/T residual dip | Yes | Mouth / hole cue → yield to probe |
| Contact / float / along | Yes (force + seat metrics) | SEAT priority (Tip-Hybrid) |
| Grasp slip / finger lag | Yes (proprio / tactile) | Grasp-restore trip |
| Hole pose `h` as attractor target | **Privileged** | **Forbidden in STAR control** |
| `priv_planar` / `priv_mouth` | Privileged | Diag only |

MSAR’s `α·e_mouth` is scientifically fine as an **oracle invent diagnosis**, but **not** a deployable method claim. STAR never adds `e_mouth` into the wrist target.

---

## 3. Track-saturate gate (anti-regression)

Define tip radius about the **same spiral planner center** already used by Tip-Hybrid (not a new hole PD):

```
ρ_tip = ‖Π(tip − c_spiral)‖
e_tip = Π(p*_spiral − tip)
```

**Track failure** (must all hold):

```
sat = (‖e_tip‖ > τ_e) ∧ (r_cmd > ρ_tip + δ) ∧ (seat_ok) ∧ (mode == SPIRAL)
```

Hysteresis:

```
streak ← streak+1 if sat else 0
track_fail ← streak ≥ N_sat
```

**Invariance on successes (ep1,4,5,6 argument):**

- Near mouth with tip following the ring: `r_cmd ≲ ρ_tip + δ` → `track_fail = false` → STAR idle → identical to Tip-Hybrid planar_C.
- ep04 baseline finds hole in ~836 steps with tip already at mouth: STAR must stay idle → **no ep04 regression path**.
- MSAR broke this by blending whenever `ρ > ρ_mouth` and by tip_err-only slip trips.

**Priority (highest first):**

1. Tip-Hybrid `SEAT` / `UPRIGHT` (unchanged)
2. STAR `GRASP_RESTORE` if slip / coupling collapse
3. STAR `YIELD_MOUTH` if mouth-local deployable cue
4. STAR `LOCAL_RECOVER` if `track_fail`
5. else Tip-Hybrid `SPIRAL_TRACK`

---

## 4. Mechanisms → fail types

### 4.1 LOCAL_RECOVER (Type A: ep2,3,8 saturators)

On `track_fail`:

```
ṙ ← 0
# Deployable local target: keep spiral *phase*, pin radius to tip ring
u_phase = unit(Π(p*_spiral − c))
p_local = c + ρ_tip · u_phase
target ← p_local     # NOT hole attractor
```

Effect: stop expanding unreachable Archimedean radius; servo tip onto its current ring with correct phase so C-gated steps can catch up. Expected: lower `priv_planar_min` on A-type without privileged mouth PD.

When `r_cmd ≤ ρ_mouth` **or** force residual mouth cue → leave LOCAL_RECOVER (see 4.3).

### 4.2 GRASP_RESTORE (Type B: ep2,7,10)

Trip (stricter than MSAR):

```
slip_trip = (s > τ_s) ∨ (‖Δw_xy‖>ε ∧ ‖Δtip‖ < γ‖Δw‖)
# Do NOT trip on tip_err alone when ρ_tip ≤ r_cmd + δ (healthy lag near ring)
```

Action for `N_r` frames:

```
ṙ ← 0
Δw_xy ← 0 (or seat-only)
C ← I, clear LS
left grasp scale ← max(scale, scale·κ)   # real tighten, once per trip
optional: short pose-hold GraspQP if already enabled
then resume SPIRAL / LOCAL_RECOVER
```

Expected: reduce slip explosion; give planar_C a chance to re-ID. Incomplete GraspQP-only freeze was MSAR’s B failure mode.

### 4.3 YIELD_MOUTH (ep03/09: at mouth, no force hole)

Deployable mouth-local cue (no `e_mouth` servo):

```
yield = (r_cmd ≤ ρ_mouth + ε_m) ∨ force_hole_residual_cue ∨ (ρ_tip ≤ ρ_mouth ∧ ‖e_tip‖ < τ_near)
```

On yield:

```
STAR planar override OFF
allow Tip-Hybrid SEAT + existing mouth_probe / unload schedule
ṙ ← 0
```

Expected: keep MSAR’s mouth-reach wins (ep08) while **not fighting** SEAT/probe so enter gate/along can fire. STAR does **not** invent a new enter law.

### 4.4 Type C along-drift

Unchanged: Tip-Hybrid SEAT priority + force-on-normal. STAR never overrides SEAT.

---

## 5. Why this should beat 4/10 (expected map)

| Episode | MSAR outcome | STAR expected |
|---------|--------------|---------------|
| 1,5,6 | keep | keep (STAR idle near mouth) |
| 4 | **lost** | **restore** (no always-on mouth blend / no tip_err-only reseat) |
| 8 | gained mouth+enter | keep via LOCAL_RECOVER then yield (no priv attractor) |
| 3,9 | mouth yes, enter no | mouth keep + yield to probe → **possible +1 enter** |
| 2,7,10 | still Type B | grasp tighten may help; not required for first rate win |

Minimal honest win: **restore ep04 + keep ep8** → **5/10**, without needing Type B solved. Stretch: convert one of ep03/09.

Pass bar: rate **> 4/10**, no regression on ep1/5/6, ep04 not worse than baseline success.

---

## 6. Landing (code / config)

| Piece | File |
|-------|------|
| Theory (this) | `docs/TIP_TRACK_THEORY_NEXT.md` |
| Gates + local target | `src/pci/tip_tracking_baselines.py` (`tip_star_*`) |
| Spiral hooks | `src/pci/sim_runner.py` (STAR supersedes MSAR when enabled) |
| Config | `configs/scheme_l3/S2_star.yaml` |

Eval policy (user gate):

1. Theory + code first.
2. Fail-subset smoke: ep2,3,4,8,9 — need improvement **and** no ep04 regression.
3. Only then full ep1–10.

```bash
# smoke (manual / parallel subset) — do not claim until ep04 OK
PYTHONPATH=src python scripts/parallel_force_enter_ep10.py \
  --config configs/scheme_l3/S2_star.yaml \
  --out-root outputs/scheme_l3/S2_force_enter_ep1_10_star \
  # restrict to fail-subset when runner supports it

PYTHONPATH=src python scripts/diag_spiral_priv.py \
  outputs/scheme_l3/S2_force_enter_ep1_10_star/ep*/ep*_summary.json
```

---

## 7. Explicit non-goals

- Do not loosen SUCCESS_STANDARD.
- Do not use privileged hole attractor in deployable STAR path.
- Do not replace Tip-Hybrid upright / seat.
- Do not claim progress from mouth geometry alone without rate > 4/10.
