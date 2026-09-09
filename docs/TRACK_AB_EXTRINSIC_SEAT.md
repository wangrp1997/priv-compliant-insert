# Track A/B: extrinsic / tip-contact SEAT (reuse)

> Gap: START_FLOAT — wrist `|resid|` ≠ tip–tray seat under non-rigid grasp.  
> Success: `docs/SUCCESS_STANDARD.md` (hard seat) — do not loosen.  
> Gate: `docs/METHOD_GATE_NO_PATCH.md`.

## Math object

Observe **true tip–tray contact** before enabling planar spiral/approach  
(hybrid **SEAT ≻ position**; Raibert/Craig 1981; ConnTact seeking).

## Search → reuse

| Track | Observation | Cite |
|-------|-------------|------|
| **A** | Peg–tray contact cluster `seated` (`ContactTipEstimator`) | Kim TEC / TEXterity contact factors |
| **B** | Wrist **residual** `|n·F_resid|` + CLEP (`F=hole_u*(fz-fz_base)`) | Kim Active Extrinsic; Doshi; CLEP |
| Shared | Freeze planar while unseated; ConnTact seek + Pfanne grasp stiffen | Raibert; ConnTact; Pfanne RA-L |

**Evidence fix (2026-09-05):** absolute `|n·F|~6N` false-seated AC ep07 — must use residual.

**Not tuned:** `min_contact_frac` / SUCCESS_STANDARD.  
**Not used:** tip XY GT as seat bit; ep-specific switches.

## Config

- `surface_seat_obs_mode`: `tip_contact` | `extrinsic` | `auto`
- Type-A default: `tip_contact` (create seat sensor if needed)
- GMHQP-AC default: `extrinsic`

Module: `src/pci/extrinsic_seat.py`.
