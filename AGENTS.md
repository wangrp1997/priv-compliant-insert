# Agent notes (priv_compliant_insert)

## Before any tip-spiral / force-enter work

1. Read `docs/MULTI_AGENT_BRIEF.md` §1 (privileged scientific problem).
2. Read `docs/SUCCESS_STANDARD.md` (do not loosen).
3. Run or cite `scripts/diag_spiral_priv.py` on your eval outputs.

## Roles

- **Reuse:** upgrade from `refs/` + `docs/REF_REUSE_MAP.md` (Agent-R).
- **Invent:** design from privileged failure types (Agent-I).
- **Oracle:** privileged tip feedback upper bound (Agent-D).

## Current honest rate

- Tip-Hybrid + planar_C: **4/10** on `outputs/scheme_l3/S2_force_enter_ep1_10v10/`.
- Type-A: **5/10** on `outputs/scheme_l3/S2_typeA_ep1_10/`.
- Track-B GMHQP: **6/10** on `outputs/scheme_l3/S2_trackB_gmhqp_ep1_10/` (ok 1,4,6,8,9,10).
- **Track-B GMHQP-AC (best deployable): 7/10** on `outputs/scheme_l3/S2_trackB_gmhqp_ac_ep1_10/` (ok 1,4,6,7,8,9,10; fail 2,3,5). Disclose in-hand tip-geom surrogate.
- Track-A tip_obs full **4/10** STOP; wave-6 tip_fuse smoke **1/7** keep-FAIL STOP (`docs/WAVE6_TRACK_A_PRIV.md`).
- Paper tables: `docs/PAPER_BASELINE_MATRIX_SUCCESS.md`.

## Oracle (Agent-D, not deployable)

Privileged tip→mouth servo: `docs/ORACLE_TIP_SERVO.md`, `configs/scheme_l3/S2_oracle_tip.yaml` (v1) / `S2_oracle_tip_v2.yaml` (v2).
Latest diagnostic upper bound: **9/10** on `outputs/scheme_l3/S2_oracle_tip_v2_ep1_10/` (v1 was **7/10**).
Do **not** put oracle RATE in the deployable table.
