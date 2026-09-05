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

> **Hard seat (2026-09-04):** 浮空搜孔不算成功。主表走 `docs/SUCCESS_STANDARD.md`（early≥0.85, cfrac≥0.80, n≥30）。旧论文 RATE（`min_contact_frac:0`）仅旁注。

离线重扫（旧跑、未重采）诚实贴面：Hybrid **1/10**、Type-A **1/10**、GMHQP **0/10**、GMHQP-AC **0/10**（论文曾 4/5/6/7）。
控制侧已加 pre-spiral SEAT + spiral 禁浮空搜；重跑 Rate 待 smoke/ep1–10。

- Paper tables: `docs/PAPER_BASELINE_MATRIX_SUCCESS.md`（须标 seat vs legacy）。

## Oracle (Agent-D, not deployable)

Privileged tip→mouth servo: `docs/ORACLE_TIP_SERVO.md`, `configs/scheme_l3/S2_oracle_tip.yaml` (v1) / `S2_oracle_tip_v2.yaml` (v2).
Latest diagnostic upper bound: **9/10** on `outputs/scheme_l3/S2_oracle_tip_v2_ep1_10/` (v1 was **7/10**).
Do **not** put oracle RATE in the deployable table.
