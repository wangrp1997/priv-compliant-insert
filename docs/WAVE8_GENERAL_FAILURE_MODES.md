# Wave-8: general failure modes (F1–F4) + next upgrades

> Mandate: `docs/GENERAL_ALGO_MANDATE.md` — ep IDs are **diagnostic probes only**, not controller branches.  
> Gate: `docs/METHOD_GATE_NO_PATCH.md` · SUCCESS_STANDARD locked.  
> Base deployable: GMHQP **6/10** (`S2_trackB_gmhqp_ep1_10`). Oracle-v2 **9/10** diagnostic only.  
> Sources: `FAIL_PRIV_TAXONOMY.md`, `GMHQP_RESIDUAL_PRIV_RX.md`, `WAVE6_TRACK_{A,B}_PRIV.md`, `TRACK_B_GRASP_MAP_HQP.md`.  
> Diag: `PYTHONPATH=src python scripts/diag_spiral_priv.py <run>/ep*/ep*_summary.json`.  
> **No ep-specific controller design in this doc.**

---

## 0. Scientific problem (locked)

Known tip→mouth planar path under non-rigid grasp (\(C(q)\neq I\), slip): keep seat + axis≤25° + true force-enter.  
Control ring: wrist F/T, finger/extrinsic contact, proprioception — **no** tip GT / tip_gt+noise.

Oracle-v2 9/10 ⇒ physics reachable when tip↔mouth observed with correct law. Residual gap = **general observability + optimal control**, not numbered patches.

---

## 1. Failure modes as general phenomena

Episode IDs below are **examples only** (cite from priv diags). Rates are context; modes are cross-run.

| Mode | General phenomenon | Priv signature (cited numbers) | Example probes (not design set) | What Oracle-v2 shows when tip known |
|------|--------------------|--------------------------------|----------------------------------|-------------------------------------|
| **F1** | **Planar saturate / never mouth** — tip task (proxy or wrist) closes while true tip stays off mouth | `priv_planar_min` **~6–12 mm**, `priv_mouth=False`, often `spiral_timeout`; under GMHQP: `bl_tip_err` **~2–6 mm** ≪ `tip_spiral_err` **~12–34 mm** → **false tip closure** | Type-A: 3,9 (8.4 / 7.5 mm); GMHQP residual: 2,3,5,7 (8.9–11.7 mm); Joint smoke: 3,8,10 | tip→mouth PD → planar~4.5 + mouth on same probes |
| **F2** | **Grasp slip + tip lag** — in-hand slip displaces tip faster than seat/path recover | slip **~600–1032 mm** (Type-A class); tip_spiral_err max **~79–167 mm**; `priv_mouth=False`. Partial under GMHQP: slip **~164–187** still no mouth | Type-A: 7,10; Joint: 7; GMHQP mix: 5,7 | `slip_reseat` + SEAT + grasp tighten then tip PD → enter |
| **F3** | **Mouth-ish, axis / tilt / press fail** — planar near mouth but upright or force-enter fails | `priv_planar_min`~**4.5–5.9 mm** or mouth once; `priv_axis_ok=False` and/or `surface_tilt_or_tray` / `surface_press`; no force-hole | Oracle-v2: 4; Track-A tip_hat smoke: 4; Type-A: 8 | `hold_enter_seat` + upright hold before press; tip PD alone insufficient |
| **F4** | **Estimator upright / tip_hat inconsistency** — estimated tip drives law with wrong orientation cue | mouth / near_mouth True but axis fail; tip_hat upright cue ~**73°** (Track-A smoke) | Track-A est→oracle smoke: 3 | Same upright/hold law needs **better \(\hat t\)/axis**, not tip_gt+noise |

### Cross-mode facts (do not invent)

| Fact | Cite |
|------|------|
| GMHQP RATE **6/10** ok {1,4,6,8,9,10}, fail {2,3,5,7} all F1-class planar never mouth | `TRACK_B_GRASP_MAP_HQP.md` §5 |
| Type-A **5/10**; Oracle-v2 **9/10** (diag) fail only ep04 (F3) | `FAIL_PRIV_TAXONOMY.md` |
| tip_obs full **4/10** STOP (enter ep05; destroy 4/8/9); tip_fuse smoke **1/7** keep-FAIL STOP | `WAVE6_TRACK_A_PRIV.md` |
| FTIP smoke **0/4**; Hybrid planar_C **0/4** | `WAVE6_TRACK_B_PRIV.md`, `WAVE7_ON_GMHQP.md` |
| F2↓ ≠ F1 enter: ep07 slip 1032→198 (Joint) or 672→187 (GMHQP) still planar ~12 mm | taxonomy + residual RX |

**Dominant deployable bottleneck today:** F1 (false tip closure under \(C^{+}\) geom proxy). F2 is secondary/partial. F3 rare on GMHQP fails (axis OK-ish 4.8–14°). F4 is Track-A estimation failure class.

---

## 2. Per-mode literature → REUSE vs INVENT

Search (2026-09-03): TEC joint est–ctrl, Active Extrinsic, contact MPC, object impedance, factor graphs, Doshi contact config, Escande HQP. Nature/TRO/RSS: LeTac-MPC **TRO 2024**; Escande **IJRR 2014**; TacGraph / TEC / TEXterity / Active Extrinsic / Doshi are ICRA/MERL-class (cite). No Nature assembly paper displaces these for our math objects.

### F1 — false planar closure / tip–wrist inconsistency

| Candidate | Cite | Fits math object? | 1–2 wk on GMHQP | Verdict |
|-----------|------|-------------------|-----------------|---------|
| Kim **TEC** joint est–ctrl (ContactMotion + PoseDiff + wrench factors) | ICRA 2023; `refs/Tactile-Estimator-Controller` | Yes — estimate tip/contact while controlling estimable motions | Med–High (slim EKF/FG port; wave-7 in flight) | **REUSE** (Track A current) |
| Kim **Active Extrinsic** contact-line + regulated pivot | ICRA 2022 arXiv:2110.03555 | Yes — contact locus for insertion without full tip GT | Med | **REUSE** (Track B meas / cost; not hard tip_obj swap) |
| **Contact-opt / COMPC** short-horizon tip+contact residual on Escande+\(C^{+}\) | LeTac-MPC TRO spirit; Active Extrinsic \(e_c\); `TRACK_B_CONTACT_OPT_ON_GMHQP.md` | Yes — cost on \(\hat c\) not tip swap | Med (wave-7 in flight) | **REUSE compose** (Track B current) |
| Doshi contact-configuration regulation (wrench/motion constraints) | ICRA 2022 arXiv:2203.01203 | Partial — planar contact constraints without object pose | Med | **REUSE** meas/constraints after COMPC |
| Open-loop \(\hat t\)→tip task (tip_obs / tip_fuse / FTIP) | tried | Wrong structure | — | **Reject** (rates ≤ baseline) |
| Ep-if / keep-set switches | — | Forbidden | — | Forbidden |

**Invent?** Only if TEC-joint **and** COMPC both fail the *general* F1 object after full ep1–10: then invent **observability-gated tip residual** with new equation vs failed free-run — still cite TEC/Active Extrinsic ancestors. Prefer not invent while REUSE in flight.

### F2 — slip + tip lag

| Candidate | Cite | Fits? | 1–2 wk | Verdict |
|-----------|------|-------|--------|---------|
| Pfanne **object-level impedance** + GraspQP internal force | RA-L 2020; `refs/graspqp` | Yes — maintain grasp wrench / reseat | High (GraspQP in tree) | **REUSE** (Track B next after path OCP) |
| LeTac-MPC tactile-reactive grasp MPC | TRO 2024 arXiv:2403.04934 | Yes — slip-aware grasp width/force over horizon | Low–Med (GelSight-centric; adapt to proprio+F/T) | **REUSE structure** / cite |
| BGF / CouplingEKF slip reset | `refs/bgf`; in-tree EKF | Partial — covariance reset ≠ reseat law | High | Keep as meas hygiene |
| TEC energy / min grasp-wrench factors | Kim ICRA 2023 | Yes — slip prevention in joint FG | With TEC port | Bundle under Track A |
| FASR / mouth_press soups | — | No | — | Reject |

**Invent?** Unlikely in 1–2 wk: Pfanne+GraspQP already maps. Invent only if impedance+QP cannot cut slip while preserving seat.

### F3 — axis / tilt / press at mouth

| Candidate | Cite | Fits? | 1–2 wk | Verdict |
|-----------|------|-------|--------|---------|
| Escande **hard HQP** seat≻upright≻path | IJRR 2014 | Yes — path must not eat axis | High (HardHQP skeleton in tree) | **REUSE** (Track B if F3 returns after F1 fix) |
| Oracle-v2 `hold_enter_seat` priority (deployable rewrite without tip GT) | `ORACLE_TIP_SERVO.md` | Priority law yes; needs non-priv upright cue | Med | **REUSE priority**, not tip GT |
| TEXterity continuous pose for upright | ICRA 2024 arXiv:2403.00049 | Axis from extrinsic pose | Low (heavy iSAM) | Cite / long-horizon |
| tip_gt+noise | — | Forbidden | — | Forbidden |

**Invent?** Low priority until F1 closed; if needed = upright cone as hard Escande level with F/T+FK axis (already GMHQP spirit).

### F4 — tip_hat / axis estimate wrong

| Candidate | Cite | Fits? | 1–2 wk | Verdict |
|-----------|------|-------|--------|---------|
| TEC PoseDiff + NIS/contact gate (joint est–ctrl) | Kim ICRA 2023; wave-7 TEC-joint | Yes — only feed \(\hat t\) when estimable | Med | **REUSE** (current A) |
| **TacGraph** in-hand pose + extrinsic contact FG (force balance, non-penetration, kinematics) | arXiv:2512.23856; tacgraph.github.io | Yes — joint pose+contact; fixes free-run | Partial (no full FG in 2 wk); soft factors first | **REUSE structure** (next A if TEC-slim stalls) |
| TEXterity full iSAM | ICRA 2024 | Yes | No (heavy) | Cite ancestor |
| Always hard tip_obj←\(\hat t\) | tip_obs | No | — | Failed 4/10 |

**Invent?** Only if slim TEC+TacGraph soft factors cannot bound upright cue — then invent **axis-factor** (peg axis in FG) with TacGraph as ancestor.

---

## 3. Ranked **next** general upgrades (after TEC-joint / COMPC)

Do **not** condition this ranking on TEC-joint or COMPC rates. Assume those wave-7 lines either finish or STOP under METHOD_GATE; pick the **next** general law on GMHQP base.

### Track A (estimation) — next after TEC-joint

| Rank | Method | Attacks | Why next | 1–2 wk |
|------|--------|---------|----------|--------|
| **A1** | **TacGraph / fuller FG factors** (force-balance, non-penetration, contact kinematics) into gated tip — build on in-tree `tip_factor_graph.py` + TEC-joint gate; still soft→gate before GMHQP tip task | F4, residual F1 observability | TEC-joint = estimability gate; TacGraph physics fills tip_fuse gap; FG skeleton already landed | Med |
| **A2** | **Doshi / Active Extrinsic wrench–line as measurement class** into same FG/EKF (not tip_obj hard replace) | F1 meas when extrinsic contact intermittent | Complements TEC ContactMotion when tray contact sparse | High |
| **A3** | TEXterity continuous pose (cite-only until A1 fails) | F3/F4 pose | Heavy | Cite |

**Stop / reject for A:** free-run \(\hat t\); variance soups; ep switches; tip_gt+noise.

### Track B (control) — next after COMPC

| Rank | Method | Attacks | Why next | 1–2 wk |
|------|--------|---------|----------|--------|
| **B1** | **Pfanne object impedance + GraspQP reseat** as Escande-adjacent grasp law (spiral-wide \(w_{\mathrm{des}}\), friction-cone \(\lambda\); reseat on slip detect) | F2 → enables F1 recovery | COMPC targets path/contact residual; orthogonal gap is grasp wrench under slip | High |
| **B2** | **Active Extrinsic contact-line regulation** (pivot / consistent contact mode) **in hierarchy**, cost/constraint only — never FTIP-style tip_obj:=\(\hat c\) | F1 when proxy false-closes | COMPC already uses \(e_c\) in cost; next is **active** regulation of contact mode (Kim ICRA’22 controller half) | Med |
| **B3** | **HardHQP** seat≻upright≻path if mouth reached but F3 returns | F3 | Priority already partial in GMHQP; harden upright vs path | High |

**Stop / reject for B:** FTIP hard tip swap; Hybrid switch on `bl_tip_err`; FASR/OIGS/PHIG restack; ep keep-set structure inside controller.

### Dual-track dependency (general)

```
Oracle upper bound: tip known → 9/10
        │
   F1 observability ──Track A──► ˆt / ĉ consistent
        │                         │
   F1/F2 control law ──Track B──► GMHQP + (COMPC) + impedance/active contact
        │
   F3 only after mouth
```

---

## 4. Eval discipline (reporting ≠ patches)

1. Full ep1–10 RATE (disclose); fail IDs only in diag tables.  
2. Keep-set regression vs GMHQP 6/10 is a **reporting gate**, not `if ep in keep` in code (`GENERAL_ALGO_MANDATE.md`).  
3. RATE ≤ baseline → STOP retune; revise **algorithm class**, not thresholds.  
4. No Nature claim required; TRO/ICRA/IJRR structures above are sufficient cites.

---

## 5. One-page summary for agents

| Mode | One-line science | Prefer |
|------|------------------|--------|
| F1 | Proxy tip false-closes; true planar 9–12 mm | A: TEC→TacGraph; B: COMPC then Active Extrinsic regulate |
| F2 | Slip hundreds of mm; reseat before path | B: Pfanne+GraspQP |
| F3 | Near mouth, axis/press | B: Hard Escande upright |
| F4 | \(\hat t\) upright wrong | A: gated TEC / TacGraph factors |

**Top next (post wave-7):** Track A → **TacGraph-slim**; Track B → **Pfanne object impedance + GraspQP reseat**.
