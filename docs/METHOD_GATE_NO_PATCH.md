# Method gate: theory for general diagnostic gap — no engineering patches

> User lock (2026-09-03): 新方案必须针对**通用特权诊断问题**，有**理论/文献依据**；禁止调参工程补丁。  
> **通用算法（强制）:** `docs/GENERAL_ALGO_MANDATE.md` — 失败 ep 只是探针；禁止按 ep 打补丁；做通用估计/优化/控制。

## Allowed

| Must | Meaning |
|------|---------|
| General gap | From priv taxonomy / Oracle contrast (F1 planar lag, F2 slip, F3 mouth axis, F4 tip_hat) — same class across eps, not one-ep hotfix |
| Theory | Named control/estimation structure + cite (hybrid force, impedance, HQP priority, TEC/extrinsic contact, object impedance, …) |
| Dual track | A = estimate Oracle inputs; B = algorithm without tip GT |
| Honest disclose | Residual privilege named; SUCCESS_STANDARD unchanged |

## Forbidden (patch)

- Threshold / gain soups: `seat_force_n`, outlier gates, freeze flags, mouth_r, press_scale retunes without new law
- Ep-specific switches / “if ep03 …”
- Re-stacking Type-A + Type-B + mouth under a new acronym with no new equation
- `tip_gt+noise` or peg xpos in control
- Claiming progress from a patch smoke that regresses then “we learned to tune less”

## Before coding checklist

1. Write/cite priv numbers for the gap.
2. One sentence: **estimation or control problem** this is (math object).
3. **Search first:** WebSearch + `refs/` + prior docs for an **existing published method** that matches that math object. If one fits (cite + feasibility 1–2 wk), **reuse/adapt it** — do not invent a parallel acronym.
4. **Invent only if search fails:** no adequate published structure for this gap → design a new equation with clear novelty vs tried methods (PHIG/CLEP/FASR/OIGS/…); still cite nearest ancestors.
5. Cite ≥1 paper/ref for the structure (reuse or ancestor).
6. State what you will **not** tune.
7. Only then implement; smoke validates the theory, not knob search.

## Reuse-vs-invent rule (user 2026-09-03)

| If | Then |
|----|------|
| Priv diagnosis names a gap **and** literature/refs already solve that gap | **Use / port** that scheme (honest cite + disclosure) |
| No adequate existing scheme for the diagnosed gap | **Invent** a principled optimization/control/estimation method (equations + anti-patch) |
| Forbidden either way | Threshold soups, ep-specific hacks, tip_gt+noise |
