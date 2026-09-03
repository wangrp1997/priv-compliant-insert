# Parallel paper-baseline campaign (2026-09-03)

## Rule
- Open-source → faithful port (ConnTact, Franka).
- No code → paper reimplementation, label honesty (PSFT).
- SUCCESS_STANDARD never loosened.
- Privileged diag maps each prior failure → our tip-referenced fix.

## Agents
| Agent | Task | Out |
|-------|------|-----|
| Done | ConnTact SpiralToFindHole | `S3_conntact_faithful_ep1_10` **0/10** |
| Done | Franka jam+spiral | `S3_franka_faithful_ep1_10` **0/10** |
| Done | Park PSFT paper reimpl | `S3_psft_faithful_ep1_10` **0/10** |
| Done | S0/S1/S2live/S6/S8 SUCCESS | all **0/10** → `PAPER_BASELINE_MATRIX_SUCCESS.md` |
| Done | priv map + v11 smoke | smoke **1/5** → **no full** |

## Reference rows already known
| Method | Rate | Note |
|--------|------|------|
| ConnTact faithful | 0/10 | Z-drop FP; wrist≠tip |
| Franka jam+spiral faithful | 0/10 | TCP spiral; tip-Z FP; jam float |
| PSFT faithful (paper reimpl) | 0/10 | eq(10) TCP; Z-drop FP; tip lag |
| S0 / S1 / S2live / S6 | 0/10 each | ablation/internal |
| S8 planar_C alone | 0/10 | ours without Tip-Hybrid |
| Tip-Hybrid+planar_C (v10) | **4/10** | current deployable best |
| Ours v11 smoke (2,3,4,8,9) | 1/5 | ep04 kept; no extra win |
| S9 EKF+QP | 1/10 | worse |
| Oracle tip→mouth | 7/10 | privileged upper bound |
