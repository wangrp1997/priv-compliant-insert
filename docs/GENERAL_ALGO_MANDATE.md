# General algorithm mandate (not episode patches)

> User lock (2026-09-03): 特权诊断里的失败 ep **只是探针/特例**；失败案例还有很多。  
> **禁止**按 ep 打补丁、keep-set 工程、fail-subset 专用开关。  
> **必须**针对通用科学问题设计估计/优化/控制算法；用全分布（ep1–10 / 更多种子）评 RATE。

## Scientific problem (general)

已知 tip→口平面轨迹下，非刚性 dexterous grasp（\(C(q)\neq I\)、滑移）中：

\[
\min \; J_{\mathrm{track}}(p_{\mathrm{tip}},p^*) + J_{\mathrm{seat}} + J_{\mathrm{upright}} + J_{\mathrm{enter}}
\]

约束：贴面、轴锥≤25°、真力入孔（SUCCESS_STANDARD）。  
观测：腕 F/T、指尖/外接触、本体感觉；**控制环禁用 tip GT / tip_gt+noise**。

Oracle-v2 9/10 只说明：**若 tip↔口可观测且律正确，物理上可达**。算法要解决的是 **一般** 的可观测性 + 最优控制，不是修某几个编号。

## What episode lists are for

| OK | NOT OK |
|----|--------|
| Use fails as **diagnostic evidence** of failure modes (F1/F2/F3/F4) | Design `if ep in {2,3,5,7}` logic |
| Report RATE on ep1–10 (or larger) | Optimize only fail-subset then claim solved |
| Ablate on a mode class | “Keep-set must stay green” as **algorithm structure** (eval gate ≠ controller) |

Eval gates may still refuse expand if RATE regresses vs GMHQP 6/10 — that is **reporting**, not a patch inside the controller.

## Track A — general estimation

**Problem:** estimate extrinsic tip / contact / axis under non-rigid grasp for any episode.  
**Algorithm class:** TEC / TEXterity joint estimation–control, factor graph, consistent contact observer.  
**Not:** tip_hat that only helps ep05; fuse knobs tuned on {4,8,9}.

## Track B — general optimal control

**Problem:** tip-referenced tracking + seat/upright/enter under \(C\neq I\) for any grasp/slip realization.  
**Algorithm class:** grasp-map / object-task HQP or contact MPC / active extrinsic regulation **as a general law**.  
**Base:** GMHQP is a candidate general structure — improve the **law**, don’t episode-patch on top.  
**Not:** FT tip only for residual eps; hybrid switch trained on fail IDs.

## Deliverable standard

1. Equations that do not mention episode indices.  
2. Cite (reuse) or invent with nearest ancestors.  
3. Evaluate on full ep1–10 (and disclose).  
4. If RATE ≤ baseline: STOP retune; revise **algorithm**, not thresholds.
