# Wave-7: optimize **on** GMHQP 6/10 (stop regressing)

> User pushback (2026-09-03): 要用算法解决特权诊断问题；不能越做越差；可在最好方案上继续优化；双路并行；顶刊级估计/控制。

## 1. Why recent work got worse (honest)

| Attempt | RATE | Why worse |
|---------|------|-----------|
| tip_obs → GMHQP | **4/10** | Free \(\hat t\) drifted; destroyed GMHQP wins 4/8/9 |
| tip_fuse | **1/7** keep-fail | Soft prior still let \(\hat t\) wreck keep-set |
| GMHQP-FTIP | **0/4** | Wrist F/T tip proxy; tip_err exploded 60–80 mm |
| Hybrid planar_C | **0/4** | No enter-win; did not fix false tip closure |

**Root:** we swapped the **tip argument** of GMHQP without a calibrated observer that stays consistent with true tip. Privileged gap is still F1: `priv_planar` 9–12 mm while `bl_tip_err` small → **proxy tip false closure**.

## 2. Non-negotiable this wave

1. **Base = GMHQP 6/10** (`S2_trackB_gmhqp.yaml` / `S2_trackB_gmhqp_ep1_10`).  
2. **Keep-set gate:** episodes {1,4,6,8,9,10} must remain enter in any full run; fail-subset must not be bought by destroying keep-set.  
3. **Algorithm, not REUSE-of-a-broken-proxy:** factor-graph / joint est-ctrl / contact-line MPC / horizon optimization with equations.  
4. METHOD_GATE: search first; invent only if needed; no threshold soups.

## 3. Diagnosed math problems

- **A (estimation):** recover extrinsic tip (and contact) consistent with tray contact + grasp, s.t. \(\|\hat t - t_{\mathrm{true}}\|\) small enough that GMHQP tip task closes planar to &lt;5 mm — **Kim TEC joint estimation–control** (ICRA’23), not open-loop EKF free-run.  
- **B (control):** given imperfect tip, optimize wrist+grasp over a horizon / contact line so true tip tracks \(p^*\) — **Active Extrinsic Contact** (Kim ICRA’22) + / or **contact-centric MPC** (TACTIC RSS / LeTac-MPC TRO spirit on path residual), layered **on** GMHQP HQP.

## 4. Dual track

| Track | Build on | Target algorithm class | Cite |
|-------|----------|------------------------|------|
| A | GMHQP tip task input | Simultaneous tactile estimation **and** control (factor graph / TEC) | Kim ICRA 2023 TEC; TEXterity; Active Extrinsic |
| B | GMHQP Escande+\(C^{+}\) | Contact-line / short-horizon tip residual MPC + grasp impedance | Kim Active Extrinsic; LeTac-MPC TRO; TACTIC RSS; GraspQP |

Smoke order: fail-subset **2,3,5,7** + **keep smoke 1,4,6** (cheap keep check) before any full 10.
