# GMHQP residual fails — priv RX (ep2/3/5/7)

> Baseline now: GMHQP **6/10**. Oracle **9/10** diag. Type-A **5/10**.  
> Sources: `diag_spiral_priv.py` on `S2_trackB_gmhqp_ep1_10` vs Type-A / Oracle same eps.  
> Method gate: `docs/METHOD_GATE_NO_PATCH.md`.

## 1. Numbers

| ep | GMHQP planar | mouth | slip | tip_err_max | Type-A same | Oracle same | Class |
|----|--------------|-------|------|-------------|-------------|-------------|-------|
| **02** | **11.2** | F | 24.5 | 19.7 | **enter** planar 4.4 mouth T | enter 4.5 | **F1 + regress** (lost Type-A win) |
| **03** | **11.5** | F | 16.9 | 18.2 | fail 8.4 mouth F slip 18.6 | enter 4.5 | **F1 pure** (low slip; GMHQP worse than Type-A) |
| **05** | **8.9** | F | 164.5 | 13.3 | **enter** 4.5 mouth T | enter 4.4 | **F1/F2 mix + regress** |
| **07** | **11.7** | F | 186.8 | 34.1 | fail 10.2 slip **672** | enter 4.5 slip 4.6 | **F2→F1 residual** (slip↓ vs Type-A, planar stuck) |

All four: `spiral_timeout`, `priv_mouth=False`, axis OK-ish (4.8–14°) — **bottleneck is planar never mouth**, not F3 axis-at-mouth.

Oracle on all four: short spiral, planar~4.5, mouth T, enter — tip PD closes the gap.

## 2. Scientific problems (residual)

1. **F1 residual under GMHQP:** tip task via \(C^{+}\) still saturates planar 9–12 mm on {2,3,5,7}; bl_tip_err often small (2–6 mm) while tip_spiral_err ~12–34 mm → **proxy tip / \(C\) used in law ≠ true tip path**.  
2. **Regress vs Type-A on {2,5}:** Type-A planar_C entered; GMHQP did not — grasp-map tip proxy **hurt** episodes Type-A already solved.  
3. **F2 partial on {5,7}:** slip 164–187 ≪ Type-A 672 on ep7, still no mouth — slip fix ≠ planar enter (same lesson as HardHQP).

## 3. Dual-track prescription

### Track A — observability of tip used by Oracle / by GMHQP

| | |
|--|--|
| **病** | Control tip proxy wrong → planar lag; Oracle tip GT succeeds same eps |
| **药** | **Deployable tip observer for GMHQP residual eps**: estimate \(\hat t\) from extrinsic contact + finger kinematics (TEC), **replace in-hand geom proxy** in tip task \(e=\Pi(\hat t-p^*)\); same GMHQP or Oracle-v2 law |
| **Cite** | Kim TEC / TEXterity / Active Extrinsic |
| **禁** | tip_gt in control; variance soups; freeze patches |
| **验** | smoke eps **2,3,5,7**; gate: enter-win ≥1 vs GMHQP fails OR planar→&lt;5+mouth |

### Track B — control when proxy tip saturates (no tip GT mode)

| | |
|--|--|
| **病** | \(C^{+}\) tip task with bad \(p_{\mathrm{tip}}\) saturates; lost Type-A wins |
| **药** | **Hybrid tip-task + planar_C fallback HQP**: when tip-spiral residual saturates / mouth not approached, Escande-switch to Type-A-style planar_C (known path wrist via estimated \(C\)) without abandoning seat/axis hard constraints; OR **online \(C\) from ContactMotion** (not frozen geom tip) |
| **Cite** | Escande HQP hierarchy; Montana/Pfanne; CouplingEKF |
| **禁** | FASR; mouth_press scale soup; rebrand OIGS |
| **验** | same eps **2,3,5,7**; must not lose GMHQP wins on 1,4,6,8,9,10 if full later; fail-subset first |

## 4. Parallel launch

Wave-5 agents cite this file as RX.
