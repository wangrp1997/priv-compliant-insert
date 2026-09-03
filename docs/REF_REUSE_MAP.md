# Reference reuse map (proxy-audited)

Pull/update refs: `source /opt/nros_tools/nros_rc && nros-proxy-on && bash scripts/clone_scheme_refs.sh`

Self-implemented modules below were checked against `refs/`; no runtime dependency on ref code.

## Port (math/logic copied or translated)

| Our file | Ref path | License | What |
|----------|----------|---------|------|
| `src/pci/conntact_repro.py` | `refs/ConnTact/.../spiral_search.py` `SpiralToFindHole` | Apache-2.0 | **Faithful** time spiral + 0.4mm height drop |
| `src/pci/franka_spiral_repro.py` | `refs/franka-peg-in-hole/scripted_expert_peg_v1.py` jam spiral | MIT | **Faithful** jam spiral r/θ + lift predicate |
| `scripts/validate_conntact_spiral_ref.py` | `refs/ConnTact/src/conntact/spiral_search.py` `SpiralToFindHole.get_spiral_search_pose` | Apache-2.0 | Archimedean spiral XY, validation only |
| `scripts/validate_conntact_faithful.py` | same | Apache-2.0 | Numeric lock for paper baseline |
| `src/pci/sim_runner.py` (`conntact_time_spiral`) | same ConnTact spiral + seeking + height enter | Apache-2.0 | **S3_conntact_faithful** paper baseline |
| `src/pci/sim_runner.py` (`franka_spiral`) | Franka jam spiral + TCP track + height enter | MIT | **S3_franka_spiral_faithful** paper baseline |
| `src/pci/sim_runner.py` (planned_spiral target) | same ConnTact spiral + seeking force along −normal | Apache-2.0 | Legacy S3 geometric (not faithful) |
| `src/pci/compliant/priv_grasp_opt.py` | `refs/graspqp` friction cone + bounded LS QP pattern | (see repo) | Phase-A finger QP |
| `src/pci/track_b_phig.py` | GraspQP + Escande HQP (cite); hybrid F/P (Raibert) | cite | Track-B wrist path + grasp co-opt (no tip GT) |
| `src/pci/track_b_hybrid_gmhqp.py` | Escande HQP + GMHQP tip-task + S8 `PlanarCouplingEstimator` | cite | **REUSE compose** Level-2 tip_task∨planar_C (docs/TRACK_B_HYBRID_GMHQP.md §0); no invent |
| `src/pci/track_b_clep.py` | Kim Active Extrinsic Contact (cite); Doshi F/T lever | cite | Track-B contact-line pivot from wrist wrench (no tip GT) |
| `src/pci/tip_theory_estimator.py` | Kim TEC / TEXterity / Active Extrinsic (cite); Doshi wrench; CouplingEKF slip | cite | Track-A TEC-slim tip+C EKF (no tip=wrist+o; no freeze patch) |
| `sim_runner` tip_hat→GMHQP (`track_a_gmhqp_tip_obs`) | TEC-slim + `track_b_gmhqp` (Montana/Escande) | cite | Wave-5 **reuse**: replace geom tip proxy in GMHQP tip task; no new acronym (`docs/TRACK_A_GMHQP_TIP_OBS.md`) |
| `tip_theory_estimator` + `track_a_gmhqp_tip_fuse` | TEC PoseDiff/DispDiff + TacGraph in-hand prior + GMHQP | cite | Wave-6 **reuse**: soft geom tip meas in TEC-slim; stop ˆt free-run (`docs/TRACK_A_GMHQP_TIP_FUSE.md`) |
| `track_a_tec_joint_gmhqp` (wave-7) | Kim TEC ICRA 2023 joint est–ctrl + GMHQP | cite | NIS/contact-gated \(\hat t\) into tip task (`docs/TRACK_A_TEC_JOINT_ON_GMHQP.md`) |
| `track_b_gmhqp_compc` (wave-7) | Active Extrinsic \(e_c\) + LeTac-MPC-style short-horizon \(J\) on GMHQP | cite | Contact residual in cost, not tip_obj swap (`docs/TRACK_B_CONTACT_OPT_ON_GMHQP.md`) |
| `src/pci/tip_factor_graph.py` | `refs/Tactile-Estimator-Controller` FACTORS.md + Kim TEC ICRA 2023 | cite | Track-A **fuller** factor-graph skeleton (ContactMotion/Wrench/Torq/Energy/Pen*); orthogonal to gated TEC-slim EKF (`docs/TRACK_A_FACTOR_GRAPH_TIP.md`) |

## Adapt (concept + simplified sim substitute)

| Our file | Ref path | What |
|----------|----------|------|
| `src/pci/tip_tracking_baselines.py` `PlanarCouplingEstimator._slip_reset` | `refs/bgf` slip gating idea | dw large & dt small → reset C=I |
| `src/pci/coupling_ekf.py` | `refs/bgf` + Pfanne IROS'17 | EKF on tip+C; slip resets covariance |
| `src/pci/tip_tracking_qp.py` | `refs/graspqp` + Escande HQP (cite) | ridge LS wrist step with step clip |
| `src/pci/compliant/left_wrist_admit.py` | `refs/irl_control` OSC admittance | wrist F/T → small displacement |
| `src/pci/tip_tracking_baselines.py` `spiral_force_gate` | ConnTact compliance + franka-peg F threshold | Frozen grasp: scale track when resid/slip high |

## Cite-only (no code port)

| Topic | Ref |
|-------|-----|
| Full factor-graph extrinsic pose | `Tactile-Estimator-Controller`, Kim/TEXterity |
| TacGraph in-hand + extrinsic contact FG | arXiv:2512.23856 (MERL 2026); force balance / non-penetration factors — Wave-8 **A1** candidate |
| Active Extrinsic contact-line | Kim ICRA 2022 arXiv:2110.03555 — Wave-8 **B2** / COMPC \(e_c\) |
| Doshi contact-configuration regulation | ICRA 2022 arXiv:2203.01203 — Wave-8 **A2** wrench/line meas |
| Object impedance (Pfanne) | RA-L 2020 — Wave-8 **B1** reseat/grasp law |
| NLP MPC slip recovery | `refs/bgf`, `LeTac-MPC` (TRO 2024) — COMPC \(J\) ancestor |
| Escande hierarchical QP | IJRR 2014 — HardHQP / GMHQP priority |
| DP tactile servo | `refs/dpse` |
| Parallel-gripper slip control | `refs/Slip-Aware-Object-Manipulation` |
| In-hand regrasp | `refs/in_hand_manipulation_2` |

## Wave-8 general-mode map

See `docs/WAVE8_GENERAL_FAILURE_MODES.md` (F1–F4 phenomena; next A=TacGraph-slim, B=Pfanne+GraspQP after TEC-joint/COMPC).

## Scheme → implementation

| Scheme | Mode | Module |
|--------|------|--------|
| S3 | `conntact_wrist` | Legacy geometric spiral (not paper-faithful) |
| S3f | `conntact_time_spiral` + `conntact_tcp` | **Faithful** ConnTact SpiralToFindHole (`S3_conntact_faithful.yaml`) |
| S3fr | `franka_spiral` + `franka_tcp` | **Faithful** Franka jam spiral (`S3_franka_spiral_faithful.yaml`) |
| S1/S2 | `tip_servo_ff` / `tip_track_live` | offset FF / live tip track |
| S6 | `priv_coupling` | `TipCouplingEstimator` scalar α |
| S8 | `planar_C` | `PlanarCouplingEstimator` 2×2 LS |
| S9 | `ekf_qp` | `CouplingEKF` + `solve_wrist_qp` + Tip-Hybrid + slip-freeze-adv |

## S9 Agent-R upgrade (2026-09)

- `coupling_ekf.slip_reset`: inflate `P_c`, boost `process_c`, expose `just_slipped` / `tip_innov_norm`
- `sim_runner` planned_spiral: freeze `planned_i` on EKF slip or tip_err > 8 mm
- `parallel_force_enter_ep10.py --config` for S9 yaml
- Success standard unchanged (`docs/SUCCESS_STANDARD.md`)

## Proxy audit checklist

- [x] Refs cloned under `refs/` via `nros-proxy-on`
- [x] ConnTact spiral cross-checked in `validate_conntact_spiral_ref.py`
- [x] Slip reset threshold aligned with `PlanarCouplingEstimator.slip_reset_dw_m` (0.0012 m)
- [x] S9 shares S8 surface invariants (plane lock, along suppress) via `_is_planar_c_mode`
