# Privileged Sensing Disclosure (paper-ready English)

## Simulator-only oracle

During simulation, we optionally read ground-truth peg-tip pose and tip–tray contact kinematics from the physics engine. These quantities are **not** available to deployable baselines and are used to (i) validate online coupling estimates $\hat{\mathbf{C}}(\mathbf{q})$, and (ii) report diagnostic metrics. They do **not** enter the control loop of ConnTact-style baselines (S0–S3) or ablations (S1/S2/S6), which use wrist F/T and proprioception only.

**Agent-D control oracle (`priv_oracle_tip_servo`, configs `S2_oracle_tip.yaml` / `S2_oracle_tip_v2.yaml`):** privileged tip + hole also close the **planar tip→mouth PD loop** (identity wrist mapping); v2 adds privileged mouth axial press + slip reseat. This is an **upper-bound diagnostic** for “if tip planar tracking to mouth were solved.” Exclude its RATE from deployable comparisons; see `docs/ORACLE_TIP_SERVO.md`.

**`obs_noise` / `priv_oracle_tip_obs` (`tip_gt+N(0,σ)`):** **REJECTED for paper claims** (user 2026-09-03). Hooks/configs may remain in tree but are **invalid** as an observability bridge. Valid path: real tip/contact estimation from eligible sensors, or methods that do not need privileged tip.

**Track A — contact tip estimator (`track_a_est_oracle`, config `S2_trackA_est_oracle_ctrl.yaml`):** same Oracle-v2 **control law**, tip input = `tip_hat` from force-weighted MuJoCo **peg↔tray contact positions** (+ wrist holdover), **not** peg body xpos and **not** tip_gt+noise. Design: `docs/TRACK_A_TIP_ESTIMATOR.md`. **Residual privilege (must disclose):** (i) sim `contact.pos` between named peg/tray geoms is a stand-in for extrinsic/tactile localization, not a deployable tactile array; (ii) slip-reseat still uses privileged grasp COM slip magnitude; mouth/socket may use known-path object TF. Report under Track A table, not deployable, until residual privilege is removed.

**Track-B GMHQP (`track_b_gmhqp`, config `S2_trackB_gmhqp.yaml`, RATE **6/10** on `S2_trackB_gmhqp_ep1_10`):** tip-task residual through planar grasp map \(C\) + Escande HQP (not Oracle tip_gt PD). **Residual privilege (must disclose in paper footnote):** in-hand **peg tip geom** used as an **object-localization surrogate** (same class as OIGS/Pfanne), **not** claimed tip-free deployable and **not** privileged tip→mouth GT in the control loop. See `docs/TRACK_B_GRASP_MAP_HQP.md`.

## Fair comparison protocol

All schemes share: handoff grasp, GraspQP stabilization, planned inward Archimedean spiral, and episode protocol (ep01–10). Only the **wrist-from-tip mapping layer** differs.

## Tactile replacement path (hardware)

Privileged tip kinematics are a **simulator-native oracle** for contact geometry. The deployable replacement is fingertip tactile (contact onset, shear, slip) + wrist F/T history + finger FK—the same estimator interface, without simulator ground truth. Do **not** use `tip_gt+noise` as that stand-in for paper claims.

## Table caption template

**Table X — Deployable spiral-search enter rate (SUCCESS_STANDARD; wrist F/T + proprioception + disclosed surrogates).** Include Tip-Hybrid / Type-A / GMHQP. Row *Oracle-v2 (diagnostic)* is an **upper bound** with privileged tip↔mouth; **exclude from deployable success-rate comparisons**. Footnote GMHQP in-hand tip-geom surrogate.
