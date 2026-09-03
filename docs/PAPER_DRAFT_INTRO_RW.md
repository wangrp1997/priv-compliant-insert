# Grasp-Aware Extrinsic Tip Tracking for Compliant Spiral Hole Search with Dexterous Hands

> **Methodology draft (publication lock)** — problem formulation, method layers, and aggregate results fixed; hardware tactile extension noted where applicable.  
> Last sync: SUCCESS_STANDARD rates (`docs/PAPER_BASELINE_MATRIX_SUCCESS.md`); legacy lat metrics (`docs/BASELINE_TABLE.md`, `outputs/scheme_l3/baseline_matrix.json`).  
> **Deployable headline RATE:** Track-B GMHQP **6/10** (Type-A 5/10; Tip-Hybrid 4/10; published baselines 0/10). Oracle-v2 **9/10** diagnostic only.

---

## 中文摘要（草稿）

工业 peg-in-hole 的螺旋搜孔默认 peg 与末端执行器刚性固连。灵巧手非刚性抓握下，腕部轨迹不等于尖端在托盘面上的接触轨迹；配置相关的 grasp coupling $\mathbf{C}(\mathbf{q})$ 使 ConnTact 式腕螺旋与恒力贴面会楔入、抬离或完全失效。本文形式化 **extrinsic tip-referenced spiral search**：在 tip 接触平面上跟踪 Archimedean 螺旋，在线估计平面 coupling，通过 $\mathbf{C}^{-1}$ 将 tip 跟踪误差映射为腕命令，并施加贴面、力界与 slip 重置约束。我们在 DexJoCo 双臂 **handoff grasp** 实例上对比 ConnTact（S3）、冻结/在线 offset（S1/S2）、标量 $\alpha$ 消融（S6）与 **planar $\mathbf{C}$** 完整方法；aggregate 表显示 ConnTact 在 dexterous grasp 下系统性失败，tip-reference 与 coupling 可改善横向接近，但 along 漂移与进孔率仍开放。真机路线为指尖触觉 + 腕 F/T；视觉仅用于螺旋前粗对齐。

---

## Abstract

Peg-in-hole assembly with **Archimedean spiral search** is standard in industrial force-controlled insertion (ConnTact, PSFT), under the assumption that the peg is **rigidly coupled** to the robot end-effector. Dexterous multi-finger grasps violate this assumption: configuration-dependent **grasp coupling** $\mathbf{C}(\mathbf{q})$ decouples wrist motion from **extrinsic tip–tray contact**, causing wedge forces, loss of surface contact, and missed hole detection. We formulate **grasp-aware extrinsic tip tracking** for on-surface spiral search: command motion in a tip-referenced contact frame while estimating online **planar coupling** $\mathbf{C}(\mathbf{q}) \in \mathbb{R}^{2\times 2}$ and inverting it to map tip tracking error to wrist commands under surface, force, and slip-reset constraints. In bimanual Allegro simulation (DexJoCo), on a **handoff-grasp instantiation** of tray insertion, we compare ConnTact-style wrist spiral (S3), frozen and live tip–wrist offset tracking (S1/S2), scalar-$\alpha$ ablation (S6), and our **planar-$\mathbf{C}$** method against a surface-lock negative baseline (S0). Across ten episodes, ConnTact fails systematically; tip-referenced tracking and explicit coupling improve lateral approach metrics, but along-surface drift and hole entry remain open. Vision is used only for pre-contact alignment; spiral closure relies on **fingertip tactile and wrist F/T**, not wrist wrench alone.

**Keywords:** dexterous manipulation, peg-in-hole, spiral search, extrinsic contact, grasp coupling, tactile insertion

---

## 1. Introduction

### 1.1 Motivation

High-precision insertion after coarse visual alignment typically relies on **compliant spiral search** on a contact surface: the peg tip executes an Archimedean trajectory while maintaining light normal force until the hole mouth is detected by a force signature (e.g., axial drop) [ConnTact; Watson et al., 2020; Park et al., RA-L 2020]. Open-source frameworks such as **ConnTact** (SwRI) implement this pipeline with Cartesian compliance and `SpiralToFindHole` state machines, and have been validated on NIST assembly boards with parallel-jaw or fixed peg–TCP coupling [GitHub: swri-robotics/ConnTact].

**Dexterous hands** introduce a qualitatively different failure mode. The controlled body during search is not the wrist frame but the **extrinsic contact** between the peg tip and the tray surface. Under non-rigid grasp, the mapping from wrist commands to tip motion is **configuration-dependent**: grasp coupling $\mathbf{C}(\mathbf{q})$ varies with finger joint configuration $\mathbf{q}$, slip, stiction, and internal load redistribution. Wrist-mounted F/T does not equal tip–tray contact wrench when fingers redistribute load. Methods that spiral the **wrist** or blindly **force-seat** the surface therefore wedge, slide, or lift the peg—behaviors we observe consistently in simulation.

### 1.2 Problem statement

We target the **surface spiral search phase** of bimanual tray insertion, instantiated here with a **handoff grasp**: after PBVS coarse alignment, the right hand holds a peg while the left hand stabilizes the tray (DexJoCo / PCI). The formulation generalizes beyond this setup; the experiments evaluate this instantiation only.

The scientific question is:

> **How to execute extrinsic tip-referenced spiral search when planar grasp coupling $\mathbf{C}(\mathbf{q})$ between wrist and tip is unknown and time-varying?**

This is **not** a force-scalar tuning problem. It is **contact geometry + configuration-dependent coupling + constrained compliance**, with three coupled requirements:

1. **Tip-referenced spiral:** track a planned Archimedean trajectory in the **tip contact plane**, not the wrist frame.
2. **Grasp coupling:** model planar wrist-to-tip motion as $\Delta \mathbf{p}_{\text{tip}} \approx \mathbf{C}(\mathbf{q})\, \Delta \mathbf{q}_{\text{wrist}}$ (equivalently $\Delta \mathbf{p}_{\text{tip}} \approx \mathbf{C}(\mathbf{q})\, \Delta \mathbf{p}_{\text{wrist}}$ in the contact tangent basis).
3. **Observability & recovery:** detect slip and over-force; reset coupling estimates on slip events; wrist F/T alone is insufficient for closure on hardware—fingertip tactile is required.

### 1.3 Approach

We decompose the controller into three layers (estimation, inversion, constraints; §4). At a high level:

- **Pre-spiral:** PBVS / socket bias for vision-assisted coarse alignment.
- **Spiral phase:** inward Archimedean waypoints on the tip contact plane; wrist commands obtained by inverting estimated $\mathbf{C}(\mathbf{q})$.
- **Grasp stabilization:** GraspQP-style contact force optimization [`priv_grasp_opt.py`] during search—orthogonal to the tip trajectory layer.
- **Compliance:** light axial press and left-wrist admittance as guards, not the primary search mechanism.

**Evaluation family** (`src/pci/tip_tracking_baselines.py`): S0–S3 are baselines; S1/S2 are tip-tracking ablations; S6 is scalar-$\alpha$ coupling ablation; the **proposed method** is **planar $\mathbf{C}(\mathbf{q})$** with surface/along constraints and slip-triggered reset.

### 1.4 Contributions

1. **Problem formulation:** extrinsic tip-referenced spiral search under non-rigid dexterous grasp, with explicit planar coupling $\mathbf{C}(\mathbf{q})$ and constraint set (§3).
2. **Baseline suite:** reproducible ConnTact-port and tip-servo ablations in unified PCI simulator (ep01–10 L3 harness).
3. **Method:** three-layer grasp-aware tip tracking—coupling estimation, wrist inversion, surface/force/slip constraints—with scalar-$\alpha$ ablation (S6) isolating the value of full planar $\mathbf{C}$.
4. **Evaluation:** SUCCESS_STANDARD enter rates on handoff-grasp ep01–10: published baselines **0/10**, Tip-Hybrid **4/10**, Type-A **5/10**, GMHQP **6/10** (best deployable; tip-geom surrogate disclosed); Oracle-v2 **9/10** diagnostic only (§5.2).

### 1.5 Paper positioning

| Venue fit | Scope |
|-----------|--------|
| ICRA / IROS / RA-L (sim) | Formulation + baselines + planar-$\mathbf{C}$ method; honest limitation on entry rate |
| Stronger | Multi-grasp generalization + tactile sim2real |
| Not claiming | SOTA full bimanual assembly; Nature-level without broad real generalization |

---

## 2. Related Work

Organized by proximity to our **spiral search + non-rigid grasp** gap. Local code refs: `priv_compliant_insert/refs/`.

### 2.1 Industrial spiral search & Cartesian compliance (rigid EE)

**ConnTact** [SwRI, 2021] — modular force-controlled assembly; `SpiralToFindHole` combines seeking wrench and time-parametric Archimedean pose offsets with Z compliance for hole drop detection.  
→ **Gap:** peg–TCP rigidity; no finger slip.  
[https://github.com/swri-robotics/ConnTact](https://github.com/swri-robotics/ConnTact)

**Watson et al., Advanced Robotics 2020** — WRS assembly challenge; spiral search + tilt insertion primitives with F/T thresholds.  
→ **Gap:** industrial grippers; cites Van Wyk dexterous hand but not spiral under grasp compliance.  
[https://arxiv.org/abs/2002.02580](https://arxiv.org/abs/2002.02580)

**Park et al., RA-L 2020 (PSFT)** — Partial Spiral Force Trajectory; compliance-based blind search without F/T.  
→ **Gap:** arm/joint compliance only; dexterous hand adds DOF.  
[http://dyros.snu.ac.kr/wp-content/uploads/2020/06/RAL_Hyeonjun.pdf](http://dyros.snu.ac.kr/wp-content/uploads/2020/06/RAL_Hyeonjun.pdf)

**Jasim et al., Procedia CIRP 2014** — Archimedean spiral parameters vs clearance.  
→ **Gap:** tip frame assumed fixed to TCP.

**Kang et al., RA-L 2022** — uncertainty-driven spiral from visual pose.  
→ **Gap:** uncertainty model omits grasp compliance.

**Alt et al., IROS 2022 (∂PSE / dpse)** — data-driven optimization of search strategies.  
→ **Gap:** optimizes rigid TCP search; local repo: `refs/dpse`.  
[https://arxiv.org/abs/2207.07524](https://arxiv.org/abs/2207.07524)

**Our use:** S3 baseline ports ConnTact spiral + compliance pattern; PSFT/∂PSE cite as classical search literature, not direct competitors on dexterous grasp.

---

### 2.2 Dexterous / multi-finger peg-in-hole (grasp ≠ rigid EE)

**Van Wyk et al., IEEE TRO 2018** — NIST benchmark; four-finger **force-based manipulation** for PiH; hand adjusts peg pose in grasp.  
→ **Closest industrial dexterous PiH**; does not focus on tray-surface spiral.  
[https://pmc.ncbi.nlm.nih.gov/articles/PMC11008496/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11008496/)

**Choi et al., RA-L 2021** — in-hand **kinesthetic sensing** estimates hole direction before search; reduces blind spiral variance for three-finger gripper.  
→ **Highly relevant:** finger-level sensing replaces rigid TCP spiral; different task geometry (pre-align vs online spiral).  
[https://doi.org/10.1109/lra.2021.3107938](https://doi.org/10.1109/lra.2021.3107938)

**Our use:** motivates that **multi-finger PiH requires hand-level state**, not wrist-only spiral; we differ by **online tip tracking during spiral** on a held tray.

---

### 2.3 Extrinsic contact estimation & tactile extrinsic dexterity

**Kim et al., ICRA 2023** — Simultaneous Tactile Estimation and Control of Extrinsic Contact; factor graph (GTSAM/iSAM), limit surface.  
[https://arxiv.org/abs/2303.03385](https://arxiv.org/abs/2303.03385) | Code: [Tactile-Estimator-Controller](https://github.com/sangwkim/Tactile-Estimator-Controller) (`refs/Tactile-Estimator-Controller`)

**Bronars et al., ICRA 2024 / IEEE TASE 2025 (TEXterity)** — Viterbi discrete pose + continuous iSAM estimator-controller; extrinsic regrasp for tool use / insertion prep.  
[https://arxiv.org/abs/2403.00049](https://arxiv.org/abs/2403.00049)

**Van der Merwe et al., RA-L 2026 (TacGraph)** — distributed tactile factor graph for in-hand + extrinsic contact.  
[https://arxiv.org/abs/2512.23856](https://arxiv.org/abs/2512.23856)

**Kim & Rodríguez, ICRA 2022** — active extrinsic contact sensing with GelSlim.

**Liang et al., ICRA 2024** — robust in-hand manipulation with extrinsic contacts (motion cones).

→ **Common gap:** estimate/control **known-environment** contacts or regrasp; **no tray-plane blind spiral search** with large initial lateral error.  
→ **Our use:** coupling estimation and tactile closure follow the spiral contact phase; full factor-graph estimation is future work.

---

### 2.4 Slip-aware manipulation & tactile insertion

**Nazari et al., Nature MI 2025 (bgf)** — ACTP tactile prediction + MPC **trajectory modulation** (not grip-force-only); uSkin tactile.  
[https://www.nature.com/articles/s42256-025-01062-2](https://www.nature.com/articles/s42256-025-01062-2) | `refs/bgf`

**Mandil et al., RSS 2022 (ACTP)** — action-conditioned tactile prediction.  
[https://arxiv.org/abs/2205.09430](https://arxiv.org/abs/2205.09430) | `refs/action_conditioned_tactile_prediction`

**Dong et al., ICRA 2021 (Tactile-RL insertion)** — GelSlim tactile flow vs wrist F/T ablation; tactile generalizes better for unknown geometry.  
[https://arxiv.org/abs/2104.01167](https://arxiv.org/abs/2104.01167)

**Lenz et al., CoRL 2024** — vision vs tactile Shapley analysis for dexterous insertion; tactile critical for z/jam at 0.5 mm tolerance.  
[https://arxiv.org/abs/2410.23860](https://arxiv.org/abs/2410.23860)

**Xu et al., IEEE TRO 2024 (LeTac-MPC)** — differentiable tactile MPC.  
[https://doi.org/10.1109/tro.2024.3463470](https://doi.org/10.1109/tro.2024.3463470) | `refs/LeTac-MPC`

**Costanzo / Slip-Aware-Object-Manipulation** — controlled slippage for reorientation.  
→ `refs/Slip-Aware-Object-Manipulation`

→ **Our use:** slip detection and re-seat during spiral; scalar $\alpha$ ablation (S6) shows isotropic coupling is insufficient—motivating full planar $\mathbf{C}$.

---

### 2.5 Grasp force optimization & contact-implicit planning

**Zurbrügg et al., CoRL 2025 (GraspQP)** — differentiable QP for force closure; dexterous grasps.  
[https://arxiv.org/abs/2508.15002](https://arxiv.org/abs/2508.15002) | `refs/graspqp`

**in_hand_manipulation_2** — contact-implicit MPC (Crocoddyl); extrinsic contact modes.  
→ `refs/in_hand_manipulation_2`

**Our use:** GraspQP-style layer in `priv_grasp_opt.py` stabilizes peg-in-hand during spiral; **orthogonal** to tip trajectory layer.

---

### 2.6 Vision for insertion

**Lee et al., ICRA 2019** — multimodal RL peg insertion (vision + wrist F/T).  
**Kang et al., RA-L 2022** — visual uncertainty spiral.

→ **Our use:** vision for **pre-spiral PBVS only**; spiral closure is contact-dominated (occlusion, sub-mm).

---

### 2.7 Summary: literature gap → our niche

| Line of work | Controls | Assumes | Missing for our task |
|--------------|----------|---------|----------------------|
| ConnTact / PSFT | wrist spiral + F/T | rigid peg–EE | tip ≠ wrist under grasp |
| TEXterity / TacGraph | tactile FG, regrasp | known env / alignment | blind tray spiral search |
| Tactile-RL / Touch2Insert | tactile insert | local align phase | bimanual + spiral on tray |
| GraspQP | pre-grasp forces | static grasp | extrinsic tip motion |
| bgf / ACTP | slip-modulated traj | pick-place / pivot | spiral + hole detect |

**We fill:** *extrinsic tip-referenced spiral search* with **explicit planar grasp coupling** $\mathbf{C}(\mathbf{q})$ between wrist commands and tip contact, benchmarked against ConnTact and tip-servo ablations in dexterous bimanual sim.

---

## 3. Problem Formulation

### 3.1 Frames and state

Let $\{W\}$ denote the controlled wrist frame and $\{T\}$ the extrinsic tip contact frame. The tray contact plane is $(\mathbf{p}_0, \mathbf{n})$ with unit normal $\mathbf{n}$. Joint configuration is $\mathbf{q}$. State includes wrist pose $\mathbf{x}_w \in SE(3)$, tip contact point $\mathbf{p}_t \in \mathbb{R}^3$, and grasp offset $\mathbf{o}_{wt} = \mathbf{x}_w - \mathbf{p}_t$ in the contact tangent basis.

### 3.2 Tip-referenced spiral on the contact plane

Project tip position onto the contact plane: $\mathbf{p}_t^\parallel = \mathbf{p}_t - (\mathbf{n}^\top (\mathbf{p}_t - \mathbf{p}_0))\,\mathbf{n}$. Choose orthonormal tangents $\mathbf{t}_1, \mathbf{t}_2$ spanning the plane. The planned **Archimedean spiral** in tip coordinates $(u, v) = (\mathbf{t}_1^\top (\mathbf{p} - \mathbf{p}_0),\, \mathbf{t}_2^\top (\mathbf{p} - \mathbf{p}_0))$ is

$$
r(\theta) = r_0 - b\,\theta, \qquad
u^*(\theta) = r(\theta)\cos\theta, \quad v^*(\theta) = r(\theta)\sin\theta,
$$

with pitch $b > 0$, initial radius $r_0$, and monotonically increasing $\theta$ during search. The reference tip trajectory is $\mathbf{p}_t^*(\theta) = \mathbf{p}_0 + u^*(\theta)\,\mathbf{t}_1 + v^*(\theta)\,\mathbf{t}_2$.

**Key distinction from ConnTact:** $\mathbf{p}_t^*$ is defined in the **tip contact plane**, not by applying spiral offsets to the wrist pose.

### 3.3 Configuration-dependent planar grasp coupling

For small motions in the contact tangent plane, wrist incremental motion $\Delta \mathbf{q}_{\text{wrist}}$ (or equivalently planar wrist displacement $\Delta \mathbf{p}_{\text{wrist}} \in \mathbb{R}^2$ in $(\mathbf{t}_1, \mathbf{t}_2)$) maps to tip displacement via a **configuration-dependent coupling matrix**

$$
\Delta \mathbf{p}_{\text{tip}} \approx \mathbf{C}(\mathbf{q})\, \Delta \mathbf{q}_{\text{wrist}}, \qquad \mathbf{C}(\mathbf{q}) \in \mathbb{R}^{2 \times 2}.
$$

$\mathbf{C}(\mathbf{q})$ captures finger compliance, contact redistribution, and slip/stiction; it is **not** assumed constant. A **scalar ablation** sets $\mathbf{C}(\mathbf{q}) = \alpha(\mathbf{q})\,\mathbf{I}$ (scheme S6); the **proposed model** estimates full planar $\mathbf{C}(\mathbf{q})$.

### 3.4 Control: invert coupling for wrist commands

Let $\mathbf{e}_t = \mathbf{p}_t^{*\parallel} - \mathbf{p}_t^\parallel$ be the planar tip tracking error. The wrist command is obtained by **inverting** the estimated coupling:

$$
\Delta \mathbf{q}_{\text{wrist}} = \hat{\mathbf{C}}(\mathbf{q})^{-1}\, \mathbf{K}\, \mathbf{e}_t,
$$

with gain $\mathbf{K} \succ 0$ and regularization when $\hat{\mathbf{C}}$ is ill-conditioned. Baselines that spiral the wrist (S3) or hold a fixed/live offset (S1/S2) omit this inversion; they serve as ablations isolating tip reference and coupling.

### 3.5 Frozen grasp, peg-in-hand slip, and force gating

**Frozen grasp (spiral phase):** finger joints are held at the firm-grasp command $\mathbf{q}_f = \mathbf{q}_{f,0}$; only the wrist executes spiral tracking. This matches industrial practice (ConnTact rigid coupling) extended to dexterous hands where in-hand compliance is passive.

**Peg-in-hand slip:** latch peg position in the right-wrist frame, $\mathbf{p}_{peg/r,0}$, and monitor $s(t) = \|\mathbf{p}_{peg/r}(t) - \mathbf{p}_{peg/r,0}\|$. World-frame peg motion alone does not indicate slip under wrist motion.

**Force gating (not finger motion):** when wrist residual $|F_r| > F_{\mathrm{gate}}$ or $s > \tau_s$, scale planar track/step by $\gamma \in (0,1]$ and apply normal unload—`spiral_force_gate` in `baseline_hold_r`. Objectives (tip tracking via $\hat{\mathbf{C}}^{-1}$) are separated from compliance constraints (light $F_d$, no wedge).

### 3.6 Constraints

Search proceeds under explicit constraints:

1. **Frozen grasp:** $\mathbf{q}_f = \mathbf{q}_{f,0}$ during $\mathcal{T}_{\mathrm{spiral}}$ (no finger re-grasp).
2. **Surface contact:** $|(\mathbf{p}_t - \mathbf{p}_0)^\top \mathbf{n}| \leq \epsilon_n$.
3. **Normal force bounds:** $F_{\min} \leq F_n \leq F_{\max}$; gated unload when $|F_r| \gg F_d$.
4. **Along-surface drift:** $\text{along}(\mathbf{p}_t) \leq \text{along}_0 + \epsilon_{\text{along}}$.
5. **Slip abort:** $s > \tau_s \Rightarrow$ halt; coupling reset on slip events.

Violation of (1)+(un-gated tracking) yields high $s$ and tip–spiral decoupling (§5 diagnostic metrics: `grasp_slip_mm`, `force_scale`, `priv_planar_min`).

---

## 4. Method

The proposed controller comprises **three layers**. Scheme IDs label simulator implementations in `src/pci/tip_tracking_baselines.py` (`surface_tip_baseline` config key).

### 4.1 Layer 1 — Coupling estimation

**Goal:** maintain $\hat{\mathbf{C}}(\mathbf{q}) \in \mathbb{R}^{2\times 2}$ relating planar wrist motion to tip motion.

- **Proposed (planar $\mathbf{C}$):** estimate $\hat{\mathbf{C}}$ from paired $(\Delta \mathbf{q}_{\text{wrist}}, \Delta \mathbf{p}_{\text{tip}})$ samples during search; in simulation, privileged tip read enables ground-truth validation; on hardware, fingertip tactile + wrist history replace privileged sensing.
- **Ablation (S6, scalar $\alpha$):** $\hat{\mathbf{C}} = \hat{\alpha}\,\mathbf{I}$ with online scalar $\hat{\alpha}$—isolates whether isotropic coupling suffices.
- **Baselines (no explicit $\mathbf{C}$):** S1 frozen offset, S2 live offset tracking; S3 ConnTact wrist spiral assumes implicit $\mathbf{C} = \mathbf{I}$; S0 surface lock ignores tip geometry.

On **slip events** ($s > \tau_s$), discard stale samples and re-estimate $\hat{\mathbf{C}}$ after re-seat (§3.5).

### 4.2 Layer 2 — Wrist command inversion

**Goal:** track $\mathbf{p}_t^*(\theta)$ by commanding the wrist through $\hat{\mathbf{C}}^{-1}$ (§3.4).

| Scheme | Role | Inversion |
|--------|------|-----------|
| **S0** | Negative baseline | Blind normal press; no spiral inversion |
| **S3** | ConnTact baseline | Wrist spiral; no $\mathbf{C}^{-1}$ |
| **S1** | Ablation | Frozen tip–wrist offset feedforward |
| **S2** | Ablation | Live offset; no full $\mathbf{C}$ |
| **S6** | Scalar ablation | $\hat{\mathbf{C}} = \hat{\alpha}\mathbf{I}$ inverted |
| **Planar $\mathbf{C}$** | **Proposed** | Full $\hat{\mathbf{C}}(\mathbf{q})^{-1}$ with regularization |

Commands are saturated and filtered; axial compliance is applied separately from planar inversion.

### 4.3 Layer 3 — Surface, force, and slip constraints

**Goal:** keep the tip on the tray, within force limits, and recover from grasp slip.

- **Surface lock:** project commanded tip motion onto the contact plane; enforce $\epsilon_n$ band (§3.5).
- **Along constraint:** limit insertion-axis drift during spiral (reduces slide-off near hole mouth).
- **Force guarding:** clip $F_n$ to $[F_{\min}, F_{\max}]$ via admittance on the left (tray) wrist and light axial press on the right hand.
- **Slip reset:** on $s > \tau_s$, trigger unload → re-estimate $\hat{\mathbf{C}}$ → resume spiral from current $\theta$.

**Grasp stabilization** (`priv_grasp_opt.py`, GraspQP-style) runs in parallel to maintain peg-in-hand forces; it does not replace tip-referenced spiral tracking.

### 4.4 Scheme summary

| ID | Config mode | Paper role |
|----|-------------|------------|
| S0 | `surface_lock` | Negative baseline (blind press) |
| S3 | `conntact_wrist` | ConnTact rigid-EE baseline |
| S1 | `tip_servo_ff` | Ablation: frozen offset |
| S2 | `tip_track_live` | Ablation: live offset |
| S6 | `priv_coupling` | Ablation: scalar $\alpha$ |
| — | planar $\mathbf{C}$ + constraints | **Proposed method** |

*Note:* Early engineering configs labeled S7 in the codebase collected deprecated patch combinations; the published method is **planar $\mathbf{C}(\mathbf{q})$** with the three layers above, not ad-hoc S7 parameter patches.

---

## 5. Results

### 5.1 Experimental setup

- **Simulator:** DexJoCo bimanual Allegro, PCI `sim_runner.py`.
- **Task instantiation:** handoff grasp—right hand holds peg, left hand stabilizes tray; episodes 01–10.
- **Primary success metric (locked):** `docs/SUCCESS_STANDARD.md` — force/ConnTact hole enter + tip seat on tray + `axis_err ≤ 25°`. Do not loosen.
- **Legacy diagnostics:** `lat_min`, `along`, $|F|_{\text{peak}}$, grasp `slip` (Table 5.3; from early L3 harness).
- **Harness:** SUCCESS_STANDARD runs under `outputs/scheme_l3/`; paper matrix [`docs/PAPER_BASELINE_MATRIX_SUCCESS.md`](PAPER_BASELINE_MATRIX_SUCCESS.md). Legacy lat table: [`docs/BASELINE_TABLE.md`](BASELINE_TABLE.md).

### 5.2 Deployable enter rate (SUCCESS_STANDARD, ep01–10)

**Main comparison (deployable only).** Oracle excluded.

| Method | Out root | RATE |
|--------|----------|------|
| ConnTact / Franka / PSFT (faithful) | `S3_*_faithful_ep1_10` | **0/10** |
| Tip-Hybrid + planar_C (v10) | `S2_force_enter_ep1_10v10` | **4/10** |
| Type-A | `S2_typeA_ep1_10` | **5/10** |
| **Track-B GMHQP (ours, best)** | `S2_trackB_gmhqp_ep1_10` | **6/10**† |

† GMHQP uses in-hand peg-tip **geom as object-localization surrogate** (disclose; not tip-free claim; not Oracle tip GT). See `docs/PRIVILEGED_SENSING_DISCLOSURE.md`.

**Per-ep GMHQP vs Type-A:** win ep08/09/10; lose ep02/05; keep ok 01/04/06; keep fail 03/07 → net **+1**.

**Smoke-only (not main RATE):** Track-A tip_hat v0 **2/4** (`S2_trackA_est_oracle_smoke`).

**Appendix — diagnostic ceiling (not deployable):** Oracle-v1 **7/10** (`S2_oracle_tip_ep1_10`); Oracle-v2 **9/10** (`S2_oracle_tip_v2_ep1_10`). Never place in the main deployable table.

### 5.3 Legacy aggregate (lat / along / slip; early L3 harness)

| Tier | Method | lat_min ↓ (mm) | along ↓ (mm) | \|F\|_peak ↓ (N) | slip ↓ (mm) | enter rate | n |
|------|--------|----------------|--------------|------------------|-------------|------------|---|
| neg | Surface lock + force seat (S0) | med 22.6 / mean 125.6 | 93.3 | 1.78 | 51 | 0% | 10 |
| A | ConnTact rigid-EE spiral (S3) | med 999.0 / mean 999.0 | 147.9 | 0.00 | 27 | 0% | 10 |
| B | Frozen tip–wrist offset (S1) | med 26.0 / mean 129.4 | 106.3 | 1.36 | 59 | 0% | 10 |
| B' | Live offset tracking (S2) | med 13.0 / mean 122.0 ** | 109.5 | 0.88 | 57 | 0% | 10 |
| C-α | Scalar α coupling ablation (S6) | med 13.9 / mean 122.2 | 110.0 | 0.90 | 56 | 0% | 10 |
| C | Planar **C** + surface/along constraints (**ours**) | med 14.5 / mean 123.4 | 102.8 | 1.33 | 81 | **10%** | 10 |

**Best median lat_min** is S2 (13.0 mm); mean lat_min favors S2 due to episode variance (ep07 all-fail grasp slip affects all schemes equally).

### 5.4 Interpretation

**SUCCESS_STANDARD headline.** Published rigid-EE ports (ConnTact / Franka / PSFT) are **0/10**. Tip-Hybrid stack reaches **4/10**; Type-A **5/10**; Track-B GMHQP **6/10** (best deployable; tip-geom surrogate disclosed). Oracle-v2 **9/10** is a diagnostic ceiling only (§5.2 appendix).

**Legacy lat table (Table 5.3).** ConnTact (S3) fails systematically on `lat_min` (999 mm); tip reference (S2/S6) helps lateral approach to ≈13–14 mm, but early planar-$\mathbf{C}$ alone still showed weak entry under that harness. Prefer §5.2 for paper enter-rate claims.

**Remaining gap.** GMHQP fails ep02/03/05/07 still saturate planar away from mouth; Oracle-v2 shows tip↔mouth known + seat/press/reseat can reach 9/10—motivating Track-A estimators without promoting Oracle into the main table.

---

## 6. Discussion

### 6.1 What generalizes

The **problem formulation** (§3)—tip-referenced spiral on the contact plane, configuration-dependent planar coupling $\mathbf{C}(\mathbf{q})$, inversion-based wrist control, and surface/force/slip constraints—applies to any dexterous peg-in-hole search where the peg tip, not the wrist, contacts the search surface. The three-layer architecture (estimation → inversion → constraints) separates concerns that ConnTact collapses under rigid-EE assumptions.

### 6.2 What is handoff-specific

Our **experiments** instantiate one bimanual setup: Allegro handoff grasp on a held tray (PCI/DexJoCo, ep01–10). Grasp geometry, tray compliance, and PBVS pre-alignment quality are fixed. Entry rate, slip statistics, and ConnTact failure modes should not be extrapolated to parallel-jaw PiH or in-hand regrasp without re-evaluation. Sim privileged tip sensing validates coupling inversion; hardware requires fingertip tactile + wrist F/T as stated in §4.1.

### 6.3 Limitations and next steps

- **Entry rate:** deployable best is GMHQP **6/10** under SUCCESS_STANDARD; Oracle-v2 **9/10** remains a non-deployable ceiling.
- **Coupling observability:** scalar $\alpha$ ablation shows isotropy is limiting; full planar $\mathbf{C}$ with tactile history is the hardware path.
- **Vision scope:** PBVS pre-spiral only; spiral closure is contact-dominated.
- **Method upgrade (V2):** replace sliding-window LS with **slip-aware EKF** (Pfanne IROS'17; Piga diff-EKF) and one-step **contact-constrained QP** (Escande HQP; Lynch & Park Ch.12). See [`docs/METHODOLOGY_V2.md`](METHODOLOGY_V2.md) and [`docs/LITERATURE_BASELINES.md`](LITERATURE_BASELINES.md).
- **Privileged sim:** disclose per [`docs/PRIVILEGED_SENSING_DISCLOSURE.md`](PRIVILEGED_SENSING_DISCLOSURE.md); diagnostic oracle separate from deployable table.

---

## 7. References (BibTeX keys `[TODO]`)

```bibtex
@misc{conntact2021,
  author = {{Southwest Research Institute}},
  title = {ConnTact Assembly Framework},
  year = {2021},
  url = {https://github.com/swri-robotics/ConnTact}
}
@article{watson2020assembly,
  title={Autonomous industrial assembly using force, torque, and RGB-D sensing},
  author={Watson, James and Miller, Austin and Correll, Nikolaus},
  journal={Advanced Robotics},
  year={2020},
  url={https://arxiv.org/abs/2002.02580}
}
@article{park2020psft,
  title={Compliant Peg-in-Hole Assembly Using Partial Spiral Force Trajectory With Tilted Peg Posture},
  author={Park, Hyeonjun and others},
  journal={IEEE RA-L},
  year={2020}
}
@inproceedings{kim2023simtact,
  title={Simultaneous Tactile Estimation and Control of Extrinsic Contact},
  author={Kim, Sangwoon and Jha, Devesh K and Romeres, Diego and Patre, PM and Rodriguez, Alberto},
  booktitle={IEEE ICRA},
  year={2023},
  url={https://arxiv.org/abs/2303.03385}
}
@inproceedings{bronars2024texterity,
  title={TEXterity: Tactile Extrinsic deXterity},
  author={Bronars, Antonia and Kim, Sangwoon and Patre, PM and Rodriguez, Alberto},
  booktitle={IEEE ICRA},
  year={2024},
  url={https://arxiv.org/abs/2403.00049}
}
@article{vanwyk2018nist,
  title={Comparative Peg-in-Hole Testing of a Force-Based Manipulation Controlled Robotic Hand},
  author={Van Wyk, Karl and others},
  journal={IEEE TRO},
  year={2018}
}
@article{choi2021kinesthetic,
  title={Kinesthetic Sensing for Peg-In-Hole Assembly Based on In-Hand Manipulation},
  author={Choi, Myoung-Su and others},
  journal={IEEE RA-L},
  year={2021}
}
@article{nazari2025bgf,
  title={Bioinspired trajectory modulation for slip-aware manipulation},
  author={Nazari, Behrooz and others},
  journal={Nature Machine Intelligence},
  year={2025}
}
@inproceedings{dong2021tactilerl,
  title={Tactile-RL for Insertion: Generalization to Objects of Unknown Geometry},
  author={Kim, Sangwoon and Jha, Devesh K and Romeres, Diego and Rodriguez, Alberto and Patre, PM},
  booktitle={IEEE ICRA},
  year={2021}
}
@inproceedings{alt2022dpse,
  title={Heuristic-free Optimization of Force-Controlled Robot Search Strategies in Stochastic Environments},
  author={Alt, Benjamin and Katic, Darko and J{\"a}kel, Rainer and Beetz, Michael},
  booktitle={IEEE/RSJ IROS},
  year={2022}
}
@inproceedings{zurbrugg2025graspqp,
  title={GraspQP: Differentiable Optimization of Force Closure for Diverse and Robust Dexterous Grasping},
  author={Zurbr{\"u}gg, Remo and Cramariuc, Andrei and Hutter, Marco},
  booktitle={CoRL},
  year={2025}
}
```

---

## Appendix A: Local repo map

| Path | Role |
|------|------|
| `refs/ConnTact` | SpiralToFindHole reference |
| `refs/Tactile-Estimator-Controller` | Extrinsic contact FG |
| `refs/graspqp` | Grasp QP |
| `refs/bgf`, `refs/action_conditioned_tactile_prediction` | Slip / ACTP |
| `refs/dpse` | Search strategy optimization |
| `refs/in_hand_manipulation_2` | Contact-implicit MPC |
| `src/pci/tip_tracking_baselines.py` | Paper baselines |
| `configs/scheme_l3/S*.yaml` | L3 ablation configs |

---

## Appendix B: Experiment changelog

| Date | Change | Notes |
|------|--------|-------|
| 2026-09-02 | ep01–10 aggregate baselines | See `docs/BASELINE_TABLE.md` |
| 2026-09-02 | Methodology lock | Planar $\mathbf{C}$ proposed; S6 scalar ablation; S7 deprecated as patch label |
| 2026-09-03 | SUCCESS_STANDARD paper matrix | Tip-Hybrid 4/10 → Type-A 5/10 → **GMHQP 6/10**; Oracle-v2 9/10 appendix only; `docs/PAPER_BASELINE_MATRIX_SUCCESS.md` |
