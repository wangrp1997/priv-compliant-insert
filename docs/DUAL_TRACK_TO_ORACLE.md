# Dual track to Oracle-level performance (9/10)

> Problem lock: known tip–mouth path; study seat + upright + force enter under non-rigid grasp.  
> Ceiling: Oracle-v2 **9/10** (`S2_oracle_tip_v2_ep1_10`). Deployable now: **Track-B GMHQP-AC 7/10**（was GMHQP **6/10** / Type-A **5/10**）.  
> SUCCESS_STANDARD never loosened. Oracle rate never goes in deployable table until Path A/B matches it **without** tip/hole GT in the loop.  
> **Method gate:** `docs/METHOD_GATE_NO_PATCH.md` — 通用诊断缺口 + 理论依据；**检索优先 → 没有再创新**；禁止调参工程补丁。  
> **Latest RX:** `docs/GENERAL_ALGO_MANDATE.md` — 失败 ep 仅探针；**通用**估计/优化控制；在 GMHQP 结构上升级定律，禁止按 ep 补丁。wave-7 已按此重定向。

## Goal

Reach **same enter rate as Oracle-v2 (9/10)** under SUCCESS_STANDARD with a **non-GT** method.

## Track A — Observe what Oracle needs

Estimate Oracle inputs from sensors / estimators, **same control law** as Oracle-v2.

| Oracle input | Estimate how |
|--------------|--------------|
| tip planar pose | tactile extrinsic / TEC / tip filter from F/T+proprio |
| mouth / hole XY | frozen vision coarse (already “known path”) |
| seat / float | wrist F/T + fingertip normal |
| axis | extrinsic orientation + FK |
| slip | shear / innov |

Milestone: `tip_hat` (no peg xpos in control) + Oracle-v2 controller → rate vs 9/10.

## Track B — Replace or jointly optimize without those estimates

Search & invent methods that **do not require** online tip GT / full tip estimator:

- joint estimation–control (Kim TEC spirit, but force-enter task)
- impedance / hybrid force–position with known path only
- MPC with contact constraints (Escande HQP / LeTac ideas)
- learning residual on wrist F/T along known spiral
- grasp–arm co-optimization (GraspQP + wrist)

**Primary pick (2026-09-03):** **PHIG** — Path-constrained Hybrid Impedance + Grasp co-opt.  
Survey + equations + FSM: `docs/TRACK_B_ALTERNATIVES.md`.  
Skeleton: `src/pci/track_b_phig.py`, `configs/scheme_l3/S2_trackB_phig.yaml`, `scripts/smoke_track_b_phig.py`.

Milestone: method M with eligible sensors only → rate → 9/10.

## Reporting

| Table | Allowed |
|-------|---------|
| Oracle-v2 | privileged diagnostic ceiling |
| Track A | estimator + same ctrl (disclose estimator) |
| Track B | no tip/hole GT |
| Published ConnTact/Franka/PSFT | 0/10 baselines |

## Parallel wave (user 2026-09-03)

| Line | Agent | Status |
|------|-------|--------|
| 1 PHIG wire | Done [PHIG wire](4f5901d4-a81c-48e3-b62d-240b3c4a5e12) + smoke monitor | fail-subset **0/6**（Type-A 同集 **1/6**）；lose ep04；wins=0；**不开全量** |
| 2 tip_hat improve | Done [Track-A tip_hat](2380347e-7814-44aa-87e7-ee1eb52c71c8) | smoke_v2 **0/4**（劣于 v0 **2/4**）；根因 wrist_delta holdover；**暂停改参** |
| 3 fail→algo | Done [Fail-driven](6f606a67-d60d-4cb2-83e7-8d79060cd657) + [HardHQP smoke](907a186c-dd91-4338-9c02-5f15301e4038) | HardHQP；单元 OK；fail-subset **0/3**；**不扩** ep1–10 |

## Parallel wave-2 (user 继续两路)

| Line | Agent | Status |
|------|-------|--------|
| A holdover fix | Done [Track-A v3](a5043e2e-b64d-46df-905e-8a8a8d11c5cc) | smoke_v3 **1/4** / v3b **0/4**（劣于 v0 **2/4**）；freeze 病理；**STOP 不扩** |
| A priv+gated freeze | This wave | Phase-0 `TRACK_A_PRIV_DIAG.md`；v4 **0/4** vs v0 **2/4**；**STOP 不扩** |
| B F1 redesign | Done [Track-B F1](55296445-c074-41ed-ab4d-c1550b62d1dd) | CLEP **0/4**（planar 更差）；gate FAIL；**暂停** |

## Parallel wave-3 (先特权诊断再动手)

| Line | Agent | Status |
|------|-------|--------|
| A | Done [Track-A priv→v4](67e3ec0b-ff46-4ea2-8170-b82a6f927e3f) | v4 **0/4** STOP；补丁线结束 |
| A cont | Done [Track-A theory](312d9b68-86f9-40b3-9a03-4a4f23ac61f9) + [wire smoke](93e42833-36a0-47f4-9efe-84736ef0710f) | TEC-slim **wired**；smoke **0/4** vs v0 **2/4**；**STOP 不扩** |
| B | Done [Track-B priv→design](75ecd3b5-f390-4cd4-8e13-5ad4dc69296f) | FASR 弃；OIGS **0/3** gate FAIL；**STOP** |

## Parallel wave-4 (priv RX 对症)

| Line | Agent | Disease → medicine | Status |
|------|-------|-------------------|--------|
| A | Done [Track-A tip+axis](57cd5d18-2114-4f13-a132-5d958a0c1dee) | F4 → tip+axis；**2/4=v0**；ep03 轴~77°；**STOP** |
| B | Done [Track-B grasp-map](fca1a8f8-0abd-4603-ac98-7243fd23fc87) + [ep1-10](ff64dc9c-a432-474f-8d2b-d46dc30292a9) | smoke **2/3 PASS** → full **6/10**；vs Type-A +1；**STOP 不调参** |

RX: `docs/DUAL_TRACK_PRIV_RX.md` · A detail: `docs/TRACK_A_TIP_AXIS_OBSERVER.md` · B: `docs/TRACK_B_GRASP_MAP_HQP.md`

## Parallel wave-5 (GMHQP 余败 + 论文表)

| Line | Agent | Role |
|------|-------|------|
| Priv RX | `docs/GMHQP_RESIDUAL_PRIV_RX.md` | ep2/3/5/7 F1 / 回退 / F2→F1 |
| A | Done [Track-A tip-obs](b2eedaa8-91be-4290-87f5-25ddaa594642) + [ep1-10](77d6af02-d42a-4b9d-b505-89c54aa21276) | smoke **1/4 PASS** → full **4/10**；毁 4/8/9 赢 5；≤GMHQP → **STOP** |
| B | Done [Track-B hybrid](aaeaf1b2-35df-44b5-a5c0-c7794f1a6154) | REUSE Escande GMHQP⊕planar_C；smoke **0/4**；**STOP** |
| Paper | Done [Paper tables](d5238460-c33d-42f6-b0ee-0e976b6bd448) | 主表 0→4→5→**GMHQP 6/10**；Oracle 附录 |

## Parallel wave-6 (priv→检索→迭代)

| Line | Agent | Role |
|------|-------|------|
| A | Done tip_fuse | smoke **1/7** keep FAIL → **STOP** |
| B | Done [Wave6 Track-B](b08c5991-42dd-4736-8767-0822ce69367e) | GMHQP-FTIP REUSE；smoke **0/4**；**STOP** |

## Parallel wave-7（在 GMHQP 上升级**通用律**）

| Line | Agent | Algorithm |
|------|-------|-----------|
| A | Done [Wave7 TEC-joint](03413760-4520-4846-a99e-f4e9516b319b) + [Mon TEC](60406a12-d831-4ede-bcd8-19707738ff89) | full **6/10=GMHQP**；`gated_on_frac=0`；**STOP 不调参** |
| B | Done [Wave7 contact-opt](54a6c3b4-546e-46be-81ce-8f459fdb5fa7) + [COMPC impl](da137cfe-6894-44b7-8292-80c1833b8a21) + [Mon COMPC](e6b7325e-012e-4ea0-9ee4-7b403721612c) | full **4/10** ≤GMHQP 6/10（失 4,9）；**STOP 不调参** |

**Mandate:** `docs/GENERAL_ALGO_MANDATE.md` — 失败 ep 仅探针；控制器无 fail/keep 分支。  
**Track-B design:** `docs/TRACK_B_CONTACT_OPT_ON_GMHQP.md` — 主评 **full ep1–10 RATE** vs GMHQP 6/10；RATE≤基线则 STOP 改方程。  
（旧 WAVE7 fail/keep 硬门禁作废为设计目标；仅作历史对照。）

### Track-A TEC-joint full (wave-7/8 eval, 2026-09-03)

- Out: `S2_trackA_tec_joint_gmhqp_ep1_10` — **RATE 6/10**（ok **1,4,6,8,9,10**；fail **2,3,5,7**）.
- vs GMHQP **6/10**：**tie**（ok/fail 集相同）；vs Type-A **5/10**：+1。
- `tec_joint_gated_on_frac=0` 全 ep；fail diag ≡ GMHQP（planar 8.9–11.7 / no mouth）→ 行为退化到 \(z_g\)。
- **METHOD_GATE STOP** — 不调参；不改论文主表。Deployable 仍 **GMHQP 6/10**。Oracle 9/10 diagnostic only。

### Track-B GMHQP-COMPC full (wave-7/8 eval, 2026-09-03)

- Out: `outputs/scheme_l3/S2_trackB_gmhqp_compc_ep1_10` · **RATE 4/10**（ok **1,6,8,10**；fail **2,3,4,5,7,9**）。
- vs GMHQP **6/10**：kept 1,6,8,10；**lost 4,9**；new wins ∅；净 **−2**。
- Fail probes: residual {2,3,5,7} planar 8.7–15.7 mm no mouth；keep-regress {4,9} planar 15.4 / 9.2（GMHQP 曾 ~4.5+mouth）。
- **METHOD_GATE STOP** — ≤ baseline；不旋钮搜参；不改论文主表。Deployable 仍 **GMHQP 6/10**。

## Parallel wave-8（多 agent）

| Line | Agent | Role |
|------|-------|------|
| A eval | Done [Mon TEC](60406a12-d831-4ede-bcd8-19707738ff89) | full **6/10=GMHQP**；gated_on=0；**STOP**；不改论文表 |
| B eval | Done [Mon COMPC](e6b7325e-012e-4ea0-9ee4-7b403721612c) | **4/10** ≤6/10；失 4,9；**STOP**；不改论文表 |
| Priv+lit | Done [Modes+lit](13cefcd9-b063-4fd8-b213-bd97870c6817) | `WAVE8_GENERAL_FAILURE_MODES.md`；下优先 A FG / B Pfanne |
| B2 invent | Done [Adaptive-C](c0413611-b4a1-4ac7-af7e-c2b748ca379a) + full eval | AC 全量 **7/10**（+ep07 vs GMHQP）；更新论文主表；STOP 调参 |
| A2 invent | Done [Factor-graph](27d38d45-0982-4094-a554-375ac91136a1) | TEC FG 骨架+单测 OK；TEC-joint full **STOP**（gated_on=0）→ FG 为下步结构候选 |

## Track-A TEC→GMHQP tip_obs (wave-5, 2026-09-03)

- **Priv gap:** GMHQP residual ep2/3/5/7 planar 8.9–11.7 mouth F；proxy tip ≠ true tip；Oracle tip GT 全过。
- **Search→reuse:** Kim TEC (`refs/Tactile-Estimator-Controller`) + existing GMHQP — **no invent acronym** (`docs/TRACK_A_GMHQP_TIP_OBS.md` §2).
- Design/code: TEC-slim `ExtrinsicTipStateEKF` → GMHQP tip_obj；config `S2_trackA_gmhqp_tip_obs.yaml`.
- Smoke `S2_trackA_gmhqp_tip_obs_smoke` eps **2,3,5,7**: **RATE 1/4**；**enter-win ep05**（planar 4.42 + mouth）；ep02/03/07 仍 F1。Gate **PASS**.
- Full `S2_trackA_gmhqp_tip_obs_ep1_10`: **RATE 4/10**（ok **1,5,6,10**；fail **2,3,4,7,8,9**）.
  - vs GMHQP **6/10**：+ep05；**毁 4,8,9**（保留 1,6,10）；净 −2。
  - vs Type-A **5/10**：−1。
- **METHOD_GATE STOP** — rate≤6/10 且无净增益；不搜 EKF 方差；不改论文主表。Deployable 仍 **GMHQP 6/10**。Oracle 9/10 diagnostic only。

## Fail-driven wave (2026-09-03)

- Taxonomy (priv): `docs/FAIL_PRIV_TAXONOMY.md` — F1 planar saturate, F2 slip, F3 mouth axis, F4 tip_hat upright.
- Primary: **HardHQP** Track-B (Escande seat≻upright≻path) — `docs/FAIL_DRIVEN_HARD_HQP.md`
- Code: `src/pci/fail_driven_hard_hqp.py`, `configs/scheme_l3/S2_failDriven_hardHQP.yaml`, `scripts/smoke_fail_driven_hard_hqp.py`
- Not JSE restack; not tip_gt+noise. Complements PHIG soft weights with hard priority.
- Fail-subset smoke `S2_failDriven_hardHQP_smoke` eps **3,7,10**: **RATE 0/3 = 0%** (vs Type-A same eps 0/3).
  - ep07: force_hole + mouth but `force_hole_along_reject`; planar/slip better than Type-A, still no enter-win.
  - Gate enter-win≥1 **not met** → **do not expand to ep1–10**.

## Track-B CLEP wave (2026-09-03)

- Design: `docs/TRACK_B_CLEP.md` — Kim Active Extrinsic contact-line via wrist F/T (no tip GT); attacks F1.
- Distinct from PHIG / HardHQP / JSE: **contact estimate → path**, not wrist→path.
- Code: `src/pci/track_b_clep.py`, `configs/scheme_l3/S2_trackB_clep.yaml`, `scripts/smoke_track_b_clep.py`
- Fail-subset `S2_trackB_clep_smoke` eps **3,8,9,10**: **RATE 0/4 = 0%** (Type-A same eps **0/4**).
  - lat_min 11.4–15.7 mm（劣于 Type-A 5.9–12.3 mm）；无 enter-win；无 `priv_planar_min`<5 mm + mouth。
  - Gate FAIL → **do not expand** ep1–10.
- Oracle 9/10 stays diagnostic-only.


## Track-B OIGS wave (2026-09-03 hard correction)

- **Priv gap:** F1 wrist≠tip (\(C\neq I\)) → `priv_planar_min` (see `docs/TRACK_B_PRIV_DIAG.md`).
- **FASR:** rejected as engineering patch; smoke **0/3** on `S2_trackB_fasr_smoke` (recorded only).
- **Primary theory method:** **OIGS** — Pfanne RA-L 2020 object impedance + GraspQP; wrist path \(\alpha(\lVert e_x\rVert)\).
- Docs: `docs/TRACK_B_OIGS.md` (equations + anti-patch).
- Code: `src/pci/track_b_oigs.py`, `configs/scheme_l3/S2_trackB_oigs.yaml`, pose-hold QP spiral-wide.
- Smoke `S2_trackB_oigs_smoke` eps **3,9,10**: **RATE 0/3**.
  - ep03: planar 8.4→5.5 mm + mouth (priv improve, not &lt;5 mm).
  - ep10: slip 1032→15 mm; planar 12.3→9.8 mm; still no enter.
  - ep09: planar/slip worse.
  - Gate FAIL → **STOP**, no ep1–10.

## Agents (this wave)

| Agent | Track | Role |
|-------|-------|------|
| Done [Track-A](55266fa1-32bc-4c97-89b2-892412b0d002) | A | tip_hat + Oracle-v2 ctrl；smoke **2/4** < Oracle 同子集 **3/4**；**未扩 ep1–10** |
| Done [Track-B lit](88b37127-aef0-4f30-a7d7-a04628b6c68d) | B | PHIG chosen; skeleton landed |
| Done [Track-B PHIG wire](4f5901d4-a81c-48e3-b62d-240b3c4a5e12) + smoke monitor | B | fail-subset **0/6**；lose ep04；gate FAIL；**未扩 ep1–10** |
| Done [Track-B joint](1fd09631-a7b0-41e5-9d94-8f45c86e194f) | B | JSE smoke **2/6** BORDERLINE；**暂不开全量** |
| Track-B CLEP (this) | B | CLEP F1 contact-line；smoke **0/4**；gate FAIL；**未扩 ep1–10** |

## Track-A smoke (2026-09-03)

- Design: `docs/TRACK_A_TIP_ESTIMATOR.md`；est: `src/pci/tip_contact_estimator.py`
- Config / out: `S2_trackA_est_oracle_ctrl.yaml` → `S2_trackA_est_oracle_smoke`
- **2/4** (ok 1,2; fail 3,4) vs Oracle-v2 same subset **3/4** (fail only 4)
- Control: tip_hat（无 peg xpos / 无 tip_gt+noise）；残差特权：peg↔tray `contact.pos` + priv grasp slip（已披露）
- ep03：到口附近但 axis≈73°、无力入 → tip_hat 质量不够撑满 Oracle 律
- Next only if user authorizes: 改进估计器 / 更大 smoke；**不**自动扩 ep1–10

## Track-A tip_hat improve (2026-09-03)

- Out: `S2_trackA_est_oracle_smoke_v2` — **0/4**（劣于 v0 **2/4**）
- 根因：`wrist_delta` holdover 把非刚性抓取当刚性偏移
- PCA 等调参曾打穿成功率；**不**自动再改 / 扩全量；v0 仍为 Track-A 基线

## Track-A holdover v3 (2026-09-03)

- Fix: `freeze_planar`（seat/接触后冻平面 tip；禁 wristΔ push；未接触则 wrist+offset bridge）
- Config flags: `surface_tip_est_holdover_mode`, `holdover_decay`, `seat_only_offset`；PCA 仍关
- Out: `S2_trackA_est_oracle_smoke_v3` — **1/4**；`..._smoke_v3b` — **0/4**
- vs v0 **2/4** / Oracle 同子集 **3/4** → **gate FAIL**；**未扩** ep1–10
- ep03：v0 axis≈73°；v3 axis≈9° 但 planar timeout（tip_hat 冻坏初值 err~400 mm）
- 诚实基线仍为 v0 **2/4**；代码保留 holdover 开关供下一轮接触观测修复

## Track-A priv-diag + gated freeze v4 (2026-09-03)

- Phase-0: `docs/TRACK_A_PRIV_DIAG.md` — 证实长螺旋 **tip_est_seated≈0** 且腕力 cfrac≈1；勿混 hyb_seat / wrist cfrac
- Fix: `freeze_require_seat` + first strong seat hard-snap（禁弱接触冻平面）
- Out: `S2_trackA_est_oracle_smoke_v4` — **RATE 0/4**（劣于 v0 **2/4**）→ **STOP 不扩** ep1–10
- ep01/02 已相对 v0 丢分；诚实 Track-A RATE 仍为 v0 **2/4**

## Parallel wave-4 (priv RX → dual parallel, 2026-09-03)

| Line | Agent | Status |
|------|-------|--------|
| A tip+axis | Done | F4(+F3): `ExtrinsicTipAxisEKF` tip+\(\hat a\); smoke **2/4** = v0；**&lt;3/4 → STOP 不扩**；无方差搜参 |
| B grasp-map HQP | Done | F1(+F2): smoke **2/3 PASS** → full **6/10**；vs Type-A 5/10 **+1**；**STOP 不调参** |

## Track-B GMHQP (wave-4, 2026-09-03)

- **Priv gap:** F1 \(C\neq I\)；Type-A ep3 planar **8.4** / slip 18.6；OIGS planar↓但 tip_err **77** mm（腕 TCP 任务）。
- **Theory:** Montana grasp map + Pfanne object impedance + Escande HQP；\(e=\Pi(p_{\mathrm{tip}}-p^*)\) → \(C^{+}\)腕 + GraspQP 指。
- Design: `docs/TRACK_B_GRASP_MAP_HQP.md`；code `src/pci/track_b_gmhqp.py`
- Config / out: `S2_trackB_gmhqp.yaml` → smoke `S2_trackB_gmhqp_smoke` eps **3,9,10** → full `S2_trackB_gmhqp_ep1_10`
- Smoke **RATE 2/3** gate **PASS**（enter ep09/10）；ep03 planar 11.5 mm
- Full ep1–10 **RATE 6/10**（ok 1,4,6,8,9,10；fail 2,3,5,7）vs Type-A **5/10**（+1；win 8/9/10，lose 2/5）
- Diag: fails planar 8.9–11.7 mm no mouth；tip_err_max ≤34 mm；Oracle **9/10** diagnostic only
- **METHOD_GATE STOP** — no retune

## Parallel wave-7 (optimize **on** GMHQP; general algo)

> Mandate: `docs/GENERAL_ALGO_MANDATE.md` — fail eps = probes only; no keep-set / fail-subset controller structure.  
> RX: `docs/WAVE7_ON_GMHQP.md`

| Line | Agent | Disease → medicine | Status |
|------|-------|-------------------|--------|
| A | Track-A TEC joint on GMHQP | free-run ˆt → Kim TEC joint + NIS/contact gate → GMHQP | full **6/10=GMHQP**; `gated_on_frac=0` → **STOP retune**；下步改算法非松门 |
| B | (parallel) contact MPC / Active Extrinsic on GMHQP | F1 under imperfect tip | see wave-7 Track-B |

- A design: `docs/TRACK_A_TEC_JOINT_ON_GMHQP.md`
- A code: `src/pci/track_a_tec_joint.py` · config `S2_trackA_tec_joint_gmhqp.yaml`
- Full `S2_trackA_tec_joint_gmhqp_ep1_10`: **RATE 6/10** = GMHQP；`gated_on_frac=0`；**METHOD_GATE STOP**；no paper-table bump.
- Prior tip_obs **4/10** / tip_fuse keep-fail = structural free-run ˆt (lesson), not ep patches to copy.

## Parallel wave-6 (priv → search → iterate, 2026-09-03)

| Line | Agent | Disease → medicine | Status |
|------|-------|-------------------|--------|
| A | Track-A tip_fuse | ˆt free-run destroy wins → TEC PoseDiff soft geom prior | smoke **1/7**; keep {4,8,9} **FAIL** → **STOP** |
| B | **REUSE** GMHQP-FTIP | proxy tip 假闭合 → Doshi FT \(\hat c\)→GMHQP | smoke **0/4 FAIL** → **STOP** |

- Phase-0 A: `docs/WAVE6_TRACK_A_PRIV.md` — GMHQP fails + tip_obs destroy {4,8,9}
- Search→**REUSE** A: Kim TEC PoseDiff / TacGraph in-hand prior (`docs/TRACK_A_GMHQP_TIP_FUSE.md`) — **no invent**
- Code A: `geom_tip_prior` in `tip_theory_estimator.py`; config `S2_trackA_gmhqp_tip_fuse.yaml`
- Smoke A `S2_trackA_gmhqp_tip_fuse_smoke` eps **2,3,4,5,7,8,9**: **RATE 1/7**
  - enter-win ep05 (GMHQP fail); ep07 planar↓+mouth (no enter)
  - **lost keep 4,8,9** → gate FAIL; **STOP** no retune / no ep1–10
- Phase-0 B: `docs/WAVE6_TRACK_B_PRIV.md`；design `docs/TRACK_B_GMHQP_FTIP.md`
- Code B: `src/pci/track_b_gmhqp_ftip.py` · `S2_trackB_gmhqp_ftip.yaml`
- Smoke B `S2_trackB_gmhqp_ftip_smoke` eps **2,3,5,7**: **RATE 0/4**
  - ep02 planar **4.40**+mouth but `force_hole_along_reject`（非 enter）
  - ep03/05/07 tip_err 60–80 mm（劣于 GMHQP）
  - Gate FAIL → **STOP**；不扩 ep1–10
- Deployable still **GMHQP 6/10**. Oracle 9/10 diagnostic only.

## Parallel wave-5 (GMHQP residual RX, 2026-09-03)

| Line | Agent | Disease → medicine | Status |
|------|-------|-------------------|--------|
| A | tip-obs for GMHQP residual | proxy tip wrong → TEC tip_hat | smoke **1/4** → full **4/10** ≤GMHQP **STOP** |
| B | **REUSE** Escande GMHQP⊕planar_C | F1 saturate → Level-2 switch to S8 | smoke **0/4 FAIL**；ep05 planar 5.6+mouth；**STOP** |

- RX: `docs/GMHQP_RESIDUAL_PRIV_RX.md`
- B design: `docs/TRACK_B_HYBRID_GMHQP.md` §0 **Search→reuse**（不 invent 平行缩写）
- B code: `src/pci/track_b_hybrid_gmhqp.py` · `S2_trackB_hybrid_gmhqp.yaml`
- Gate: enter-win≥1 on {2,3,5,7} or planar&lt;5+mouth；else STOP
- Smoke result: **RATE 0/4**；ep02/03/07 同 GMHQP F1；ep05 planar 8.9→5.6 + mouth（未&lt;5）→ **gate FAIL / STOP 不调参**

## Track-B Escande GMHQP⊕planar_C smoke (wave-5 REUSE)

- Design §0 Search→reuse: Escande + GMHQP tip-task + S8 planar_C（非 invent）
- Out: `S2_trackB_hybrid_gmhqp_smoke` eps **2,3,5,7**
- **RATE 0/4** vs GMHQP 同集 0/4；未收回 Type-A ep2/5
- Oracle 9/10 stays diagnostic-only

## Track-A tip+axis observer (wave-4, 2026-09-03)

- **Priv gap:** v0 tip_hat 可到口（ep3 planar~4.5, mouth T）但 axis **73°**；tip-only theory EKF **0/4**；Oracle ep3 axis **4.3°**+enter。
- **Theory:** TEC/TEXterity continuous extrinsic pose — state \(\hat t_\parallel,\hat a_{\mathrm{peg}},C\); ContactMotion + finger proprio / wrench / intermittent contact. Cites Kim TEC / TEXterity / Active Extrinsic.
- Design: `docs/TRACK_A_TIP_AXIS_OBSERVER.md`；code `ExtrinsicTipAxisEKF` in `tip_theory_estimator.py`
- Config / out: `S2_trackA_tip_axis.yaml` → `S2_trackA_tip_axis_smoke`
- Control: tip_hat + **axis_hat** → same Oracle-v2 law（无 `feat.peg_axis` GT / 无 tip_gt+noise）
- Residual privilege disclosed: intermittent peg↔tray `contact.pos` + wrench + finger proprio axis + wrist approach prior + priv grasp slip
- Smoke ep1–4: **RATE 2/4**（ok 1,2；fail 3,4）= v0 **2/4**；ep03 仍 axis≈77° / `tip_est_axis_err_peak≈70°`；**未达 ≥3/4 → STOP 不扩 ep1–10；不搜方差旋钮**
- Honest Track-A RATE baseline remains v0 **2/4**


## Track-B PHIG smoke (2026-09-03)

- Config / out: `S2_trackB_phig.yaml` → `outputs/scheme_l3/S2_trackB_phig_smoke`（eps 3,4,7,8,9,10）
- **RATE 0/6** vs Type-A same subset **1/6** (only ep04)
- keep/win/lose: keep Type-A fails 3/7/8/9/10；**lose ep04**；wins on Type-A fails = **0**
- Gate (keep ep04 + ≥2 enter wins) **FAIL** → **未扩 ep1–10**
- Diag (`diag_spiral_priv.py`): `priv_planar_min` 5.7–15.6 mm；全 `spiral_timeout`；grasp_slip 65–228 mm；mouth_press 未触发
- Details: `docs/TRACK_B_ALTERNATIVES.md` §6

## Track-B joint smoke (2026-09-03)

- Config / out: `S2_trackB_joint.yaml` → `outputs/scheme_l3/S2_trackB_joint_smoke`
- **2/6**: keep ep04, win ep09; fail ep03/07/08/10 still planar 6–12 mm
- Gate fail-set enter-win ≥2 → only 1 → **BORDERLINE**; slip ep10 1032→198 mm
- Next only if user authorizes: tighter LOCAL_RECOVER / mouth press, or wait PHIG wire
