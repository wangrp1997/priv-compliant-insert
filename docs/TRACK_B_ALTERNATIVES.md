# Track-B alternatives toward Oracle-level 9/10 (no tip GT)

> **合规特权诊断:** Oracle-v2 = **9/10** 上界（tip↔口 GT）；Type-A = **5/10** 可部署。  
> **科学问题:** 已知 tip–口平面轨迹下，非刚性抓取如何保贴面 + 轴≤25° + 真力入孔。  
> **Track-B 约束:** 控制环**不用** `tip_gt` / `tip_gt+noise`；**不放宽** `SUCCESS_STANDARD`。  
> **对照:** Track-A = 估 tip 后复用 Oracle-v2 控制律；本文件 = 换算法 / 联立优化。

---

## 1. Ranked options

| Rank | Method | Oracle privilege avoided | Fit to DexJoCo handoff | 1–2 wk implement | Cite / repo |
|------|--------|--------------------------|------------------------|------------------|-------------|
| **1 (primary)** | **PHIG** — Path-constrained Hybrid Impedance + Grasp co-opt | Full tip XY (and tip_hat). Uses frozen mouth/spiral + wrist F/T + proprio + grasp QP | High: Tip-Hybrid FSM + GraspQP + wrist admit already in tree | **High** — skeleton landed | Raibert&Craig 1981 hybrid; Hogan 1985 impedance; GraspQP (`refs/graspqp`); Escande HQP soft weights |
| 2 | Kim TEC / TEXterity joint est–ctrl (force-enter retarget) | Does **not** avoid tip/object pose — jointly estimates it. Borderline Track-A if tip_hat enters PD | Medium: `refs/Tactile-Estimator-Controller`; sim needs tactile substitute | Medium — FG/iSAM heavy; slim factor version ~2 wk | Kim et al. ICRA 2023 *Simultaneous Tactile Estimation and Control*; Bronars et al. ICRA 2024 TEXterity (arXiv:2403.00049); Active extrinsic PiH Kim ICRA 2022 |
| 3 | Escande-style **hard** HQP stack (seat ≻ upright ≻ path ≻ grasp) | Tip XY; constraints on F/T + orientation + path residual in wrist frame | High: `tip_tracking_qp.py` already ridge LS | High for soft weighted; medium for true hierarchical inequalities | Escande, Mansard, Wieber IJRR 2014 |
| 4 | LeTac-MPC / bgf trajectory modulation | Tip GT; uses tactile/slip prediction to reshape wrist spiral | Medium: `refs/LeTac-MPC`, `refs/bgf`; need action-conditioned slip cue | Low–medium — MPC NLP port cost | Xu et al. TRO 2024 LeTac-MPC; Nazari et al. Nature MI 2025 bgf |
| 5 | Hybrid force/position **alone** on known spiral (ConnTact / Franka / PSFT class) | Tip GT (uses TCP) | Already faithful baselines | Done — **0/10** under SUCCESS_STANDARD | ConnTact (`refs/ConnTact`); Park PSFT RA-L 2020; Franka jam spiral |
| 6 | Learning residual Δwrist from wrist F/T along known spiral | Tip GT; learns mouth approach residual | Medium if offline rollouts exist | Medium — data + train loop | Dong tactile-RL ICRA 2021 (cite); no in-repo policy yet |
| 7 | Active extrinsic pivot (contact-line estimate, no full tip) | Full tip pose; uses extrinsic contact geometry | Medium: Kim ICRA 2022 idea; needs contact Jacobian | Medium | Kim & Rodriguez ICRA 2022 Active Extrinsic Contact |
| 8 | Contact-implicit MPC (in-hand → extrinsic) | Tip GT if wrench/contact constraints suffice | Low–medium: `refs/in_hand_manipulation_2` ROS2/Leap | Low in 1–2 wk | scheme_eval `D_contact_implicit_mpc`; Kim IROS 2024 / IMPACT |

**Reject for Track-B primary:** anything that needs stepwise `tip_gt` or `tip_gt+noise` in the control law (Oracle / Track-A obs bridge).

---

## 2. Why rank-1 PHIG can approach 9/10

Oracle-v2 成功三件套（相对 v1）: tip→口、mouth axial press、slip reseat + seat/axis hold。

| Oracle 成分 | Track-B PHIG 替代 |
|-------------|-------------------|
| tip→口 PD（需 tip） | 已知螺旋 `p*(θ)` 上 **腕路径阻抗** + GraspQP 保 `C≈I`，使腕运动≈tip 运动 |
| mouth press | 可部署：`r_cmd` 近口 + F/T 门 → 轴向加压（不依赖 tip lat GT） |
| slip reseat | 可部署：proprio slip / 剪切残差 → SEAT + grasp tighten（与 Oracle-v2 同族，去掉 tip_lag 特权） |
| hold enter | 可部署：axis/along/float 审计未过则不停（SUCCESS_STANDARD） |

发表基线腕螺旋 **0/10** 证明「仅已知路径」不够；Oracle **9/10** 证明「贴面+扶正+加压」够。  
PHIG = 已知路径腕跟踪 + Tip-Hybrid 贴面/扶正 + 抓取联立保刚 + Oracle-v2 可部署加压/reseat。  
若抓取把 `C` 钉在 `I` 附近，则无需 tip 估计即可逼近 Oracle 跟踪效果。

---

## 3. Primary method: PHIG (equations + FSM)

### 3.1 Signals (eligible)

- Frozen mouth / spiral center `c`（视觉粗定位，问题锁允许）  
- Planned spiral `p*(θ)` on contact plane  
- Wrist pose `w`, peg orientation from FK / extrinsic grasp  
- Wrist F/T residual `f`（法向 `f_n`、切向 `f_t`）  
- Grasp slip proxy（指关节 / 相对位姿）  
- **Forbidden in loop:** peg tip GT, hole GT beyond frozen `c`

### 3.2 Soft hierarchy (Escande spirit, 1-step)

Priority (hard → soft):

1. **SEAT:** \(f_n \ge f_{\mathrm{seat}}\)；沿 \(-n\) 加压 / 卸载  
2. **UPRIGHT:** \(\angle(a_{\mathrm{peg}}, a_{\mathrm{hole}}) \le \theta_{\mathrm{soft}}\)（tip-pivot）  
3. **PATH:** \(\Pi_n(w - p^*(\theta)) \to 0\)  
4. **GRASP:** squeeze / cone QP（GraspQP）使腕→tip 近似恒等  

One-step wrist command:

\[
\Delta w_\parallel = \mathrm{clip}\!\Big(
  K_p\,\Pi_n\big(p^*(\theta)-w\big)
  + K_f\,\Pi_n(f)
  ,\; \Delta_{\max}\Big)
\]

\[
\Delta w_\perp = k_n\,(f_{\mathrm{des}} - f_n)\,n
\]

\[
w^+ = w + \Delta w_\parallel + \Delta w_\perp
\]

Grasp co-opt (existing `priv_grasp_opt` / GraspQP pattern):

\[
\min_\lambda \; \|G\lambda - w_{\mathrm{obj}}\|^2 + \lambda_{\mathrm{reg}}\|\lambda-\lambda_0\|^2
\quad\text{s.t.}\quad \lambda\ge 0,\; f_{\min}\le \|f_i\|\le f_{\max}
\]

with \(w_{\mathrm{obj}}\) from upright + squeeze (Pfanne-style object impedance cite).

Mouth press (no tip lat):

\[
\text{near\_mouth} \iff
  r_{\mathrm{cmd}} \le r_{\mathrm{mouth}}
  \;\wedge\;
  (f_n \ge f_{\mathrm{press}} \lor \text{force\_hole\_cue})
\]

\[
\Delta w_\perp \leftarrow \Delta w_\perp + \delta_{\mathrm{press}}\,n
\quad\text{when near\_mouth}
\]

### 3.3 FSM landing

```
SEAT ──(f_n ok)──► UPRIGHT ──(axis≤soft)──► PATH_FOLLOW
   ▲                      │                      │
   │                      │                      ▼
   └──── SLIP_RESEAT ◄────┴──── (slip trip)  MOUTH_PRESS
                                                    │
                                                    ▼
                                              ENTER_HOLD
                                         (no stop if axis/along/float fail)
```

| Mode | Action | Advance spiral? |
|------|--------|-----------------|
| SEAT | axial press; planar scale ↓ | No |
| UPRIGHT | tip-pivot ω; planar scale ↓ | No |
| PATH_FOLLOW | PHIG Δw toward `p*(θ)` | Yes if seat+upright |
| MOUTH_PRESS | raise unload bar + axial press | Freeze ρ |
| SLIP_RESEAT | grasp tighten + SEAT | Freeze |
| ENTER_HOLD | wait SUCCESS_STANDARD | Stop only if audit OK |

Maps onto existing `TipHybridState` modes (`seat` / `upright` / `spiral`) + PHIG submodes in `PhigState`.

### 3.4 Code / config anchors

| Item | Path |
|------|------|
| Design + survey | `docs/TRACK_B_ALTERNATIVES.md` (this file) |
| Controller skeleton | `src/pci/track_b_phig.py` |
| Baseline hook | `baseline_hold_r(..., mode="track_b_phig")` |
| Config | `configs/scheme_l3/S2_trackB_phig.yaml` |
| Unit smoke | `scripts/smoke_track_b_phig.py` |

---

## 4. Next coding checklist (after skeleton)

1. ~~Wire `surface_tip_phig_*` mouth-press / slip-reseat in `sim_runner` (mirror Oracle-v2 flags, **without** tip_lat GT gate).~~ **Done** (`phig_near_mouth` / `phig_slip_trip` + retrip Δ / grasp bypass freeze / hold-enter).  
2. ~~Strengthen spiral-phase GraspQP / left squeeze when `phig` (bypass `freeze_left_hand` like STAR).~~ **Done** on slip-reseat.  
3. ~~Smoke fail-subset ep3,4,7,8,9,10~~ **Done** → **0/6** (Type-A same eps **1/6**). Gate fail → **不开** ep1–10.  
   - Out: `outputs/scheme_l3/S2_trackB_phig_smoke`；diag: tip 未到口（`priv_planar_min` 5.7–15.6 mm）；全 `spiral_timeout`。  
   - vs Type-A: **lose ep04**；wins on Type-A fails = **0**（需 keep ep04 + ≥2 wins）。  
4. Ablate: PHIG path-only vs +grasp vs +mouth_press vs +reseat.  
5. Optional week-2: soft→hard Escande inequality levels if seat fights path.

---

## 5. Disclosure

- Oracle-v2 rate stays **diagnostic only**.  
- PHIG is Track-B deployable candidate if sensors = vision mouth + wrist F/T + proprio (+ sim grasp wrench as disclosed).  
- No `tip_gt+noise` path in PHIG.

---

## 6. Smoke RATE (fail-subset)

| Run | eps | RATE | vs Type-A same eps | Notes |
|-----|-----|------|--------------------|-------|
| `S2_trackB_phig_smoke` (wiring v1, slip_τ=0.035) | 3,4,7,8,9,10 | **0/6** | Type-A **1/6** (ep04 only) | slip thrash froze ρ; mouth_press=0; ep04 lost |
| `S2_trackB_phig_smoke` (retrip Δ + τ=0.12 + seed armed) | 3,4,7,8,9,10 | **0/6** | Type-A **1/6** (ep04 only) | final monitor 2026-09-03；全 `priv_tip_spiral_gate_fail` / `spiral_timeout` |

**Monitor verdict (Track-B PHIG smoke):** RATE **0/6**. keep/win/lose vs Type-A {3,4,7,8,9,10}: keep fails 3/7/8/9/10；**lose ep04**；wins=0。  
Gate (keep ep04 + ≥2 enter wins on Type-A fails) **FAIL** → **do not expand** to ep1–10.  
Bottleneck: tip never reaches mouth (`priv_planar_min` 5.7–15.6 mm)；grasp_slip 65–228 mm；mouth_press 未触发。  
Next only if user authorizes: path/grasp co-opt fix or HardHQP，**不**自动全量。
