# Track-A: general tip/contact factor-graph estimator (Kim TEC spirit)

> Mandate: `docs/GENERAL_ALGO_MANDATE.md` — 通用估计律；失败 ep 仅探针，**禁止**按 ep / keep-set 补丁。  
> Gate: `docs/METHOD_GATE_NO_PATCH.md` — Search→REUSE.  
> Orthogonal to current full eval: gated TEC-slim EKF + NIS → GMHQP (`docs/TRACK_A_TEC_JOINT_ON_GMHQP.md`, `S2_trackA_tec_joint_gmhqp`).  
> This doc is the **next** Track-A upgrade: fuller factor set (ContactMotion, grasp wrench, extrinsic contact, …).  
> Code skeleton: `src/pci/tip_factor_graph.py` (no GTSAM runtime dep; no ep1–10 launch yet).

---

## 1. Scientific problem (general)

Under non-rigid dexterous grasp (\(C(q)\neq I\), slip), known tip→mouth planar path \(p^*\):

\[
\min \; J_{\mathrm{track}}(p_{\mathrm{tip}},p^*) + J_{\mathrm{seat}} + J_{\mathrm{upright}} + J_{\mathrm{enter}}
\]

**Estimation object (any episode):** recover extrinsic tip / contact / grasp wrench so tip-referenced control is information-consistent — not tip = wrist + const offset, not ep-tuned \(\hat t\).

Control ring: **no** tip GT / tip_gt+noise / peg xpos.

---

## 2. Search → REUSE

| Candidate | What | Verdict |
|-----------|------|---------|
| **Kim et al. ICRA 2023 TEC** — Simultaneous Tactile Estimation and Control of Extrinsic Contact ([arXiv:2303.03385](https://arxiv.org/abs/2303.03385)); `refs/Tactile-Estimator-Controller` + `FACTORS.md` | Factor graph joint past-est / future-ctrl; ContactMotion, PoseDiff, Wrench, Torq*, EnergyElastic, Pen* | **REUSE** (spirit + factor catalog; slim numpy NLS, no GTSAM required) |
| Kim & Rodriguez ICRA 2022 Active Extrinsic Contact | Wrench / contact geometry cues | **REUSE** (meas factors) |
| Bronars TEXterity | Continuous extrinsic pose iSAM | Cite ancestor (heavier) |
| Current TEC-slim EKF (`tip_theory_estimator.py`) + NIS gate (`track_a_tec_joint.py`) | Recursive ContactMotion + tip/wrench/geom updates; gate before GMHQP | **Keep running** as orthogonal baseline; this module does **not** replace it mid-eval |
| Ep / keep-set switches | Forbidden | Forbidden |

**Decision:** REUSE Kim TEC factor catalog, mapped onto peg tip / tray extrinsic contact + wrist F/T (GelSlim → sim surrogates: intermittent peg↔tray contact, in-hand geom tip, wrist wrench). Wiring name only when eval-ready: `track_a_tip_fg` / `tip_factor_graph` — **not** a new invent acronym.

---

## 3. State (variables)

Per timestep \(i\) (sliding window \(i\in\{t-W{+}1,\ldots,t\}\) for estimation; optional future \(t{+}1{:}t{+}T\) for control — control half deferred until wired):

| Symbol | Dim (planar slim) | Meaning |
|--------|-------------------|---------|
| \(g_i\) | SE(2)/SE(3) meas | Gripper / wrist pose (FK; often treated as **known**) |
| \(t_i\) | \(\mathbb{R}^2\) | Extrinsic tip on tray plane (tangent coords) |
| \(C_i\) or shared \(C\) | \(2\times 2\) | Wrist→tip planar coupling (\(C\neq I\)) |
| \(c_i\) | \(\mathbb{R}^2\) (+ formation) | Extrinsic contact locus (point / line / patch) |
| \(w_i\) | \(\mathbb{R}^{3}\) or \(6\) | Grasp / intrinsic wrench (force; torque optional) |
| \(a_i\) | \(\mathbb{R}^2\) (optional) | Peg axis in tangent coords (TEXterity continuous pose) |
| \(K\) | grasp params (optional) | Stiffness / compliance center prior (TEC \(K\)) |

**Planar tip convention** (same as TEC-slim EKF): tray normal \(n\), tangent basis \((e_1,e_2)\),

\[
t_\parallel = (t\cdot e_1,\; t\cdot e_2),\qquad
\Delta w_\parallel = \Pi_n(\Delta w_{\mathrm{wrist}}).
\]

Full TEC uses \(g,o,w,c,K\in SE(3)\times\cdots\); we keep the **same factor roles**, projected to tip/contact planar + wrench for force-enter.

---

## 4. Factor catalog (equations)

Kim TEC solves one NLS (Eq.1 in paper):

\[
\hat x = \arg\min_x \sum_f \big\| F_f(x_{1:t},x_{t+1:t+T};\, z_{1:t},z_{t+1:t+T}) \big\|_{\Sigma_f}^2
\]

Past = estimation; future = control. Skeleton implements **estimation residuals** first.

### 4.1 From `refs/Tactile-Estimator-Controller/FACTORS.md` → our residuals

| TEC factor | Our residual (planar / wrench slim) | Role |
|------------|-------------------------------------|------|
| **ContactMotion** | \(F_{\mathrm{cm}} = (t_i - t_{i-1}) - C\,\Delta w_{\parallel,i}\) | Tip moves through grasp map, not rigid \(\Delta w\) |
| **PoseDiff** | \(F_{\mathrm{pd}} = t_i - \Pi_n(z_{\mathrm{geom},i})\) | In-hand object tip prior (TacGraph / geom surrogate) |
| **DispDiff** | \(F_{\mathrm{dd}} = (t_i-t_{i-1}) - \Pi_n(z_{\mathrm{geom},i}-z_{\mathrm{geom},i-1})\) | Relative in-hand motion consistency |
| **Wrench** | \(F_{\mathrm{wr}} = w_i - \hat w_{\mathrm{lin}}(K,\delta_i)\) or Doshi lever: \(\tau \approx r\times F\) with \(r=c-g\) | Grasp wrench regression / lever-arm contact |
| **WrenchInc** | \(F_{\mathrm{wri}} = (w_i-w_{i-1}) - \hat w_{\mathrm{lin}}(K,\delta_i-\delta_{i-1})\) | Incremental wrench–displacement |
| **TorqPoint** | \(F_{\mathrm{tp}} = M - r(g,c)\times F\) (soft → 0) | Point extrinsic: no torque about contact |
| **TorqLine** | \(F_{\mathrm{tl}} = F_{\mathrm{tp}}\cdot a_x(c)\) | Line extrinsic: torque along contact line |
| **EnergyElastic** | \(F_E = w \oslash \sqrt{K}\) (quadratic energy proxy) | Prefer low grasp deformation (slip-aware regularizer) |
| **PenHinge** | \(F_{\mathrm{pen}} = \mathrm{hinge}(d_{\min}-d_{\mathrm{pen}})\) | Maintain minimum extrinsic penetration / seat |
| **PenEven** | even penetration along line/patch | Line/patch formations (later) |
| **DispVar** | variance / smoothness on tactile displacement | Optional temporal regularizer |

### 4.2 Measurement / prior factors (eligible + residual-priv disclosed)

| Factor | \(z\) | Privilege |
|--------|-------|-----------|
| GripperPrior | wrist FK \(g^*\) | eligible |
| ExtrinsicContact | intermittent peg↔tray contact pos | residual-priv in sim (disclosed) |
| WrenchMeas | wrist F/T | eligible |
| GeomTipPrior | in-hand tip surrogate | eligible (GMHQP geom class) |
| AxisPrior (opt.) | finger proprio / approach | eligible |
| ContactFormation | point→line→patch switch costs | theory; soft |

### 4.3 Core equations (estimation window)

**ContactMotion (process / binary):**

\[
F_{\mathrm{cm}}(t_{i-1},t_i,C;\Delta w_i)
  = t_i - t_{i-1} - C\,\Delta w_{\parallel,i}
\]

**Extrinsic contact measurement (unary, intermittent):**

\[
F_{\mathrm{ec}}(t_i;\,z_c) = t_i - \Pi_n(z_c)
\]

**Grasp wrench / Active Extrinsic (unary soft):**

\[
F_{\mathrm{lev}}(c_i;\,g,F,\tau)
  = \tau - (c_i - g)\times F
  \quad\text{(or planar lever residual → tip/contact)}
\]

**Object-fixed contact (TEC \(F_{oc}\) spirit):**

\[
F_{\mathrm{oc}}(t_{i-1},c_{i-1},t_i,c_i)
  = (c_i - t_i) - (c_{i-1} - t_{i-1})
\]

(contact offset in object/tip frame sticky unless formation change.)

**Environment contact (TEC \(F_{cc}\) spirit):**

\[
F_{\mathrm{cc}}(c_{i-1},c_i;\,n)
  = \big[(c_i-c_{i-1})\cdot n\big]\;\text{(strong)}
  +\; \Pi_n(c_i-c_{i-1})\;\text{(weaker slip)}
\]

Shared \(C\) across window (or random-walk \(C_i\)) with slip inflate on large NIS — **never** snap tip←wrist+offset.

---

## 5. vs current TEC-slim (orthogonal)

| | **TEC-slim + joint gate** (running) | **This factor-graph (next A)** |
|--|--------------------------------------|--------------------------------|
| Backend | Recursive EKF (`ExtrinsicTipStateEKF`) | Sliding-window NLS over factors |
| Factors used | ContactMotion predict; contact / geom / wrench tip updates | Full catalog: ContactMotion, PoseDiff/DispDiff, Wrench/WrenchInc, Torq*, Energy, Pen*, \(F_{oc}\)/\(F_{cc}\) |
| Grasp wrench | Only as soft tip lever meas | Explicit \(w\) variables + EnergyElastic |
| Extrinsic contact | Intermittent tip \(z_c\) | \(c\) + tip + torque-about-contact + formation |
| Control coupling | NIS/contact hard gate → GMHQP tip_obj | Future: TEC-style future horizon factors (deferred); est output can still feed GMHQP |
| Eval now | `S2_trackA_tec_joint_gmhqp` **do not restart** | Skeleton + unit test only; **no** ep1–10 yet |

Both obey GENERAL_ALGO_MANDATE: general state/factors, no episode indices in the law.

---

## 6. Implementation plan (skeleton → wire)

1. **Now:** `src/pci/tip_factor_graph.py` — residual APIs + window Gauss–Newton; unit test (no MuJoCo).  
2. **Later:** wire optional backend behind tip_est mode (orthogonal flag); smoke then full ep1–10 vs GMHQP 6/10 and tec_joint RATE.  
3. If RATE ≤ baseline: STOP retune; revise factors / observability, not thresholds.

---

## 7. Cite / reuse map

| Artifact | Path |
|----------|------|
| FACTORS.md | `refs/Tactile-Estimator-Controller/FACTORS.md` |
| Paper | Kim et al., ICRA 2023 / arXiv:2303.03385 |
| TEC-slim EKF | `src/pci/tip_theory_estimator.py` |
| TEC joint gate | `src/pci/track_a_tec_joint.py` |
| Clep / Doshi lever | `src/pci/track_b_clep.py` |
| This skeleton | `src/pci/tip_factor_graph.py` |

Update: `docs/REF_REUSE_MAP.md` (cite Port/Adapt row for tip_factor_graph).
