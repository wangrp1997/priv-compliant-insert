# PSFT faithful reproduction protocol (Park et al. RA-L 2020)

> **Paper reimplementation, no official open source.**

Park et al., “Compliant Peg-in-Hole Assembly Using Partial Spiral Force Trajectory With Tilted Peg Posture,” IEEE RA-L 2020, DOI [10.1109/LRA.2020.3000428](https://doi.org/10.1109/lra.2020.3000428).  
Recovered PDF (author lab DYROS/SNU preprint): `refs/papers/park_psft_ral2020.pdf`.  
IEEE / arXiv do **not** ship code; this tree ports published equations only.

## Source of truth

| Item | Path |
|------|------|
| Paper PDF | `refs/papers/park_psft_ral2020.pdf` |
| Our port | `src/pci/psft_repro.py` |
| Config | `configs/scheme_l3/S3_psft_faithful.yaml` |
| Search mode | `surface_tip_search_mode: psft_spiral` |
| Eval | `scripts/parallel_force_enter_ep10.py --config .../S3_psft_faithful.yaml` |

## Locked equations (paper)

1. **Spring force (eq 3)**  
   \(f_t = k_p (p_t - p_p)\)

2. **PSFT polar update (eq 10)**  
   \(\theta_t \leftarrow \theta_{t-1} \pm \mathrm{atan}(h/r_t)\) with bounce in \([0,\theta_{\max}]\)  
   \(\hat\theta_t \leftarrow \hat\theta_{t-1} + \mathrm{atan}(h/r_t)\)  
   \(r_t = r_{\min} + \hat\theta_t \cdot d / (2\theta_{\max})\), clipped to \(r_{\max}\)

3. **Wrench (eq 13)**  
   \(f = \Omega f_t + \Omega f_a\), \(m = k_\omega \delta\phi\)  
   Sim: \(\Omega \approx I\) on planar task; seeking owns \(f_a\) along contact normal.

4. **Geometry helpers (eqs 8–9)** implemented in `psft_repro.py` (shift / \(\theta_{\max}\) lower bound); full shift+tilt FSM before contact is only partially adapted (Phase-A already seats the peg).

Fig. 8 defaults used: \(h=0.0005\), \(\theta_{\max}=\pi/2\), \(\Delta t=0.003\). Experiment gains \(k_p=300\), \(k_\omega=3\), \(f_a=7\) N (§IV-B).

## Explicit adaptations (must disclose)

| Park (rigid manipulator + fixed peg) | Our DexJoCo handoff |
|--------------------------------------|---------------------|
| Rigid peg–EE | Allegro **non-rigid** grasp (stress test) |
| Cartesian compliance via \(k_p\) force | Soft sim: **TCP tracks** \(p_t\) (`psft_tcp`) + light seeking around `surface_planned_spiral_f_des_n` (~0.025 N) — same *role* as \(f_a\), **force scale** differs |
| Absolute world frame | Tray-plane tangent basis \((t_1,t_2)\) |
| Full init→shift→tilt→PSFT→wiggle FSM | Phase-A PBVS already seated; we reproduce **PSFT search eqs (3)/(10)/(13)** |
| Hole found ≈ peg velocity → 0 at 3-pt contact | PCI enter marker = height drop 0.4 mm (ConnTact-compatible) **plus** locked `SUCCESS_STANDARD` |
| Intentional tilt \(\alpha\) for ARIE | Config `surface_psft_tilt_rad` recorded; wrist tilt not fully servoed under grasp (disclose) |

## Not this baseline

- Tip-Hybrid / MSAR / STAR / planar_C / S9 → **ours / invent**, disabled in `S3_psft_faithful.yaml`
- ConnTact `conntact_time_spiral` / Franka jam spiral → separate S3 rows

## Numeric lock

```bash
PYTHONPATH=src python -c "from pci.psft_repro import assert_matches_paper_sample; assert_matches_paper_sample(); print('OK')"
```

## Eval command

```bash
MUJOCO_GL=egl PYTHONPATH=src:/home/wangrenpeng/dexjoco \
  /home/wangrenpeng/miniconda3/envs/dexjoco/bin/python \
  scripts/parallel_force_enter_ep10.py \
  --config configs/scheme_l3/S3_psft_faithful.yaml \
  --eps 1-10 --workers 3 \
  --out-root outputs/scheme_l3/S3_psft_faithful_ep1_10
```

Report: SUCCESS_STANDARD rate + per-ep reason / lat_min / axis / grasp_slip (diag).

## Eval result (SUCCESS_STANDARD)

- RATE: **0/10** on `outputs/scheme_l3/S3_psft_faithful_ep1_10/`
- Many `force_hole=1` from height-drop are **false enters** (along still ~100–160 mm) → rejected by seat/along/axis audit (`force_hole_along_reject` / `force_hole_axis_reject`).
- ep07: `grasp_slip_pre_spiral` — never reached PSFT search.
- Tip planar motion under TCP PSFT often tiny (`tip_xy` 0.04–4 mm on fails) while wrist spiral advances — wrist≠tip.
- Paper claim: faithful PSFT under non-rigid handoff grasp does not achieve tip-on-surface upright force-enter (same failure class as ConnTact faithful 0/10).

## Honesty (non-rigid)

PSFT assumes compliance maps wrist force → peg tip. Under dexterous grasp, slip decouples wrist spiral from tip planar motion (`priv_planar` / tip lag). Height-drop enter markers fire without seat → SUCCESS_STANDARD rejects.
