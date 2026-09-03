# Fail-driven HardHQP (Track B)

> **合规特权诊断:** Type-A 5/10 vs Oracle-v2 9/10；主缺口 F1 planar saturate + F3 mouth axis。  
> **科学问题:** 已知螺旋路径上，用硬优先级保贴面/直立后再跟路径、力入孔。  
> **Track:** B（无 tip GT / tip_gt+noise）。  
> **Taxonomy:** `docs/FAIL_PRIV_TAXONOMY.md`

## Method

Escande-style **hard** HQP cascade (cite: Escande, Mansard, Wieber IJRR 2014):

1. **SEAT (hard):** if wrist F/T \(f_n < f_{\mathrm{seat}}\) → axial press only; path scale = 0  
2. **UPRIGHT (hard):** if axis > soft deg → tip-pivot proxy from FK peg vs hole axis; path frozen  
3. **PATH / MOUTH:** known spiral waypoint wrist impedance; near mouth → axial press  

Differs from PHIG: PHIG uses soft impedance weights; HardHQP **blocks** path until higher levels clear.

## Files

| Item | Path |
|------|------|
| Taxonomy | `docs/FAIL_PRIV_TAXONOMY.md` |
| Module | `src/pci/fail_driven_hard_hqp.py` |
| Config | `configs/scheme_l3/S2_failDriven_hardHQP.yaml` |
| Unit smoke | `scripts/smoke_fail_driven_hard_hqp.py` |
| Hook | `baseline_hold_r(..., mode="fail_driven_hard_hqp")` + `sim_runner` |

## Unit smoke

```bash
cd /home/wangrenpeng/priv_compliant_insert
PYTHONPATH=src /home/wangrenpeng/miniconda3/envs/dexjoco/bin/python \
  scripts/smoke_fail_driven_hard_hqp.py
```

## Fail-subset sim smoke (2026-09-03)

Worst F1/F2 eps: **3,7,10** (TypeA fails). Out: `outputs/scheme_l3/S2_failDriven_hardHQP_smoke`.

```bash
cd /home/wangrenpeng/priv_compliant_insert
export MUJOCO_GL=egl
PYTHONPATH=src:/home/wangrenpeng/dexjoco:/home/wangrenpeng/dexjoco/dexjoco:/home/wangrenpeng/dexjoco/embodied_grasp_insertion \
  /home/wangrenpeng/miniconda3/envs/dexjoco/bin/python \
  scripts/parallel_force_enter_ep10.py \
  --config configs/scheme_l3/S2_failDriven_hardHQP.yaml \
  --out-root outputs/scheme_l3/S2_failDriven_hardHQP_smoke \
  --eps 3,7,10 --workers 2 --video

PYTHONPATH=src /home/wangrenpeng/miniconda3/envs/dexjoco/bin/python \
  scripts/diag_spiral_priv.py \
  outputs/scheme_l3/S2_failDriven_hardHQP_smoke/ep*/ep*_summary.json
```

### RATE (SUCCESS_STANDARD locked)

| Run | eps | enter-win RATE | note |
|-----|-----|----------------|------|
| **HardHQP smoke** | 3,7,10 | **0/3 = 0%** | gate fail |
| Type-A same eps | 3,7,10 | **0/3** | Type-A overall still 5/10 |

Per-ep (HardHQP vs Type-A):

| ep | HardHQP reason | planar_min | slip | Type-A planar / slip |
|----|----------------|------------|------|----------------------|
| 03 | `priv_tip_spiral_gate_fail` | 6.4 mm | 65 mm | 8.4 / 19 |
| 07 | `force_hole_along_reject` (force_hole=1) | 4.5 mm | 108 mm | 10.2 / 673 |
| 10 | `surface_tilt_or_tray` | 14.7 mm | 33 mm | 12.3 / 1032 |

Diag: ep07 got `priv_mouth` + `hole_detected` but along-reject; ep03/10 still planar saturate / no enter. Slip much better than Type-A on 7/10; **no enter-win**.

Gate (honest): enter-win ≥1 on this subset before ep1–10 → **NOT met** → **do not expand**. Oracle 9/10 stays diagnostic-only.

## Disclosure

- No tip GT in control loop.  
- SUCCESS_STANDARD unchanged.  
- Do not put Oracle rate in deployable table.
