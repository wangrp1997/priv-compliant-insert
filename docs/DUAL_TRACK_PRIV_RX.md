# Dual-track prescription from privileged diagnosis (2026-09-03)

> Method gate: `docs/METHOD_GATE_NO_PATCH.md`  
> Sources: `diag_spiral_priv.py` on Type-A / Oracle-v2 / Track-A v0 / theory EKF / OIGS  
> SUCCESS_STANDARD locked; Oracle **9/10** diagnostic only.

## 1. Failure taxonomy (numbers)

| ID | Scientific failure | Signature (priv) | Oracle contrast |
|----|-------------------|------------------|-----------------|
| **F1** | Non-rigid **腕≠tip**（\(C\neq I\)）→ tip 跟不住已知螺旋 | TypeA ep3: slip only **18.6** mm 仍 `priv_planar_min=8.4` mm, mouth F; ep9: 7.5 mm | Oracle ep3: planar **4.47** + mouth，slip 11 — **tip PD 关掉平面缝** |
| **F2** | Grasp **大滑移** → tip 滞后 / 轴崩 | TypeA ep7/10: slip **672 / 1032** mm, tip_err_max 79–167, axis~90° | Oracle 同集 slip **4.6 / 6.0**，planar~4.5 — **reseat + tip PD** |
| **F3** | **到口后**轴/加压失败 | TypeA ep8: mouth T, planar 5.9, 无力入; Oracle **唯一败** ep4: mouth T, axis **42.8°** | tip 到口不够；需 upright + press |
| **F4** | Track-A **tip_hat 观测不足 / 轴错** | v0 ep3: mouth T 但 axis **73°**; tip_est_err_peak **46** mm; theory EKF 无接触 ~**90** mm → **0/4** | Oracle tip GT → ep3 axis **4.3°** 且进孔 |

Rejected method lessons (not to repeat):

| Try | Result | Lesson |
|-----|--------|--------|
| PHIG / CLEP / FASR | planar 更差或补丁 | 腕路径/力阈值 **不能替代 tip 任务** |
| HardHQP | slip↓, ep7 along_reject | 只压滑移不够 F1 |
| OIGS | ep3 planar 8.4→5.5+mouth 但仍 timeout; tip_err **77** mm | 物体阻抗若未把 **任务写在 tip/物体坐标** 会漂 |
| theory EKF wrench-only | **0/4** | 纯腕扳手 **不够** tip 信息；需接触/指尖因子 |

## 2. Scientific problems (one sentence each)

1. **F1 控制问题:** 已知 \(p^*(\theta)\) 下，如何在不测 tip GT 时让 **物体 tip** 跟踪螺旋（经 grasp map \(G\) / \(C\)），而不是腕 TCP 跟踪。  
2. **F2 控制问题:** 大滑移时如何 **恢复接触约束**（reseat）并重建 \(C\approx I\) 后再跟路径。  
3. **F3 控制问题:** tip 已在口邻域时，如何在座面约束下完成 **轴锥≤25° + 力入孔**。  
4. **F4 估计问题:** 间歇 peg↔tray 接触下，如何估计 **tip 平面位姿 + peg 轴**，供 Oracle 同款律使用。

## 3. Dual-track prescription (对症)

### Track A — 对 F4（兼及 F3）

| | |
|--|--|
| **病** | tip_hat 平面偶可达口，但轴错；wrench-only EKF 信息不足 |
| **药** | **TEC/TEXterity 风格多传感外接触观测器**：状态 \(\hat t_\parallel,\hat a_{\mathrm{peg}}\)；测量 = 间歇接触点/法向 + 指尖运动学/力；过程 = ContactMotion（禁止 \(t=w+o\)） |
| **Cite** | Kim TEC ICRA’23；TEXterity ICRA’24；Active Extrinsic ICRA’22 |
| **禁** | freeze / 降 seat_force / 搜 EKF 方差补丁 |
| **验** | tip_hat→Oracle-v2；对照 v0 **2/4**；目标 ≥3/4 同子集再谈全量 |
| **Status (2026-09-03)** | Landed `ExtrinsicTipAxisEKF` + `S2_trackA_tip_axis`；smoke **2/4** = v0；ep03 仍 axis≈77°（`axis_hat` peak err≈70°）；**&lt;3/4 → STOP 不扩 / 不搜参**。Doc: `TRACK_A_TIP_AXIS_OBSERVER.md` |

### Track B — 对 F1（主），F2 为约束

| | |
|--|--|
| **病** | \(C\neq I\) → `priv_planar_min` 饱和；OIGS 未真正把任务写在 tip |
| **药** | **Grasp-map 操作空间 / 物体任务 HQP**：任务 \(e = \Pi(p_{\mathrm{tip}}-p^*)\) 经 \(G\)（或估 \(C\)）映射到腕+指；座面/轴为硬优先；已知螺旋为 tip 参考 |
| **Cite** | Pfanne RA-L 2020 object impedance；GraspQP；Escande HQP；Montana grasp map |
| **禁** | 再堆 mouth_press_scale / FASR 力偏置 / 腕螺旋换皮 |
| **验** | fail-subset **3,9,10**；目标 planar→&lt;5 mm + mouth 或 enter-win≥1 |
| **落地** | `docs/TRACK_B_GRASP_MAP_HQP.md` · `src/pci/track_b_gmhqp.py` · `S2_trackB_gmhqp.yaml` |
| **Smoke** | `S2_trackB_gmhqp_smoke` **RATE 2/3** gate PASS（enter ep09/10） |
| **Full** | `S2_trackB_gmhqp_ep1_10` **RATE 6/10**（ok 1,4,6,8,9,10；fail 2,3,5,7）vs Type-A **5/10** +1；wins 8/9/10；lose 2/5；fails planar 8.9–11.7；**METHOD_GATE STOP 不调参**；Oracle 9/10 diagnostic only |

## 4. Parallel agents

Launch A and B against this document; Phase 0 only to **cite these numbers**, not re-invent taxonomy.
