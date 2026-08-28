# DexJoCo 复用索引

## 主用（Phase A 粗对准）

| 模块 | 路径 | 复用什么 |
|------|------|----------|
| hybrid PBVS 几何 | `dexjoco/dexjoco/hybrid_insert/geometry.py` | `pbvs_tip_feature_error`, `pbvs_tip_velocity`, tip/socket |
| **hybrid 双臂控制器** | `dexjoco/dexjoco/hybrid_insert/controller.py` | **ALIGN 段** `_pbvs_servo(mode="align")`、双臂相对 PBVS、无 snap |
| hybrid 配置 | `dexjoco/dexjoco/hybrid_insert/config.py` | `pbvs_standoff_m`, `pbvs_lambda_*`, `pos_tol_m` |
| 接触判定 | `dexjoco/dexjoco/hybrid_insert/assembly_contacts.py` | `insert_ok`, peg/tray lost |
| eval 集成 | `dexjoco/dexjoco/hybrid_insert/integration.py` | handoff hook 参考 |

## Phase B 参考（柔顺 — 文献，非 echo_insert）

| 文献/模块 | 路径或链接 | 复用什么 |
|-----------|------------|----------|
| **双臂 insert RA-L 2022** | [Lee et al.](https://doi.org/10.1109/lra.2022.3187497) | 双臂分工、insert 阶段力控 |
| **UR3 三段柔顺** | [ISR 2020](https://doi.org/10.1007/s41315-020-00136-1) | linear + spiral search + impedance |
| **PSFT 螺旋** | [RA-L 2020](https://doi.org/10.1109/lra.2020.3000428) | 部分螺旋、tilted peg |
| **任务空间 compliance** | [TUM dual-arm](https://mediatum.ub.tum.de/doc/1703739/) | 对准/插入分阶段 |
| 接触判定 | `dexjoco/dexjoco/hybrid_insert/assembly_contacts.py` | `insert_ok` |
| hybrid RELEASE 阈值 | `hybrid_insert/config.py` | `release_insert_socket_dist_m` 等 |

详见 [`docs/LITERATURE.md`](LITERATURE.md)。

## handoff / sim

| 模块 | 路径 | 说明 |
|------|------|----------|
| handoff 回放 | `dexjoco/interaction_retarget/skill_replay/insert.py` | demo → peg_lift_end |
| handoff eval | `dexjoco/scripts/eval_lerobot_demo_handoff_insert.py` | `_prepare_handoff` |
| 特权几何特征 | `reach_insert_rl/reach_insert_rl/env/full_obs.py` | `privileged_full_features` |

## 明确不用

| 模块 | 原因 |
|------|------|
| **`priv_snap_insert/`** | snap/O2H/手指粘物体 |
| **`echo_insert/`** | 本地实现有问题；Phase B 改文献 admittance+螺旋 |
| hybrid INSERT 硬 PBVS | 近孔易 ram |

**原则**：Phase A **直接复用或薄封装** `HybridInsertController` 的 ALIGN；Phase B 新写 `pci/compliant/`。
