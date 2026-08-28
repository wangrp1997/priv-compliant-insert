# SoftContact Bias Probe is Phase A of PCI — dual-arm privileged compliant insert

仓库：[wangrp1997/priv-compliant-insert](https://github.com/wangrp1997/priv-compliant-insert)  
本地目录：`priv_compliant_insert`  
简称：**PCI**（Privileged Compliant Insert）

目标：双臂 bimanual assembly **handoff 后柔顺插孔**。  
当前里程碑：**偏置孔 + PBVS ALIGN + 力门控贴面**（插不进、轻触停），再进入 Phase B 双臂柔顺精插。

## 快速跑（贴面 smoke）

```bash
# 需先 clone: https://github.com/wangrp1997/dexjoco
export DEXJOCO=/path/to/dexjoco
cd /path/to/priv-compliant-insert
MUJOCO_GL=egl PYTHONPATH="src:${DEXJOCO}:${DEXJOCO}/dexjoco:${DEXJOCO}/embodied_grasp_insertion:${DEXJOCO}/../reach_insert_rl" \
  python scripts/smoke_pbvs_surface.py --episode 1 --seed 0
```

输出：`outputs/smoke/epXX_pbvs_surface.mp4` + `.json`

## 文档

- [`docs/PLAN.md`](docs/PLAN.md) — 总体方案与阶段
- [`docs/LITERATURE.md`](docs/LITERATURE.md) — Phase B 文献
- [`docs/REF_NOTES.md`](docs/REF_NOTES.md) — 参考仓库笔记

## 依赖

- DexJoCo：[wangrp1997/dexjoco](https://github.com/wangrp1997/dexjoco)
- Python 3.11+，`PYTHONPATH` 含本仓 `src` 与 DexJoCo
