#!/usr/bin/env python3
"""Quick feasibility scan of reference repos for PCI tip-tracking schemes."""
from __future__ import annotations

import json
import re
from pathlib import Path

REFS = Path("/home/wangrenpeng/priv_compliant_insert/refs")

SCHEMES = {
    "A_conntact_spiral": {
        "repo": "ConnTact",
        "need": ["SpiralToFindHole", "seeking_force", "comply"],
        "pci_fit": "B1 贴面螺旋+孔检测；刚性夹持假设",
        "blocker": "无 grasp slip 模型；需改接 dexjoco",
    },
    "B_tip_servo_offset": {
        "repo": "ConnTact",
        "need": ["offset", "tip", "wrist"],
        "pci_fit": "已有 _tip_servo_lift_wrist_cmd；wrist_ff 曾 lat→0.5mm",
        "blocker": "缺贴面约束+在线 offset 更新",
    },
    "C_extrinsic_estimator": {
        "repo": "Tactile-Estimator-Controller",
        "need": ["factor", "gtsam", "extrinsic", "tactile"],
        "pci_fit": "TEXterity 前身；估 peg-in-hand + 外接触",
        "blocker": "ROS+GelSight；仿真可用 priv 替",
    },
    "D_contact_implicit_mpc": {
        "repo": "in_hand_manipulation_2",
        "need": ["mpc", "crocoddyl", "contact", "mujoco"],
        "pci_fit": "手内 contact-implicit MPC；可扩 extrinsic",
        "blocker": "ROS2+Leap hand；非双臂 peg-tray",
    },
    "E_letac_mpc": {
        "repo": "LeTac-MPC",
        "need": ["mpc", "tactile", "gelsight"],
        "pci_fit": "触觉+可微 MPC 防 slip",
        "blocker": "平行夹爪+GelSight；非 insertion spiral",
    },
    "F_actp_slip_predict": {
        "repo": "action_conditioned_tactile_prediction",
        "need": ["lstm", "slip", "tactile", "action"],
        "pci_fit": "action→触觉/滑移预测；粘滞 recovery",
        "blocker": "需数据采集；真机 tactile",
    },
    "G_nature_slip_traj": {
        "repo": "bgf",
        "need": ["slip", "trajectory", "modulation"],
        "pci_fit": "Nature MI：改轨迹而非只加抓力",
        "blocker": "pick-place；需改 spiral 场景",
    },
    "H_slip_aware_inhand": {
        "repo": "Slip-Aware-Object-Manipulation",
        "need": ["slip", "force", "controlled_slippage"],
        "pci_fit": "controlled slippage 重定位",
        "blocker": "平行夹爪+自定义传感器",
    },
    "I_dpse_search": {
        "repo": "dpse",
        "need": ["spiral", "search", "probe"],
        "pci_fit": "螺旋/probe 搜索策略优化",
        "blocker": "长期调参；不解决跟手",
    },
    "J_irl_admittance": {
        "repo": "irl_control",
        "need": ["admit", "osc", "mujoco"],
        "pci_fit": "双臂 MuJoCo admittance 参考",
        "blocker": "无 tip 闭环；可作底层",
    },
}


def scan_repo(repo_path: Path) -> dict:
    if not repo_path.exists():
        return {"exists": False, "files": 0, "py": 0, "readme": False}
    files = list(repo_path.rglob("*"))
    py = [p for p in files if p.suffix == ".py" and p.is_file()]
    readme = any(p.name.lower().startswith("readme") for p in repo_path.iterdir() if p.is_file())
    text = ""
    for p in list(py)[:200]:
        try:
            if p.stat().st_size < 120_000:
                text += p.read_text(errors="ignore").lower() + "\n"
        except OSError:
            pass
    if readme:
        for p in repo_path.iterdir():
            if p.name.lower().startswith("readme"):
                try:
                    text += p.read_text(errors="ignore").lower()
                except OSError:
                    pass
    return {
        "exists": True,
        "files": len([f for f in files if f.is_file()]),
        "py": len(py),
        "readme": readme,
        "text_kb": len(text) // 1024,
        "_text": text,
    }


def score_scheme(meta: dict, scan: dict) -> tuple[int, list[str], list[str]]:
    if not scan.get("exists"):
        return 0, [], ["repo missing"]
    text = scan.pop("_text", "")
    hits = [k for k in meta["need"] if k.lower() in text]
    miss = [k for k in meta["need"] if k.lower() not in text]
    score = len(hits) * 20 + (10 if scan.get("py", 0) > 5 else 0)
    deps = []
    for d in ("ros", "ros2", "gelsight", "libfranka", "crocoddyl", "gtsam", "pybullet", "isaac"):
        if d in text:
            deps.append(d)
    if "mujoco" in text:
        deps.append("mujoco")
    score += 10 if "mujoco" in deps else 0
    score -= min(20, len([x for x in deps if x in ("ros", "ros2", "gelsight", "libfranka")]))
    return max(0, min(100, score)), hits, miss + ([f"deps:{','.join(sorted(set(deps)))}"] if deps else [])


def main() -> None:
    rows = []
    for sid, meta in SCHEMES.items():
        rp = REFS / meta["repo"]
        scan = scan_repo(rp)
        score, hits, notes = score_scheme(meta, scan)
        rows.append(
            {
                "id": sid,
                "repo": meta["repo"],
                "score": score,
                "hits": hits,
                "notes": notes,
                "pci_fit": meta["pci_fit"],
                "blocker": meta["blocker"],
                "py_files": scan.get("py", 0),
                "exists": scan.get("exists", False),
            }
        )
    rows.sort(key=lambda r: -r["score"])

    # recommend tier
    for r in rows:
        if r["score"] >= 60:
            r["tier"] = "A 可快速嫁接"
        elif r["score"] >= 40:
            r["tier"] = "B 需改编"
        elif r["score"] >= 20:
            r["tier"] = "C 仅参考"
        else:
            r["tier"] = "D 不可用/缺仓库"

    out = REFS / "scheme_eval.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"Wrote {out}\n")
    print(f"{'ID':<22} {'score':>5}  {'tier':<12} repo")
    print("-" * 72)
    for r in rows:
        print(f"{r['id']:<22} {r['score']:>5}  {r['tier']:<12} {r['repo']}")
    print("\n--- PCI 推荐组合 ---")
    print("短期: B_tip_servo_offset + A_conntact_spiral(孔检测) + J_irl_admittance(底层)")
    print("中期: C_extrinsic_estimator(priv仿真) + B 闭环")
    print("论文向: C + F/G(slip预测/recovery) + 短视界 MPC(D/E 思想)")


if __name__ == "__main__":
    main()
