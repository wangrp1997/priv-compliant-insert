#!/usr/bin/env python3
"""Scheme D sweep: tip-servo lift + tip-spiral gates (privileged_diagnostic).

Sweeps tip lift target, wrist step, f_des on ep=3. No video.
Writes outputs/pk_lift_D_result.json.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
PY = "/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python"
EP = 3

PYTHONPATH = ":".join(
    [
        str(REPO / "src"),
        "/home/wangrenpeng/dexjoco",
        "/home/wangrenpeng/dexjoco/dexjoco",
        "/home/wangrenpeng/dexjoco/embodied_grasp_insertion",
        "/home/wangrenpeng/reach_insert_rl",
    ]
)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_one(tag: str, cfg: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
    env = {
        **os.environ,
        "PYTHONPATH": PYTHONPATH,
        "MUJOCO_GL": "egl",
    }
    log_path = out_dir / "run.log"
    cmd = [
        PY,
        "-u",
        str(REPO / "scripts" / "smoke_ep0.py"),
        "--episode",
        str(EP),
        "--config",
        str(cfg_path),
        "--out-dir",
        str(out_dir),
    ]
    with log_path.open("w", encoding="utf-8") as logf:
        subprocess.run(cmd, cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT, check=False)
    summary_path = out_dir / f"ep{EP:02d}_summary.json"
    row: dict = {
        "tag": tag,
        "ok": False,
        "success": False,
        "fail_reason": "missing_summary",
        "params": {
            "tip_servo_lift_m": float(cfg["approach"]["surface_tip_servo_lift_m"]),
            "wrist_step_m": float(cfg["approach"]["surface_tip_servo_lift_wrist_step_m"]),
            "f_des_n": float(cfg["approach"]["surface_const_force_des_n"]),
        },
    }
    if not summary_path.is_file():
        return row
    with summary_path.open(encoding="utf-8") as f:
        j = json.load(f)
    sm = j.get("surface_meta") or {}
    gate = j.get("priv_tip_spiral_gate") or sm.get("priv_tip_spiral_gate") or {}
    row.update(
        {
            "success": bool(j.get("success")),
            "fail_reason": j.get("fail_reason", ""),
            "ok": bool(gate.get("ok")),
            "lift_ok": bool(gate.get("lift_ok")),
            "spiral_ok": bool(gate.get("spiral_ok")),
            "force_ok": bool(gate.get("force_ok")),
            "tip_up_mm": float(gate.get("tip_up_m") or sm.get("surface_tip_up_m") or 0.0) * 1e3,
            "along_rise_mm": float(gate.get("along_rise_m") or sm.get("surface_tip_along_rise_m") or 0.0)
            * 1e3,
            "tip_xy_peak_mm": float(gate.get("tip_xy_peak_mm") or sm.get("spiral_tip_xy_peak_mm") or 0.0),
            "spiral_mean_resid_n": float(gate.get("spiral_mean_resid_n") or float("nan")),
            "tilt_peak_deg": float(sm.get("phase_a_tray_tilt_peak_deg") or 0.0),
            "gate": gate,
        }
    )
    return row


def main() -> int:
    base = _load(REPO / "configs" / "pk_lift_D.yaml")
    tmp = REPO / "outputs" / "pk_lift_D_sweep"
    tmp.mkdir(parents=True, exist_ok=True)

    # Tip only follows ~20% of wrist; need ~8mm tip_target for tip_up≥1.5mm.
    # Early-stop on first pass. Grasp tighten (>=1) helps tip follow.
    lifts = [0.008, 0.010, 0.012]
    wrist_steps = [0.008, 0.012]
    f_dess = [0.03, 0.05]

    rows: list[dict] = []
    best: dict | None = None
    for lift_m in lifts:
        for wstep in wrist_steps:
            for fdes in f_dess:
                tag = f"L{lift_m*1e3:.1f}_W{wstep*1e3:.1f}_F{fdes:.2f}"
                cfg = copy.deepcopy(base)
                a = cfg.setdefault("approach", {})
                a["surface_tip_servo_lift_enable"] = True
                a["surface_tip_servo_lift_m"] = float(lift_m)
                a["surface_tip_servo_lift_need_m"] = 0.0015
                a["surface_tip_servo_lift_wrist_step_m"] = float(wstep)
                a["surface_tip_servo_lift_frames"] = int(
                    a.get("surface_tip_servo_lift_frames", 120)
                )
                a["surface_const_force_des_n"] = float(fdes)
                # Keep axial lift off; scheme D owns lift. Grasp stay firm (>=1).
                a["surface_light_lift_m"] = 0.0
                a["surface_light_lift_frames"] = 0
                a["surface_tip_lift_m"] = 0.0
                if float(a.get("surface_lift_right_grasp_scale", 1.0)) < 1.0:
                    a["surface_lift_right_grasp_scale"] = 1.0
                if float(a.get("surface_right_grasp_scale", 1.0)) < 1.0:
                    a["surface_right_grasp_scale"] = 1.0
                print(f"\n=== {tag} ===", flush=True)
                row = _run_one(tag, cfg, tmp / tag)
                rows.append(row)
                print(
                    f"{tag}: ok={int(row['ok'])} "
                    f"up={row.get('tip_up_mm', float('nan')):.2f}mm "
                    f"alongΔ={row.get('along_rise_mm', float('nan')):.2f}mm "
                    f"xy={row.get('tip_xy_peak_mm', float('nan')):.2f}mm "
                    f"mean|r|={row.get('spiral_mean_resid_n', float('nan')):.3f}N "
                    f"fail={row.get('fail_reason')}",
                    flush=True,
                )
                if row.get("ok") and (best is None or row.get("tip_xy_peak_mm", 0) > best.get("tip_xy_peak_mm", 0)):
                    best = row
                # Early stop on first full gate pass.
                if row.get("ok"):
                    break
            if best is not None and best.get("ok"):
                break
        if best is not None and best.get("ok"):
            break

    passed = [r for r in rows if r.get("ok")]
    result = {
        "scheme": "D_tip_servo_lift",
        "experiment_tag": "privileged_diagnostic",
        "episode": EP,
        "gates": {
            "tip_up_or_along_rise_m": 0.0015,
            "tip_xy_peak_mm": 6.0,
            "spiral_mean_resid_n": [0.05, 0.45],
        },
        "n_tried": len(rows),
        "n_pass": len(passed),
        "passed": bool(passed),
        "best": best,
        "rows": rows,
    }
    out = REPO / "outputs" / "pk_lift_D_result.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out} passed={int(bool(passed))} n={len(rows)}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
