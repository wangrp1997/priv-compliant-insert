#!/usr/bin/env python3
"""Parallel ep3 sweep: priv-lat search variants."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
PY = "/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python"
BASE = REPO / "configs" / "demo_spiral_J_lissajous.yaml"
OUT = REPO / "outputs" / "pk_priv_lat_sweep"
CFG_DIR = REPO / "outputs" / "pk_priv_lat_sweep_cfgs"

TRIALS = [
    {
        "id": "K_lat_liss",
        "desc": "priv_lat_lissajous axis_ff=0.5 scale=1.30",
        "over": {
            "surface_tip_search_mode": "priv_lat_lissajous",
            "surface_priv_lat_frames": 480,
            "surface_priv_lat_step_m": 0.002,
            "surface_priv_lat_axis_ff": 0.5,
            "surface_priv_lat_dither_ax_m": 0.004,
            "surface_lift_right_grasp_scale": 1.3,
            "surface_right_grasp_scale": 1.3,
            "surface_tip_spiral_freeze_offset": True,
        },
    },
    {
        "id": "L_lat_only",
        "desc": "priv_lat pure radial axis_ff=0.5",
        "over": {
            "surface_tip_search_mode": "priv_lat",
            "surface_priv_lat_frames": 480,
            "surface_priv_lat_step_m": 0.002,
            "surface_priv_lat_axis_ff": 0.5,
        },
    },
    {
        "id": "M_scale115",
        "desc": "priv_lat_liss scale=1.15 less peg tilt",
        "over": {
            "surface_tip_search_mode": "priv_lat_lissajous",
            "surface_priv_lat_axis_ff": 0.5,
            "surface_lift_right_grasp_scale": 1.15,
            "surface_right_grasp_scale": 1.15,
        },
    },
    {
        "id": "N_nofreeze",
        "desc": "priv_lat_liss live wrist-tip offset",
        "over": {
            "surface_tip_search_mode": "priv_lat_lissajous",
            "surface_priv_lat_axis_ff": 0.5,
            "surface_tip_spiral_freeze_offset": False,
        },
    },
    {
        "id": "O_strong_ff",
        "desc": "priv_lat_liss axis_ff=1.2 step=3mm",
        "over": {
            "surface_tip_search_mode": "priv_lat_lissajous",
            "surface_priv_lat_step_m": 0.003,
            "surface_priv_lat_axis_ff": 1.2,
            "surface_priv_lat_dither_ax_m": 0.003,
        },
    },
]


def _run_one(trial: dict) -> dict:
    tid = trial["id"]
    out_dir = OUT / tid
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = CFG_DIR
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{tid}.yaml"
    cfg = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    cfg.setdefault("approach", {}).update(trial["over"])
    cfg["_demo"] = f"priv_lat_{tid}"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    log_path = out_dir / "run.log"
    cmd = [
        PY,
        str(REPO / "scripts" / "smoke_ep0.py"),
        "--episode",
        "3",
        "--config",
        str(cfg_path),
        "--out-dir",
        str(out_dir),
        "--video",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    summary_path = out_dir / "ep03_summary.json"
    row = {
        "id": tid,
        "desc": trial["desc"],
        "ok": proc.returncode == 0,
        "video": str(out_dir / "ep03_ego.mp4"),
    }
    if summary_path.is_file():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        sm = s.get("surface_meta") or {}
        g = sm.get("priv_tip_spiral_gate") or {}
        row.update(
            {
                "gate": g.get("ok"),
                "insert_ok": s.get("insert_ok"),
                "final_lat_mm": round(float(s.get("final_lat_m", 0)) * 1000, 2),
                "spiral_lat_min_mm": sm.get("spiral_lat_min_mm"),
                "spiral_reason": sm.get("spiral_reason"),
                "search_mode": sm.get("search_mode"),
                "grasp_slip_mm": round(
                    float(sm.get("grasp_slip_peak_m") or 0) * 1000, 1
                ),
                "peg_tilt_peak_deg": sm.get("phase_a_peg_tilt_peak_deg"),
            }
        )
    else:
        row["error"] = "no summary"
    return row


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(_run_one, t): t["id"] for t in TRIALS}
        for fut in as_completed(futs):
            tid = futs[fut]
            try:
                row = fut.result()
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            except Exception as e:
                rows.append({"id": tid, "error": str(e)})
    rows.sort(key=lambda r: r.get("id", ""))
    result = OUT / "result.json"
    result.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {result}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
