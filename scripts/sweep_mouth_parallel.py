#!/usr/bin/env python3
"""Parallel ep3 sweep: mouth-wiggle variants after priv-lat search."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
PY = "/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python"
BASE = REPO / "configs" / "demo_spiral_J_lissajous.yaml"
OUT = REPO / "outputs" / "pk_mouth_sweep"
CFG_DIR = REPO / "outputs" / "pk_mouth_sweep_cfgs"

M_BASE = {
    "surface_tip_search_mode": "priv_lat_lissajous",
    "surface_priv_lat_frames": 480,
    "surface_priv_lat_step_m": 0.002,
    "surface_priv_lat_axis_ff": 0.5,
    "surface_lift_right_grasp_scale": 1.15,
    "surface_right_grasp_scale": 1.15,
    "surface_tip_spiral_freeze_offset": True,
}

MOUTH_BASE = {
    "surface_mouth_wiggle_enable": True,
    "surface_mouth_continue_on_near": True,
    "surface_mouth_wiggle_frames": 200,
    "surface_mouth_wiggle_xy_m": 0.0018,
    "surface_mouth_wiggle_ax_step_m": 0.00008,
    "surface_mouth_jam_fxy_n": 0.35,
    "surface_mouth_jam_frames": 6,
}

TRIALS = [
    {
        "id": "P1_mouth_peg",
        "desc": "M + mouth wiggle (peg only)",
        "over": {**M_BASE, **MOUTH_BASE},
    },
    {
        "id": "P2_mouth_tray",
        "desc": "M + mouth wiggle + tray admit 0.35",
        "over": {
            **M_BASE,
            **MOUTH_BASE,
            "surface_mouth_tray_admit_scale": 0.35,
            "surface_mouth_tray_admit_max_m": 0.0005,
        },
    },
    {
        "id": "P3_nofreeze_hold",
        "desc": "N live offset + along hold + mouth wiggle",
        "over": {
            **M_BASE,
            **MOUTH_BASE,
            "surface_tip_spiral_freeze_offset": False,
            "surface_priv_lat_along_hold": True,
            "surface_priv_lat_along_hold_max_m": 0.102,
        },
    },
    {
        "id": "P4_jam_early",
        "desc": "M + mouth wiggle + early jam recovery",
        "over": {
            **M_BASE,
            **MOUTH_BASE,
            "surface_mouth_jam_fxy_n": 0.28,
            "surface_mouth_jam_frames": 4,
            "surface_mouth_wiggle_rot_rad": 0.012,
        },
    },
    {
        "id": "P5_handoff_insert",
        "desc": "M + mouth wiggle then Phase-B insert",
        "over": {
            **M_BASE,
            **MOUTH_BASE,
            "surface_handoff_insert_after_gate": True,
        },
        "cfg_over": {
            "compliant": {"search": {"mouth_hold_stop": False}},
        },
    },
    {
        "id": "P6_chamfer_press",
        "desc": "M + mouth wiggle + strong chamfer press",
        "over": {
            **M_BASE,
            **MOUTH_BASE,
            "surface_mouth_wiggle_ax_step_m": 0.00016,
            "surface_mouth_wiggle_lat_step_m": 0.0008,
            "surface_mouth_wiggle_axis_ff": 0.5,
        },
    },
]


def _render_force_combo(out_dir: Path, summary_path: Path) -> None:
    ego = out_dir / "ep03_ego.mp4"
    if not ego.is_file() or not summary_path.is_file():
        return
    force_mp4 = out_dir / "ep03_force.mp4"
    combo = out_dir / "ep03_ego_force.mp4"
    subprocess.run(
        [
            PY,
            str(REPO / "scripts" / "render_force_curve_anim.py"),
            "--summary",
            str(summary_path),
            "--ego",
            str(ego),
            "--out",
            str(force_mp4),
            "--fps",
            "30",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if not force_mp4.is_file():
        return
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(ego),
            "-i",
            str(force_mp4),
            "-filter_complex",
            "[0:v]scale=-2:640[ego];[1:v]scale=-2:640[fc];[ego][fc]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-shortest",
            "-movflags",
            "+faststart",
            str(combo),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _run_one(trial: dict) -> dict:
    tid = trial["id"]
    out_dir = OUT / tid
    out_dir.mkdir(parents=True, exist_ok=True)
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = CFG_DIR / f"{tid}.yaml"
    cfg = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    cfg.setdefault("approach", {}).update(trial["over"])
    for key, val in trial.get("cfg_over", {}).items():
        if isinstance(val, dict) and isinstance(cfg.get(key), dict):
            for subk, subv in val.items():
                if isinstance(subv, dict) and isinstance(cfg[key].get(subk), dict):
                    cfg[key][subk].update(subv)
                else:
                    cfg[key][subk] = subv
        else:
            cfg[key] = val
    cfg["_demo"] = f"mouth_{tid}"
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
        gate_ok = bool(g.get("ok"))
        insert_ok = bool(s.get("insert_ok"))
        row.update(
            {
                "gate": gate_ok,
                "insert_ok": insert_ok,
                "success": bool(s.get("success")),
                "final_lat_mm": round(float(s.get("final_lat_m", 0)) * 1000, 2),
                "final_along_mm": round(float(s.get("final_along_m", 0)) * 1000, 2),
                "spiral_lat_min_mm": sm.get("spiral_lat_min_mm"),
                "spiral_reason": sm.get("spiral_reason"),
                "mouth_wiggle_ok": sm.get("mouth_wiggle_ok"),
                "mouth_wiggle_reason": sm.get("mouth_wiggle_reason"),
                "mouth_frames": sm.get("mouth_wiggle_frames"),
                "grasp_slip_mm": round(
                    float(sm.get("grasp_slip_peak_m") or 0) * 1000, 1
                ),
                "peg_tilt_peak_deg": sm.get("phase_a_peg_tilt_peak_deg"),
            }
        )
        if gate_ok or insert_ok:
            _render_force_combo(out_dir, summary_path)
            combo = out_dir / "ep03_ego_force.mp4"
            if combo.is_file():
                row["force_video"] = str(combo)
    else:
        row["error"] = "no summary"
    return row


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=6) as ex:
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
