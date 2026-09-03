#!/usr/bin/env python3
"""Run ep08 planned-spiral with priv metrics; exit 0 only when PRIV_OK."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def priv_metrics(summary_path: Path) -> dict:
    s = json.loads(summary_path.read_text())
    sm = s.get("surface_meta") or {}
    sp = [r for r in sm.get("force_trace", []) if r.get("phase") == "planned_spiral"]
    if not sp:
        return {
            "ok": False,
            "reason": sm.get("spiral_reason", "no_spiral"),
            "n": 0,
        }
    grace = 120
    resid = np.array([float(r["resid_r"]) for r in sp[grace:]]) if len(sp) > grace else np.array([])
    lat = np.array([float(r["lat_mm"]) for r in sp])
    along = np.array([float(r["along_mm"]) for r in sp])
    reason = sm.get("spiral_reason", "")
    drift = float(sm.get("priv_planned_along_drift_peak_mm") or 0.0)
    priv_enter = bool(sm.get("priv_planned_hole_entered"))
    lat_min = float(lat.min())
    resid_p90 = float(np.percentile(resid, 90)) if resid.size else 999.0
    resid_max = float(resid.max()) if resid.size else 999.0
    ok = (
        reason not in ("priv_overforce", "priv_along_drift", "spiral_tilt_or_tray", "grasp_slip")
        and len(sp) > 200
        and lat_min < 14.0
        and (not resid.size or resid_p90 < 0.10)
        and drift < 8.0
    )
    return {
        "ok": ok,
        "reason": reason,
        "n": len(sp),
        "lat_min": lat_min,
        "along_d": float(along[-1] - along[0]) if along.size else 0.0,
        "resid_p90": resid_p90,
        "resid_max": resid_max,
        "drift_mm": drift,
        "priv_enter": priv_enter,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/P1_tip_surface_spiral.yaml")
    ap.add_argument("--name", default="P1_planned_spiral_surface_priv")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--out-root", default="outputs/pose_hold_ablation/ep08")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--hdd-video", default="/mnt/hdd/dexjoco/outputs/pk_mouth_view/pose_hold_ablation")
    args = ap.parse_args()

    out_dir = ROOT / args.out_root / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", f"src:{env.get('PYTHONPATH', '')}")
    env.setdefault("MUJOCO_GL", "egl")
    py = os.environ.get(
        "DEXJOCo_PYTHON",
        "/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python",
    )
    cmd = [
        py,
        "-u",
        str(ROOT / "scripts/smoke_ep0.py"),
        "--episodes",
        str(args.episodes),
        "--config",
        str(ROOT / args.config),
        "--out-dir",
        str(out_dir),
    ]
    if args.video:
        cmd.append("--video")

    with log_path.open("w") as logf:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"smoke_ep0 failed rc={proc.returncode} log={log_path}", file=sys.stderr)
        return proc.returncode

    summary = out_dir / f"ep{args.episodes:02d}_summary.json"
    if not summary.exists():
        print(f"missing summary {summary}", file=sys.stderr)
        return 2

    m = priv_metrics(summary)
    print(
        f"PRIV {'OK' if m['ok'] else 'FAIL'} | reason={m['reason']} n={m['n']} "
        f"lat_min={m.get('lat_min', float('nan')):.2f}mm "
        f"resid_p90={m.get('resid_p90', float('nan')):.3f}N "
        f"drift={m.get('drift_mm', float('nan')):.1f}mm priv_enter={m.get('priv_enter')}"
    )

    if args.video and args.hdd_video:
        src = out_dir / f"ep{args.episodes:02d}_ego.mp4"
        if src.exists():
            dst_dir = Path(args.hdd_video) / args.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / "ego.mp4"
            try:
                import imageio.v2 as iio

                r = iio.get_reader(str(src))
                w = iio.get_writer(str(dst), fps=30)
                for frame in r:
                    w.append_data(np.asarray(frame))
                r.close()
                w.close()
                print(f"video -> {dst}")
            except Exception as exc:
                print(f"video copy skip: {exc}", file=sys.stderr)

    return 0 if m["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
