#!/usr/bin/env python3
"""Privileged spiral diagnostics from ep summary force_trace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_spiral(summary_path: Path) -> tuple[dict, list[dict]]:
    s = json.loads(summary_path.read_text())
    sm = s.get("surface_meta") or s
    sp = [r for r in sm.get("force_trace") or [] if r.get("phase") == "planned_spiral"]
    return sm, sp


def diagnose(sm: dict, sp: list[dict]) -> dict:
    if not sp:
        return {"status": "no_spiral"}
    priv_p = np.array([float(r.get("priv_planar_mm", 999)) for r in sp])
    lat = np.array([float(r.get("lat_mm", 999)) for r in sp])
    along = np.array([float(r.get("along_mm", 0)) for r in sp])
    tip_err = np.array([float(r.get("tip_spiral_err_mm", 0)) for r in sp])
    if tip_err.max() <= 0:
        tip_err = np.abs(
            np.array([float(r.get("spiral_r_mm", 0)) for r in sp])
            - np.array([float(r.get("tip_xy_mm", 0)) for r in sp])
        )
    resid = np.array([float(r.get("resid_r", 0)) for r in sp])
    bl_tip = np.array([float(r.get("bl_tip_err_mm", 0)) for r in sp])
    return {
        "n_spiral": len(sp),
        "priv_planar_min_mm": float(priv_p.min()),
        "lat_min_mm": float(lat.min()),
        "lat_priv_gap_mm": float(lat.min() - priv_p.min()),
        "along_at_start_mm": float(along[0]),
        "along_at_end_mm": float(along[-1]),
        "along_drift_mm": float(along[-1] - along[0]),
        "tip_spiral_err_mean_mm": float(tip_err.mean()),
        "tip_spiral_err_max_mm": float(tip_err.max()),
        "resid_mean_n": float(resid.mean()),
        "resid_p90_n": float(np.percentile(resid, 90)),
        "bl_tip_err_mean_mm": float(bl_tip.mean()) if bl_tip.any() else None,
        "priv_enter": bool(max(int(r.get("priv_hole_entered", 0)) for r in sp)),
        "priv_mouth": bool(max(int(r.get("priv_at_mouth", 0)) for r in sp)),
        "grasp_slip_mm": float(sm.get("grasp_slip_peak_m") or 0) * 1000.0,
        "spiral_reason": sm.get("spiral_reason"),
        "bl_path": sp[-1].get("bl_path") if sp else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path, nargs="+")
    args = ap.parse_args()
    for p in args.summary:
        sm, sp = load_spiral(p)
        d = diagnose(sm, sp)
        print(f"\n=== {p.parent.name} / {p.parent.parent.name} ===")
        for k, v in d.items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
