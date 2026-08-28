#!/usr/bin/env python3
"""Privileged post-run check: did Phase B truly contact rim before spiral?"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag", type=Path, default=REPO / "outputs/diagnose/ep01_diag.json")
    parser.add_argument("--summary", type=Path, default=REPO / "outputs/smoke/ep01_summary.json")
    args = parser.parse_args()

    if not args.diag.is_file():
        print(f"[verify] missing {args.diag}", file=sys.stderr)
        return 1

    d = json.loads(args.diag.read_text(encoding="utf-8"))
    traj = d.get("traj", [])
    b = d.get("b_start", {})
    surface = next((r for r in traj if r.get("reason") == "surface_contact"), None)
    hole = next((r for r in traj if r.get("reason") == "hole_detected"), None)

    print("[verify] B-start privileged:")
    print(
        f"  tip={b.get('tip_dist_mm', float('nan')):.1f}mm "
        f"along={b.get('along_mm', float('nan')):.1f}mm "
        f"lat={b.get('lat_mm', float('nan')):.1f}mm"
    )
    if d.get("approach_noise"):
        an = d["approach_noise"]
        if "b_lat_mm" in an:
            print(f"  after A: lat={an['b_lat_mm']:.1f}mm along={an.get('b_along_mm', float('nan')):.1f}mm")
        if "pose_error_lat_m" in an:
            print(f"  pose_error={an['pose_error_lat_m']*1000:.1f}mm alpha={an.get('pose_error_alpha', 0):.2f}")

    ok = True
    if surface is None:
        print("[verify] FAIL: no surface_contact step")
        ok = False
    else:
        along = float(surface["along_mm"])
        lat = float(surface["lat_mm"])
        tip = float(surface["tip_dist_mm"])
        print(
            f"[verify] surface_contact s={surface['step']} "
            f"along={along:.2f}mm lat={lat:.1f}mm tip={tip:.1f}mm"
        )
        if not (0.0 <= along <= 6.0):
            print(f"[verify] FAIL: along={along:.1f}mm not at rim plane [0,6]mm")
            ok = False
        if lat < 4.0:
            print(f"[verify] WARN: lat={lat:.1f}mm small — may be over hole not rim")
        if lat >= 4.0:
            print(f"[verify] OK: lateral offset {lat:.1f}mm — rim contact plausible")

    spiral = [r for r in traj if r.get("reason") == "searching"]
    if spiral:
        al = [float(r["along_mm"]) for r in spiral]
        print(
            f"[verify] spiral n={len(spiral)} along range [{min(al):.1f}, {max(al):.1f}]mm"
        )
        if max(al) - min(al) > 8.0:
            print("[verify] WARN: spiral axial drift >8mm (possible air spiral / plunge)")

    if hole:
        print(
            f"[verify] hole_detected s={hole['step']} along={hole['along_mm']:.1f}mm "
            f"insert_ok={hole.get('insert_ok')}"
        )

    if args.summary.is_file():
        s = json.loads(args.summary.read_text(encoding="utf-8"))
        print(
            f"[verify] smoke insert_ok={s.get('insert_ok')} video={s.get('ego_video', 'n/a')}"
        )

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
