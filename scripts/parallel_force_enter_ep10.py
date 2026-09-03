#!/usr/bin/env python3
"""Parallel ep1-10: success = force hole + tip seat + upright enter."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = os.environ.get(
    "DEXJOCo_PYTHON", "/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python"
)
CFG = ROOT / "configs/scheme_l3/S2_force_insert.yaml"
OUT_ROOT = ROOT / "outputs/scheme_l3/S2_force_enter_ep10"


def _run_one(ep: int, out_root: Path, video: bool, cfg: Path) -> dict:
    out_dir = out_root / f"ep{ep:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    cmd = [
        PY,
        str(ROOT / "scripts/smoke_ep0.py"),
        "--config",
        str(cfg),
        "--episode",
        str(ep),
        "--out-dir",
        str(out_dir),
    ]
    if video:
        cmd.append("--video")
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    summary_path = out_dir / f"ep{ep:02d}_summary.json"
    row: dict = {"ep": ep, "ok_proc": proc.returncode == 0}
    if not summary_path.is_file():
        row["error"] = "no summary"
        row["enter_ok"] = False
        return row
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    sm = s.get("surface_meta") or s.get("approach_noise") or {}
    force_hole = bool(
        sm.get("force_hole_raw")
        or sm.get("force_planned_hole_entered")
        or sm.get("force_hole_entered")
        or s.get("mouth_enter_ok")
        or s.get("mouth_ok")
    )
    # enter_ok from summary success (force + priv audit inside runner).
    enter_ok = bool(s.get("success")) or bool(s.get("enter_ok"))
    priv_audit = bool(
        sm.get("priv_enter_audit")
        if "priv_enter_audit" in sm
        else s.get("priv_enter_audit")
    )
    row.update(
        {
            "enter_ok": enter_ok,
            "success": bool(s.get("success")),
            "force_hole": force_hole,
            "priv_audit": priv_audit,
            "priv_axis_ok": bool(s.get("priv_axis_ok", True)),
            "priv_seat_ok": bool(s.get("priv_seat_ok", True)),
            "insert_ok": bool(s.get("insert_ok")),
            "fail_reason": s.get("fail_reason") or sm.get("spiral_reason"),
            "spiral_reason": sm.get("spiral_reason"),
            "lat_min_mm": float(sm.get("spiral_lat_min_mm") or 999.0),
            "final_lat_mm": float(s.get("final_lat_m") or 0.0) * 1000.0,
            "final_along_mm": float(s.get("final_along_m") or 0.0) * 1000.0,
            "axis_err_deg": float(sm.get("final_axis_err_deg") or 0.0),
            "float_peak_mm": float(sm.get("spiral_tip_float_peak_m") or 0.0) * 1000.0,
            "contact_frac": (
                float(sm["spiral_contact_frac"])
                if sm.get("spiral_contact_frac") is not None
                else -1.0
            ),
            "grasp_slip_mm": float(sm.get("grasp_slip_peak_m") or 0.0) * 1000.0,
            "video": str(out_dir / f"ep{ep:02d}_ego.mp4"),
        }
    )
    print(
        f"[force-enter] ep={ep:02d} enter={int(enter_ok)} force_hole={int(force_hole)} "
        f"audit={int(bool(sm.get('priv_enter_audit')))} "
        f"axis={row['axis_err_deg']:.1f} float={row['float_peak_mm']:.1f}mm "
        f"cfrac={row['contact_frac']:.2f} "
        f"along={row['final_along_mm']:.1f} lat_min={row['lat_min_mm']:.1f} "
        f"reason={row.get('fail_reason') or row.get('spiral_reason')}",
        flush=True,
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="1-10", help="e.g. 1-10 or 1,2,5")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--config", type=Path, default=CFG)
    ap.add_argument("--video", action="store_true", default=True)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    video = bool(args.video) and not bool(args.no_video)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = Path(args.config)

    eps: list[int] = []
    for part in str(args.eps).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            eps.extend(range(int(a), int(b) + 1))
        else:
            eps.append(int(part))
    eps = sorted(set(eps))

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = {ex.submit(_run_one, ep, out_root, video, cfg): ep for ep in eps}
        for fut in as_completed(futs):
            rows.append(fut.result())
    rows.sort(key=lambda r: int(r["ep"]))

    ok = sum(1 for r in rows if r.get("enter_ok"))
    n = len(rows)
    table = {
        "metric": "force_hole_seat_upright_enter",
        "standard": "tip_on_surface_spiral + axis_err<=25deg + force_hole",
        "config": str(cfg),
        "n": n,
        "enter_ok": ok,
        "rate": (ok / n) if n else 0.0,
        "rows": rows,
    }
    out_json = out_root / "summary_enter10.json"
    out_json.write_text(json.dumps(table, indent=2), encoding="utf-8")

    print("\n=== force hole + tip seat + upright enter ===", flush=True)
    for r in rows:
        print(
            f"ep{r['ep']:02d} enter={int(bool(r.get('enter_ok')))} "
            f"force_hole={int(bool(r.get('force_hole')))} "
            f"axis={r.get('axis_err_deg', 0):.1f} "
            f"float={r.get('float_peak_mm', 0):.1f}mm "
            f"along={r.get('final_along_mm', 0):.1f} "
            f"lat_min={r.get('lat_min_mm', 0):.1f} "
            f"reason={r.get('fail_reason') or r.get('spiral_reason')}",
            flush=True,
        )
    print(f"RATE {ok}/{n} = {100.0 * ok / max(n, 1):.1f}%", flush=True)
    print(f"wrote {out_json}", flush=True)
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
