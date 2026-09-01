#!/usr/bin/env python3
"""Batch ep0-9 with P1 (best mouth) config; report hole-entry success rate."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = "/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python"
CFG = REPO / "outputs" / "pk_mouth_sweep_cfgs" / "P1_mouth_peg.yaml"
OUT = REPO / "outputs" / "pk_mouth_P1_ep10"
HDD = Path("/mnt/hdd/dexjoco/outputs/pk_mouth_view/P1_ep10")

# Privileged "进孔": tip/along clearly sunk below surface (~100mm) into mouth/bore.
ALONG_ENTER_M = 0.090  # P1 success ~69mm; fail-on-surface ~100–108mm
TIP_ENTER_M = 0.090


def _run_one(ep: int) -> dict:
    out_dir = OUT / f"ep{ep:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    cmd = [
        PY,
        str(REPO / "scripts" / "smoke_ep0.py"),
        "--episode",
        str(ep),
        "--config",
        str(CFG),
        "--out-dir",
        str(out_dir),
        "--video",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    summary_path = out_dir / f"ep{ep:02d}_summary.json"
    row: dict = {"ep": ep, "ok_proc": proc.returncode == 0}
    if not summary_path.is_file():
        row["error"] = "no summary"
        return row
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    sm = s.get("surface_meta") or {}
    g = sm.get("priv_tip_spiral_gate") or {}
    along = float(s.get("final_along_m") or sm.get("align_along_mm", 0) / 1000.0 or 0.0)
    tip = float(s.get("final_tip_dist_m") or 0.0)
    lat = float(s.get("final_lat_m") or 0.0)
    reason = str(sm.get("spiral_reason") or "")
    enter_ok = bool(
        tip <= TIP_ENTER_M
        or along <= ALONG_ENTER_M
        or (reason == "mouth_insert" and along <= 0.095)
    )
    row.update(
        {
            "gate": bool(g.get("ok")),
            "enter_ok": enter_ok,
            "success": bool(s.get("success")),
            "final_lat_mm": round(lat * 1000, 2),
            "final_along_mm": round(along * 1000, 2),
            "final_tip_mm": round(tip * 1000, 2),
            "spiral_reason": reason,
            "mouth_wiggle_ok": sm.get("mouth_wiggle_ok"),
            "mouth_frames": sm.get("mouth_wiggle_frames"),
            "grasp_slip_mm": round(float(sm.get("grasp_slip_peak_m") or 0) * 1000, 1),
            "video": str(out_dir / f"ep{ep:02d}_ego.mp4"),
        }
    )
    # OpenPI-style ego copy for enter_ok episodes (Cursor-friendly Lavf61.1).
    ego_src = out_dir / f"ep{ep:02d}_ego.mp4"
    if ego_src.is_file() and enter_ok:
        hdd_ep = HDD / f"episode_{ep:02d}"
        hdd_ep.mkdir(parents=True, exist_ok=True)
        ego_dst = hdd_ep / "ego.mp4"
        try:
            import imageio
            import imageio.v2 as iio
            import numpy as np

            r = iio.get_reader(str(ego_src))
            frames = [np.asarray(x) for x in r]
            r.close()
            w = imageio.get_writer(str(ego_dst), fps=30)
            for fr in frames:
                w.append_data(fr)
            w.close()
            row["hdd_ego"] = str(ego_dst)
        except Exception as e:
            row["hdd_ego_error"] = str(e)
    return row


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    HDD.mkdir(parents=True, exist_ok=True)
    if not CFG.is_file():
        print(f"missing cfg {CFG}", flush=True)
        return 1
    eps = list(range(10))
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(_run_one, ep): ep for ep in eps}
        for fut in as_completed(futs):
            ep = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"ep": ep, "error": str(e), "enter_ok": False}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    rows.sort(key=lambda r: int(r.get("ep", -1)))
    n = len(rows)
    n_enter = sum(1 for r in rows if r.get("enter_ok"))
    n_gate = sum(1 for r in rows if r.get("gate"))
    summary = {
        "algo": "P1_mouth_peg",
        "cfg": str(CFG),
        "n": n,
        "enter_ok_n": n_enter,
        "enter_rate": round(n_enter / max(n, 1), 3),
        "gate_ok_n": n_gate,
        "gate_rate": round(n_gate / max(n, 1), 3),
        "enter_rule": f"tip<={TIP_ENTER_M} or along<={ALONG_ENTER_M} or (mouth_insert and along<=0.095)",
        "rows": rows,
    }
    out_json = OUT / "result.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"ENTER {n_enter}/{n} = {100.0 * n_enter / max(n, 1):.1f}%  "
        f"GATE {n_gate}/{n} = {100.0 * n_gate / max(n, 1):.1f}%  wrote {out_json}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
