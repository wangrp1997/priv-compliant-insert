#!/usr/bin/env python3
"""L3 parallel ep08 evaluation for tip-tracking scheme comparison."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = os.environ.get(
    "DEXJOCo_PYTHON", "/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python"
)
EP = 8
OUT_ROOT = ROOT / "outputs/scheme_l3/ep08"
HDD = Path("/mnt/hdd/dexjoco/outputs/pk_mouth_view/scheme_l3")

SCHEMES: list[dict] = [
    {
        "id": "S0_baseline_surface_lock",
        "name": "baseline_surface_priv",
        "config": "configs/scheme_l3/S0_baseline_surface_lock.yaml",
        "ref_repo": "ConnTact",
        "claim": "当前 surface_lock+force_seat（对照）",
    },
    {
        "id": "S1_tip_wrist_ff",
        "name": "tip_wrist_ff",
        "config": "configs/scheme_l3/S1_tip_wrist_ff.yaml",
        "ref_repo": "ConnTact+tip_servo",
        "claim": "B: tip参考+冻结offset反算腕（wrist_ff）",
    },
    {
        "id": "S2_tip_track_live",
        "name": "tip_track_live",
        "config": "configs/scheme_l3/S2_tip_track_live.yaml",
        "ref_repo": "ConnTact",
        "claim": "B': tip+offset平面跟踪（无surface_lock）",
    },
    {
        "id": "S3_conntact_spiral",
        "name": "conntact_spiral",
        "config": "configs/scheme_l3/S3_conntact_spiral.yaml",
        "ref_repo": "ConnTact",
        "claim": "A: ConnTact式live spiral（非planned FF）",
    },
    {
        "id": "S4_mouth_wiggle",
        "name": "mouth_wiggle",
        "config": "configs/scheme_l3/S4_mouth_wiggle.yaml",
        "ref_repo": "pci",
        "claim": "mouth wiggle 近口搜孔",
    },
    {
        "id": "S5_wrist_ff_mouth",
        "name": "wrist_ff_mouth",
        "config": "configs/scheme_l3/S5_wrist_ff_mouth.yaml",
        "ref_repo": "ConnTact+tip_servo",
        "claim": "B+A: wrist_ff 后进 mouth wiggle",
        "paper_tier": "ablation",
    },
    {
        "id": "S6_priv_coupling",
        "name": "priv_coupling",
        "config": "configs/scheme_l3/S6_priv_coupling.yaml",
        "ref_repo": "Tactile-Estimator-Controller+bgf",
        "claim": "C: 在线 grasp coupling + tip 反算腕（论文主方法）",
        "paper_tier": "proposed",
    },
]


def metrics(summary_path: Path) -> dict:
    s = json.loads(summary_path.read_text())
    sm = s.get("surface_meta") or {}
    sp = [r for r in sm.get("force_trace", []) if r.get("phase") == "planned_spiral"]
    mw = [r for r in sm.get("force_trace", []) if r.get("phase") == "mouth_wiggle"]
    phases = sp or mw or sm.get("force_trace", [])
    lat_vals = [float(r.get("lat_mm", 999)) for r in phases] if phases else [999.0]
    resid_vals = [float(r.get("resid_r", 0)) for r in phases] if phases else [0.0]
    import numpy as np

    resid = np.array(resid_vals) if resid_vals else np.array([0.0])
    grace = min(120, max(0, len(resid) - 1))
    resid_tail = resid[grace:] if len(resid) > grace else resid
    return {
        "success": bool(s.get("success")),
        "spiral_reason": sm.get("spiral_reason") or sm.get("mouth_wiggle_reason"),
        "lat_min_mm": float(min(lat_vals)),
        "final_lat_mm": float(s.get("final_lat_m", 0)) * 1000.0,
        "final_along_mm": float(s.get("final_along_m", 0)) * 1000.0,
        "resid_peak_n": float(resid.max()) if resid.size else 0.0,
        "resid_p90_n": float(np.percentile(resid_tail, 90)) if resid_tail.size else 0.0,
        "spiral_frames": int(sm.get("spiral_frames") or len(sp)),
        "tip_xy_peak_mm": float(sm.get("spiral_tip_xy_peak_mm") or sm.get("mouth_tip_xy_peak_mm") or 0),
        "grasp_slip_mm": float(sm.get("grasp_slip_peak_m") or 0) * 1000.0,
        "priv_enter": bool(sm.get("priv_planned_hole_entered")),
        "mouth_ok": bool(sm.get("mouth_wiggle_ok")),
        "gate_ok": bool((sm.get("priv_tip_spiral_gate") or {}).get("ok")),
    }


def score(m: dict) -> float:
    s = 0.0
    if m["lat_min_mm"] < 8.0:
        s += 40
    elif m["lat_min_mm"] < 14.0:
        s += 20
    elif m["lat_min_mm"] < 20.0:
        s += 5
    if m["resid_peak_n"] < 0.12:
        s += 25
    elif m["resid_peak_n"] < 0.5:
        s += 10
    if m["grasp_slip_mm"] < 30:
        s += 15
    if m["priv_enter"] or m["mouth_ok"]:
        s += 20
    if m["spiral_reason"] in ("priv_overforce", "grasp_slip", "spiral_tilt_or_tray"):
        s -= 25
    return s


def run_one(scheme: dict, rerun: bool) -> dict:
    name = scheme["name"]
    out_dir = OUT_ROOT / name
    summary = out_dir / f"ep{EP:02d}_summary.json"
    log = out_dir / "run.log"
    out_dir.mkdir(parents=True, exist_ok=True)

    if rerun or not summary.exists():
        env = os.environ.copy()
        env.setdefault(
            "PYTHONPATH",
            f"src:{ROOT}:/home/wangrenpeng/dexjoco:/home/wangrenpeng/dexjoco/dexjoco:"
            f"/home/wangrenpeng/dexjoco/embodied_grasp_insertion:/home/wangrenpeng/reach_insert_rl",
        )
        env["MUJOCO_GL"] = "egl"
        cmd = [
            PY,
            "-u",
            str(ROOT / "scripts/smoke_ep0.py"),
            "--episodes",
            str(EP),
            "--config",
            str(ROOT / scheme["config"]),
            "--out-dir",
            str(out_dir),
        ]
        t0 = time.time()
        with log.open("w") as lf:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0
        if proc.returncode != 0 and not summary.exists():
            return {
                **scheme,
                "status": "fail_run",
                "elapsed_s": elapsed,
                "log": str(log),
            }

    if not summary.exists():
        return {**scheme, "status": "no_summary"}

    m = metrics(summary)
    m["score"] = score(m)
    m["status"] = "ok"
    # copy video
    vid = out_dir / f"ep{EP:02d}_ego.mp4"
    if vid.exists():
        dst_dir = HDD / name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "ego.mp4"
        try:
            if not dst.exists() or vid.stat().st_mtime > dst.stat().st_mtime:
                dst.write_bytes(vid.read_bytes())
        except OSError:
            pass
    return {**scheme, **m}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true", help="force re-run smokes")
    ap.add_argument("--jobs", type=int, default=3, help="parallel workers")
    ap.add_argument("--only", nargs="*", help="scheme ids to run")
    args = ap.parse_args()

    schemes = SCHEMES
    if args.only:
        schemes = [s for s in SCHEMES if s["id"] in args.only]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(run_one, s, args.rerun): s for s in schemes}
        for fut in as_completed(futs):
            results.append(fut.result())

    results.sort(key=lambda r: -float(r.get("score", -1)))
    out_json = OUT_ROOT / "l3_scheme_eval.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\nWrote {out_json}\n")
    print(f"{'ID':<26} {'score':>5} {'lat_min':>7} {'r_pk':>5} {'slip':>5} reason")
    print("-" * 78)
    for r in results:
        if r.get("status") != "ok":
            print(f"{r['id']:<26}  FAIL  {r.get('status')}")
            continue
        print(
            f"{r['id']:<26} {r['score']:5.0f} {r['lat_min_mm']:7.1f} "
            f"{r['resid_peak_n']:5.2f} {r['grasp_slip_mm']:5.0f} {r.get('spiral_reason')}"
        )

    best = next((r for r in results if r.get("status") == "ok"), None)
    if best:
        print(f"\n当前最优 L3: {best['id']} ({best['claim']}) score={best['score']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
