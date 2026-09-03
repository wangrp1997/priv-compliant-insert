#!/usr/bin/env python3
"""Parallel baseline matrix eval: schemes × episodes → JSON + markdown table."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = os.environ.get(
    "DEXJOCo_PYTHON", "/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python"
)

# Paper baselines only (skip mouth ablations for main table)
PAPER_SCHEMES: list[dict] = [
    {
        "id": "S0",
        "tier": "neg",
        "name": "surface_lock",
        "config": "configs/scheme_l3/S0_baseline_surface_lock.yaml",
        "label": "Surface lock + force seat",
    },
    {
        "id": "S3",
        "tier": "A",
        "name": "conntact_wrist",
        "config": "configs/scheme_l3/S3_conntact_spiral.yaml",
        "label": "ConnTact rigid-EE spiral",
    },
    {
        "id": "S1",
        "tier": "B",
        "name": "tip_servo_ff",
        "config": "configs/scheme_l3/S1_tip_wrist_ff.yaml",
        "label": "Frozen tip–wrist offset",
    },
    {
        "id": "S2",
        "tier": "B'",
        "name": "tip_track_live",
        "config": "configs/scheme_l3/S2_tip_track_live.yaml",
        "label": "Live offset tracking",
    },
    {
        "id": "S6",
        "tier": "C-α",
        "name": "priv_coupling",
        "config": "configs/scheme_l3/S6_priv_coupling.yaml",
        "label": "Scalar α coupling (ablation)",
    },
    {
        "id": "S8",
        "tier": "C",
        "name": "planar_C",
        "config": "configs/scheme_l3/S8_planar_C.yaml",
        "label": "Planar C + along/surface invariants",
    },
    {
        "id": "S9",
        "tier": "C+",
        "name": "ekf_qp",
        "config": "configs/scheme_l3/S9_ekf_qp.yaml",
        "label": "Slip-aware EKF + planar QP (V2)",
    },
]


def metrics(summary_path: Path) -> dict:
    s = json.loads(summary_path.read_text())
    sm = s.get("surface_meta") or {}
    sp = [r for r in sm.get("force_trace", []) if r.get("phase") == "planned_spiral"]
    lat_vals = [float(r.get("lat_mm", 999)) for r in sp] if sp else [999.0]
    resid_vals = [float(r.get("resid_r", 0)) for r in sp] if sp else [0.0]
    import numpy as np

    resid = np.array(resid_vals) if resid_vals else np.array([0.0])
    grace = min(120, max(0, len(resid) - 1))
    resid_tail = resid[grace:] if len(resid) > grace else resid
    priv_p = (
        [float(r.get("priv_planar_mm", 999)) for r in sp] if sp else [999.0]
    )
    f_scales = [float(r.get("bl_force_scale", 1.0)) for r in sp] if sp else [1.0]
    f_des_vals = [float(r.get("f_des", 0.025)) for r in sp] if sp else [0.025]
    fd = float(np.mean(f_des_vals)) if f_des_vals else 0.025
    return {
        "episode": int(s.get("episode", -1)),
        "success": bool(s.get("success")),
        "spiral_reason": sm.get("spiral_reason") or sm.get("mouth_wiggle_reason"),
        "lat_min_mm": float(min(lat_vals)),
        "priv_planar_min_mm": float(min(priv_p)),
        "final_lat_mm": float(s.get("final_lat_m", 0)) * 1000.0,
        "final_along_mm": float(s.get("final_along_m", 0)) * 1000.0,
        "resid_peak_n": float(resid.max()) if resid.size else 0.0,
        "resid_mean_n": float(resid.mean()) if resid.size else 0.0,
        "resid_fdes_ratio": float(resid.mean() / fd) if resid.size and fd > 0 else 0.0,
        "resid_p90_n": float(np.percentile(resid_tail, 90)) if resid_tail.size else 0.0,
        "grasp_slip_mm": float(sm.get("grasp_slip_peak_m") or 0) * 1000.0,
        "force_scale_mean": float(np.mean(f_scales)) if f_scales else 1.0,
        "force_gated_frac": float(np.mean([1.0 if x < 0.99 else 0.0 for x in f_scales]))
        if f_scales
        else 0.0,
        "priv_enter": bool(sm.get("priv_planned_hole_entered")),
        "mouth_ok": bool(sm.get("mouth_wiggle_ok")),
    }


def run_job(scheme: dict, ep: int, rerun: bool) -> dict:
    name = scheme["name"]
    out_dir = ROOT / "outputs/scheme_l3" / f"ep{ep:02d}" / name
    summary = out_dir / f"ep{ep:02d}_summary.json"
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
            str(ep),
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
                "episode": ep,
                "status": "fail_run",
                "elapsed_s": elapsed,
            }

    if not summary.exists():
        return {**scheme, "episode": ep, "status": "no_summary"}

    m = metrics(summary)
    return {**scheme, **m, "status": "ok"}


def aggregate(rows: list[dict]) -> list[dict]:
    by_scheme: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        by_scheme.setdefault(r["id"], []).append(r)

    out = []
    for s in PAPER_SCHEMES:
        rs = by_scheme.get(s["id"], [])
        if not rs:
            out.append({**s, "n": 0})
            continue

        def mean(k: str) -> float:
            return float(statistics.mean(float(x[k]) for x in rs))

        def median_lat() -> float:
            vals = sorted(float(x["lat_min_mm"]) for x in rs)
            vals = [v for v in vals if v < 900.0] or vals
            return float(statistics.median(vals))

        def median_priv() -> float:
            vals = sorted(float(x["priv_planar_min_mm"]) for x in rs)
            vals = [v for v in vals if v < 900.0] or vals
            return float(statistics.median(vals))

        enters = sum(1 for x in rs if x.get("priv_enter") or x.get("mouth_ok"))
        out.append(
            {
                **s,
                "n": len(rs),
                "lat_min_mean": mean("lat_min_mm"),
                "lat_min_median": median_lat(),
                "priv_planar_median": median_priv(),
                "along_mean": mean("final_along_mm"),
                "resid_fdes_mean": mean("resid_fdes_ratio"),
                "force_scale_mean": mean("force_scale_mean"),
                "slip_mean": mean("grasp_slip_mm"),
                "enter_rate": enters / len(rs),
                "episodes": sorted(int(x["episode"]) for x in rs),
            }
        )
    return out


def markdown_table(agg: list[dict], detail_rows: list[dict]) -> str:
    lines = [
        "# Baseline Comparison Table (auto-generated)",
        "",
        "## Aggregate (handoff grasp, DexJoCo PCI)",
        "",
        "| Tier | Method | lat_min ↓ | priv_planar ↓ | slip ↓ | γ_mean | resid/f_des | enter | n |",
        "|------|--------|-----------|---------------|--------|--------|-------------|-------|---|",
    ]
    best_lat = min((a["lat_min_median"] for a in agg if a.get("n")), default=999.0)
    for a in agg:
        if not a.get("n"):
            lines.append(f"| {a['tier']} | {a['label']} | — | — | — | — | — | — | 0 |")
            continue
        star = " **" if a["lat_min_median"] <= best_lat + 0.01 else ""
        lines.append(
            f"| {a['tier']} | {a['label']} | "
            f"med {a['lat_min_median']:.1f}{star} | "
            f"{a['priv_planar_median']:.1f} | "
            f"{a['slip_mean']:.0f} | "
            f"{a['force_scale_mean']:.2f} | "
            f"{a['resid_fdes_mean']:.1f} | "
            f"{100*a['enter_rate']:.0f}% | {a['n']} |"
        )
    lines.extend(["", "## Per-episode", "", "| ep | scheme | lat | priv_p | slip | γ | resid/f_d | enter | reason |"])
    lines.append("|----|--------|-----|--------|------|---|-----------|-------|--------|")
    for r in sorted(detail_rows, key=lambda x: (x.get("episode", 0), x.get("id", ""))):
        if r.get("status") != "ok":
            lines.append(f"| {r.get('episode','?')} | {r.get('id','?')} | FAIL | — | — | — | — | — | {r.get('status')} |")
            continue
        ent = "Y" if (r.get("priv_enter") or r.get("mouth_ok")) else "N"
        lines.append(
            f"| {r['episode']:02d} | {r['id']} | {r['lat_min_mm']:.1f} | "
            f"{r.get('priv_planar_min_mm',0):.1f} | {r['grasp_slip_mm']:.0f} | "
            f"{r.get('force_scale_mean',1):.2f} | {r.get('resid_fdes_ratio',0):.1f} | {ent} | {r.get('spiral_reason','')} |"
        )
    lines.append("")
    lines.append(f"_Generated by `scripts/parallel_baseline_matrix_eval.py`_")
    return "\n".join(lines)


def collect_from_disk(schemes: list[dict], eps: list[int]) -> list[dict]:
    rows: list[dict] = []
    for s in schemes:
        for ep in eps:
            out_dir = ROOT / "outputs/scheme_l3" / f"ep{ep:02d}" / s["name"]
            summary = out_dir / f"ep{ep:02d}_summary.json"
            if not summary.exists():
                rows.append({**s, "episode": ep, "status": "no_summary"})
                continue
            m = metrics(summary)
            rows.append({**s, **m, "status": "ok"})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="8", help="comma list or range 1-10")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only", nargs="*", help="scheme ids S0 S1 ...")
    ap.add_argument(
        "--md-out",
        default="docs/BASELINE_TABLE.md",
        help="markdown table output",
    )
    args = ap.parse_args()

    if "-" in args.episodes and "," not in args.episodes:
        a, b = args.episodes.split("-", 1)
        eps = list(range(int(a), int(b) + 1))
    else:
        eps = [int(x.strip()) for x in args.episodes.split(",") if x.strip()]

    schemes = PAPER_SCHEMES
    if args.only:
        schemes = [s for s in PAPER_SCHEMES if s["id"] in args.only]

    if args.rerun:
        jobs = [(s, ep) for s in schemes for ep in eps]
        with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            futs = {ex.submit(run_job, s, ep, True): (s, ep) for s, ep in jobs}
            for fut in as_completed(futs):
                r = fut.result()
                if r.get("status") == "ok":
                    print(
                        f"OK ep{r['episode']:02d} {r['id']} lat={r['lat_min_mm']:.1f} "
                        f"along={r['final_along_mm']:.1f}",
                        flush=True,
                    )
                else:
                    print(
                        f"FAIL ep{r.get('episode')} {r.get('id')} {r.get('status')}",
                        flush=True,
                    )

    results = collect_from_disk(schemes, eps)

    agg = aggregate(results)
    out_root = ROOT / "outputs/scheme_l3"
    out_json = out_root / "baseline_matrix.json"
    out_json.write_text(
        json.dumps({"aggregate": agg, "detail": results}, indent=2, ensure_ascii=False)
    )

    md = markdown_table(agg, results)
    md_path = ROOT / args.md_out
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")

    print(f"\nWrote {out_json}\nWrote {md_path}\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
