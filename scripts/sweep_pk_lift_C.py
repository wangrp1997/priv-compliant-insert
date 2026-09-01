#!/usr/bin/env python3
"""Scheme C sweep: left yield during tip-lift (privileged_diagnostic).

Sweeps surface_lift_left_share / admit / extra unload / f_des / lift_m on ep=3.
Writes outputs/pk_lift_C_result.json. No video.

Gates:
  tip_up>=1.5mm OR along_rise>=1.5mm
  spiral tip_xy_peak>=6mm
  spiral mean |resid_r| in [0.05, 0.45]N
  right grasp scale never < 1
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
PY = "/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python"
BASE = REPO / "configs" / "pk_lift_C.yaml"
OUT_ROOT = REPO / "outputs" / "pk_lift_C"
RESULT = REPO / "outputs" / "pk_lift_C_result.json"
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

# Compact grid: left yield share/admit/extra + right f_des + lift cmd.
GRID = [
    # share, admit, extra_mm, f_des, lift_mm
    (0.5, 0.0, 0.0, 0.03, 6.0),
    (0.7, 0.0, 0.2, 0.03, 8.0),
    (0.9, 0.0, 0.3, 0.03, 10.0),
    (0.5, 0.35, 0.0, 0.05, 6.0),
    (0.7, 0.35, 0.2, 0.05, 8.0),
    (0.9, 0.35, 0.3, 0.05, 10.0),
    (1.0, 0.5, 0.25, 0.05, 10.0),
    (0.85, 0.45, 0.35, 0.06, 8.0),
    (0.7, 0.6, 0.2, 0.04, 10.0),
    (1.0, 0.25, 0.4, 0.03, 10.0),
    (0.9, 0.55, 0.15, 0.08, 8.0),
    (0.6, 0.4, 0.25, 0.05, 12.0),
]


def _env() -> dict:
    return {
        **os.environ,
        "MUJOCO_GL": "egl",
        "PYTHONPATH": PYTHONPATH,
    }


def _gate_from_meta(sm: dict, cfg: dict) -> dict:
    """Prefer runner-embedded gate; else recompute without importing sim_runner."""
    embedded = sm.get("priv_tip_spiral_gate")
    if isinstance(embedded, dict) and "ok" in embedded:
        g = dict(embedded)
        # Ensure numeric fields exist for scoring / result.
        g.setdefault("tip_up_m", sm.get("surface_tip_up_m") or 0.0)
        g.setdefault("along_rise_m", sm.get("surface_tip_along_rise_m") or 0.0)
        g.setdefault("tip_xy_peak_mm", sm.get("spiral_tip_xy_peak_mm") or 0.0)
        if "spiral_mean_resid_n" not in g or g["spiral_mean_resid_n"] is None:
            tr = sm.get("force_trace") or []
            sp = [x for x in tr if str(x.get("phase")) == "spiral"]
            if sp:
                g["spiral_mean_resid_n"] = float(
                    sum(float(x.get("resid_r", 0.0)) for x in sp) / len(sp)
                )
            else:
                g["spiral_mean_resid_n"] = float("nan")
        return g

    a = cfg.get("approach", {})
    lift_need = float(a.get("priv_gate_tip_lift_m", 0.0015))
    xy_need = float(a.get("priv_gate_tip_xy_mm", 6.0))
    f_lo = float(a.get("priv_gate_force_lo_n", 0.05))
    f_hi = float(a.get("priv_gate_force_hi_n", 0.45))
    tip_up = float(sm.get("surface_tip_up_m") or 0.0)
    along_rise = float(sm.get("surface_tip_along_rise_m") or 0.0)
    tip_xy = float(sm.get("spiral_tip_xy_peak_mm") or 0.0)
    tr = sm.get("force_trace") or []
    sp = [x for x in tr if str(x.get("phase")) == "spiral"]
    if sp:
        mean_r = float(sum(float(x.get("resid_r", 0.0)) for x in sp) / len(sp))
    else:
        mean_r = float("nan")
    lift_ok = (tip_up >= lift_need) or (along_rise >= lift_need)
    spiral_ok = tip_xy >= xy_need
    force_ok = bool(sp) and (f_lo <= mean_r <= f_hi)
    ok = bool(lift_ok and spiral_ok and force_ok)
    reasons: list[str] = []
    if not lift_ok:
        reasons.append(
            f"tip_lift_fail up={tip_up*1e3:.2f}mm alongΔ={along_rise*1e3:.2f}mm "
            f"need>={lift_need*1e3:.1f}mm"
        )
    if not spiral_ok:
        reasons.append(f"tip_spiral_fail xy={tip_xy:.2f}mm need>={xy_need:.1f}mm")
    if not force_ok:
        reasons.append(
            f"surface_force_fail mean|r|={mean_r:.3f}N need∈[{f_lo:.2f},{f_hi:.2f}]N"
        )
    return {
        "ok": ok,
        "lift_ok": lift_ok,
        "spiral_ok": spiral_ok,
        "force_ok": force_ok,
        "tip_up_m": tip_up,
        "along_rise_m": along_rise,
        "tip_xy_peak_mm": tip_xy,
        "spiral_mean_resid_n": mean_r,
        "reasons": reasons,
    }


def _score(g: dict) -> float:
    """Higher is better; gate pass wins; else partial credit."""
    tip_up = float(g.get("tip_up_m") or 0.0)
    along = float(g.get("along_rise_m") or 0.0)
    xy = float(g.get("tip_xy_peak_mm") or 0.0)
    mean_r = float(g.get("spiral_mean_resid_n") or float("nan"))
    lift = max(tip_up, along) * 1000.0
    force_pen = 0.0
    if mean_r == mean_r:
        if mean_r < 0.05:
            force_pen = (0.05 - mean_r) * 20.0
        elif mean_r > 0.45:
            force_pen = (mean_r - 0.45) * 20.0
    else:
        force_pen = 5.0
    base = lift + 0.5 * min(xy, 12.0) - force_pen
    return (1000.0 + base) if g.get("ok") else base


def _overrides(share, admit, extra_mm, f_des, lift_mm) -> dict:
    return {
        "surface_lift_left_share": float(share),
        "surface_lift_left_admit_scale": float(admit),
        "surface_lift_left_extra_unload_m": float(extra_mm) * 1e-3,
        "surface_const_force_des_n": float(f_des),
        "surface_light_lift_m": float(lift_mm) * 1e-3,
        "surface_tip_lift_m": 0.0015,
        "surface_light_lift_frames": max(40, int(lift_mm / 0.15)),
        "surface_lift_right_grasp_scale": 1.0,
        "surface_tip_servo_lift_enable": False,
        "surface_right_grasp_scale": 1.08,
    }


def _metrics(g: dict) -> dict:
    mean_r = g.get("spiral_mean_resid_n")
    if mean_r is not None and isinstance(mean_r, float) and math.isnan(mean_r):
        mean_r = None
    return {
        "tip_up_mm": float(g.get("tip_up_m") or 0.0) * 1e3,
        "along_rise_mm": float(g.get("along_rise_m") or 0.0) * 1e3,
        "tip_xy_peak_mm": float(g.get("tip_xy_peak_mm") or 0.0),
        "spiral_mean_resid_n": mean_r,
    }


def main() -> int:
    with BASE.open(encoding="utf-8") as f:
        base = yaml.safe_load(f)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = OUT_ROOT / "_cfgs"
    tmp.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    best: dict | None = None

    for i, (share, admit, extra_mm, f_des, lift_mm) in enumerate(GRID):
        tag = (
            f"s{share:.2f}_a{admit:.2f}_e{extra_mm:.2f}_"
            f"f{f_des:.2f}_L{lift_mm:.0f}"
        )
        overrides = _overrides(share, admit, extra_mm, f_des, lift_mm)
        cfg = copy.deepcopy(base)
        a = cfg.setdefault("approach", {})
        for k, v in overrides.items():
            a[k] = v
        a["surface_right_grasp_scale"] = max(
            1.0, float(a.get("surface_right_grasp_scale", 1.0))
        )
        cfg.setdefault("compliant", {}).setdefault("search", {})[
            "contact_f_des_n"
        ] = float(f_des)

        cfg_path = tmp / f"{tag}.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, sort_keys=False)

        out_dir = OUT_ROOT / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / f"ep{EP:02d}_summary.json"

        # Reuse existing PASS/complete summary if present (resume after crash).
        need_run = True
        if summary_path.is_file():
            try:
                with summary_path.open(encoding="utf-8") as f:
                    preview = json.load(f)
                sm0 = preview.get("surface_meta") or {}
                g0 = _gate_from_meta(sm0, cfg)
                if g0.get("ok"):
                    need_run = False
                    print(
                        f"\n=== [{i+1}/{len(GRID)}] {tag} === (reuse PASS summary)",
                        flush=True,
                    )
            except Exception:
                need_run = True

        if need_run:
            print(f"\n=== [{i+1}/{len(GRID)}] {tag} ===", flush=True)
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
            subprocess.run(cmd, cwd=REPO, env=_env(), check=False)

        row: dict = {
            "tag": tag,
            "overrides": overrides,
            "params": {
                "surface_lift_left_share": share,
                "surface_lift_left_admit_scale": admit,
                "surface_lift_left_extra_unload_m": extra_mm * 1e-3,
                "surface_const_force_des_n": f_des,
                "surface_light_lift_m": lift_mm * 1e-3,
                "surface_lift_right_grasp_scale": 1.0,
            },
            "ok": False,
            "error": None,
        }
        if not summary_path.is_file():
            row["error"] = "missing_summary"
            rows.append(row)
            continue
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
        sm = summary.get("surface_meta") or {}
        g = _gate_from_meta(sm, cfg)
        # Grasp contract: config forbids scale<1; record for audit.
        right_grasp_min = min(
            float(overrides["surface_lift_right_grasp_scale"]),
            float(a.get("surface_right_grasp_scale", 1.0)),
        )
        grasp_ok = right_grasp_min >= 1.0 - 1e-9
        if not grasp_ok:
            g = dict(g)
            g["ok"] = False
            reasons = list(g.get("reasons") or [])
            reasons.append(f"right_grasp_loosen min_scale={right_grasp_min:.3f}")
            g["reasons"] = reasons
        row.update(
            {
                "ok": bool(g.get("ok")),
                "lift_ok": bool(g.get("lift_ok")),
                "spiral_ok": bool(g.get("spiral_ok")),
                "force_ok": bool(g.get("force_ok")),
                "grasp_ok": grasp_ok,
                "tip_up_m": g.get("tip_up_m"),
                "along_rise_m": g.get("along_rise_m"),
                "tip_xy_peak_mm": g.get("tip_xy_peak_mm"),
                "spiral_mean_resid_n": g.get("spiral_mean_resid_n"),
                "metrics": _metrics(g),
                "gates": {
                    "lift_ok": bool(g.get("lift_ok")),
                    "spiral_ok": bool(g.get("spiral_ok")),
                    "force_ok": bool(g.get("force_ok")),
                    "grasp_ok": grasp_ok,
                },
                "reasons": g.get("reasons"),
                "score": _score(g),
                "fail_reason": summary.get("fail_reason"),
                "final_phase": summary.get("final_phase"),
                "summary_path": str(summary_path),
            }
        )
        rows.append(row)
        print(
            f"  gate={row['ok']} lift={row['lift_ok']} spiral={row['spiral_ok']} "
            f"force={row['force_ok']} tip_up={float(row['tip_up_m'] or 0)*1e3:.2f}mm "
            f"along={float(row['along_rise_m'] or 0)*1e3:.2f}mm "
            f"xy={float(row['tip_xy_peak_mm'] or 0):.2f}mm "
            f"mean|r|={row['spiral_mean_resid_n']}",
            flush=True,
        )
        if best is None or float(row["score"]) > float(best.get("score") or -1e9):
            best = row
        if row["ok"]:
            print(f"  PASS — stop early at {tag}", flush=True)
            break

    rows_sorted = sorted(
        rows, key=lambda r: float(r.get("score") or -1e9), reverse=True
    )
    passed = bool(best and best.get("ok"))
    result = {
        "status": "passed" if passed else "failed",
        "experiment_tag": "privileged_diagnostic",
        "scheme": "C_left_yield_lift",
        "path": "stop_after_surface",
        "episode": EP,
        "gate": {
            "tip_up_or_along_rise_m": 0.0015,
            "tip_xy_peak_mm": 6.0,
            "spiral_mean_resid_n": [0.05, 0.45],
            "forbid_right_loosen": True,
        },
        "passed": passed,
        "overrides": (best or {}).get("overrides"),
        "metrics": (best or {}).get("metrics"),
        "gates": (best or {}).get("gates"),
        "best": best,
        "n_tried": len(rows),
        "rows": rows_sorted,
        "fail_reasons": None
        if passed
        else (best or {}).get("reasons") or ["all_trials_failed"],
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {RESULT} status={result['status']}", flush=True)
    if best:
        print(
            f"best tag={best.get('tag')} ok={best.get('ok')} "
            f"metrics={best.get('metrics')}",
            flush=True,
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
