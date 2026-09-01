#!/usr/bin/env python3
"""Scheme A privileged_diagnostic sweep: bleed → firm-grasp axial tip-lift → tip-spiral.

Gates (all required):
  tip_up>=1.5mm OR along_rise>=1.5mm
  spiral tip_xy_peak>=6mm
  spiral mean |resid_r| in [0.05, 0.45]N
No grasp_scale<1. Writes outputs/pk_lift_A_result.json.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
DEXJOCO = Path("/home/wangrenpeng/dexjoco")

for p in (
    str(REPO / "src"),
    str(DEXJOCO),
    str(DEXJOCO / "dexjoco"),
    str(DEXJOCO / "embodied_grasp_insertion"),
    str(DEXJOCO.parent / "reach_insert_rl"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("MUJOCO_GL", "egl")


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _metrics(summary: dict) -> dict:
    # surface fields live under approach_noise / surface_meta / top-level
    buckets = [summary]
    for k in ("approach_noise", "surface_meta", "meta"):
        v = summary.get(k)
        if isinstance(v, dict):
            buckets.append(v)

    def g(key, default=None):
        for b in buckets:
            if key in b and b[key] is not None:
                return b[key]
        return default

    tip_up = float(g("surface_tip_up_m", 0.0) or 0.0)
    along_rise = float(g("surface_tip_along_rise_m", 0.0) or 0.0)
    tip_xy = float(g("spiral_tip_xy_peak_mm", 0.0) or 0.0)
    tr = g("force_trace", []) or []
    sp = [x for x in tr if str(x.get("phase")) == "spiral"]
    if sp:
        rr = np.asarray([float(x.get("resid_r", 0.0)) for x in sp], dtype=np.float64)
        mean_r = float(rr.mean())
        med_r = float(np.median(rr))
        peak_r = float(rr.max())
    else:
        mean_r = float("nan")
        med_r = float("nan")
        peak_r = float("nan")
    lift_ok = (tip_up >= 0.0015) or (along_rise >= 0.0015)
    spiral_ok = tip_xy >= 6.0
    force_ok = bool(sp) and (0.05 <= mean_r <= 0.45)
    ok = bool(lift_ok and spiral_ok and force_ok)
    reasons = []
    if not lift_ok:
        reasons.append(
            f"lift_fail tip_up={tip_up*1e3:.2f}mm along_rise={along_rise*1e3:.2f}mm"
        )
    if not spiral_ok:
        reasons.append(f"spiral_fail tip_xy={tip_xy:.2f}mm")
    if not force_ok:
        reasons.append(f"force_fail mean|r|={mean_r:.3f}N n_sp={len(sp)}")
    return {
        "ok": ok,
        "lift_ok": lift_ok,
        "spiral_ok": spiral_ok,
        "force_ok": force_ok,
        "tip_up_mm": tip_up * 1e3,
        "along_rise_mm": along_rise * 1e3,
        "tip_xy_peak_mm": tip_xy,
        "spiral_mean_resid_n": mean_r,
        "spiral_median_resid_n": med_r,
        "spiral_peak_resid_n": peak_r,
        "spiral_frames": len(sp),
        "force_bleed_ok": bool(g("force_bleed_ok", False)),
        "lift_cmd_m": float(g("surface_light_lift_m", 0.0) or 0.0),
        "reasons": reasons,
    }


def _run_one(cfg: dict, ep: int, out_dir: Path) -> dict:
    from pci.sim_runner import build_env_and_controllers, run_pci_episode, write_summary

    out_dir.mkdir(parents=True, exist_ok=True)
    env, hybrid, pipeline, force_labeler = build_env_and_controllers(
        cfg, episode_indices=[ep]
    )
    try:
        env.reset(episode_index=ep, video_cb=None)
        hybrid.on_reset(env._env)
        hold44 = __import__(
            "reach_insert_rl.env.full_obs", fromlist=["current_action44"]
        ).current_action44(env.unwrapped)
        hybrid.observe(env._env, hold44)
        summary = run_pci_episode(
            env,
            cfg=cfg,
            hybrid=hybrid,
            pipeline=pipeline,
            force_labeler=force_labeler,
            ego_recorder=None,
            episode_index=ep,
        )
        summary["episode"] = ep
        write_summary(out_dir / f"ep{ep:02d}_summary.json", summary)
        return summary
    finally:
        env.close()


def main() -> int:
    base = _load(REPO / "configs/demo_surface_only.yaml")
    out_root = REPO / "outputs" / "pk_lift_A"
    out_root.mkdir(parents=True, exist_ok=True)
    ep = 3

    # Scheme A focused grid (bleed low → axial lift closed-loop; no tip_servo / no loosen).
    # Prioritize historically good f≈0.03 + lift≈6–10mm (tip rose ~2mm, tip_xy~9mm).
    candidates = [
        (0.03, 10.0, 50),
        (0.03, 6.0, 40),
        (0.02, 10.0, 50),
        (0.05, 10.0, 50),
        (0.03, 15.0, 70),
        (0.02, 15.0, 70),
        (0.05, 6.0, 40),
        (0.08, 10.0, 50),
        (0.02, 6.0, 40),
        (0.05, 15.0, 70),
    ]
    grid = []
    for f_des, lift_mm, frames in candidates:
        grid.append(
            {
                "surface_const_force_des_n": float(f_des),
                "surface_force_bleed_tol_n": max(0.02, float(f_des)),
                "surface_force_bleed_step_m": 0.0005,
                "surface_force_bleed_max_frames": 120,
                "surface_force_bleed_confirm": 4,
                "surface_light_lift_m": float(lift_mm) * 1e-3,
                "surface_tip_lift_m": 0.0015,
                "surface_light_lift_frames": int(frames),
                "surface_lift_right_grasp_scale": 1.0,
                "surface_light_retouch_frames": 0,
                "surface_tip_servo_lift_enable": False,
                "surface_const_force_frames": 10,
            }
        )

    trials = []
    best = None
    for i, ov in enumerate(grid):
        cfg = copy.deepcopy(base)
        cfg.setdefault("approach", {}).update(ov)
        cfg["_pk"] = {"name": "pk_lift_A", "trial": i, **ov}
        tag = (
            f"f{ov['surface_const_force_des_n']:.2f}"
            f"_L{ov['surface_light_lift_m']*1e3:.0f}"
            f"_n{ov['surface_light_lift_frames']}"
        )
        print(f"\n===== [{i+1}/{len(grid)}] {tag} =====", flush=True)
        try:
            summary = _run_one(cfg, ep=ep, out_dir=out_root / tag)
            m = _metrics(summary)
        except Exception as e:  # noqa: BLE001
            m = {
                "ok": False,
                "error": repr(e),
                "reasons": [f"exception:{e}"],
            }
            print(f"FAIL exception: {e}", flush=True)
        row = {"tag": tag, "overrides": ov, **m}
        trials.append(row)
        print(
            f"→ ok={m.get('ok')} tip_up={m.get('tip_up_mm')} "
            f"along={m.get('along_rise_mm')} tip_xy={m.get('tip_xy_peak_mm')} "
            f"mean_r={m.get('spiral_mean_resid_n')} reasons={m.get('reasons')}",
            flush=True,
        )
        (out_root / "sweep_partial.json").write_text(
            json.dumps(trials, indent=2), encoding="utf-8"
        )
        if m.get("ok"):
            score = float(m["tip_xy_peak_mm"]) - abs(
                float(m["spiral_mean_resid_n"]) - 0.2
            )
            if best is None or score > best[0]:
                best = (score, row)
            # Early stop on first solid pass with tip_xy>=8
            if float(m["tip_xy_peak_mm"]) >= 8.0:
                print("early-stop: strong pass", flush=True)
                break

    result_path = REPO / "outputs" / "pk_lift_A_result.json"
    if best is not None:
        row = best[1]
        result = {
            "status": "passed",
            "scheme": "A_bleed_then_axial_lift",
            "privileged_diagnostic": True,
            "episode": ep,
            "yaml_overrides": row["overrides"],
            "metrics": {
                "tip_up_mm": row["tip_up_mm"],
                "along_rise_mm": row["along_rise_mm"],
                "tip_xy_peak_mm": row["tip_xy_peak_mm"],
                "spiral_mean_resid_n": row["spiral_mean_resid_n"],
                "spiral_median_resid_n": row["spiral_median_resid_n"],
                "spiral_peak_resid_n": row["spiral_peak_resid_n"],
                "spiral_frames": row["spiral_frames"],
                "lift_cmd_m": row["lift_cmd_m"],
                "force_bleed_ok": row["force_bleed_ok"],
            },
            "gates": {
                "lift_ok": row["lift_ok"],
                "spiral_ok": row["spiral_ok"],
                "force_ok": row["force_ok"],
            },
            "n_trials": len(trials),
            "all_trials": trials,
        }
    else:
        # Best-effort nearest (not claimed pass)
        def _near(t: dict) -> float:
            if "tip_up_mm" not in t:
                return -1e9
            lift = max(float(t["tip_up_mm"]), float(t["along_rise_mm"]))
            xy = float(t.get("tip_xy_peak_mm") or 0.0)
            fr = float(t.get("spiral_mean_resid_n") or 0.0)
            force_pen = 0.0 if 0.05 <= fr <= 0.45 else 5.0
            return lift + 0.5 * xy - force_pen

        ranked = sorted(trials, key=_near, reverse=True)
        top = ranked[0] if ranked else {}
        result = {
            "status": "failed",
            "scheme": "A_bleed_then_axial_lift",
            "privileged_diagnostic": True,
            "episode": ep,
            "reason": (top.get("reasons") or ["no_trial"])
            if top
            else ["no_trial"],
            "best_attempt": top,
            "n_trials": len(trials),
            "all_trials": trials,
        }

    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {result_path} status={result['status']}", flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
