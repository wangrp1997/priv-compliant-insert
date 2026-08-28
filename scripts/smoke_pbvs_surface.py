#!/usr/bin/env python3
"""偏置孔 + 双臂 hybrid ALIGN + soft 下压贴面（不进 INSERT），录 ego 视频交付。"""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--config", type=Path, default=REPO / "configs/default.yaml")
    parser.add_argument("--out-dir", type=Path, default=REPO / "outputs/smoke")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    from reach_insert_rl.env.full_obs import current_action44

    from pci.ego_video import EgoVideoRecorder, make_gym_ego_video_cb
    from pci.sim_runner import (
        build_env_and_controllers,
        run_pbvs_biased_surface_press,
        write_summary,
    )

    video_path = args.out_dir / f"ep{args.episode:02d}_pbvs_surface.mp4"
    recorder = EgoVideoRecorder(video_path)
    video_cb = make_gym_ego_video_cb(recorder)

    env, hybrid, _pipeline, _fl = build_env_and_controllers(
        cfg, episode_indices=[args.episode]
    )
    gym_env = env._env
    min_lat = float(cfg.get("approach", {}).get("surface_min_lat_m", 0.006))

    try:
        env.reset(episode_index=args.episode, video_cb=video_cb)
        hybrid.on_reset(gym_env)
        hybrid.observe(gym_env, current_action44(env.unwrapped))

        steps, reason, meta = run_pbvs_biased_surface_press(
            env,
            hybrid,
            gym_env,
            cfg=cfg,
            ego_recorder=recorder,
            rng=rng,
        )
        insert_ok = bool(meta.get("insert_ok", False))
        lat_mm = float(meta.get("final_lat_mm", 0.0))
        along_mm = float(meta.get("final_along_mm", 0.0))
        ok = (
            reason == "surface_press"
            and (not insert_ok)
            and (lat_mm >= min_lat * 1000 * 0.5)
            and bool(meta.get("force_gated", False))
        )

        summary = {
            "episode": args.episode,
            "steps": steps,
            "reason": reason,
            "ok": ok,
            "socket_bias": {
                k: meta[k]
                for k in (
                    "socket_bias_lat_m",
                    "socket_bias_x_m",
                    "socket_bias_y_m",
                    "socket_bias_z_m",
                )
                if k in meta
            },
            "force": {
                "fz_baseline": meta.get("fz_baseline"),
                "fz_contact": meta.get("fz_contact"),
                "fz_delta": meta.get("fz_delta"),
                "cmd_site_drift_mm": meta.get("cmd_site_drift_mm"),
                "synced_to_site": meta.get("synced_to_site"),
            },
            "align": {
                "lat_mm": meta.get("align_lat_mm"),
                "along_mm": meta.get("align_along_mm"),
            },
            "final": {
                "lat_mm": lat_mm,
                "along_mm": along_mm,
                "tip_mm": float(meta.get("final_tip_mm", 0.0)),
                "axis_err_deg": float(meta.get("final_axis_err_deg", 0.0)),
                "hole_tilt_deg": meta.get("final_hole_tilt_deg"),
            },
            "entered_insert": False,
            "insert_ok": insert_ok,
            "ego_video": str(video_path),
        }
        out_path = args.out_dir / f"ep{args.episode:02d}_pbvs_surface.json"
        write_summary(out_path, summary)
        recorder.close()

        print(
            f"[pbvs-surface] reason={reason} ok={ok} "
            f"lat={lat_mm:.1f}mm along={along_mm:.1f}mm insert_ok={insert_ok}",
            flush=True,
        )
        print(f"[pbvs-surface] wrote {out_path}", flush=True)
        print(f"[pbvs-surface] video={video_path}", flush=True)
        return 0 if ok else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
