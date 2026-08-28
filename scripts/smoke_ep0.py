#!/usr/bin/env python3
"""P0 smoke: handoff → hybrid ALIGN → PCI compliant B1/B2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="comma-separated episode indices (overrides --episode)",
    )
    parser.add_argument("--config", type=Path, default=REPO / "configs/default.yaml")
    parser.add_argument("--out-dir", type=Path, default=REPO / "outputs/smoke")
    parser.add_argument("--video", action="store_true", help="write epXX_ego.mp4")
    args = parser.parse_args()

    if args.episodes:
        episodes = [int(x.strip()) for x in args.episodes.split(",") if x.strip()]
    else:
        episodes = [args.episode]

    cfg = _load_config(args.config)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from pci.ego_video import EgoVideoRecorder, make_gym_ego_video_cb
    from pci.sim_runner import build_env_and_controllers, run_pci_episode, write_summary

    exit_code = 0
    for ep in episodes:
        video_path = out_dir / f"ep{ep:02d}_ego.mp4" if args.video else None
        recorder = EgoVideoRecorder(video_path) if video_path else None
        video_cb = make_gym_ego_video_cb(recorder)

        env, hybrid, pipeline, force_labeler = build_env_and_controllers(
            cfg, episode_indices=[ep]
        )
        try:
            env.reset(episode_index=ep, video_cb=video_cb)
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
                ego_recorder=recorder,
            )
            summary["episode"] = ep
            summary["refs"] = ["ConnTact", "franka-peg-in-hole", "irl_control", "hybrid_insert"]
            if video_path is not None:
                summary["ego_video"] = str(video_path)
            out_path = out_dir / f"ep{ep:02d}_summary.json"
            write_summary(out_path, summary)
            if recorder is not None:
                recorder.close()

            print(f"[pci-smoke] ep={ep} success={summary['success']} insert_ok={summary['insert_ok']}")
            print(
                f"[pci-smoke] align_steps={summary['align_steps']} align_phase={summary['align_phase']} "
                f"control_steps={summary['control_steps']} final={summary['final_phase']}"
            )
            print(
                f"[pci-smoke] tip={summary['final_tip_dist_m']*1000:.1f}mm "
                f"lat={summary['final_lat_m']*1000:.1f}mm along={summary['final_along_m']*1000:.1f}mm"
            )
            print(f"[pci-smoke] wrote {out_path}")
            if video_path is not None and video_path.is_file():
                print(f"[pci-smoke] ego_video={video_path}")
            if not summary["success"]:
                exit_code = 1
        finally:
            env.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
