#!/usr/bin/env python3
"""Privileged Phase-B trajectory dump (analysis only, not in control loop)."""

from __future__ import annotations

import argparse
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


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _delta_along_hole(delta_xyz: np.ndarray, hole_axis: np.ndarray) -> float:
    """Positive = motion along hole axis (deeper toward socket in typical frame)."""
    axis = np.asarray(hole_axis, dtype=np.float64).reshape(3)
    axis /= np.linalg.norm(axis) + 1e-12
    return float(np.dot(np.asarray(delta_xyz, dtype=np.float64).reshape(3), axis))


def _delta_along_tool(delta_xyz: np.ndarray, approach_axis: np.ndarray) -> float:
    axis = np.asarray(approach_axis, dtype=np.float64).reshape(3)
    axis /= np.linalg.norm(axis) + 1e-12
    return float(np.dot(np.asarray(delta_xyz, dtype=np.float64).reshape(3), axis))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--config", type=Path, default=REPO / "configs/default.yaml")
    parser.add_argument("--out-dir", type=Path, default=REPO / "outputs/diagnose")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    from reach_insert_rl.env.full_obs import current_action44

    from pci.features import features_from_raw
    from pci.pipeline import PipelinePhase
    from pci.sensors import read_right_finger_force12, read_wrist_wrench_world
    from pci.sim_runner import (
        _sim_dt,
        build_env_and_controllers,
        run_pbvs_biased_surface_press,
        step_action44,
    )
    from pci.task_frame import TaskFrame
    from pci.wrist import apply_tip_delta44, hole_task_basis

    env, hybrid, pipeline, force_labeler = build_env_and_controllers(
        cfg, episode_indices=[args.episode]
    )
    traj: list[dict] = []
    try:
        env.reset(episode_index=args.episode)
        hybrid.on_reset(env._env)
        raw = env.unwrapped
        gym_env = env._env
        max_ctrl = int(cfg.get("sim", {}).get("max_control_steps", 1200))

        align_steps, align_reason, surface_meta = run_pbvs_biased_surface_press(
            env, hybrid, gym_env, cfg=cfg, rng=rng
        )
        if align_reason not in ("surface_press", "surface_press_but_bad_geom"):
            feat = features_from_raw(raw)
            print(f"[diag] Phase A failed: {align_reason} tip={feat.tip_socket_dist_m*1000:.1f}mm")
            return 1

        feat0 = features_from_raw(raw)
        action44 = current_action44(raw)
        task_frame = TaskFrame.from_hole_axis(
            action44[0:3],
            feat0.hole_axis,
            peg_axis_world=feat0.peg_axis,
        )
        wrench = read_wrist_wrench_world(raw)
        finger12 = read_right_finger_force12(raw, force_labeler)
        hold_fingers = action44[6:22].copy()
        pipeline.begin_compliant(
            task_frame,
            wrench[0],
            finger12,
            hold_fingers,
            action44[0:3],
            already_on_surface=True,
        )
        wh0 = task_frame.wrench_tool(wrench[0])
        b_start_feat = feat0

        hole_axis = feat0.hole_axis.copy()
        tip0 = feat0.tip_pos.copy()
        socket0 = feat0.socket_pos.copy()
        tool_hole_dot = float(np.dot(task_frame.approach_axis, hole_axis))
        peg_hole_dot = float(np.dot(feat0.peg_axis, hole_axis))

        for step in range(max_ctrl):
            outcome = env._labeler.compute(raw)
            action44 = current_action44(raw)
            wrench = read_wrist_wrench_world(raw)
            finger12 = read_right_finger_force12(raw, force_labeler)
            feat = features_from_raw(raw)
            wh = task_frame.wrench_tool(wrench[0])
            wh_raw = task_frame.wrench_tool(wrench[0])
            tip_before = feat.tip_pos.copy()
            pr = pipeline.step(
                wrench[0],
                action44[0:3],
                finger12,
                insert_ok_eval=bool(outcome.insert_ok),
                dt=_sim_dt(cfg),
            )
            delta = np.asarray(pr.delta_xyz, dtype=np.float64).reshape(3)
            max_step = float(cfg.get("sim", {}).get("max_pos_step_m", 0.004))
            dn = float(np.linalg.norm(delta))
            if dn > max_step:
                delta = delta * (max_step / dn)
                dn = float(np.linalg.norm(delta))

            action44 = apply_tip_delta44(action44, delta)
            action44[6:22] = action44[6:22] + pr.delta_hand16
            if pr.finger_open > 0.0:
                action44[6:22] = hold_fingers * (1.0 - pr.finger_open)
            step_action44(gym_env, action44)

            feat1 = features_from_raw(raw)
            tip_move = feat1.tip_pos - tip_before
            tip_dist_delta = (feat1.tip_socket_dist_m - feat0.tip_socket_dist_m) * 1000

            traj.append(
                {
                    "step": step,
                    "phase": pr.phase.name,
                    "reason": pr.reason,
                    "insert_ok": bool(outcome.insert_ok),
                    "peg_ok": bool(outcome.peg_ok),
                    "tip_dist_mm": feat1.tip_socket_dist_m * 1000,
                    "tip_dist_change_mm": feat1.tip_socket_dist_m * 1000 - feat.tip_socket_dist_m * 1000,
                    "lat_mm": feat1.lateral_m * 1000,
                    "along_mm": feat1.along_m * 1000,
                    "axis_err_deg": np.degrees(feat1.axis_error_rad),
                    "wrist_fz_tool": float(wh[2]),
                    "wrist_fz_raw": float(task_frame.wrench_tool(wrench[0])[2]),
                    "wrist_f_lat": float(np.linalg.norm(wh[:2])),
                    "push_sign": float(pipeline.search._push_sign),
                    "on_surface": bool(pipeline.search._on_surface),
                    "delta_norm_mm": dn * 1000,
                    "delta_xyz_mm": (delta * 1000).tolist(),
                    "delta_along_hole_mm": _delta_along_hole(delta, hole_axis) * 1000,
                    "delta_along_tool_mm": _delta_along_tool(delta, task_frame.approach_axis) * 1000,
                    "tip_move_along_hole_mm": _delta_along_hole(tip_move, hole_axis) * 1000,
                    "along_mm_priv_eval": feat1.along_m * 1000,
                    "f_des_z": float(pipeline.search._push_sign * pipeline.search.config.push_force_n)
                    if pr.phase == PipelinePhase.COMPLIANT_SEARCH
                    else float(pipeline.insert._push_sign * pipeline.insert.config.f_insert_des_n),
                }
            )
            feat0 = feat1
            if pr.done:
                break

        reasons: dict[str, int] = {}
        for row in traj:
            reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1

        seek = [r for r in traj if r["reason"] in ("seeking_surface", "surface_contact", "surface_seek_timeout")]
        spiral = [r for r in traj if r["reason"] in ("searching", "force_retreat", "hole_detected")]

        out = {
            "episode": args.episode,
            "align_steps": align_steps,
            "surface_reason": align_reason,
            "surface_meta": surface_meta,
            "b_start": {
                "tip_dist_mm": b_start_feat.tip_socket_dist_m * 1000,
                "lat_mm": b_start_feat.lateral_m * 1000,
                "along_mm": b_start_feat.along_m * 1000,
                "axis_err_deg": np.degrees(b_start_feat.axis_error_rad),
                "wrist_fz_tool": float(wh0[2]),
                "push_sign": float(pipeline.search._push_sign),
                "tool_approach_dot_hole": tool_hole_dot,
                "peg_axis_dot_hole": peg_hole_dot,
                "hole_axis": hole_axis.tolist(),
                "tool_approach_axis": task_frame.approach_axis.tolist(),
                "already_on_surface": True,
            },
            "reason_counts": reasons,
            "traj": traj,
        }
        out_path = args.out_dir / f"ep{args.episode:02d}_diag.json"
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

        b0_tip = traj[0]["tip_dist_mm"] if traj else float("nan")
        b1_tip = traj[-1]["tip_dist_mm"] if traj else float("nan")
        print(f"[diag] ep={args.episode} align={align_steps} B_steps={len(traj)}")
        print(
            f"[diag] B-start tip={out['b_start']['tip_dist_mm']:.1f}mm "
            f"tool·hole={tool_hole_dot:+.3f} peg·hole={peg_hole_dot:+.3f} "
            f"Fz0={wh0[2]:+.2f}N push_sign={pipeline.search._push_sign:+.0f}"
        )
        if seek:
            d0, d1 = seek[0]["tip_dist_mm"], seek[-1]["tip_dist_mm"]
            along_sum = sum(r["delta_along_hole_mm"] for r in seek)
            print(
                f"[diag] contact_seek n={len(seek)} tip {d0:.1f}->{d1:.1f}mm "
                f"sum_delta_hole={along_sum:+.1f}mm "
                f"on_surface_end={seek[-1].get('on_surface')}"
            )
        if spiral:
            d0, d1 = spiral[0]["tip_dist_mm"], spiral[-1]["tip_dist_mm"]
            along_sum = sum(r["delta_along_hole_mm"] for r in spiral)
            retreat_n = sum(1 for r in spiral if r["reason"] == "force_retreat")
            print(
                f"[diag] spiral n={len(spiral)} tip {d0:.1f}->{d1:.1f}mm "
                f"sum_delta_hole={along_sum:+.1f}mm retreats={retreat_n}"
            )
        print(f"[diag] reasons={reasons}")
        if traj:
            worst = sorted(traj, key=lambda r: r["tip_dist_change_mm"], reverse=True)[:5]
            print("[diag] largest tip_dist increases:")
            for r in worst:
                print(
                    f"  s={r['step']} {r['reason']} d_tip=+{r['tip_dist_change_mm']:.2f}mm "
                    f"delta_hole={r['delta_along_hole_mm']:+.2f}mm "
                    f"delta_tool={r['delta_along_tool_mm']:+.2f}mm Fz={r['wrist_fz_tool']:+.1f}N"
                )
        print(f"[diag] wrote {out_path}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
