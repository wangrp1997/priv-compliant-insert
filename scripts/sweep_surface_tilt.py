#!/usr/bin/env python3
"""Sweep surface grasp/admit knobs; privileged_diagnostic tilt/mouth report."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
EPISODES = "0,3,4,5,6"


def _load_base(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _apply(cfg: dict, patch: dict) -> dict:
    out = copy.deepcopy(cfg)
    for k, v in patch.get("approach", {}).items():
        out.setdefault("approach", {})[k] = v
    for k, v in patch.get("priv_grasp", {}).items():
        out.setdefault("compliant", {}).setdefault("priv_grasp_opt", {})[k] = v
    return out


def _run_one(tag: str, cfg_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **dict(__import__("os").environ),
        "PYTHONPATH": ":".join(
            [
                str(REPO / "src"),
                "/home/wangrenpeng/dexjoco",
                "/home/wangrenpeng/dexjoco/dexjoco",
                "/home/wangrenpeng/dexjoco/embodied_grasp_insertion",
                "/home/wangrenpeng/reach_insert_rl",
            ]
        ),
        "MUJOCO_GL": "egl",
    }
    cmd = [
        "/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python",
        "-u",
        str(REPO / "scripts" / "smoke_ep0.py"),
        "--episodes",
        EPISODES,
        "--config",
        str(cfg_path),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=REPO, env=env, check=False)
    rows = []
    for p in sorted(out_dir.glob("ep*_summary.json")):
        with p.open(encoding="utf-8") as f:
            j = json.load(f)
        sm = j.get("surface_meta") or {}
        rows.append(
            {
                "ep": p.stem.split("_")[0],
                "mouth": bool(j.get("mouth_ok")),
                "tray_ok": j.get("tray_ok"),
                "tilt": sm.get("settle_tilt_peak_deg"),
                "tray_peak_a": sm.get("phase_a_tray_tilt_peak_deg"),
                "peg_peak_a": sm.get("phase_a_peg_tilt_peak_deg"),
                "rel_contact": sm.get("rel_rot_at_contact_rad"),
                "grasp_peak": sm.get("grasp_ramp_peak_scale"),
                "admit_peak": sm.get("soft_admit_peak_scale"),
                "fail": j.get("fail_reason", ""),
            }
        )
    mouths = sum(1 for r in rows if r["mouth"])
    tilts = [r["tilt"] for r in rows if r["tilt"] is not None]
    summary = {
        "tag": tag,
        "mouth_rate": mouths / max(len(rows), 1),
        "tilt_mean": sum(tilts) / len(tilts) if tilts else None,
        "tilt_max": max(tilts) if tilts else None,
        "rows": rows,
    }
    with (out_dir / "sweep_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(REPO / "configs" / "theory_pose_qp.yaml"))
    args = parser.parse_args()
    base = _load_base(Path(args.base))
    tmp = REPO / "outputs" / "_sweep_cfgs"
    tmp.mkdir(parents=True, exist_ok=True)

    grid = [
        ("r23a", {"approach": {"surface_soft_left_admit_scale": 0.18, "surface_left_grasp_scale": 1.45, "surface_left_grasp_scale_init": 1.25}}),
        ("r23b", {"approach": {"surface_soft_left_admit_scale": 0.22, "surface_left_grasp_scale": 1.50, "surface_left_grasp_scale_init": 1.28}}),
        ("r23c", {"approach": {"surface_soft_left_admit_scale": 0.25, "surface_left_grasp_scale": 1.55, "surface_left_grasp_scale_init": 1.30, "surface_grasp_ramp_max_scale": 1.70}}),
        ("r23d", {"approach": {"surface_soft_left_admit_scale": 0.22, "surface_left_grasp_scale": 1.52, "surface_left_grasp_scale_init": 1.32, "surface_grasp_ramp_step": 0.06}, "priv_grasp": {"f_squeeze_n": 7.0}}),
    ]

    all_summaries = []
    for tag, patch in grid:
        cfg = _apply(base, patch)
        cfg_path = tmp / f"{tag}.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, sort_keys=False)
        print(f"\n=== sweep {tag} ===", flush=True)
        s = _run_one(tag, cfg_path, REPO / "outputs" / f"theory_pose_qp_{tag}")
        all_summaries.append(s)
        print(
            f"{tag}: mouth={s['mouth_rate']:.0%} tilt_mean={s['tilt_mean']:.1f}° "
            f"tilt_max={s['tilt_max']:.1f}°",
            flush=True,
        )

    out = REPO / "outputs" / "surface_tilt_sweep_report.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
