#!/usr/bin/env python3
"""Privileged mouth-finding batch: success = mouth_ok ∧ tray_ok ∧ peg_ok."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
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


def main() -> int:
    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text(encoding="utf-8"))
    out = REPO / "outputs" / "mouth_batch20"
    out.mkdir(parents=True, exist_ok=True)
    episodes = list(range(20))

    from pci.sim_runner import build_env_and_controllers, run_pci_episode, write_summary

    rows = []
    for ep in episodes:
        env, hybrid, pipeline, force_labeler = build_env_and_controllers(
            cfg, episode_indices=[ep]
        )
        env.reset(episode_index=ep)
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
        write_summary(out / f"ep{ep:02d}_summary.json", summary)
        row = {
            "ep": ep,
            "success": bool(summary.get("success")),
            "mouth_ok": bool(summary.get("mouth_ok")),
            "tray_ok": bool(summary.get("tray_ok")),
            "peg_ok": bool(summary.get("peg_ok")),
            "fail_reason": summary.get("fail_reason"),
            "lat": summary.get("final_lat_m"),
            "along": summary.get("final_along_m"),
            "steps": summary.get("control_steps"),
        }
        rows.append(row)
        print(
            f"[mouth-batch] ep={ep} success={row['success']} mouth={row['mouth_ok']} "
            f"fail={row['fail_reason']!r} lat={(row['lat'] or 0)*1e3:.1f}mm",
            flush=True,
        )

    n = len(rows)
    ok = sum(1 for r in rows if r["success"])
    batch = {
        "experiment_tag": "privileged_diagnostic_mouth",
        "n": n,
        "mouth_success": ok,
        "rate": ok / n if n else 0.0,
        "fail_reasons": dict(Counter(r["fail_reason"] or "OK" for r in rows)),
        "rows": rows,
    }
    (out / "batch_summary.json").write_text(json.dumps(batch, indent=2), encoding="utf-8")
    print(f"[mouth-batch] MOUTH {ok}/{n} = {100 * ok / max(n, 1):.0f}%", flush=True)
    return 0 if ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
