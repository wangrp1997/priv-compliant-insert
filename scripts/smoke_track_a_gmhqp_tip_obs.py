#!/usr/bin/env python3
"""Unit smoke: TEC tip_hat → GMHQP tip task (reuse wire; no MuJoCo).

Search decision: docs/TRACK_A_GMHQP_TIP_OBS.md §2 — REUSE Kim TEC + GMHQP.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pci.tip_tracking_baselines import baseline_hold_r
from pci.track_b_gmhqp import (
    GmhqpConfig,
    GmhqpState,
    gmhqp_wrist_delta,
    is_gmhqp_mode,
    is_gmhqp_tip_obs_mode,
)


def main() -> int:
    assert is_gmhqp_tip_obs_mode("track_a_gmhqp_tip_obs")
    assert is_gmhqp_mode("track_a_gmhqp_tip_obs")
    assert is_gmhqp_mode("track_b_gmhqp")

    n = np.array([0.0, 0.0, 1.0])
    press = -n
    path = np.array([0.08, 0.0, 0.05])
    site = np.array([0.12, 0.0, 0.05])
    tip_geom = np.array([0.095, 0.002, 0.05])
    tip_hat = np.array([0.11, 0.025, 0.05])

    st = GmhqpState()
    _, meta_g, st = gmhqp_wrist_delta(
        site_xyz=site,
        tip_obj=tip_geom,
        path_target=path,
        normal=n,
        press_ax=press,
        contact_resid_n=0.2,
        axis_err_deg=5.0,
        r_cmd_m=0.012,
        state=st,
        cfg=GmhqpConfig(),
    )
    _, meta_h, st = gmhqp_wrist_delta(
        site_xyz=site,
        tip_obj=tip_hat,
        path_target=path,
        normal=n,
        press_ax=press,
        contact_resid_n=0.2,
        axis_err_deg=5.0,
        r_cmd_m=0.012,
        state=st,
        cfg=GmhqpConfig(),
    )
    # tip_hat farther from path → larger tip-task error (structure check).
    assert float(meta_h["tip_task_err_mm"]) > float(meta_g["tip_task_err_mm"]) + 5.0
    assert meta_h["path"] == "track_b_gmhqp"

    hold, meta, _ = baseline_hold_r(
        "track_a_gmhqp_tip_obs",
        target=path,
        tip=tip_hat,
        site_xyz=site,
        offset_frozen=np.zeros(3),
        coupling=None,
        spiral_n=n,
        press_ax=press,
        ax_step=0.0001,
        track=1.0,
        max_step_m=0.012,
        along_now_m=0.10,
        along0_m=0.10,
        hole_axis=n,
        along_slack_m=0.01,
        contact_resid_n=0.2,
        gmhqp_cfg=GmhqpConfig(),
        gmhqp_state=GmhqpState(),
        r_cmd_m=0.012,
        force_gate_enable=False,
        axis_err_deg=5.0,
    )
    assert meta.get("path") == "track_b_gmhqp"
    assert float(meta["tip_task_err_mm"]) > 1.0

    cfg = yaml.safe_load(
        (REPO / "configs/scheme_l3/S2_trackA_gmhqp_tip_obs.yaml").read_text(
            encoding="utf-8"
        )
    )
    a = cfg["approach"]
    assert a["surface_tip_baseline"] == "track_a_gmhqp_tip_obs"
    assert a["surface_tip_gmhqp_use_tip_obs"] is True
    assert a["surface_tip_est_backend"] == "theory_ekf"
    assert a.get("surface_tip_oracle_mouth") is False
    print("smoke_track_a_gmhqp_tip_obs: OK (reuse TEC→GMHQP)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
