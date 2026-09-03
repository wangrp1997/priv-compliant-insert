#!/usr/bin/env python3
"""Unit smoke for Track-B GMHQP-FTIP (REUSE, no MuJoCo)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pci.tip_tracking_baselines import baseline_hold_r
from pci.track_b_gmhqp_ftip import (
    GmhqpFtipConfig,
    GmhqpFtipState,
    gmhqp_ftip_wrist_delta,
    is_gmhqp_ftip_mode,
)


def main() -> int:
    assert is_gmhqp_ftip_mode("track_b_gmhqp_ftip")
    n = np.array([0.0, 0.0, 1.0])
    press = -n
    path = np.array([0.08, 0.0, 0.05])
    site = np.array([0.10, 0.0, 0.05])
    tip_geom = np.array([0.085, 0.0, 0.05])  # geom near path (~5 mm)
    # FT lever puts contact farther from path than geom tip
    force = np.array([0.0, 0.0, 0.2])
    torque = np.array([0.0, -0.004, 0.0])  # n×τ/fn → +x ⇒ tip ~[0.12,0,0.05]

    cfg = GmhqpFtipConfig()
    st = GmhqpFtipState()
    hold, meta, st = gmhqp_ftip_wrist_delta(
        site_xyz=site,
        tip_geom=tip_geom,
        path_target=path,
        normal=n,
        press_ax=press,
        contact_resid_n=0.2,
        axis_err_deg=5.0,
        force_xyz=force,
        torque_xyz=torque,
        tip_anchor=tip_geom,
        r_cmd_m=0.02,
        state=st,
        cfg=cfg,
    )
    assert meta["path"] == "track_b_gmhqp_ftip"
    assert meta["tip_source"] == "ft_contact"
    assert meta["ft_valid"]
    assert float(meta["ft_tip_err_mm"]) > float(meta["geom_tip_err_mm"]) + 1.0
    assert hold.shape == (3,)
    # Wrist should move toward closing larger FT tip error (roughly −x)
    assert float(hold[0]) < float(site[0])

    # Hook through baseline_hold_r
    hold2, meta2, _ = baseline_hold_r(
        "track_b_gmhqp_ftip",
        target=path,
        tip=tip_geom,
        site_xyz=site,
        offset_frozen=site - tip_geom,
        coupling=None,
        spiral_n=n,
        press_ax=press,
        ax_step=0.0,
        track=1.0,
        max_step_m=0.012,
        along_now_m=0.09,
        along0_m=0.09,
        hole_axis=n,
        along_slack_m=0.01,
        tip_anchor=tip_geom,
        contact_resid_n=0.2,
        force_gate_enable=False,
        wrench_xyz=force,
        wrench_tau_xyz=torque,
        gmhqp_ftip_cfg=cfg,
        gmhqp_ftip_state=GmhqpFtipState(),
        axis_err_deg=5.0,
        r_cmd_m=0.02,
    )
    assert meta2["path"] == "track_b_gmhqp_ftip"
    assert hold2.shape == (3,)

    yml = yaml.safe_load(
        (REPO / "configs/scheme_l3/S2_trackB_gmhqp_ftip.yaml").read_text()
    )
    ap = yml["approach"]
    assert ap["surface_tip_baseline"] == "track_b_gmhqp_ftip"
    assert float(ap["surface_planned_priv_enter_max_axis_err_deg"]) == 25.0
    print("smoke_track_b_gmhqp_ftip: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
