#!/usr/bin/env python3
"""Unit smoke for Track-B GMHQP-CompC (general law; no MuJoCo)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pci.tip_tracking_baselines import baseline_hold_r
from pci.track_b_gmhqp_compc import (
    GmhqpCompcConfig,
    GmhqpCompcState,
    contact_horizon_u,
    gmhqp_compc_wrist_delta,
    is_gmhqp_compc_mode,
    residual_contact_weight,
)


def main() -> int:
    assert is_gmhqp_compc_mode("track_b_gmhqp_compc")
    assert not is_gmhqp_compc_mode("track_b_gmhqp")
    assert not is_gmhqp_compc_mode("track_b_gmhqp_ftip")

    # α: large tip err → near 0; small tip err → near 1
    assert residual_contact_weight(0.05, 0.005) < 0.05
    assert residual_contact_weight(0.0005, 0.005) > 0.9

    cfg = GmhqpCompcConfig()
    et = np.array([0.01, 0.0, 0.0])
    ec = np.array([0.0, 0.008, 0.0])
    u_tip_only, info0 = contact_horizon_u(
        e_tip=et, e_contact=None, contact_valid=False, cfg=cfg
    )
    assert float(info0["wc_eff"]) == 0.0
    assert float(np.linalg.norm(u_tip_only - et)) > 1e-6  # damped vs raw e
    u_both, info1 = contact_horizon_u(
        e_tip=et * 0.05, e_contact=ec, contact_valid=True, cfg=cfg
    )
    assert float(info1["alpha_contact"]) > 0.5
    assert float(info1["wc_eff"]) > 0.0
    # Contact pulls +y when tip residual is falsely small
    assert float(u_both[1]) > 0.0

    n = np.array([0.0, 0.0, 1.0])
    press = -n
    path = np.array([0.08, 0.0, 0.05])
    mouth = np.array([0.08, 0.0, 0.05])
    site = np.array([0.10, 0.0, 0.05])
    tip = np.array([0.085, 0.0, 0.05])  # in-hand tip near path
    force = np.array([0.0, 0.0, 0.2])
    torque = np.array([0.0, -0.004, 0.0])

    st = GmhqpCompcState()
    hold, meta, st = gmhqp_compc_wrist_delta(
        site_xyz=site,
        tip_obj=tip,
        path_target=path,
        mouth_xyz=mouth,
        normal=n,
        press_ax=press,
        contact_resid_n=0.2,
        axis_err_deg=5.0,
        force_xyz=force,
        torque_xyz=torque,
        tip_anchor=tip,
        r_cmd_m=0.02,
        state=st,
        cfg=cfg,
    )
    assert meta["path"] == "track_b_gmhqp_compc"
    assert meta.get("mpc_active") is True
    assert hold.shape == (3,)
    # tip_obj never replaced: tip_task_err reflects geom tip, not FT swap
    assert "tip_source" not in meta or meta.get("tip_source") != "ft_contact"

    hold2, meta2, _ = baseline_hold_r(
        "track_b_gmhqp_compc",
        target=path,
        tip=tip,
        site_xyz=site,
        offset_frozen=site - tip,
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
        tip_anchor=tip,
        contact_resid_n=0.2,
        force_gate_enable=False,
        wrench_xyz=force,
        wrench_tau_xyz=torque,
        gmhqp_compc_cfg=cfg,
        gmhqp_compc_state=GmhqpCompcState(),
        axis_err_deg=5.0,
        r_cmd_m=0.02,
        mouth_xyz=mouth,
    )
    assert meta2["path"] == "track_b_gmhqp_compc"
    assert hold2.shape == (3,)

    yml = yaml.safe_load(
        (REPO / "configs/scheme_l3/S2_trackB_gmhqp_compc.yaml").read_text()
    )
    ap = yml["approach"]
    assert ap["surface_tip_baseline"] == "track_b_gmhqp_compc"
    assert bool(ap.get("surface_tip_gmhqp_compc_enable"))
    assert float(ap["surface_planned_priv_enter_max_axis_err_deg"]) == 25.0
    print("smoke_track_b_gmhqp_compc: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
