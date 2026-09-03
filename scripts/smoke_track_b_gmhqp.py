#!/usr/bin/env python3
"""Unit smoke for Track-B GMHQP (no MuJoCo).

Distinct from OIGS: tip on path + wrist off path → small Δw;
tip off path → tip-task error drives wrist via C (not wrist TCP chase).
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
)
from pci.track_b_oigs import OigsConfig, oigs_wrist_delta


def main() -> int:
    assert is_gmhqp_mode("track_b_gmhqp")
    n = np.array([0.0, 0.0, 1.0])
    press = -n
    path = np.array([0.08, 0.0, 0.05])
    site_off = np.array([0.12, 0.0, 0.05])  # wrist far from path
    tip_on = path.copy()  # tip already on path
    tip_off = np.array([0.10, 0.02, 0.05])

    st = GmhqpState()
    hold_on, meta_on, st = gmhqp_wrist_delta(
        site_xyz=site_off,
        tip_obj=tip_on,
        path_target=path,
        normal=n,
        press_ax=press,
        contact_resid_n=0.2,
        axis_err_deg=5.0,
        r_cmd_m=0.012,
        state=st,
        cfg=GmhqpConfig(),
    )
    hold_off, meta_off, st = gmhqp_wrist_delta(
        site_xyz=site_off,
        tip_obj=tip_off,
        path_target=path,
        normal=n,
        press_ax=press,
        contact_resid_n=0.2,
        axis_err_deg=5.0,
        r_cmd_m=0.012,
        state=st,
        cfg=GmhqpConfig(),
    )
    assert meta_on["path"] == "track_b_gmhqp"
    assert meta_on["privileged"] is False
    # Tip on path → near-zero planar wrist move; tip off → larger.
    d_on = float(np.linalg.norm((hold_on - site_off)[:2]))
    d_off = float(np.linalg.norm((hold_off - site_off)[:2]))
    assert d_on < 1e-4, d_on
    assert d_off > d_on + 1e-3, (d_on, d_off)

    # OIGS (broken for this case): wrist off path → large move even if tip on path.
    hold_oigs, meta_o = oigs_wrist_delta(
        site_xyz=site_off,
        path_target=path,
        normal=n,
        press_ax=press,
        contact_resid_n=0.2,
        r_cmd_m=0.012,
        obj_rot_err_rad=0.0,
        cfg=OigsConfig(),
    )
    d_oigs = float(np.linalg.norm((hold_oigs - site_off)[:2]))
    assert d_oigs > 1e-3, d_oigs
    assert d_oigs > d_on + 1e-3

    hold2, meta2, _ = baseline_hold_r(
        "track_b_gmhqp",
        target=path,
        tip=tip_off,
        site_xyz=site_off,
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
    assert meta2["path"] == "track_b_gmhqp"
    assert float(meta2["tip_task_err_mm"]) > 1.0

    cfg = yaml.safe_load((REPO / "configs/scheme_l3/S2_trackB_gmhqp.yaml").read_text())
    a = cfg["approach"]
    assert a["surface_tip_baseline"] == "track_b_gmhqp"
    assert a["surface_tip_gmhqp_enable"] is True
    assert a["surface_tip_search_pose_hold"] is True
    assert a["surface_planned_priv_enter_max_axis_err_deg"] == 25.0
    assert a["surface_planned_priv_enter_max_along_m"] == 0.100
    g = cfg["compliant"]["priv_grasp_opt"]
    assert g["enable"] is True

    print("smoke_track_b_gmhqp: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
