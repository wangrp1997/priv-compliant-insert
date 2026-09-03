#!/usr/bin/env python3
"""Unit smoke for Track-B OIGS (no MuJoCo).

Checks: path alpha from object residual; tip GT ignored; SUCCESS knobs locked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pci.tip_tracking_baselines import baseline_hold_r
from pci.track_b_oigs import (
    OigsConfig,
    OigsState,
    is_oigs_mode,
    oigs_path_alpha,
    oigs_wrist_delta,
)


def main() -> int:
    assert is_oigs_mode("track_b_oigs")
    a0 = oigs_path_alpha(obj_rot_err_rad=0.0, cfg=OigsConfig(e_rigid_rad=0.12))
    a1 = oigs_path_alpha(obj_rot_err_rad=0.12, cfg=OigsConfig(e_rigid_rad=0.12))
    assert a0 > 0.9
    assert a1 <= 0.05 + 1e-9  # floor

    n = np.array([0.0, 0.0, 1.0])
    site = np.array([0.10, 0.0, 0.05])
    tip = np.array([9.0, 9.0, 0.05])  # absurd tip must not drive
    path = np.array([0.08, 0.0, 0.05])
    press = -n

    hold_stiff, meta_s = oigs_wrist_delta(
        site_xyz=site,
        path_target=path,
        normal=n,
        press_ax=press,
        contact_resid_n=0.2,
        r_cmd_m=0.012,
        obj_rot_err_rad=0.0,
        cfg=OigsConfig(),
    )
    hold_soft, meta_w = oigs_wrist_delta(
        site_xyz=site,
        path_target=path,
        normal=n,
        press_ax=press,
        contact_resid_n=0.2,
        r_cmd_m=0.012,
        obj_rot_err_rad=0.20,
        cfg=OigsConfig(),
    )
    assert meta_s["path"] == "track_b_oigs"
    assert meta_s["privileged"] is False
    assert meta_s["oigs_alpha"] > meta_w["oigs_alpha"]
    assert float(np.linalg.norm((hold_stiff - site)[:2])) > float(
        np.linalg.norm((hold_soft - site)[:2])
    )

    st = OigsState(obj_rot_err_rad=0.02)
    hold2, meta2, _ = baseline_hold_r(
        "track_b_oigs",
        target=path,
        tip=tip,
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
        oigs_cfg=OigsConfig(),
        oigs_state=st,
        r_cmd_m=0.012,
        force_gate_enable=False,
    )
    assert meta2["path"] == "track_b_oigs"
    assert float(np.linalg.norm(hold2 - tip)) > 1.0

    cfg = yaml.safe_load((REPO / "configs/scheme_l3/S2_trackB_oigs.yaml").read_text())
    a = cfg["approach"]
    assert a["surface_tip_baseline"] == "track_b_oigs"
    assert a["surface_tip_search_pose_hold"] is True
    assert a["surface_planned_priv_enter_max_axis_err_deg"] == 25.0
    assert a["surface_planned_priv_enter_max_along_m"] == 0.100
    g = cfg["compliant"]["priv_grasp_opt"]
    assert g["enable"] is True
    assert float(g["k_pos"]) >= 40.0

    print("smoke_track_b_oigs: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
