#!/usr/bin/env python3
"""Unit smoke for Track-B PHIG (no MuJoCo).

Checks: wrist path law ignores tip; mouth/slip gates; SUCCESS_STANDARD knobs untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pci.tip_tracking_baselines import baseline_hold_r
from pci.track_b_phig import (
    PhigConfig,
    PhigState,
    phig_axial_override,
    phig_near_mouth,
    phig_select_mode,
    phig_slip_trip,
    phig_wrist_delta,
)


def main() -> int:
    n = np.array([0.0, 0.0, 1.0])
    site = np.array([0.10, 0.00, 0.05])
    tip = np.array([0.12, 0.02, 0.05])  # deliberately lagged tip
    path = np.array([0.08, 0.00, 0.05])  # known spiral waypoint
    press = -n

    hold, meta = phig_wrist_delta(
        site_xyz=site,
        path_target=path,
        normal=n,
        press_ax=press,
        ax_step=0.0002,
        wrench_xyz=np.array([0.05, 0.0, 0.06]),
        cfg=PhigConfig(k_path=1.0, max_step_m=0.02),
    )
    d = hold - site
    # Planar move must be toward path, not toward tip.
    assert float(d[0]) < 0.0, d
    assert abs(float(d[1])) < 1e-6, d
    assert meta["path"] == "track_b_phig"
    assert meta.get("privileged") is False

    # baseline_hold_r: tip lag must not flip wrist direction vs path.
    hold_r, meta2, ax = baseline_hold_r(
        "track_b_phig",
        target=path,
        tip=tip,
        site_xyz=site,
        offset_frozen=site - tip,
        coupling=None,
        spiral_n=n,
        press_ax=press,
        ax_step=0.0002,
        track=1.0,
        max_step_m=0.02,
        along_now_m=0.09,
        along0_m=0.09,
        hole_axis=n,
        along_slack_m=0.01,
        contact_resid_n=0.04,
        f_des_n=0.04,
        force_gate_enable=False,
    )
    assert meta2["path"] == "track_b_phig"
    assert float((hold_r - site)[0]) < 0.0

    assert phig_near_mouth(
        r_cmd_m=0.003,
        contact_resid_n=0.1,
        force_hole_cue=False,
        mouth_r_m=0.0045,
        f_mouth_press_n=0.08,
    )
    assert not phig_near_mouth(
        r_cmd_m=0.012,
        contact_resid_n=0.1,
        force_hole_cue=False,
        mouth_r_m=0.0045,
        f_mouth_press_n=0.08,
    )
    assert phig_slip_trip(grasp_slip_m=0.15, slip_tau_m=0.12, slip_armed_m=-1.0)
    assert not phig_slip_trip(grasp_slip_m=0.15, slip_tau_m=0.12, slip_armed_m=0.15)
    assert not phig_slip_trip(
        grasp_slip_m=0.17, slip_tau_m=0.12, slip_armed_m=0.15, retrip_delta_m=0.05
    )
    assert phig_slip_trip(
        grasp_slip_m=0.21, slip_tau_m=0.12, slip_armed_m=0.15, retrip_delta_m=0.05
    )

    st = PhigState()
    st = phig_select_mode(
        st,
        hybrid_mode="spiral",
        r_cmd_m=0.003,
        contact_resid_n=0.1,
        force_hole_cue=False,
        grasp_slip_m=0.0,
        axis_err_deg=10.0,
        cfg=PhigConfig(),
    )
    assert st.mode == "mouth_press"
    assert phig_axial_override(mode="mouth_press", ax_step=0.0001) >= 0.00045

    cfg_path = REPO / "configs/scheme_l3/S2_trackB_phig.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    a = cfg["approach"]
    assert a["surface_tip_baseline"] == "track_b_phig"
    assert a.get("surface_tip_phig_enable") is True
    assert float(a["surface_planned_priv_enter_max_axis_err_deg"]) == 25.0
    assert float(a["surface_planned_priv_enter_max_float_m"]) == 0.004
    assert a.get("surface_tip_oracle_mouth") in (None, False)

    print("smoke_track_b_phig: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
