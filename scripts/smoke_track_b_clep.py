#!/usr/bin/env python3
"""Unit smoke for Track-B CLEP (no MuJoCo).

Checks: contact estimate ignores tip GT; wrist moves with contact→path error;
SUCCESS_STANDARD knobs untouched in config.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pci.tip_tracking_baselines import baseline_hold_r
from pci.track_b_clep import (
    ClepConfig,
    ClepState,
    clep_wrist_delta,
    estimate_contact_from_wrench,
    is_clep_mode,
)


def main() -> int:
    n = np.array([0.0, 0.0, 1.0])
    site = np.array([0.10, 0.00, 0.05])
    tip = np.array([0.12, 0.02, 0.05])  # lagged tip (must not drive law)
    path = np.array([0.08, 0.00, 0.05])
    press = -n
    # Contact under tip: lever arm from site toward tip, normal force +z
    force = np.array([0.0, 0.0, 0.2])
    # τ = r × F with r = tip-site ≈ (0.02, 0.02, 0); F=(0,0,0.2)
    # τ = (0.02,0.02,0)×(0,0,0.2) = (0.004, -0.004, 0)
    torque = np.array([0.004, -0.004, 0.0])

    c, em = estimate_contact_from_wrench(
        site_xyz=site,
        force_xyz=force,
        torque_xyz=torque,
        normal=n,
        tip_anchor=site,
        cfg=ClepConfig(),
    )
    assert c is not None, em
    assert float(np.linalg.norm(c[:2] - tip[:2])) < 0.005, (c, tip)

    hold, meta, st = clep_wrist_delta(
        site_xyz=site,
        path_target=path,
        normal=n,
        press_ax=press,
        contact_resid_n=0.08,
        r_cmd_m=0.012,
        ax_step=0.0002,
        force_xyz=force,
        torque_xyz=torque,
        tip_anchor=site,
        state=ClepState(),
        cfg=ClepConfig(k_contact=1.0, max_step_m=0.05),
    )
    d = hold - site
    # Contact near tip; path is −x of tip → planar Δw should go −x, not toward tip lag +y alone
    assert float(d[0]) < 0.0, d
    assert meta["path"] == "track_b_clep"
    assert meta.get("privileged") is False
    assert meta.get("clep_valid") is True
    assert is_clep_mode("track_b_clep")

    hold_r, meta2, ax = baseline_hold_r(
        "track_b_clep",
        target=path,
        tip=tip,
        site_xyz=site,
        offset_frozen=site - tip,
        coupling=None,
        spiral_n=n,
        press_ax=press,
        ax_step=0.0002,
        track=1.0,
        max_step_m=0.05,
        along_now_m=0.09,
        along0_m=0.09,
        hole_axis=n,
        along_slack_m=0.01,
        tip_anchor=site,
        contact_resid_n=0.08,
        f_des_n=0.04,
        force_gate_enable=False,
        wrench_xyz=force,
        wrench_tau_xyz=torque,
        clep_cfg=ClepConfig(k_contact=1.0, max_step_m=0.05),
        clep_state=ClepState(),
    )
    assert meta2["path"] == "track_b_clep"
    assert float((hold_r - site)[0]) < 0.0

    cfg_path = REPO / "configs/scheme_l3/S2_trackB_clep.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    a = cfg["approach"]
    assert a["surface_tip_baseline"] == "track_b_clep"
    assert a.get("surface_tip_clep_enable") is True
    assert float(a["surface_planned_priv_enter_max_axis_err_deg"]) == 25.0
    assert float(a["surface_planned_priv_enter_max_float_m"]) == 0.004
    assert a.get("surface_tip_oracle_mouth") in (None, False)

    print("smoke_track_b_clep: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
