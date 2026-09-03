#!/usr/bin/env python3
"""Unit smoke for fail-driven HardHQP (no MuJoCo).

Checks: seat freezes path; upright before path; path ignores tip lag; no tip GT.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pci.fail_driven_hard_hqp import (
    HardHqpConfig,
    hard_hqp_select_level,
    hard_hqp_wrist_delta,
    is_hard_hqp_mode,
)
from pci.tip_tracking_baselines import baseline_hold_r


def main() -> int:
    assert is_hard_hqp_mode("fail_driven_hard_hqp")
    n = np.array([0.0, 0.0, 1.0])
    site = np.array([0.10, 0.00, 0.05])
    tip = np.array([0.14, 0.03, 0.05])  # lagged tip must not steer
    path = np.array([0.08, 0.00, 0.05])
    press = -n
    cfg = HardHqpConfig(f_seat_n=0.05, upright_soft_deg=16.0, max_step_m=0.02)

    # SEAT level: low contact → no planar toward path.
    hold_s, meta_s = hard_hqp_wrist_delta(
        site_xyz=site,
        path_target=path,
        normal=n,
        press_ax=press,
        contact_resid_n=0.01,
        axis_err_deg=5.0,
        r_cmd_m=0.02,
        ax_step=0.0,
        cfg=cfg,
    )
    assert meta_s["hqp_level"] == "seat", meta_s
    assert abs(float((hold_s - site)[0])) < 1e-9, hold_s - site
    assert float(meta_s["ax_step_m"]) > 0.0

    # UPRIGHT: seat ok, axis large → pivot, path frozen.
    peg = np.array([0.5, 0.0, 0.866])  # ~30° tilt
    hold_u, meta_u = hard_hqp_wrist_delta(
        site_xyz=site,
        path_target=path,
        normal=n,
        press_ax=press,
        contact_resid_n=0.08,
        axis_err_deg=30.0,
        peg_axis=peg,
        hole_axis=n,
        r_cmd_m=0.02,
        cfg=cfg,
    )
    assert meta_u["hqp_level"] == "upright", meta_u
    assert meta_u.get("path_scale", 1.0) == 0.0

    # PATH: tip lag must not flip wrist vs known waypoint (−x).
    hold_p, meta_p = hard_hqp_wrist_delta(
        site_xyz=site,
        path_target=path,
        normal=n,
        press_ax=press,
        contact_resid_n=0.08,
        axis_err_deg=5.0,
        r_cmd_m=0.02,
        cfg=cfg,
    )
    assert meta_p["hqp_level"] == "path", meta_p
    assert float((hold_p - site)[0]) < 0.0, hold_p - site
    assert meta_p.get("privileged") is False

    hold_r, meta2, _ax = baseline_hold_r(
        "fail_driven_hard_hqp",
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
        contact_resid_n=0.08,
        f_des_n=0.04,
        force_gate_enable=False,
        hard_hqp_cfg=cfg,
        axis_err_deg=5.0,
        peg_axis=n,
        r_cmd_m=0.02,
    )
    assert meta2["path"] == "fail_driven_hard_hqp"
    assert float((hold_r - site)[0]) < 0.0

    # SUCCESS_STANDARD knobs untouched in config stub.
    yml = yaml.safe_load(
        (REPO / "configs/scheme_l3/S2_failDriven_hardHQP.yaml").read_text()
    )
    a = yml["approach"]
    assert float(a["surface_planned_priv_enter_max_axis_err_deg"]) == 25.0
    assert float(a["surface_planned_priv_enter_max_along_m"]) == 0.100
    assert a["surface_tip_baseline"] == "fail_driven_hard_hqp"
    assert a.get("surface_tip_hard_hqp_enable") is True

    st = hard_hqp_select_level(
        contact_resid_n=0.08, axis_err_deg=5.0, r_cmd_m=0.003, cfg=cfg
    )
    assert st.level == "mouth"

    print("smoke_fail_driven_hard_hqp: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
