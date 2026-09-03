#!/usr/bin/env python3
"""Unit smoke for Track-B Escande GMHQP⊕planar_C (REUSE, no MuJoCo).

Checks: mode hook; tip_task when tip on path; switch to planar_c_fb after
tip residual plateau; config baseline + SUCCESS_STANDARD knobs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pci.tip_tracking_baselines import baseline_hold_r
from pci.track_b_hybrid_gmhqp import (
    HybridGmhqpConfig,
    HybridGmhqpState,
    hybrid_gmhqp_wrist_delta,
    is_hybrid_gmhqp_mode,
)


def main() -> int:
    assert is_hybrid_gmhqp_mode("track_b_hybrid_gmhqp")
    n = np.array([0.0, 0.0, 1.0])
    press = -n
    path = np.array([0.08, 0.0, 0.05])
    site = np.array([0.10, 0.0, 0.05])
    tip_off = np.array([0.10, 0.02, 0.05])  # ~22 mm tip err

    cfg = HybridGmhqpConfig(sat_err_m=0.008, sat_frames=5, sat_progress_m=0.01)
    st = HybridGmhqpState()
    # Force plateau: constant large tip err for sat_frames
    for _ in range(12):
        hold, meta, st = hybrid_gmhqp_wrist_delta(
            site_xyz=site,
            tip_obj=tip_off,
            path_target=path,
            normal=n,
            press_ax=press,
            contact_resid_n=0.2,
            axis_err_deg=5.0,
            r_cmd_m=0.012,
            state=st,
            cfg=cfg,
        )
    assert meta["path"] == "track_b_hybrid_gmhqp"
    assert meta["privileged"] is False
    assert st.task == "planar_c_fb", st.task
    assert meta["hybrid_task"] == "planar_c_fb"
    assert float(np.linalg.norm((hold - site)[:2])) > 1e-4

    # Healthy tip-on-path stays tip_task (no switch)
    st2 = HybridGmhqpState()
    hold_on, meta_on, st2 = hybrid_gmhqp_wrist_delta(
        site_xyz=site,
        tip_obj=path.copy(),
        path_target=path,
        normal=n,
        press_ax=press,
        contact_resid_n=0.2,
        axis_err_deg=5.0,
        r_cmd_m=0.012,
        state=st2,
        cfg=HybridGmhqpConfig(),
    )
    assert st2.task == "tip_task"
    assert float(np.linalg.norm((hold_on - site)[:2])) < 1e-4

    hold_b, meta_b, _ = baseline_hold_r(
        "track_b_hybrid_gmhqp",
        target=path,
        tip=tip_off,
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
        hybrid_gmhqp_cfg=HybridGmhqpConfig(),
        hybrid_gmhqp_state=HybridGmhqpState(),
        r_cmd_m=0.012,
        force_gate_enable=False,
        axis_err_deg=5.0,
    )
    assert meta_b["path"] == "track_b_hybrid_gmhqp"

    yml = yaml.safe_load(
        (REPO / "configs/scheme_l3/S2_trackB_hybrid_gmhqp.yaml").read_text()
    )
    a = yml["approach"]
    assert a["surface_tip_baseline"] == "track_b_hybrid_gmhqp"
    assert a["surface_tip_hybrid_gmhqp_enable"] is True
    assert a["surface_planned_priv_enter_max_axis_err_deg"] == 25.0
    assert a["surface_planned_priv_enter_max_along_m"] == 0.100
    assert yml["_pk"].get("reuse") is True

    print("smoke_track_b_hybrid_gmhqp: OK (reuse Escande GMHQP⊕planar_C)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
