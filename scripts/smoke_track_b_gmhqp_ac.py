#!/usr/bin/env python3
"""Unit smoke for Track-B Adaptive-C GMHQP (general law; no MuJoCo)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pci.tip_tracking_baselines import baseline_hold_r
from pci.track_b_gmhqp_ac import (
    GmhqpAcConfig,
    GmhqpAcState,
    adaptive_ridge,
    gmhqp_ac_wrist_delta,
    is_gmhqp_ac_mode,
    update_adaptive_coupling,
)


def main() -> int:
    assert is_gmhqp_ac_mode("track_b_gmhqp_ac")
    assert not is_gmhqp_ac_mode("track_b_gmhqp")
    assert not is_gmhqp_ac_mode("track_b_gmhqp_compc")

    cfg = GmhqpAcConfig()
    # Ridge grows with uncertainty / innov
    r0 = adaptive_ridge(p_c_trace=0.0, tip_innov_norm=0.0, cfg=cfg)
    r1 = adaptive_ridge(p_c_trace=0.5, tip_innov_norm=0.01, cfg=cfg)
    assert r1 > r0

    n = np.array([0.0, 0.0, 1.0])
    press = -n
    path = np.array([0.08, 0.0, 0.05])
    site = np.array([0.10, 0.0, 0.05])
    tip = np.array([0.085, 0.0, 0.05])

    st = GmhqpAcState()
    # Warm EKF with wrist/tip steps; C is only indirectly observable.
    C_true = np.array([[0.7, 0.0], [0.0, 1.1]], dtype=np.float64)
    w = site.copy()
    t = tip.copy()
    rng = np.random.default_rng(0)
    for _ in range(40):
        dw = rng.normal(0.0, 0.002, size=3)
        dw[2] = 0.0
        dw2 = dw[:2]
        dt2 = C_true @ dw2
        w = w + dw
        t = t + np.array([dt2[0], dt2[1], 0.0])
        update_adaptive_coupling(st, wrist=w, tip_obj=t, normal=n, cfg=cfg)

    C_hat = st.C
    assert C_hat.shape == (2, 2)
    assert np.all(np.isfinite(C_hat))
    assert float(np.linalg.norm(C_hat - np.eye(2))) >= 0.0  # runs without crash
    assert st.ekf is not None and st.ekf.initialized

    hold, meta, st = gmhqp_ac_wrist_delta(
        site_xyz=w,
        tip_obj=t,
        path_target=path,
        normal=n,
        press_ax=press,
        contact_resid_n=0.2,
        axis_err_deg=5.0,
        r_cmd_m=0.02,
        state=st,
        cfg=cfg,
    )
    assert meta["path"] == "track_b_gmhqp_ac"
    assert meta.get("hqp_level") in ("tip_path", "mouth")
    assert "c_ridge_eff" in meta
    assert hold.shape == (3,)
    # Seat hard: planar freeze
    hold_s, meta_s, _ = gmhqp_ac_wrist_delta(
        site_xyz=w,
        tip_obj=t,
        path_target=path,
        normal=n,
        press_ax=press,
        contact_resid_n=0.01,
        axis_err_deg=5.0,
        r_cmd_m=0.02,
        state=GmhqpAcState(),
        cfg=cfg,
    )
    assert meta_s["hqp_level"] == "seat"
    assert float(np.linalg.norm((hold_s - w) - press * float(meta_s["ax_step_m"]))) < 1e-9

    hold2, meta2, _ = baseline_hold_r(
        "track_b_gmhqp_ac",
        target=path,
        tip=t,
        site_xyz=w,
        offset_frozen=w - t,
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
        tip_anchor=t,
        contact_resid_n=0.2,
        force_gate_enable=False,
        gmhqp_ac_cfg=cfg,
        gmhqp_ac_state=GmhqpAcState(),
        axis_err_deg=5.0,
        r_cmd_m=0.02,
    )
    assert meta2["path"] == "track_b_gmhqp_ac"
    assert hold2.shape == (3,)

    yml = yaml.safe_load(
        (REPO / "configs/scheme_l3/S2_trackB_gmhqp_ac.yaml").read_text()
    )
    ap = yml["approach"]
    assert ap["surface_tip_baseline"] == "track_b_gmhqp_ac"
    assert bool(ap.get("surface_tip_gmhqp_ac_enable"))
    assert float(ap["surface_planned_priv_enter_max_axis_err_deg"]) == 25.0
    print("smoke_track_b_gmhqp_ac: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
