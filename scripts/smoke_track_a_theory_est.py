#!/usr/bin/env python3
"""Unit smoke for Track-A TEC-slim ExtrinsicTipStateEKF (no MuJoCo, no ep RATE).

Checks process + measurement theory:
1. Rigid C=I: tip tracks wrist planar motion through predict.
2. Contact measurement pulls tip_hat toward z_c (not wrist+offset).
3. Wrench lever-arm soft measurement updates tip without tip GT.
4. Slip reset inflates C, does not snap tip to wrist.
5. Config points at theory backend; SUCCESS_STANDARD knobs untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pci.tip_theory_estimator import (
    ExtrinsicTipStateEKF,
    TheoryEstConfig,
    is_track_a_theory_mode,
)


def main() -> int:
    n = np.array([0.0, 0.0, 1.0])
    cfg = TheoryEstConfig(process_tip=1e-8, process_c=1e-8, meas_wrench_var=1e-6)
    ekf = ExtrinsicTipStateEKF(cfg)

    tip0 = np.array([0.10, 0.02, 0.05])
    w0 = np.array([0.08, 0.00, 0.05])
    est = ekf.reset(wrist_pos=w0, plane_n=n, tip_seed=tip0)
    assert float(np.linalg.norm(est.tip_hat[:2] - tip0[:2])) < 1e-9

    # Predict: wrist +Δx with C=I → tip +Δx
    w1 = w0 + np.array([0.01, 0.0, 0.0])
    est = ekf.step(wrist_pos=w1, plane_n=n)
    assert abs(float(est.tip_hat[0] - (tip0[0] + 0.01))) < 1e-4, est.tip_hat
    assert est.source == "predict"

    # Contact meas far from wrist offset — tip must move toward contact, not wrist+o
    zc = np.array([0.14, 0.05, 0.05])
    est = ekf.step(
        wrist_pos=w1,
        plane_n=n,
        contact_tip=zc,
        contact_n=1,
        contact_force_n=0.2,
    )
    assert est.source.startswith("contact"), est.source
    assert float(np.linalg.norm(est.tip_hat[:2] - zc[:2])) < 0.01, est.tip_hat

    # Wrench soft meas: lever toward (0.12, 0.02)
    site = np.array([0.10, 0.00, 0.05])
    force = np.array([0.0, 0.0, 0.2])
    torque = np.array([0.004, -0.004, 0.0])  # r≈(0.02,0.02)
    ekf2 = ExtrinsicTipStateEKF(cfg)
    ekf2.reset(wrist_pos=site, plane_n=n, tip_seed=site)
    est2 = ekf2.step(
        wrist_pos=site,
        plane_n=n,
        force_xyz=force,
        torque_xyz=torque,
    )
    assert "wrench" in est2.source, est2.source
    assert float(est2.tip_hat[0]) > site[0], est2.tip_hat

    # Slip reset does not teleport tip to wrist
    tip_before = ekf.tip_world(n).copy()
    ekf.slip_reset()
    tip_after = ekf.tip_world(n)
    assert float(np.linalg.norm(tip_before[:2] - tip_after[:2])) < 1e-9
    assert ekf.just_slipped is True

    assert is_track_a_theory_mode("track_a_theory_est")
    assert not is_track_a_theory_mode("track_a_est_oracle")

    yml = yaml.safe_load(
        (REPO / "configs/scheme_l3/S2_trackA_theory_est.yaml").read_text()
    )
    a = yml["approach"]
    assert a.get("surface_tip_est_backend") == "theory_ekf"
    assert a.get("surface_tip_baseline") == "track_a_theory_est"
    # SUCCESS_STANDARD knobs unchanged (locked)
    assert float(a.get("surface_planned_priv_enter_max_axis_err_deg", 25)) == 25.0

    print("[ok] track_a_theory_est unit smoke (equations only; no ep RATE)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
