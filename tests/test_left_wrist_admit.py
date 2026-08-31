"""Unit tests for left-wrist F/T admittance (no MuJoCo)."""

from __future__ import annotations

import numpy as np

from pci.compliant.left_wrist_admit import LeftWristAdmitConfig, LeftWristAdmitController


def test_yields_opposite_to_force() -> None:
    cfg = LeftWristAdmitConfig(enable=True, k_force=0.001, f_deadband_n=0.0, k_torque=0.0)
    c = LeftWristAdmitController(cfg)
    w = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    d = c.step(w)
    assert d[0] < 0.0
    assert abs(d[1]) < 1e-9


def test_torque_gives_lateral() -> None:
    cfg = LeftWristAdmitConfig(
        enable=True, k_force=0.0, k_torque=0.01, tau_deadband_nm=0.0, f_deadband_n=0.0
    )
    c = LeftWristAdmitController(cfg)
    # tau along x, approach along z → tip in ±y
    w = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    d = c.step(w, approach_axis=np.array([0.0, 0.0, 1.0]))
    assert abs(d[1]) > abs(d[0])
    assert abs(d[2]) < 1e-9


def test_total_travel_cap() -> None:
    cfg = LeftWristAdmitConfig(
        enable=True,
        k_force=0.01,
        k_torque=0.0,
        f_deadband_n=0.0,
        max_step_m=0.001,
        max_total_m=0.0015,
    )
    c = LeftWristAdmitController(cfg)
    w = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    acc = np.zeros(3)
    for _ in range(20):
        acc = acc + c.step(w)
    assert float(np.linalg.norm(acc)) <= 0.0015 + 1e-9
    assert float(np.linalg.norm(c.step(w))) < 1e-12
