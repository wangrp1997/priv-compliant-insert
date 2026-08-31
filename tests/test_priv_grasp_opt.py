"""Unit tests for privileged grasp-opt QP (no MuJoCo)."""

from __future__ import annotations

import numpy as np

from pci.compliant.priv_grasp_opt import (
    PrivGraspOptConfig,
    PrivGraspOptController,
    build_grasp_matrix,
    solve_contact_forces_qp,
)
from pci.priv_geom import PrivGraspGeom


def _geom() -> PrivGraspGeom:
    I = np.eye(3)
    tips_r = np.array(
        [[0.02, 0.02, 0.0], [0.02, -0.02, 0.0], [-0.02, 0.02, 0.0], [-0.02, -0.02, 0.0]],
        dtype=np.float64,
    )
    tips_l = tips_r + np.array([0.0, 0.25, 0.0])
    peg = np.array([0.4, 0.0, 0.9])
    tray = np.array([0.4, 0.25, 0.9])
    return PrivGraspGeom(
        peg_pos=peg,
        peg_rot=I.copy(),
        tray_pos=tray,
        tray_rot=I.copy(),
        right_wrist_pos=peg + np.array([0.0, 0.0, 0.05]),
        right_wrist_rot=I.copy(),
        left_wrist_pos=tray + np.array([0.0, 0.0, 0.05]),
        left_wrist_rot=I.copy(),
        right_tip_pos=tips_r + peg,
        left_tip_pos=tips_l + tray - tips_r[0] * 0,  # already absolute-ish
    )


def test_qp_returns_positive_forces() -> None:
    tips = np.array(
        [[0.02, 0.02, 0.0], [0.02, -0.02, 0.0], [-0.02, 0.02, 0.0], [-0.02, -0.02, 0.0]]
    )
    f12 = np.zeros(12)
    for i in range(4):
        f12[i * 3 : i * 3 + 3] = -tips[i] / (np.linalg.norm(tips[i]) + 1e-9)
    G, _ = build_grasp_matrix(tips, f12, np.zeros(3), mu=0.4, n_cone=4, torque_weight=5.0)
    assert G.shape == (6, 16)
    w = np.array([0.0, 0.0, 0.0, 0.05, 0.0, 0.0])
    mags = solve_contact_forces_qp(
        G, w, f_squeeze=2.0, n_cone=4, f_min=0.3, f_max=8.0, reg=0.2
    )
    assert mags.shape == (4,)
    assert float(mags.min()) >= 0.3 - 1e-9


def test_rel_pose_error_produces_hand_delta() -> None:
    cfg = PrivGraspOptConfig(enable=True, k_rel_rot=5.0, k_admit=0.002)
    ctrl = PrivGraspOptController(cfg)
    g0 = _geom()
    hold = np.ones(16) * 0.5
    ctrl.reset(g0, hold, hold)
    # Twist peg relative to tray.
    from scipy.spatial.transform import Rotation as R

    g1 = PrivGraspGeom(
        peg_pos=g0.peg_pos.copy(),
        peg_rot=R.from_rotvec([0.15, 0.0, 0.0]).as_matrix(),
        tray_pos=g0.tray_pos.copy(),
        tray_rot=g0.tray_rot.copy(),
        right_wrist_pos=g0.right_wrist_pos.copy(),
        right_wrist_rot=g0.right_wrist_rot.copy(),
        left_wrist_pos=g0.left_wrist_pos.copy(),
        left_wrist_rot=g0.left_wrist_rot.copy(),
        right_tip_pos=g0.right_tip_pos.copy(),
        left_tip_pos=g0.left_tip_pos.copy(),
    )
    f12 = np.zeros(12)
    for i in range(4):
        f12[i * 3 + 2] = -1.0
    out = ctrl.step(g1, f12, f12)
    assert out.rel_rot_err_rad > 0.05
    assert float(np.linalg.norm(out.delta_right_hand16)) > 0.0
    assert float(np.linalg.norm(out.delta_left_hand16)) > 0.0


def test_zero_rel_rot_still_squeezes() -> None:
    """Search + frozen left: k_rel_rot=0 must still admit toward squeeze."""
    cfg = PrivGraspOptConfig(enable=True, k_rel_rot=0.0, k_admit=0.002, f_squeeze_n=2.5)
    ctrl = PrivGraspOptController(cfg)
    g0 = _geom()
    hold = np.ones(16) * 0.5
    ctrl.reset(g0, hold, hold)
    f12 = np.zeros(12)  # measured ~0 → want close toward squeeze
    out = ctrl.step(g0, f12, f12)
    assert float(np.linalg.norm(out.delta_right_hand16)) > 0.0
    assert float(np.linalg.norm(out.delta_left_hand16)) > 0.0
    assert float(out.right_f_des.min()) >= cfg.f_min_n - 1e-9
