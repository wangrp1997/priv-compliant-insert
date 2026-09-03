"""Unit tests: tip_factor_graph residuals + window NLS (no MuJoCo).

General law: docs/TRACK_A_FACTOR_GRAPH_TIP.md — REUSE Kim TEC factors.
"""

from __future__ import annotations

import numpy as np

from pci.tip_factor_graph import (
    TEC_FACTOR_NAMES,
    TipContactFactorGraph,
    TipFGConfig,
    TipFGFrame,
    is_tip_factor_graph_mode,
    residual_contact_motion,
    residual_energy_elastic,
    residual_extrinsic_contact,
    residual_object_fixed_contact,
    residual_pen_hinge,
    residual_pose_diff,
    residual_torq_line,
    residual_torq_point,
    residual_wrench_lever_tip,
)


def test_mode_names() -> None:
    assert is_tip_factor_graph_mode("track_a_tip_fg")
    assert is_tip_factor_graph_mode("tip_factor_graph")
    assert not is_tip_factor_graph_mode("track_a_tec_joint_gmhqp")


def test_factor_registry_covers_tec_spirit() -> None:
    needed = {
        "ContactMotion",
        "PoseDiff",
        "DispDiff",
        "Wrench",
        "TorqPoint",
        "TorqLine",
        "EnergyElastic",
        "PenHinge",
        "ExtrinsicContact",
    }
    assert needed.issubset(set(TEC_FACTOR_NAMES))


def test_contact_motion_zero_when_consistent() -> None:
    c = np.array([[0.8, 0.1], [-0.05, 0.9]], dtype=np.float64)
    dw = np.array([0.002, -0.001], dtype=np.float64)
    t0 = np.array([0.01, 0.02], dtype=np.float64)
    t1 = t0 + c @ dw
    r = residual_contact_motion(t0, t1, c, dw)
    assert float(np.linalg.norm(r)) < 1e-12


def test_pose_diff_and_extrinsic() -> None:
    a = np.array([0.1, -0.2])
    b = np.array([0.1, -0.2])
    assert float(np.linalg.norm(residual_pose_diff(a, b))) < 1e-15
    assert float(np.linalg.norm(residual_extrinsic_contact(a, b))) < 1e-15


def test_object_fixed_contact_sticky() -> None:
    tip0 = np.array([0.0, 0.0])
    tip1 = np.array([0.01, 0.0])
    c0 = tip0 + np.array([0.002, 0.0])
    c1 = tip1 + np.array([0.002, 0.0])
    r = residual_object_fixed_contact(tip0, tip1, c0, c1)
    assert float(np.linalg.norm(r)) < 1e-12


def test_torq_point_and_line() -> None:
    r = np.array([0.0, 0.0, -0.05])
    f = np.array([0.0, 0.0, -2.0])
    # Pure force through contact → zero torque
    m = np.cross(r, f)
    tp = residual_torq_point(r, f, m)
    assert float(np.linalg.norm(tp)) < 1e-12
    tl = residual_torq_line(r, f, m, np.array([1.0, 0.0, 0.0]))
    assert abs(float(tl[0])) < 1e-12


def test_energy_and_pen_hinge() -> None:
    w = np.array([0.1, 0.0, 0.0, 1.0, 0.0, 2.0])
    k = np.ones(6)
    e = residual_energy_elastic(w, k)
    assert e.shape == (6,)
    assert float(residual_pen_hinge(0.002, 0.001)[0]) == 0.0
    assert float(residual_pen_hinge(0.0005, 0.001)[0]) == 0.0005


def test_window_nls_tracks_contact_no_mujoco() -> None:
    """Synthetic planar contact: FG tip should pull toward contact meas."""
    n = np.array([0.0, 0.0, 1.0])
    cfg = TipFGConfig(window=5, max_iters=6)
    fg = TipContactFactorGraph(cfg)
    # Seed near wrist; contact offset in +x
    wrist0 = np.array([0.0, 0.0, 0.1])
    contact0 = np.array([0.008, 0.0, 0.0])
    out = fg.reset(wrist_pos=wrist0, plane_n=n, tip_seed=wrist0)
    assert out.tip_hat.shape == (3,)

    tip_last = None
    for k in range(6):
        wrist = wrist0 + np.array([0.001 * k, 0.0, 0.0])
        contact = contact0 + np.array([0.001 * k, 0.0, 0.0])
        # Lever consistent with contact under pure normal force at tip
        f = np.array([0.0, 0.0, -1.0])
        r = contact - wrist
        tau = np.cross(r, f)
        est = fg.step(
            TipFGFrame(
                wrist_pos=wrist,
                plane_n=n,
                geom_tip=contact + np.array([0.0005, 0.0, 0.0]),
                contact_tip=contact,
                contact_n=1,
                force_xyz=f,
                torque_xyz=tau,
                tip_gt=contact,
            )
        )
        tip_last = est.tip_hat
        assert np.isfinite(est.cost)
        assert est.meta["tip_est_backend"] == "tip_factor_graph"
        assert est.meta["tip_est_uses_peg_xpos"] is False

    assert tip_last is not None
    # Should be within a few mm of true contact after updates
    err = float(np.linalg.norm((tip_last - contact0 - np.array([0.005, 0.0, 0.0]))[:2]))
    assert err < 0.005, f"planar tip err {err}"
    assert float(est.tip_err_to_gt_m) < 0.005


def test_wrench_lever_residual_shape() -> None:
    tip = np.array([0.01, 0.0, 0.0])
    wrist = np.array([0.0, 0.0, 0.1])
    f = np.array([0.0, 0.0, -1.0])
    tau = np.cross(tip - wrist, f)
    n = np.array([0.0, 0.0, 1.0])
    r = residual_wrench_lever_tip(tip, wrist, f, tau, n)
    assert r.shape == (2,)
    assert float(np.linalg.norm(r)) < 1e-9
