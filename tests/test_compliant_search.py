"""Unit tests for dual-arm compliant spiral search."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from pci.compliant.search import CompliantSearchConfig, CompliantSearchController
from pci.task_frame import TaskFrame


def _frame() -> TaskFrame:
    rot = R.from_euler("xyz", [0.1, -0.05, 0.2]).as_matrix()
    return TaskFrame(
        origin_world=np.array([0.4, 0.0, 0.85]),
        rot_world_tool=rot,
        approach_axis=rot[:, 2] / np.linalg.norm(rot[:, 2]),
    )


def test_high_axial_force_retreats_not_pushes() -> None:
    cfg = CompliantSearchConfig(
        push_force_n=3.0,
        f_axial_max_n=6.0,
        retreat_step_m=0.001,
        min_search_steps=1,
        left_spiral_share=0.3,
    )
    ctrl = CompliantSearchController(cfg)
    frame = _frame()
    ctrl.reset(frame, np.zeros(6), already_on_surface=True)
    wrench = np.zeros(6)
    wrench[:3] = frame.approach_axis * 12.0
    out = ctrl.step(frame, wrench, frame.origin_world.copy())
    assert out.reason == "force_retreat"
    axial = float(np.dot(out.delta_xyz, frame.approach_axis))
    assert axial * ctrl._push_sign < 0.0
    assert out.delta_left_xyz is not None


def test_left_led_share_moves_tray_more() -> None:
    """Right-led: right lateral >= left lateral when share is small."""
    cfg = CompliantSearchConfig(
        min_search_steps=1,
        left_spiral_share=0.12,
        right_lat_scale=1.0,
        left_axial_share=0.0,
        hold_press_m=0.0,
        spiral_step_rad=0.4,
        spiral_pitch_m=0.002,
        admittance_k_xy=0.0,
        hole_detect_fz_drop_n=100.0,
        hole_f_max_n=0.05,
        contact_min_n=99.0,
        spiral_try_insert_theta=1e9,
        priv_assist=False,
        priv_seek_step_m=0.0,
        max_search_steps=500,
    )
    ctrl = CompliantSearchController(cfg)
    frame = _frame()
    wrist = frame.origin_world.copy()
    ctrl.reset(frame, np.zeros(6), wrist_xyz=wrist, already_on_surface=True)
    out = ctrl.step(frame, np.zeros(6), wrist)
    assert out.reason == "searching"
    assert out.delta_left_xyz is not None
    r_lat = out.delta_xyz - frame.approach_axis * float(
        np.dot(out.delta_xyz, frame.approach_axis)
    )
    l_lat = out.delta_left_xyz - frame.approach_axis * float(
        np.dot(out.delta_left_xyz, frame.approach_axis)
    )
    assert float(np.linalg.norm(r_lat)) + 1e-9 >= float(np.linalg.norm(l_lat))
    if float(np.linalg.norm(r_lat)) > 1e-8 and float(np.linalg.norm(l_lat)) > 1e-8:
        assert float(np.dot(r_lat, l_lat)) < 0.0


def test_force_enough_stops_press() -> None:
    """|Fz| >= contact_f_des → axial press = 0 (no hold*0.6 / near_hole boost)."""
    cfg = CompliantSearchConfig(
        min_search_steps=1,
        max_search_steps=500,
        hold_press_m=0.00035,
        contact_f_des_n=0.45,
        contact_press_gain=0.0,
        near_hole_press_m=0.00050,
        admittance_k_xy=0.0,
        hole_detect_fz_drop_n=100.0,
        hole_f_max_n=0.05,
        contact_min_n=99.0,
        spiral_try_insert_theta=1e9,
        priv_assist=False,
        priv_seek_step_m=0.0,
        left_spiral_share=0.0,
        left_axial_share=0.0,
        wrench_lpf_alpha=1.0,
        priv_enter_lat_m=0.0045,
    )
    ctrl = CompliantSearchController(cfg)
    frame = _frame()
    wrist = frame.origin_world.copy()
    ctrl.reset(frame, np.zeros(6), wrist_xyz=wrist, already_on_surface=True)
    wrench = np.zeros(6)
    wrench[:3] = frame.approach_axis * 1.0  # |Fz_tool| ≈ 1.0 > 0.45
    out = ctrl.step(frame, wrench, wrist, priv_lat_m=0.002)
    assert out.reason == "searching"
    axial = float(np.dot(out.delta_xyz, frame.approach_axis))
    assert abs(axial) < 1e-9


def test_dual_spiral_expands_relative() -> None:
    cfg = CompliantSearchConfig(
        min_search_steps=1,
        left_spiral_share=0.35,
        left_axial_share=0.0,
        hold_press_m=0.0,
        spiral_step_rad=0.35,
        spiral_pitch_m=0.002,
        spiral_radius_max_m=0.015,
        admittance_k_xy=0.0,
        max_admit_step_m=0.003,
        max_lat_step_m=0.003,
        hole_detect_fz_drop_n=100.0,
        hole_f_max_n=0.05,
        contact_min_n=99.0,
        max_search_steps=2000,
        spiral_try_insert_theta=1e9,
        priv_assist=False,
    )
    ctrl = CompliantSearchController(cfg)
    frame = _frame()
    wrench = np.zeros(6)
    right = frame.origin_world.copy()
    left = frame.origin_world.copy() + np.array([0.05, 0.0, 0.0])
    ctrl.reset(frame, wrench, wrist_xyz=right, already_on_surface=True)
    for _ in range(200):
        out = ctrl.step(frame, wrench, right)
        assert out.reason == "searching"
        right = right + out.delta_xyz
        if out.delta_left_xyz is not None:
            left = left + out.delta_left_xyz
    rel = right - left
    rel = rel - frame.approach_axis * float(np.dot(rel, frame.approach_axis))
    # Relative lateral grew from initial 0.05 offset spiral motion.
    assert float(np.linalg.norm(rel - np.array([0.05, 0.0, 0.0]))) > 0.003 or float(
        np.linalg.norm(rel)
    ) > 0.052


def test_contact_seek_then_dual_search() -> None:
    cfg = CompliantSearchConfig(
        min_search_steps=99,
        contact_min_n=5.0,
        contact_rise_n=1.5,
        contact_min_seek_steps=3,
        contact_seek_max_steps=20,
        left_spiral_share=0.3,
    )
    ctrl = CompliantSearchController(cfg)
    frame = _frame()
    wrist = frame.origin_world.copy()
    ctrl.reset(frame, np.zeros(6), wrist_xyz=wrist)
    assert ctrl.step(frame, np.zeros(6), wrist).reason == "seeking_surface"
    high = np.zeros(6)
    high[:3] = frame.approach_axis * 6.0
    hit = False
    for _ in range(cfg.contact_min_seek_steps + 2):
        if ctrl.step(frame, high, wrist).reason == "surface_contact":
            hit = True
            break
    assert hit
    out = ctrl.step(frame, high, wrist)
    assert out.reason == "searching"
    assert out.delta_left_xyz is not None


def test_search_timeout_is_failure() -> None:
    cfg = CompliantSearchConfig(
        min_search_steps=1,
        max_search_steps=2,
        contact_min_n=99.0,
        spiral_try_insert_theta=1e9,
        hole_detect_fz_drop_n=100.0,
    )
    ctrl = CompliantSearchController(cfg)
    frame = _frame()
    ctrl.reset(frame, np.zeros(6), already_on_surface=True)
    ctrl.step(frame, np.zeros(6), frame.origin_world.copy())
    out = ctrl.step(frame, np.zeros(6), frame.origin_world.copy())
    assert out.done and not out.success
    assert out.reason == "search_timeout"


def test_spiral_complete_no_longer_false_enters() -> None:
    """Spiral coverage alone must not enter INSERT (false-enter ban)."""
    cfg = CompliantSearchConfig(
        min_search_steps=1,
        max_search_steps=8,
        spiral_step_rad=1.0,
        spiral_try_insert_theta=3.0,
        spiral_try_min_radius_m=0.001,
        contact_min_n=99.0,
        hole_detect_fz_drop_n=100.0,
        priv_assist=False,
        priv_enter_require_along=False,
    )
    ctrl = CompliantSearchController(cfg)
    frame = _frame()
    wrist = frame.origin_world.copy()
    ctrl.reset(frame, np.zeros(6), wrist_xyz=wrist, already_on_surface=True)
    last = None
    for _ in range(10):
        last = ctrl.step(frame, np.zeros(6), wrist)
        if last.done:
            break
        wrist = wrist + last.delta_xyz
    assert last is not None and last.done
    assert last.reason == "search_timeout"
    assert not last.success


def test_priv_blocks_try_insert_when_far() -> None:
    cfg = CompliantSearchConfig(
        min_search_steps=1,
        max_search_steps=20,
        spiral_step_rad=2.0,
        spiral_pitch_m=0.01,
        spiral_try_min_radius_m=0.002,
        spiral_try_insert_theta=1.0,
        contact_min_n=99.0,
        hole_detect_fz_drop_n=100.0,
        priv_assist=True,
        priv_enter_lat_m=0.006,
        priv_far_lat_m=0.012,
        priv_enter_require_along=True,
        priv_enter_along_max_m=0.090,
    )
    ctrl = CompliantSearchController(cfg)
    frame = _frame()
    wrist = frame.origin_world.copy()
    ctrl.reset(frame, np.zeros(6), wrist_xyz=wrist, already_on_surface=True)
    out = ctrl.step(frame, np.zeros(6), wrist, priv_lat_m=0.020, priv_along_m=0.10)
    assert not out.done
    wrist = wrist + out.delta_xyz
    out = ctrl.step(frame, np.zeros(6), wrist, priv_lat_m=0.020, priv_along_m=0.10)
    assert not out.done
    assert out.reason == "searching"


def test_priv_along_seat_enters_not_lat_alone() -> None:
    """lat-near alone must NOT enter; along seat + lat-near does."""
    cfg = CompliantSearchConfig(
        min_search_steps=1,
        max_search_steps=50,
        spiral_step_rad=0.5,
        contact_min_n=0.0,
        hole_detect_fz_drop_n=100.0,
        hole_detect_confirm=2,
        priv_assist=True,
        priv_enter_lat_m=0.0065,
        priv_enter_require_along=True,
        priv_enter_along_max_m=0.090,
        priv_enter_require_force=False,
    )
    ctrl = CompliantSearchController(cfg)
    frame = _frame()
    wrist = frame.origin_world.copy()
    ctrl.reset(frame, np.zeros(6), wrist_xyz=wrist, already_on_surface=True)
    # Lat near but still on rim → keep searching.
    out = ctrl.step(frame, np.zeros(6), wrist, priv_lat_m=0.004, priv_along_m=0.100)
    assert not out.done
    # Seat: along drops below rim while lat near.
    out = ctrl.step(frame, np.zeros(6), wrist, priv_lat_m=0.004, priv_along_m=0.080)
    out = ctrl.step(frame, np.zeros(6), wrist, priv_lat_m=0.004, priv_along_m=0.080)
    assert out.done and out.success
    assert out.reason == "priv_along_seat"


def test_zero_priv_seek_does_not_rim_recenter() -> None:
    """priv_seek_step_m==0 must not fall back to 0.5mm rim seek."""
    cfg = CompliantSearchConfig(
        push_force_n=3.0,
        f_axial_max_n=6.0,
        retreat_step_m=0.001,
        min_search_steps=1,
        priv_assist=True,
        priv_seek_step_m=0.0,
        priv_enter_lat_m=0.006,
        priv_enter_along_max_m=0.090,
    )
    ctrl = CompliantSearchController(cfg)
    frame = _frame()
    ctrl.reset(frame, np.zeros(6), already_on_surface=True)
    wrench = np.zeros(6)
    wrench[:3] = frame.approach_axis * 12.0
    lat_vec = np.array([0.01, 0.0, 0.0], dtype=np.float64)
    out = ctrl.step(
        frame,
        wrench,
        frame.origin_world.copy(),
        priv_lat_m=0.008,
        priv_along_m=0.105,
        priv_lat_vec=lat_vec,
    )
    assert out.reason == "force_retreat"
    assert out.reason != "rim_recenter"


def test_nonzero_priv_seek_can_rim_recenter() -> None:
    cfg = CompliantSearchConfig(
        push_force_n=3.0,
        f_axial_max_n=6.0,
        retreat_step_m=0.001,
        min_search_steps=1,
        priv_assist=True,
        priv_seek_step_m=0.0005,
        priv_enter_lat_m=0.006,
        priv_enter_along_max_m=0.090,
    )
    ctrl = CompliantSearchController(cfg)
    frame = _frame()
    ctrl.reset(frame, np.zeros(6), already_on_surface=True)
    wrench = np.zeros(6)
    wrench[:3] = frame.approach_axis * 12.0
    lat_vec = frame.rot_world_tool[:, 0].copy()
    out = ctrl.step(
        frame,
        wrench,
        frame.origin_world.copy(),
        priv_lat_m=0.008,
        priv_along_m=0.105,
        priv_lat_vec=lat_vec,
    )
    assert out.reason == "rim_recenter"


def test_force_enter_requires_spiral_min_radius() -> None:
    """Force drop before spiral_try_min_radius must not enter INSERT."""
    cfg = CompliantSearchConfig(
        min_search_steps=1,
        max_search_steps=200,
        spiral_step_rad=0.05,
        spiral_pitch_m=0.001,
        spiral_radius_start_m=0.0,
        spiral_try_min_radius_m=0.020,
        contact_min_n=0.0,
        hole_detect_fz_drop_n=0.3,
        hole_f_max_n=12.0,
        hole_fz_max_after_drop_n=12.0,
        hole_fxy_max_n=12.0,
        hole_detect_confirm=2,
        priv_assist=False,
        priv_enter_require_force=True,
        priv_enter_require_along=False,
        wrench_lpf_alpha=1.0,
    )
    ctrl = CompliantSearchController(cfg)
    frame = _frame()
    wrist = frame.origin_world.copy()
    # Peak contact first, then unload (force drop).
    peak = np.zeros(6)
    peak[:3] = frame.approach_axis * 5.0
    ctrl.reset(frame, peak, wrist_xyz=wrist, already_on_surface=True)
    ctrl._peak_abs_fz = 5.0
    ctrl._abs_fz_ema = 5.0
    drop = np.zeros(6)
    drop[:3] = frame.approach_axis * 0.5
    # Radius still ~0 — must not enter.
    out = ctrl.step(frame, drop, wrist)
    out = ctrl.step(frame, drop, wrist)
    assert not out.done, f"early enter at r={ctrl._spiral_radius(ctrl._theta)}"
    # Grow spiral past min radius, then force-drop again.
    ctrl._theta = 2.0 * np.pi * 25.0  # r = pitch * 25 ≈ 0.025 > 0.020
    out = ctrl.step(frame, drop, wrist)
    out = ctrl.step(frame, drop, wrist)
    assert out.done and out.success
    assert out.reason == "hole_detected"


def test_priv_lat_reject_blocks_far_hole_detected() -> None:
    """Privileged monitor reject: far lat must not enter INSERT (continue spiral)."""
    cfg = CompliantSearchConfig(
        min_search_steps=1,
        max_search_steps=200,
        spiral_step_rad=0.05,
        spiral_pitch_m=0.001,
        spiral_radius_start_m=0.0,
        spiral_try_min_radius_m=0.002,
        spiral_try_insert_theta=1e9,
        contact_min_n=0.0,
        hole_detect_fz_drop_n=0.3,
        hole_f_max_n=12.0,
        hole_fz_max_after_drop_n=12.0,
        hole_fxy_max_n=12.0,
        hole_detect_confirm=2,
        reject_hole_if_priv_lat_m=0.025,
        priv_assist=False,
        priv_enter_require_force=True,
        priv_enter_require_along=False,
        wrench_lpf_alpha=1.0,
    )
    ctrl = CompliantSearchController(cfg)
    frame = _frame()
    wrist = frame.origin_world.copy()
    peak = np.zeros(6)
    peak[:3] = frame.approach_axis * 5.0
    ctrl.reset(frame, peak, wrist_xyz=wrist, already_on_surface=True)
    ctrl._peak_abs_fz = 5.0
    ctrl._abs_fz_ema = 5.0
    drop = np.zeros(6)
    drop[:3] = frame.approach_axis * 0.5
    # Consume surface rebias (zeros theta), then grow radius.
    out = ctrl.step(frame, drop, wrist, priv_lat_m=0.032)
    assert not out.done
    wrist = wrist + out.delta_xyz
    ctrl._theta = 2.0 * np.pi * 5.0
    out = ctrl.step(frame, drop, wrist, priv_lat_m=0.032)
    out = ctrl.step(frame, drop, wrist, priv_lat_m=0.032)
    assert not out.done
    assert out.reason == "searching"
    # Near enough (below reject 25mm) → accept.
    out = ctrl.step(frame, drop, wrist, priv_lat_m=0.020)
    out = ctrl.step(frame, drop, wrist, priv_lat_m=0.020)
    assert out.done and out.success
    assert out.reason == "hole_detected"


def test_recovery_gated_seek_activates_only_after_confirm() -> None:
    """Seek toward hole only after lat>recovery threshold for N steps."""
    cfg = CompliantSearchConfig(
        min_search_steps=1,
        max_search_steps=500,
        spiral_step_rad=0.05,
        spiral_pitch_m=0.001,
        spiral_radius_start_m=0.0,
        spiral_try_min_radius_m=0.05,
        spiral_try_insert_theta=1e9,
        contact_min_n=99.0,
        hole_detect_fz_drop_n=100.0,
        hold_press_m=0.0,
        admittance_k_xy=0.0,
        priv_assist=False,
        priv_seek_step_m=0.0006,
        priv_recovery_lat_m=0.020,
        priv_recovery_confirm=5,
        left_spiral_share=0.0,
        wrench_lpf_alpha=1.0,
    )
    ctrl = CompliantSearchController(cfg)
    frame = _frame()
    wrist = frame.origin_world.copy()
    ctrl.reset(frame, np.zeros(6), wrist_xyz=wrist, already_on_surface=True)
    lat_vec = frame.rot_world_tool[:, 0] * 0.05  # 50mm off in tool-x
    # Before confirm: no recovery.
    for _ in range(4):
        out = ctrl.step(
            frame,
            np.zeros(6),
            wrist,
            priv_lat_m=0.050,
            priv_lat_vec=lat_vec,
        )
        wrist = wrist + out.delta_xyz
        assert not ctrl._recovery_active
    # 5th far step → recovery on + rebias.
    out = ctrl.step(
        frame,
        np.zeros(6),
        wrist,
        priv_lat_m=0.050,
        priv_lat_vec=lat_vec,
    )
    assert ctrl._recovery_active
    # Next step applies seek (pull opposite lat_vec).
    out = ctrl.step(
        frame,
        np.zeros(6),
        wrist,
        priv_lat_m=0.050,
        priv_lat_vec=lat_vec,
    )
    planar = out.delta_xyz - frame.approach_axis * float(
        np.dot(out.delta_xyz, frame.approach_axis)
    )
    # Seek component should reduce lat (negative along lat_vec).
    assert float(np.dot(planar, lat_vec)) < 0.0
