"""Unit tests for force-regulated compliant search (no privileged along)."""

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
    )
    ctrl = CompliantSearchController(cfg)
    frame = _frame()
    ctrl.reset(frame, np.zeros(6))
    wrench = np.zeros(6)
    wrench[:3] = frame.approach_axis * 12.0
    wrist = frame.origin_world.copy()
    out = ctrl.step(frame, wrench, wrist)
    assert out.reason == "force_retreat"
    axial = float(np.dot(out.delta_xyz, frame.approach_axis))
    assert axial * ctrl._push_sign < 0.0


def test_contact_seek_before_spiral() -> None:
    cfg = CompliantSearchConfig(
        min_search_steps=99,
        contact_min_n=5.0,
        contact_rise_n=1.5,
        contact_min_seek_steps=3,
        contact_seek_max_steps=20,
    )
    ctrl = CompliantSearchController(cfg)
    frame = _frame()
    wrist = frame.origin_world.copy()
    ctrl.reset(frame, np.zeros(6), wrist_xyz=wrist)
    out0 = ctrl.step(frame, np.zeros(6), wrist)
    assert out0.reason == "seeking_surface"
    high = np.zeros(6)
    high[:3] = frame.approach_axis * 6.0
    out_contact = None
    for _ in range(cfg.contact_min_seek_steps + 2):
        out_mid = ctrl.step(frame, high, wrist)
        if out_mid.reason == "surface_contact":
            out_contact = out_mid
            break
    assert out_contact is not None
    assert out_contact.reason == "surface_contact"
    out2 = ctrl.step(frame, high, wrist)
    assert out2.reason == "searching"


def test_spiral_keeps_axial_force_regulation() -> None:
    cfg = CompliantSearchConfig(min_search_steps=1, push_force_n=4.0, admittance_k_z=0.001)
    ctrl = CompliantSearchController(cfg)
    frame = _frame()
    ctrl.reset(frame, np.zeros(6), already_on_surface=True)
    # Low Fz → should push along approach (constant-force Z).
    out = ctrl.step(frame, np.zeros(6), frame.origin_world.copy())
    assert out.reason == "searching"
    axial = float(np.dot(out.delta_xyz, frame.approach_axis))
    assert axial > 0.0


def test_search_timeout_is_failure() -> None:
    cfg = CompliantSearchConfig(min_search_steps=1, max_search_steps=2, contact_min_n=99.0)
    ctrl = CompliantSearchController(cfg)
    frame = _frame()
    ctrl.reset(frame, np.zeros(6), already_on_surface=True)
    ctrl.step(frame, np.zeros(6), frame.origin_world.copy())
    out = ctrl.step(frame, np.zeros(6), frame.origin_world.copy())
    assert out.done and not out.success
    assert out.reason == "search_timeout"
