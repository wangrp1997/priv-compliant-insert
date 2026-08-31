"""Apply wrist / tip deltas to 44-d rotvec actions."""

from __future__ import annotations

import numpy as np


def apply_tip_delta44(action44: np.ndarray, delta_xyz: np.ndarray) -> np.ndarray:
    out = np.asarray(action44, dtype=np.float64).reshape(44).copy()
    out[0:3] = out[0:3] + np.asarray(delta_xyz, dtype=np.float64).reshape(3)
    return out


def apply_dual_wrist_delta44(
    action44: np.ndarray,
    delta_right_xyz: np.ndarray,
    delta_left_xyz: np.ndarray | None = None,
) -> np.ndarray:
    """Right xyz at [0:3], left xyz at [22:25] (rotvec action44 layout)."""
    out = apply_tip_delta44(action44, delta_right_xyz)
    if delta_left_xyz is not None:
        out[22:25] = out[22:25] + np.asarray(delta_left_xyz, dtype=np.float64).reshape(3)
    return out


def hole_task_basis(hole_axis_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tangent1, tangent2, hole_axis (unit; insert motion uses -axis)."""
    axis = np.asarray(hole_axis_world, dtype=np.float64).reshape(3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(axis @ ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    t1 = np.cross(ref, axis)
    t1 /= np.linalg.norm(t1) + 1e-12
    t2 = np.cross(axis, t1)
    return t1, t2, axis


def wrench_in_hole_frame(wrench6: np.ndarray, hole_axis_world: np.ndarray) -> np.ndarray:
    """Express local wrist wrench force/torque in hole frame (+Z = hole axis)."""
    t1, t2, axis = hole_task_basis(hole_axis_world)
    rot = np.stack([t1, t2, axis], axis=1)
    w = np.asarray(wrench6, dtype=np.float64).reshape(6)
    out = w.copy()
    out[:3] = rot.T @ w[:3]
    out[3:6] = rot.T @ w[3:6]
    return out
