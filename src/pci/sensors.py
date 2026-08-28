"""Sim wrist F/T and fingertip contact forces (deployable sensors)."""

from __future__ import annotations

import numpy as np

_WRIST_SENSORS = (
    ("panda/wrist_force_right", "panda/wrist_torque_right"),
    ("panda/wrist_force_left", "panda/wrist_torque_left"),
)


def read_wrist_wrench_local(raw) -> np.ndarray:
    """Return ``(2, 6)`` local wrist wrench [force, torque] per arm."""
    result = np.empty((2, 6), dtype=np.float64)
    for side, (force_name, torque_name) in enumerate(_WRIST_SENSORS):
        force = np.asarray(raw._data.sensor(force_name).data, dtype=np.float64).reshape(-1)
        torque = np.asarray(raw._data.sensor(torque_name).data, dtype=np.float64).reshape(-1)
        if force.shape != (3,) or torque.shape != (3,):
            raise ValueError(
                f"wrist sensors must be 3D, got {force_name}={force.shape}, "
                f"{torque_name}={torque.shape}"
            )
        result[side] = np.concatenate([force, torque])
    if not np.isfinite(result).all():
        raise ValueError("wrist sensors must contain only finite values")
    return result.copy()


def read_right_finger_force12(raw, labeler) -> np.ndarray:
    """Right-hand fingertip contact forces (12,) from ``FingerForceLabeler``."""
    frame = labeler.compute(raw)
    return np.asarray(frame.right_finger_force, dtype=np.float64).reshape(12).copy()
