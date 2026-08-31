"""Privileged peg/tray/tip geometry for research grasp-opt (sim only).

合规备注: 本模块读 MuJoCo body/site 真值，仅用于 privileged_diagnostic /
研究版相对姿态优化；不得计入非特权成功率。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from dexjoco.sim.envs.assembly_geometry import names_from_raw
from dexquery.data.finger_contact_forces import (
    FINGER_TIP_BODIES_LEFT,
    FINGER_TIP_BODIES_RIGHT,
)


@dataclass(frozen=True, slots=True)
class PrivGraspGeom:
    """One-step privileged poses for dual-object grasp regulation."""

    peg_pos: np.ndarray  # (3,)
    peg_rot: np.ndarray  # (3, 3)
    tray_pos: np.ndarray
    tray_rot: np.ndarray
    right_wrist_pos: np.ndarray
    right_wrist_rot: np.ndarray
    left_wrist_pos: np.ndarray
    left_wrist_rot: np.ndarray
    right_tip_pos: np.ndarray  # (4, 3)
    left_tip_pos: np.ndarray  # (4, 3)


def _body_pose(raw, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    bid = int(raw._model.body(body_name).id)
    pos = np.asarray(raw._data.xpos[bid], dtype=np.float64).reshape(3).copy()
    rot = np.asarray(raw._data.xmat[bid], dtype=np.float64).reshape(3, 3).copy()
    return pos, rot


def _tip_positions(raw, tip_names: tuple[str, ...]) -> np.ndarray:
    out = np.empty((len(tip_names), 3), dtype=np.float64)
    for i, name in enumerate(tip_names):
        bid = int(raw._model.body(name).id)
        out[i] = np.asarray(raw._data.xpos[bid], dtype=np.float64).reshape(3)
    return out


def read_priv_grasp_geom(raw) -> PrivGraspGeom:
    """Read peg, tray (socket body), wrists, and fingertip body positions."""
    names = names_from_raw(raw)
    peg_pos, peg_rot = _body_pose(raw, names.peg_body)
    tray_pos, tray_rot = _body_pose(raw, names.socket_body)
    r_pos = np.asarray(raw._data.site_xpos[int(raw._site_right_id)], dtype=np.float64).copy()
    l_pos = np.asarray(raw._data.site_xpos[int(raw._site_left_id)], dtype=np.float64).copy()
    r_rot = np.asarray(raw._data.site_xmat[int(raw._site_right_id)], dtype=np.float64).reshape(3, 3).copy()
    l_rot = np.asarray(raw._data.site_xmat[int(raw._site_left_id)], dtype=np.float64).reshape(3, 3).copy()
    return PrivGraspGeom(
        peg_pos=peg_pos,
        peg_rot=peg_rot,
        tray_pos=tray_pos,
        tray_rot=tray_rot,
        right_wrist_pos=r_pos,
        right_wrist_rot=r_rot,
        left_wrist_pos=l_pos,
        left_wrist_rot=l_rot,
        right_tip_pos=_tip_positions(raw, FINGER_TIP_BODIES_RIGHT),
        left_tip_pos=_tip_positions(raw, FINGER_TIP_BODIES_LEFT),
    )


def copy_priv_grasp_geom(geom: PrivGraspGeom) -> PrivGraspGeom:
    """Deep-copy arrays so a surface latch cannot be mutated later."""
    return PrivGraspGeom(
        peg_pos=np.asarray(geom.peg_pos, dtype=np.float64).reshape(3).copy(),
        peg_rot=np.asarray(geom.peg_rot, dtype=np.float64).reshape(3, 3).copy(),
        tray_pos=np.asarray(geom.tray_pos, dtype=np.float64).reshape(3).copy(),
        tray_rot=np.asarray(geom.tray_rot, dtype=np.float64).reshape(3, 3).copy(),
        right_wrist_pos=np.asarray(geom.right_wrist_pos, dtype=np.float64).reshape(3).copy(),
        right_wrist_rot=np.asarray(geom.right_wrist_rot, dtype=np.float64).reshape(3, 3).copy(),
        left_wrist_pos=np.asarray(geom.left_wrist_pos, dtype=np.float64).reshape(3).copy(),
        left_wrist_rot=np.asarray(geom.left_wrist_rot, dtype=np.float64).reshape(3, 3).copy(),
        right_tip_pos=np.asarray(geom.right_tip_pos, dtype=np.float64).reshape(4, 3).copy(),
        left_tip_pos=np.asarray(geom.left_tip_pos, dtype=np.float64).reshape(4, 3).copy(),
    )
