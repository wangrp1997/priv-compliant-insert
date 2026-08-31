"""Privileged left-wrist follow of tray relative pose (diagnostic only).

Locks the latch-time left-wrist↔tray transform and soft-tracks it so the
holding arm moves with the tray instead of freezing in world frame.
不合规: 读 MuJoCo tray/腕真值；仅 privileged_diagnostic。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from pci.priv_geom import PrivGraspGeom


@dataclass
class LeftTrayFollowConfig:
    enable: bool = True
    max_step_m: float = 0.0012
    max_rot_step_rad: float = 0.020
    # Blend toward desired each step (1=full, lower=softer).
    track_gain: float = 0.85
    follow_rot: bool = True


class LeftTrayFollowController:
    """Maintain latch relative pose: wrist tracks tray translation (+ optional rot)."""

    def __init__(self, config: LeftTrayFollowConfig | None = None) -> None:
        self.config = config or LeftTrayFollowConfig()
        self._tray_in_l0_R: np.ndarray | None = None
        self._tray_in_l0_p: np.ndarray | None = None

    def reset(self, geom: PrivGraspGeom) -> None:
        self._tray_in_l0_R = geom.left_wrist_rot.T @ geom.tray_rot
        self._tray_in_l0_p = geom.left_wrist_rot.T @ (geom.tray_pos - geom.left_wrist_pos)

    def desired_wrist(
        self, geom: PrivGraspGeom
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (pos3, rot33) for left wrist that keeps latch relative pose."""
        if self._tray_in_l0_R is None or self._tray_in_l0_p is None:
            return None
        L_des = geom.tray_rot @ self._tray_in_l0_R.T
        p_des = geom.tray_pos - L_des @ self._tray_in_l0_p
        return p_des, L_des

    def step_hold6(
        self,
        hold_left6: np.ndarray,
        geom: PrivGraspGeom,
    ) -> tuple[np.ndarray, float]:
        """Soft-update left wrist mocap hold [pos3|rotvec3]. Returns (hold, |Δp|)."""
        cfg = self.config
        hold = np.asarray(hold_left6, dtype=np.float64).reshape(6).copy()
        if not cfg.enable:
            return hold, 0.0
        des = self.desired_wrist(geom)
        if des is None:
            return hold, 0.0
        p_des, R_des = des
        gain = float(np.clip(cfg.track_gain, 0.0, 1.0))
        dp = (p_des - hold[0:3]) * gain
        dn = float(np.linalg.norm(dp))
        if dn > cfg.max_step_m and dn > 1e-12:
            dp = dp * (cfg.max_step_m / dn)
        hold[0:3] = hold[0:3] + dp

        if cfg.follow_rot:
            R_cur = R.from_rotvec(hold[3:6]).as_matrix()
            R_err = R_des @ R_cur.T
            rv = R.from_matrix(R_err).as_rotvec() * gain
            rn = float(np.linalg.norm(rv))
            if rn > cfg.max_rot_step_rad and rn > 1e-12:
                rv = rv * (cfg.max_rot_step_rad / rn)
            hold[3:6] = R.from_matrix(R.from_rotvec(rv).as_matrix() @ R_cur).as_rotvec()
        return hold, float(np.linalg.norm(dp))
