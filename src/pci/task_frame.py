"""Frozen tool frame at Phase A→B handoff (no privileged geometry)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True, slots=True)
class TaskFrame:
    """Wrist/tool frame captured once when entering Phase B."""

    origin_world: np.ndarray
    rot_world_tool: np.ndarray
    approach_axis: np.ndarray

    @classmethod
    def from_action44(cls, action44: np.ndarray) -> TaskFrame:
        a = np.asarray(action44, dtype=np.float64).reshape(44)
        origin = a[0:3].copy()
        rot = R.from_rotvec(a[3:6]).as_matrix()
        # Tool +Z ≈ peg approach into hole (ConnTact wrist-frame push).
        approach = rot[:, 2].copy()
        approach /= np.linalg.norm(approach) + 1e-12
        return cls(origin_world=origin, rot_world_tool=rot, approach_axis=approach)

    @classmethod
    def from_hole_axis(
        cls,
        origin_world: np.ndarray,
        hole_axis_world: np.ndarray,
        *,
        peg_axis_world: np.ndarray | None = None,
    ) -> TaskFrame:
        """One-time privileged init at A→B: tool +Z aligned with insert direction."""
        from pci.wrist import hole_task_basis

        origin = np.asarray(origin_world, dtype=np.float64).reshape(3).copy()
        t1, t2, axis = hole_task_basis(hole_axis_world)
        # Insert advances along -hole_axis (see hybrid_insert.geometry.insert_along_hole_delta).
        approach = -axis
        rot = np.stack([t1, t2, approach], axis=1)
        return cls(origin_world=origin, rot_world_tool=rot, approach_axis=approach)

    def to_tool(self, vec_world: np.ndarray) -> np.ndarray:
        return self.rot_world_tool.T @ np.asarray(vec_world, dtype=np.float64).reshape(3)

    def to_world(self, vec_tool: np.ndarray) -> np.ndarray:
        return self.rot_world_tool @ np.asarray(vec_tool, dtype=np.float64).reshape(3)

    def wrench_tool(self, wrench6: np.ndarray) -> np.ndarray:
        w = np.asarray(wrench6, dtype=np.float64).reshape(6)
        out = w.copy()
        out[:3] = self.to_tool(w[:3])
        out[3:6] = self.to_tool(w[3:6])
        return out

    @staticmethod
    def lpf_wrench(
        meas6: np.ndarray,
        prev6: np.ndarray | None,
        alpha: float,
    ) -> np.ndarray:
        """First-order low-pass on wrench6. alpha in (0,1]; 1 = no filter."""
        m = np.asarray(meas6, dtype=np.float64).reshape(6)
        a = float(np.clip(alpha, 1e-6, 1.0))
        if prev6 is None or a >= 1.0 - 1e-12:
            return m.copy()
        p = np.asarray(prev6, dtype=np.float64).reshape(6)
        return a * m + (1.0 - a) * p

    def admit_step(
        self,
        F_des6: np.ndarray,
        F_meas6: np.ndarray,
        v_tool3: np.ndarray,
        *,
        K: float | np.ndarray,
        B: float | np.ndarray,
        comply_mask: np.ndarray | list[bool] | list[float],
    ) -> np.ndarray:
        """Task-space admittance (ConnTact / Ott): Δx = K(F_des−F_meas) − B v.

        Only force axes (xyz) are used. ``comply_mask`` (3,) selects soft axes
        (1=admit, 0=stiff / no force-driven motion on that tool axis).
        Returns world-frame translational delta (3,).
        """
        fd = np.asarray(F_des6, dtype=np.float64).reshape(6)
        fm = np.asarray(F_meas6, dtype=np.float64).reshape(6)
        v = np.asarray(v_tool3, dtype=np.float64).reshape(3)
        mask = np.asarray(comply_mask, dtype=np.float64).reshape(3)
        mask = np.clip(mask, 0.0, 1.0)

        k = np.asarray(K, dtype=np.float64).reshape(-1)
        if k.size == 1:
            k = np.full(3, float(k[0]), dtype=np.float64)
        else:
            k = k[:3].copy()

        b = np.asarray(B, dtype=np.float64).reshape(-1)
        if b.size == 1:
            b = np.full(3, float(b[0]), dtype=np.float64)
        else:
            b = b[:3].copy()

        err_f = fd[:3] - fm[:3]
        delta_tool = mask * (k * err_f - b * v)
        return self.to_world(delta_tool)

    def axial_world(self, step_m: float) -> np.ndarray:
        return self.approach_axis * float(step_m)

    def spiral_offset_world(self, theta: float, radius: float) -> np.ndarray:
        t1 = self.rot_world_tool[:, 0]
        t2 = self.rot_world_tool[:, 1]
        return radius * (np.cos(theta) * t1 + np.sin(theta) * t2)
