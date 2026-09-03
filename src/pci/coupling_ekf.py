"""Slip-aware planar coupling EKF for tip-referenced spiral search.

Adapted concepts (not runtime deps):
- bgf slip_control.py: slip-gated Kalman reset
- Pfanne IROS'17: recursive grasp state from proprioception
- Tactile-Estimator-Controller: buffer/update state machine (simplified)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = _unit(normal)
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(a, n))) > 0.9:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    t1 = _unit(np.cross(a, n))
    t2 = np.cross(n, t1)
    return t1, t2


def _to2(v: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([float(np.dot(v, t1)), float(np.dot(v, t2))], dtype=np.float64)


def _planar(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    n = _unit(normal)
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v - n * float(np.dot(v, n))


@dataclass
class CouplingEKF:
    """EKF state [tip_x, tip_y, c11, c12, c21, c22] in contact tangent frame."""

    process_tip: float = 1e-6
    process_c: float = 1e-5
    meas_var: float = 4e-7
    slip_ratio: float = 0.25
    slip_dw_m: float = 0.0012
    alpha_floor: float = 0.18
    alpha_ceil: float = 1.05
    _x: np.ndarray = field(default_factory=lambda: np.zeros(6))
    _P: np.ndarray = field(default_factory=lambda: np.eye(6) * 0.01)
    _initialized: bool = False
    _last_site: np.ndarray | None = None
    _last_tip_plane: np.ndarray | None = None
    _last_dw2: np.ndarray = field(default_factory=lambda: np.zeros(2))
    just_slipped: bool = False
    tip_innov_norm: float = 0.0
    slip_P_c: float = 0.25
    slip_process_boost: float = 20.0

    def reset(self, wrist: np.ndarray, tip: np.ndarray, normal: np.ndarray) -> None:
        t1, t2 = _tangent_basis(normal)
        t2p = _to2(tip, t1, t2)
        self._x = np.array([t2p[0], t2p[1], 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._P = np.eye(6) * 0.01
        self._P[2:, 2:] *= 0.1
        self._initialized = True
        self._last_site = np.asarray(wrist, dtype=np.float64).reshape(3).copy()
        self._last_tip_plane = t2p.copy()
        self._last_dw2[:] = 0.0
        self.just_slipped = False
        self.tip_innov_norm = 0.0

    def _C(self) -> np.ndarray:
        return np.array(
            [[self._x[2], self._x[3]], [self._x[4], self._x[5]]],
            dtype=np.float64,
        )

    def _clip_c(self) -> None:
        c = self._C()
        u, s, vt = np.linalg.svd(c)
        s = np.clip(s, self.alpha_floor, self.alpha_ceil)
        c = u @ np.diag(s) @ vt
        self._x[2:6] = np.array([c[0, 0], c[0, 1], c[1, 0], c[1, 1]])

    def slip_reset(self) -> None:
        """bgf/Kim-style: C←I and inflate covariance so C re-learns after slip."""
        self._x[2:6] = np.array([1.0, 0.0, 0.0, 1.0])
        self._P[2:, 2:] = np.eye(4) * float(self.slip_P_c)
        self._last_dw2[:] = 0.0
        self.just_slipped = True
        # Temporarily larger process noise on C until next clean updates.
        self.process_c = max(self.process_c, 1e-5 * float(self.slip_process_boost))

    def predict(self, dw_wrist3: np.ndarray, normal: np.ndarray) -> None:
        if not self._initialized:
            return
        t1, t2 = _tangent_basis(normal)
        dw2 = _to2(_planar(dw_wrist3, normal), t1, t2)
        c = self._C()
        self._x[0:2] = self._x[0:2] + c @ dw2
        f = np.eye(6)
        f[0:2, 2:6] = np.kron(dw2.reshape(1, 2), np.eye(2))
        q = np.diag([self.process_tip, self.process_tip] + [self.process_c] * 4)
        self._P = f @ self._P @ f.T + q
        self._last_dw2 = dw2.copy()

    def update(self, tip3: np.ndarray, wrist3: np.ndarray, normal: np.ndarray) -> None:
        t1, t2 = _tangent_basis(normal)
        z = _to2(tip3, t1, t2)
        if not self._initialized:
            self.reset(wrist3, tip3, normal)
            return
        self.just_slipped = False
        if self._last_tip_plane is not None and float(np.linalg.norm(self._last_dw2)) > self.slip_dw_m:
            dt_obs = z - self._last_tip_plane
            dt_pred = self._C() @ self._last_dw2
            if float(np.linalg.norm(dt_obs)) < self.slip_ratio * float(
                np.linalg.norm(dt_pred)
            ):
                self.slip_reset()
        h = np.zeros((2, 6))
        h[0, 0] = 1.0
        h[1, 1] = 1.0
        r = np.eye(2) * self.meas_var
        y = z - self._x[0:2]
        self.tip_innov_norm = float(np.linalg.norm(y))
        s = h @ self._P @ h.T + r
        k = self._P @ h.T @ np.linalg.inv(s)
        self._x = self._x + k @ y
        self._P = (np.eye(6) - k @ h) @ self._P
        self._clip_c()
        self._last_tip_plane = z.copy()
        # Decay process boost after a clean update post-slip.
        if not self.just_slipped and self.process_c > 1e-5:
            self.process_c = 0.9 * self.process_c + 0.1 * 1e-5

    def step_filter(
        self, wrist3: np.ndarray, tip3: np.ndarray, normal: np.ndarray
    ) -> None:
        """Predict from wrist motion then update with privileged tip."""
        w = np.asarray(wrist3, dtype=np.float64).reshape(3)
        if self._last_site is not None:
            self.predict(w - self._last_site, normal)
        self.update(tip3, w, normal)
        self._last_site = w.copy()

    @property
    def C_matrix(self) -> np.ndarray:
        return self._C().copy()

    @property
    def P_c_trace(self) -> float:
        """Trace of covariance block on vec(C) — for uncertainty-aware C⁺."""
        if not self._initialized:
            return 0.0
        return float(np.trace(self._P[2:6, 2:6]))

    @property
    def initialized(self) -> bool:
        return bool(self._initialized)
