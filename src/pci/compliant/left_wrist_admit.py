"""Left-wrist admittance from wrist F/T only (deployable sensors).

Yields to contact wrench so the tray-holding arm absorbs tip moments instead of
locking rigid. No privileged object pose in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LeftWristAdmitConfig:
    enable: bool = True
    k_force: float = 0.00012  # m/N — translate opposite to sensed force
    k_torque: float = 0.0008  # m/(N·m) — tip-moment → small lateral give
    max_step_m: float = 0.00045
    max_total_m: float = 0.004  # cap accumulated yield so tray does not wander
    wrench_lpf_alpha: float = 0.35
    # Ignore tiny sensor noise / grasp bias.
    f_deadband_n: float = 0.35
    tau_deadband_nm: float = 0.04


class LeftWristAdmitController:
    """δx = -K_f F − K_τ (τ × û) with LPF, clipped."""

    def __init__(self, config: LeftWristAdmitConfig | None = None) -> None:
        self.config = config or LeftWristAdmitConfig()
        self._w_filt = np.zeros(6, dtype=np.float64)
        self._has_filt = False
        self._acc = np.zeros(3, dtype=np.float64)

    def reset(self, wrench_left6: np.ndarray | None = None) -> None:
        self._acc[:] = 0.0
        if wrench_left6 is None:
            self._w_filt[:] = 0.0
            self._has_filt = False
        else:
            self._w_filt = np.asarray(wrench_left6, dtype=np.float64).reshape(6).copy()
            self._has_filt = True

    def step(
        self,
        wrench_left6_world: np.ndarray,
        *,
        approach_axis: np.ndarray | None = None,
    ) -> np.ndarray:
        cfg = self.config
        zero = np.zeros(3, dtype=np.float64)
        if not cfg.enable:
            return zero

        w = np.asarray(wrench_left6_world, dtype=np.float64).reshape(6)
        a = float(np.clip(cfg.wrench_lpf_alpha, 0.0, 1.0))
        if not self._has_filt:
            self._w_filt = w.copy()
            self._has_filt = True
        else:
            self._w_filt = (1.0 - a) * self._w_filt + a * w

        f = self._w_filt[:3].copy()
        tau = self._w_filt[3:6].copy()
        fn = float(np.linalg.norm(f))
        tn = float(np.linalg.norm(tau))
        if fn < cfg.f_deadband_n:
            f[:] = 0.0
        if tn < cfg.tau_deadband_nm:
            tau[:] = 0.0

        # Yield to force (admittance toward F_des=0).
        delta = -cfg.k_force * f

        # Tip moment → lateral give in plane ⟂ approach (if given).
        if approach_axis is not None and tn >= cfg.tau_deadband_nm:
            ax = np.asarray(approach_axis, dtype=np.float64).reshape(3)
            an = float(np.linalg.norm(ax))
            if an > 1e-9:
                ax = ax / an
                tip = np.cross(tau, ax)
                tip = tip - ax * float(np.dot(tip, ax))
                delta = delta + cfg.k_torque * tip

        dn = float(np.linalg.norm(delta))
        if dn > cfg.max_step_m and dn > 1e-12:
            delta = delta * (cfg.max_step_m / dn)

        # Soft total travel cap: shrink step if remaining budget is small.
        cap = float(cfg.max_total_m)
        if cap > 0.0:
            used = float(np.linalg.norm(self._acc))
            remain = max(0.0, cap - used)
            dn2 = float(np.linalg.norm(delta))
            if remain <= 1e-12:
                return zero
            if dn2 > remain:
                delta = delta * (remain / dn2)
        self._acc = self._acc + delta
        return delta
