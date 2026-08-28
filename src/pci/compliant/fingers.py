"""Per-finger joint admittance during compliant insert (PCI innovation).

Refs: FSMJIC (ROBIO 2016), DLR Hand II (IROS 2003),
Pfanne RA-L 2020 object-level impedance (simplified: admittance + force balance, no QP).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FingerCompliantConfig:
    """Joint admittance: Δq = K_f (f_des - f) - B_f Δq_prev."""

    k_admit: float = 0.0004  # K_f [rad / N]
    b_admit: float = 0.35  # B_f discrete damping on previous Δq
    max_joint_step: float = 0.025
    balance_gain: float = 0.12  # shifts per-finger f_des toward mean load
    f_balance_des_n: float = 2.0  # nominal tip-force setpoint [N]
    f_des_min_n: float = 0.3
    f_des_max_n: float = 5.0
    # Relative to hold: open (relax) more limited than close (tighten).
    hold_open_limit: float = 0.08
    hold_close_limit: float = 0.02
    enable: bool = True


class FingerCompliantController:
    """Right-hand 16d: joint admittance + Pfanne-style tip-force balance.

    High-load finger → lower f_des → negative Δq on primary flex (relax).
    Low-load finger → raise f_des slightly → share grasp load.
    Tip force norms from finger_force12 (4×3).
    """

    def __init__(self, config: FingerCompliantConfig | None = None) -> None:
        self.config = config or FingerCompliantConfig()
        self._delta_prev = np.zeros(16, dtype=np.float64)

    def reset(self, finger_force12: np.ndarray) -> None:
        _ = np.asarray(finger_force12, dtype=np.float64).reshape(12)
        self._delta_prev = np.zeros(16, dtype=np.float64)

    def step(self, finger_force12: np.ndarray, hold_hand16: np.ndarray) -> np.ndarray:
        cfg = self.config
        if not cfg.enable:
            return np.zeros(16, dtype=np.float64)

        f = np.asarray(finger_force12, dtype=np.float64).reshape(12)
        hold = np.asarray(hold_hand16, dtype=np.float64).reshape(16)
        delta = np.zeros(16, dtype=np.float64)

        norms = np.array(
            [float(np.linalg.norm(f[i * 3 : (i + 1) * 3])) for i in range(4)],
            dtype=np.float64,
        )
        mean_n = float(np.mean(norms))
        f_nom = float(cfg.f_balance_des_n)

        for i in range(4):
            fn = float(norms[i])
            # Pfanne-style balance without QP: redistribute desired tip force.
            # High load (fn > mean) → f_des ↓; low load → f_des ↑.
            f_des = f_nom + cfg.balance_gain * (mean_n - fn)
            f_des = float(np.clip(f_des, cfg.f_des_min_n, cfg.f_des_max_n))

            j = 4 * i + 1  # primary flexion joint per finger
            # Δq_i = K_f (f_des,i - f_i) - B_f Δq_prev
            delta[j] = cfg.k_admit * (f_des - fn) - cfg.b_admit * float(self._delta_prev[j])

        delta = np.clip(delta, -cfg.max_joint_step, cfg.max_joint_step)
        # Do not open beyond hold posture (release handled separately).
        proposed = hold + delta
        delta = (
            np.clip(proposed, hold - cfg.hold_open_limit, hold + cfg.hold_close_limit) - hold
        )
        self._delta_prev = delta.copy()
        return delta
