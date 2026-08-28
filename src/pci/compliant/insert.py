"""B2/B3: tool-frame admittance insert + release — F/T only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pci.task_frame import TaskFrame


@dataclass
class CompliantInsertConfig:
    admittance_k_axial: float = 0.0002
    admittance_k_lateral: float = 0.00005
    admittance_b: float = 0.0001
    wrench_lpf_alpha: float = 0.35
    f_axial_limit_n: float = 12.0
    f_lateral_limit_n: float = 10.0
    f_insert_des_n: float = 5.0
    stall_patience: int = 20
    retreat_step_m: float = 0.001
    max_admit_step_m: float = 0.0015
    max_insert_steps: int = 500
    open_rate: float = 0.15
    max_release_steps: int = 80
    dt: float = 1.0 / 30.0


@dataclass(frozen=True, slots=True)
class InsertStepResult:
    done: bool
    success: bool
    phase: str
    reason: str
    delta_xyz: np.ndarray
    finger_open: float = 0.0
    seat_detected: bool = False


class CompliantInsertController:
    """True wrist admittance along tool axes; jam → lift + spiral bias."""

    def __init__(self, config: CompliantInsertConfig | None = None) -> None:
        self.config = config or CompliantInsertConfig()
        self._steps = 0
        self._stall = 0
        self._prev_axial = 0.0
        self._release_steps = 0
        self._releasing = False
        self._peak_abs_fz = 0.0
        self._axial_at_reset = 0.0
        self._push_sign = 1.0
        self._wrench_filt: np.ndarray | None = None
        self._prev_wrist: np.ndarray | None = None

    def reset(
        self,
        frame: TaskFrame,
        wrench_right6: np.ndarray,
        wrist_xyz: np.ndarray | None = None,
    ) -> None:
        self._steps = 0
        self._stall = 0
        self._release_steps = 0
        self._releasing = False
        self._wrench_filt = TaskFrame.lpf_wrench(
            wrench_right6, None, self.config.wrench_lpf_alpha
        )
        wh = frame.wrench_tool(self._wrench_filt)
        fz0 = float(wh[2])
        self._peak_abs_fz = abs(fz0)
        self._push_sign = 1.0
        if wrist_xyz is not None:
            axial = self._axial_pos(frame, wrist_xyz)
            self._axial_at_reset = axial
            self._prev_axial = axial
        else:
            self._prev_axial = 0.0
            self._axial_at_reset = 0.0
        self._prev_wrist = (
            np.asarray(wrist_xyz, dtype=np.float64).reshape(3).copy()
            if wrist_xyz is not None
            else None
        )

    def _axial_pos(self, frame: TaskFrame, wrist_xyz: np.ndarray) -> float:
        return float(
            np.dot(
                np.asarray(wrist_xyz, dtype=np.float64).reshape(3) - frame.origin_world,
                frame.approach_axis,
            )
        )

    def _v_tool(
        self,
        frame: TaskFrame,
        wrist_xyz: np.ndarray,
        *,
        dt: float | None = None,
        prev_delta_xyz: np.ndarray | None = None,
    ) -> np.ndarray:
        w = np.asarray(wrist_xyz, dtype=np.float64).reshape(3)
        if self._prev_wrist is not None:
            step_dt = max(float(dt if dt is not None else self.config.dt), 1e-6)
            return frame.to_tool((w - self._prev_wrist) / step_dt)
        if prev_delta_xyz is not None:
            step_dt = max(float(dt if dt is not None else self.config.dt), 1e-6)
            return frame.to_tool(np.asarray(prev_delta_xyz, dtype=np.float64).reshape(3) / step_dt)
        return np.zeros(3, dtype=np.float64)

    def _clip_norm(self, delta: np.ndarray, lim: float) -> np.ndarray:
        d = np.asarray(delta, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(d))
        if n > lim > 0.0:
            return d * (lim / n)
        return d

    def step(
        self,
        frame: TaskFrame,
        wrench_right6: np.ndarray,
        wrist_xyz: np.ndarray,
        *,
        release_trigger: bool,
        dt: float | None = None,
        prev_delta_xyz: np.ndarray | None = None,
    ) -> InsertStepResult:
        cfg = self.config
        zero = np.zeros(3, dtype=np.float64)
        wrist = np.asarray(wrist_xyz, dtype=np.float64).reshape(3)

        self._wrench_filt = TaskFrame.lpf_wrench(
            wrench_right6, self._wrench_filt, cfg.wrench_lpf_alpha
        )
        wh_raw = frame.wrench_tool(wrench_right6)
        wh = frame.wrench_tool(self._wrench_filt)
        f_ax = float(wh[2])
        f_ax_raw = float(wh_raw[2])
        f_lat = float(np.linalg.norm(wh_raw[:2]))
        f_ax_abs = abs(f_ax_raw)
        self._peak_abs_fz = max(self._peak_abs_fz, f_ax_abs)
        axial_pos = self._axial_pos(frame, wrist)
        axial_travel = axial_pos - self._axial_at_reset

        if release_trigger and not self._releasing:
            self._releasing = True
            self._release_steps = 0

        if self._releasing:
            self._release_steps += 1
            open_amt = min(1.0, cfg.open_rate * self._release_steps)
            done = self._release_steps >= cfg.max_release_steps
            self._prev_wrist = wrist.copy()
            return InsertStepResult(
                done=done,
                success=release_trigger,
                phase="release",
                reason="release_done" if done else "releasing",
                delta_xyz=zero,
                finger_open=open_amt,
                seat_detected=False,
            )

        self._steps += 1
        if self._steps >= cfg.max_insert_steps:
            self._prev_wrist = wrist.copy()
            return InsertStepResult(True, False, "insert", "insert_timeout", zero)

        axial_progress = axial_pos - self._prev_axial
        # Stall: high force, or no meaningful progress since insert reset.
        min_travel_m = 0.003
        stressed = (
            f_lat > cfg.f_lateral_limit_n
            or f_ax_abs > cfg.f_axial_limit_n
            or (axial_progress < 1e-5 and axial_travel < min_travel_m)
        )
        if stressed:
            self._stall += 1
        else:
            self._stall = max(0, self._stall - 1)

        jam = self._stall >= cfg.stall_patience or f_ax_abs > cfg.f_axial_limit_n * 0.95
        if jam:
            self._stall = 0
            lift = frame.axial_world(-self._push_sign * cfg.retreat_step_m)
            self._prev_axial = axial_pos
            self._prev_wrist = wrist.copy()
            return InsertStepResult(False, False, "insert", "retreat", lift)

        self._prev_axial = axial_pos

        # Insert: Z soft, XY stiff (ConnTact axis selection).
        f_des = np.zeros(6, dtype=np.float64)
        f_des[2] = self._push_sign * cfg.f_insert_des_n
        k = np.array(
            [cfg.admittance_k_lateral, cfg.admittance_k_lateral, cfg.admittance_k_axial],
            dtype=np.float64,
        )
        comply = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        v_tool = self._v_tool(frame, wrist, dt=dt, prev_delta_xyz=prev_delta_xyz)
        admit = frame.admit_step(
            f_des, wh, v_tool, K=k, B=cfg.admittance_b, comply_mask=comply
        )
        delta = self._clip_norm(admit, cfg.max_admit_step_m)

        self._prev_wrist = wrist.copy()
        return InsertStepResult(
            False,
            False,
            "insert",
            "admitting",
            delta,
        )
