"""B2/B3: tool-frame admittance insert + release — F/T only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pci.task_frame import TaskFrame


@dataclass
class CompliantInsertConfig:
    admittance_k_axial: float = 0.00025
    admittance_k_lateral: float = 0.00004
    admittance_b: float = 0.00012
    wrench_lpf_alpha: float = 0.35
    f_axial_limit_n: float = 12.0
    f_lateral_limit_n: float = 10.0
    f_insert_des_n: float = 3.5
    # Open-loop axial nudge; grasp bias makes pure admittance stall.
    hold_press_m: float = 0.0005
    left_insert_share: float = 0.20
    stall_patience: int = 40
    retreat_step_m: float = 0.001
    max_admit_step_m: float = 0.0018
    max_insert_steps: int = 1200
    open_rate: float = 0.15
    max_release_steps: int = 80
    dt: float = 1.0 / 30.0
    # Micro-orbit when depth stalls (chamfer wedge).
    wiggle_radius_m: float = 0.00035
    wiggle_step_rad: float = 0.35
    depth_stall_eps_m: float = 0.00005
    depth_stall_patience: int = 20


@dataclass(frozen=True, slots=True)
class InsertStepResult:
    done: bool
    success: bool
    phase: str
    reason: str
    delta_xyz: np.ndarray
    finger_open: float = 0.0
    seat_detected: bool = False
    delta_left_xyz: np.ndarray | None = None


class CompliantInsertController:
    """True wrist admittance along tool axes; jam → lift + spiral bias."""

    def __init__(self, config: CompliantInsertConfig | None = None) -> None:
        self.config = config or CompliantInsertConfig()
        self._steps = 0
        self._stall = 0
        self._depth_stall = 0
        self._best_axial = 0.0
        self._wiggle_theta = 0.0
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
        self._depth_stall = 0
        self._wiggle_theta = 0.0
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
            self._best_axial = axial
        else:
            self._prev_axial = 0.0
            self._axial_at_reset = 0.0
            self._best_axial = 0.0
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

        # Jam only on true force overload — wrist mocap travel is unreliable
        # under hard contact + grasp-biased F/T (cmd moves, tip lags).
        stressed = f_lat > cfg.f_lateral_limit_n or f_ax_abs > cfg.f_axial_limit_n
        if stressed:
            self._stall += 1
        else:
            self._stall = max(0, self._stall - 1)

        jam = self._stall >= cfg.stall_patience or f_ax_abs > cfg.f_axial_limit_n * 0.98
        if jam:
            self._stall = 0
            lift = frame.axial_world(-self._push_sign * cfg.retreat_step_m)
            left_lift = -float(np.clip(cfg.left_insert_share, 0.0, 0.5)) * lift
            self._prev_axial = axial_pos
            self._prev_wrist = wrist.copy()
            return InsertStepResult(
                False,
                False,
                "insert",
                "retreat",
                lift,
                delta_left_xyz=left_lift,
            )

        self._prev_axial = axial_pos
        # Deeper = larger axial along push sign (depends on frame); track best travel.
        travel = float(axial_pos - self._axial_at_reset) * float(self._push_sign)
        best_travel = float(self._best_axial - self._axial_at_reset) * float(self._push_sign)
        if travel > best_travel + float(cfg.depth_stall_eps_m):
            self._best_axial = axial_pos
            self._depth_stall = 0
        else:
            self._depth_stall += 1

        # Seat: always open-loop press into hole. Wrist |Fz|≈6N grasp bias must NOT
        # cancel press (comparing to f_insert_des would zero axial forever).
        v_tool = self._v_tool(frame, wrist, dt=dt, prev_delta_xyz=prev_delta_xyz)
        f_des_lat = np.zeros(6, dtype=np.float64)
        k_lat = np.array(
            [cfg.admittance_k_lateral, cfg.admittance_k_lateral, 0.0], dtype=np.float64
        )
        comply_lat = np.array([1.0, 1.0, 0.0], dtype=np.float64)
        admit_lat = frame.admit_step(
            f_des_lat, wh, v_tool, K=k_lat, B=cfg.admittance_b, comply_mask=comply_lat
        )
        # Chamfer unstick: small planar chord orbit when depth stalls.
        wiggle = np.zeros(3, dtype=np.float64)
        reason = "admitting"
        if self._depth_stall >= int(cfg.depth_stall_patience) and float(cfg.wiggle_radius_m) > 0.0:
            prev_th = float(self._wiggle_theta)
            self._wiggle_theta = prev_th + float(cfg.wiggle_step_rad)
            r = float(cfg.wiggle_radius_m)
            wiggle = frame.spiral_offset_world(self._wiggle_theta, r) - frame.spiral_offset_world(
                prev_th, r
            )
            wiggle = wiggle - frame.approach_axis * float(np.dot(wiggle, frame.approach_axis))
            reason = "wiggle_press"
        press = float(cfg.hold_press_m)
        if f_lat > 0.5 * cfg.f_lateral_limit_n:
            press *= 0.5  # ease axial when jammed sideways
        if self._depth_stall >= int(cfg.depth_stall_patience):
            press *= 0.6  # less wedge while orbiting
        axial_right = frame.axial_world(+self._push_sign * press)
        share = float(np.clip(cfg.left_insert_share, 0.0, 0.5))
        axial_left = frame.axial_world(-self._push_sign * float(cfg.hold_press_m) * share)

        right = self._clip_norm(admit_lat + axial_right + wiggle, cfg.max_admit_step_m)
        left = self._clip_norm(axial_left, cfg.max_admit_step_m)

        self._prev_wrist = wrist.copy()
        return InsertStepResult(
            False,
            False,
            "insert",
            reason,
            right,
            delta_left_xyz=left,
        )
