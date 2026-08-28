"""B1: tool-frame spiral search + wrist F/T admittance (ConnTact / PSFT).

Theory: constant-force surface contact + Archimedean XY spiral; hole = Fz drop.
No privileged along/rim distance in the control loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pci.task_frame import TaskFrame


@dataclass
class CompliantSearchConfig:
    push_force_n: float = 3.0
    admittance_k_z: float = 0.00035
    admittance_k_xy: float = 0.00004
    admittance_b: float = 0.00012
    wrench_lpf_alpha: float = 0.35
    spiral_track_gain: float = 0.25
    spiral_pitch_m: float = 0.0004
    spiral_step_rad: float = 0.15
    spiral_radius_max_m: float = 0.012
    max_lat_step_m: float = 0.0008
    max_admit_step_m: float = 0.0012
    f_axial_max_n: float = 8.0
    retreat_step_m: float = 0.0008
    max_search_steps: int = 600
    min_search_steps: int = 40
    contact_min_n: float = 3.5
    contact_min_seek_steps: int = 5
    contact_rise_n: float = 1.5
    contact_seek_max_steps: int = 150
    hole_detect_fz_drop_n: float = 1.2
    hole_f_max_n: float = 5.0
    jam_spiral_boost_steps: int = 12
    jam_spiral_track_gain: float = 0.55
    dt: float = 1.0 / 30.0


@dataclass(frozen=True, slots=True)
class SearchStepResult:
    done: bool
    success: bool
    reason: str
    delta_xyz: np.ndarray


class CompliantSearchController:
    """Force-regulated contact + Archimedean spiral XY (constant-force Z)."""

    def __init__(self, config: CompliantSearchConfig | None = None) -> None:
        self.config = config or CompliantSearchConfig()
        self._frame: TaskFrame | None = None
        self._steps = 0
        self._theta = 0.0
        self._baseline_fz = 0.0
        self._peak_abs_fz = 0.0
        self._spiral_origin: np.ndarray | None = None
        self._wrench_filt: np.ndarray | None = None
        self._prev_wrist: np.ndarray | None = None
        self._push_sign = 1.0
        self._jam_spiral_boost = 0
        self._on_surface = False
        self._contact_steps = 0
        self._min_abs_fz_seek = 0.0
        self._f_push_target = float(self.config.push_force_n)

    def reset(
        self,
        frame: TaskFrame,
        wrench_right6: np.ndarray,
        *,
        wrist_xyz: np.ndarray | None = None,
        already_on_surface: bool = False,
    ) -> None:
        self._frame = frame
        self._steps = 0
        self._theta = 0.0
        self._wrench_filt = TaskFrame.lpf_wrench(
            wrench_right6, None, self.config.wrench_lpf_alpha
        )
        wh = frame.wrench_tool(self._wrench_filt)
        self._baseline_fz = float(wh[2])
        self._peak_abs_fz = abs(self._baseline_fz)
        self._push_sign = 1.0
        wrist0 = (
            np.asarray(wrist_xyz, dtype=np.float64).reshape(3).copy()
            if wrist_xyz is not None
            else frame.origin_world.copy()
        )
        self._spiral_origin = wrist0.copy()
        self._prev_wrist = None
        self._jam_spiral_boost = 0
        self._contact_steps = 0
        self._min_abs_fz_seek = abs(self._baseline_fz)
        self._on_surface = bool(already_on_surface)
        # Already in contact: hold ~current |Fz| (don't retreat because |Fz|>push_force).
        if self._on_surface:
            self._peak_abs_fz = max(self._peak_abs_fz, abs(self._baseline_fz))
            self._f_push_target = max(
                float(self.config.push_force_n), abs(self._baseline_fz)
            )
        else:
            self._f_push_target = float(self.config.push_force_n)

    def recover_from_jam(
        self,
        frame: TaskFrame,
        wrench_right6: np.ndarray,
        wrist_xyz: np.ndarray,
    ) -> None:
        self.reset(
            frame,
            wrench_right6,
            wrist_xyz=wrist_xyz,
            already_on_surface=False,
        )
        self._steps = max(0, self.config.min_search_steps - 1)
        self._jam_spiral_boost = self.config.jam_spiral_boost_steps

    def reset_steps_only(self) -> None:
        self._steps = 0
        self._theta = 0.0

    def _clip_lateral(self, frame: TaskFrame, delta: np.ndarray) -> np.ndarray:
        cfg = self.config
        d = np.asarray(delta, dtype=np.float64).reshape(3)
        lat = d - frame.approach_axis * float(np.dot(d, frame.approach_axis))
        lat_norm = float(np.linalg.norm(lat))
        if lat_norm > cfg.max_lat_step_m:
            lat = lat * (cfg.max_lat_step_m / lat_norm)
        axial = frame.approach_axis * float(np.dot(d, frame.approach_axis))
        return lat + axial

    def _clip_norm(self, delta: np.ndarray, lim: float) -> np.ndarray:
        d = np.asarray(delta, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(d))
        if n > lim > 0.0:
            return d * (lim / n)
        return d

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
            return frame.to_tool(
                np.asarray(prev_delta_xyz, dtype=np.float64).reshape(3) / step_dt
            )
        return np.zeros(3, dtype=np.float64)

    def _axial_admit_tool_z(self, fz_meas: float, v_z: float) -> float:
        cfg = self.config
        # Drive |Fz| toward contact target (ConnTact constant-force).
        f_err = float(self._f_push_target) - abs(float(fz_meas))
        return float(cfg.admittance_k_z * f_err - cfg.admittance_b * v_z)

    def step(
        self,
        frame: TaskFrame,
        wrench_right6: np.ndarray,
        wrist_xyz: np.ndarray,
        *,
        dt: float | None = None,
        prev_delta_xyz: np.ndarray | None = None,
    ) -> SearchStepResult:
        cfg = self.config
        self._steps += 1
        zero = np.zeros(3, dtype=np.float64)
        wrist = np.asarray(wrist_xyz, dtype=np.float64).reshape(3)

        self._wrench_filt = TaskFrame.lpf_wrench(
            wrench_right6, self._wrench_filt, cfg.wrench_lpf_alpha
        )
        wh_raw = frame.wrench_tool(wrench_right6)
        wh = frame.wrench_tool(self._wrench_filt)
        fz = float(wh[2])
        fz_raw = float(wh_raw[2])
        f_lat = float(np.linalg.norm(wh_raw[:2]))
        self._peak_abs_fz = max(self._peak_abs_fz, abs(fz_raw))
        self._min_abs_fz_seek = min(self._min_abs_fz_seek, abs(fz_raw))

        if abs(fz_raw) > cfg.f_axial_max_n or f_lat > cfg.f_axial_max_n:
            retreat = frame.axial_world(-self._push_sign * cfg.retreat_step_m)
            self._prev_wrist = wrist.copy()
            return SearchStepResult(False, False, "force_retreat", retreat)

        if not self._on_surface:
            self._contact_steps += 1
            v_tool = self._v_tool(frame, wrist, dt=dt, prev_delta_xyz=prev_delta_xyz)
            admit_tool = np.zeros(3, dtype=np.float64)
            admit_tool[2] = self._axial_admit_tool_z(fz, float(v_tool[2]))
            delta = self._clip_norm(frame.to_world(admit_tool), cfg.max_admit_step_m)
            force_rise = abs(fz_raw) - self._min_abs_fz_seek
            contact_ok = self._contact_steps >= cfg.contact_min_seek_steps and (
                force_rise >= cfg.contact_rise_n or abs(fz_raw) >= cfg.contact_min_n
            )
            if contact_ok:
                self._on_surface = True
                self._peak_abs_fz = abs(fz_raw)
                self._baseline_fz = fz
                self._f_push_target = max(float(cfg.push_force_n), abs(fz_raw))
                self._spiral_origin = wrist.copy()
                self._prev_wrist = wrist.copy()
                return SearchStepResult(False, False, "surface_contact", delta)
            if self._contact_steps >= cfg.contact_seek_max_steps:
                self._prev_wrist = wrist.copy()
                return SearchStepResult(True, False, "surface_seek_timeout", delta)
            self._prev_wrist = wrist.copy()
            return SearchStepResult(False, False, "seeking_surface", delta)

        if (
            self._steps >= cfg.min_search_steps
            and self._peak_abs_fz >= cfg.contact_min_n
            and abs(fz) < self._peak_abs_fz - cfg.hole_detect_fz_drop_n
            and abs(fz) <= cfg.hole_f_max_n
        ):
            self._prev_wrist = wrist.copy()
            return SearchStepResult(True, True, "hole_detected", zero)

        if self._steps >= cfg.max_search_steps:
            self._prev_wrist = wrist.copy()
            return SearchStepResult(True, False, "search_timeout", zero)

        prev_theta = self._theta
        self._theta += cfg.spiral_step_rad
        radius = min(
            cfg.spiral_radius_max_m,
            cfg.spiral_pitch_m * self._theta / (2.0 * np.pi + 1e-12),
        )
        ref_delta = frame.spiral_offset_world(self._theta, radius) - frame.spiral_offset_world(
            prev_theta, radius
        )
        track = (
            cfg.jam_spiral_track_gain if self._jam_spiral_boost > 0 else cfg.spiral_track_gain
        )
        if self._jam_spiral_boost > 0:
            self._jam_spiral_boost -= 1
        spiral_bias = ref_delta * track

        # ConnTact: constant-force Z via |Fz| + spiral XY (+ light XY admit).
        v_tool = self._v_tool(frame, wrist, dt=dt, prev_delta_xyz=prev_delta_xyz)
        f_des = np.zeros(6, dtype=np.float64)
        k_xy = np.array([cfg.admittance_k_xy, cfg.admittance_k_xy, 0.0], dtype=np.float64)
        comply_xy = np.array([1.0, 1.0, 0.0], dtype=np.float64)
        admit_lat = frame.admit_step(
            f_des, wh, v_tool, K=k_xy, B=cfg.admittance_b, comply_mask=comply_xy
        )
        admit_tool = np.zeros(3, dtype=np.float64)
        admit_tool[2] = self._axial_admit_tool_z(fz, float(v_tool[2]))
        admit_axial = frame.to_world(admit_tool)

        delta = self._clip_lateral(frame, spiral_bias + admit_lat + admit_axial)
        delta = self._clip_norm(delta, cfg.max_admit_step_m)
        self._prev_wrist = wrist.copy()
        return SearchStepResult(False, False, "searching", delta)
