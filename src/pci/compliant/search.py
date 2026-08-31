"""B1: right-led on-surface force spiral (dexterous-safe).

Right wrist does Archimedean XY + contact-force hold; left barely shares.
Privilege only gates try-insert (no direction seek by default).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pci.task_frame import TaskFrame


@dataclass
class CompliantSearchConfig:
    push_force_n: float = 0.8
    unload_max_m: float = 0.0
    unload_k_z: float = 0.0015
    # Right-led force spiral; left barely shares.
    left_spiral_share: float = 0.12
    left_axial_share: float = 0.05
    left_unload_share: float = 0.0
    right_lat_scale: float = 1.0
    hold_press_m: float = 0.00035
    contact_f_des_n: float = 2.5
    contact_press_gain: float = 0.00025
    admittance_k_z: float = 0.0008
    admittance_k_xy: float = 0.00008
    admittance_b: float = 0.00012
    wrench_lpf_alpha: float = 0.35
    spiral_track_gain: float = 1.0
    spiral_pitch_m: float = 0.0025
    spiral_step_rad: float = 0.30
    spiral_radius_max_m: float = 0.028
    # After surface contact, grow Archimedean radius from this floor.
    spiral_radius_start_m: float = 0.0
    max_lat_step_m: float = 0.0022
    max_admit_step_m: float = 0.0018
    f_axial_max_n: float = 9.0
    retreat_step_m: float = 0.0010
    max_search_steps: int = 1600
    min_search_steps: int = 30
    contact_min_n: float = 2.0
    contact_min_seek_steps: int = 5
    contact_rise_n: float = 1.5
    contact_seek_max_steps: int = 150
    hole_detect_fz_drop_n: float = 0.45
    hole_f_max_n: float = 12.0
    hole_detect_confirm: int = 2
    # After force-drop, |Fz| / |Fxy| must stay low (reject glancing unload).
    hole_fz_max_after_drop_n: float = 4.0
    hole_fxy_max_n: float = 4.0
    spiral_try_min_radius_m: float = 0.016
    spiral_try_insert_theta: float = 36.0
    # Privileged monitor reject (not seek): refuse hole_detected if tip still far.
    reject_hole_if_priv_lat_m: float = 0.0
    # Privilege: enter-gate only (seek_step=0 → no direction thrash).
    priv_assist: bool = True
    priv_enter_lat_m: float = 0.008
    priv_far_lat_m: float = 0.014
    priv_seek_step_m: float = 0.0
    # Recovery-gated seek: only when lat > threshold for N steps (not always-on).
    priv_recovery_lat_m: float = 0.0
    priv_recovery_confirm: int = 15
    # True hole seat: tip must drop along axis (rim ~0.10m → below this).
    priv_enter_along_max_m: float = 0.090
    priv_enter_require_along: bool = True
    priv_enter_require_force: bool = False
    jam_spiral_boost_steps: int = 12
    jam_spiral_track_gain: float = 1.0
    discrete_hop: bool = False
    hop_lift_m: float = 0.0025
    hop_press_m: float = 0.0030
    hop_lift_cmd_budget_m: float = 0.012
    hop_lift_step_m: float = 0.0020
    dt: float = 1.0 / 30.0
    # Light axial press when near hole so tip can fall (search usually strips Z).
    near_hole_press_m: float = 0.00035


@dataclass(frozen=True, slots=True)
class SearchStepResult:
    done: bool
    success: bool
    reason: str
    delta_xyz: np.ndarray
    delta_left_xyz: np.ndarray | None = None


class CompliantSearchController:
    """Dual-arm on-surface spiral: right search + left opposite share."""

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
        self._unload_travel_m = 0.0
        self._unload_done = False
        self._rebias_spiral = False
        # tip_resync may re-anchor origin without wiping spiral progress.
        self._rebias_keep_theta = False
        self._abs_fz_ema = 0.0
        self._hole_confirm = 0
        self._recovery_streak = 0
        self._recovery_active = False

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
        self._abs_fz_ema = abs(self._baseline_fz)
        self._hole_confirm = 0
        self._recovery_streak = 0
        self._recovery_active = False
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
        self._f_push_target = float(self.config.push_force_n)
        self._unload_travel_m = 0.0
        self._unload_done = True  # Phase-A already contacted; stay on-surface spiral
        self._rebias_spiral = bool(already_on_surface)
        if self._on_surface:
            self._peak_abs_fz = max(self._peak_abs_fz, abs(self._baseline_fz))
            self._abs_fz_ema = max(self._abs_fz_ema, abs(self._baseline_fz))

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
            already_on_surface=True,
        )
        self._steps = max(0, self.config.min_search_steps - 1)
        self._jam_spiral_boost = self.config.jam_spiral_boost_steps
        self._unload_done = True
        self._rebias_spiral = True

    def reset_steps_only(self) -> None:
        self._steps = 0
        self._theta = 0.0

    def _spiral_radius(self, theta: float) -> float:
        cfg = self.config
        r0 = max(0.0, float(getattr(cfg, "spiral_radius_start_m", 0.0)))
        r = r0 + cfg.spiral_pitch_m * float(theta) / (2.0 * np.pi + 1e-12)
        return min(cfg.spiral_radius_max_m, r)

    def _spiral_chord(self, frame: TaskFrame) -> np.ndarray:
        cfg = self.config
        prev_theta = float(self._theta)
        self._theta = prev_theta + float(cfg.spiral_step_rad)
        r = self._spiral_radius(self._theta)
        r_prev = self._spiral_radius(prev_theta)
        chord = frame.spiral_offset_world(self._theta, r) - frame.spiral_offset_world(
            prev_theta, r_prev
        )
        track = (
            cfg.jam_spiral_track_gain if self._jam_spiral_boost > 0 else cfg.spiral_track_gain
        )
        if self._jam_spiral_boost > 0:
            self._jam_spiral_boost -= 1
        chord = chord * float(track)
        return chord - frame.approach_axis * float(np.dot(chord, frame.approach_axis))

    def _clip_norm(self, delta: np.ndarray, lim: float) -> np.ndarray:
        d = np.asarray(delta, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(d))
        if n > lim > 0.0:
            return d * (lim / n)
        return d

    def _clip_lat(self, frame: TaskFrame, delta: np.ndarray) -> np.ndarray:
        cfg = self.config
        d = np.asarray(delta, dtype=np.float64).reshape(3)
        lat = d - frame.approach_axis * float(np.dot(d, frame.approach_axis))
        n = float(np.linalg.norm(lat))
        if n > cfg.max_lat_step_m:
            lat = lat * (cfg.max_lat_step_m / n)
        return lat

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

    def _axial_seek_tool_z(self, fz_meas: float, v_z: float) -> float:
        cfg = self.config
        f_err = float(self._f_push_target) - abs(float(fz_meas))
        return float(cfg.admittance_k_z * f_err - cfg.admittance_b * v_z)

    def _left_share(self) -> float:
        s = float(self.config.left_spiral_share)
        return float(np.clip(s, 0.0, 0.35))

    def _left_axial_share(self) -> float:
        s = float(self.config.left_axial_share)
        if s <= 0.0:
            s = float(self.config.left_unload_share)
        return float(np.clip(s, 0.0, 0.5))

    def _clip_lat_scaled(self, frame: TaskFrame, delta: np.ndarray, lim: float) -> np.ndarray:
        d = np.asarray(delta, dtype=np.float64).reshape(3)
        lat = d - frame.approach_axis * float(np.dot(d, frame.approach_axis))
        n = float(np.linalg.norm(lat))
        if n > lim > 0.0:
            lat = lat * (lim / n)
        return lat

    def _dual_spiral_step(
        self,
        frame: TaskFrame,
        wrench_tool6: np.ndarray,
        wrist: np.ndarray,
        v_tool: np.ndarray,
        *,
        priv_lat_vec: np.ndarray | None = None,
        priv_lat_m: float | None = None,
    ) -> SearchStepResult:
        """Right-led on-surface spiral + contact-force hold; left barely moves."""
        cfg = self.config
        if self._rebias_spiral:
            self._spiral_origin = wrist.copy()
            # Full rebias (surface/jam/tip_soft seek) zeros theta; tip_resync keeps it.
            if not bool(getattr(self, "_rebias_keep_theta", False)):
                self._theta = 0.0
            self._rebias_spiral = False
            self._rebias_keep_theta = False

        chord = self._spiral_chord(frame)
        share = self._left_share()  # small: tray almost still
        right_lim = float(cfg.max_lat_step_m) * float(np.clip(cfg.right_lat_scale, 0.15, 1.0))
        left_lim = float(cfg.max_lat_step_m) * max(share, 0.08)
        # Right does most of the spiral XY.
        right_lat = self._clip_lat_scaled(frame, chord * (1.0 - share), right_lim)
        left_lat = self._clip_lat_scaled(frame, -chord * share, left_lim)

        # Priv seek: assist near-hole OR recovery-gated far-lat (not always-on).
        seek_ok = (
            float(cfg.priv_seek_step_m) > 0.0
            and priv_lat_vec is not None
            and priv_lat_m is not None
        )
        assist_seek = (
            bool(cfg.priv_assist)
            and seek_ok
            and float(priv_lat_m) > float(cfg.priv_enter_lat_m)
        )
        rec_lat = float(getattr(cfg, "priv_recovery_lat_m", 0.0))
        recovery_seek = (
            seek_ok
            and rec_lat > 0.0
            and bool(getattr(self, "_recovery_active", False))
            and float(priv_lat_m) > rec_lat
        )
        if assist_seek or recovery_seek:
            v = np.asarray(priv_lat_vec, dtype=np.float64).reshape(3)
            v = v - frame.approach_axis * float(np.dot(v, frame.approach_axis))
            vn = float(np.linalg.norm(v))
            if vn > 1e-9:
                step = float(cfg.priv_seek_step_m)
                if recovery_seek and not assist_seek:
                    # Recovery: replace spiral with lateral pull toward hole (no axial slam).
                    step = min(max(step, 0.0005), float(priv_lat_m) * 0.40)
                    seek_lat = self._clip_lat_scaled(frame, -v * (step / vn), right_lim)
                    right_lat = seek_lat
                    left_lat = np.zeros(3, dtype=np.float64)
                else:
                    right_lat = self._clip_lat_scaled(
                        frame, right_lat - v * (step / vn), right_lim
                    )

        # Force-guided XY give (edge catch) — right only.
        f_des = np.zeros(6, dtype=np.float64)
        k_xy = np.array([cfg.admittance_k_xy, cfg.admittance_k_xy, 0.0], dtype=np.float64)
        admit_lat = frame.admit_step(
            f_des,
            wrench_tool6,
            v_tool,
            K=k_xy,
            B=cfg.admittance_b,
            comply_mask=np.array([1.0, 1.0, 0.0]),
        )
        admit_lat = self._clip_lat_scaled(frame, admit_lat, right_lim)
        right_lat = self._clip_norm(right_lat + admit_lat, cfg.max_admit_step_m)

        # Maintain surface contact force (not open-loop slam).
        # Force enough → stop press (no hold_press*0.6); avoids tip slip under load.
        fz = float(wrench_tool6[2])
        f_abs = abs(fz)
        f_des = float(cfg.contact_f_des_n)
        if f_abs < f_des:
            press = float(cfg.hold_press_m) + float(cfg.contact_press_gain) * (f_des - f_abs)
        else:
            press = 0.0
        # Near hole: extra light press only while under contact target.
        # Also boost press inside privileged reject band (monitor, not seek).
        reject_lat = float(getattr(cfg, "reject_hole_if_priv_lat_m", 0.0))
        near_band = float(cfg.priv_enter_lat_m) * 1.5
        if reject_lat > 0.0:
            near_band = max(near_band, reject_lat)
        if (
            f_abs < f_des
            and priv_lat_m is not None
            and float(priv_lat_m) <= near_band
        ):
            press = max(press, float(cfg.near_hole_press_m))
        right_ax = frame.axial_world(+self._push_sign * press)
        left_ax = frame.axial_world(-self._push_sign * press * self._left_axial_share())

        right = self._clip_norm(right_lat + right_ax, cfg.max_admit_step_m)
        left = self._clip_norm(left_lat + left_ax, cfg.max_admit_step_m)
        self._prev_wrist = wrist.copy()
        return SearchStepResult(
            False, False, "searching", right, delta_left_xyz=left
        )

    def step(
        self,
        frame: TaskFrame,
        wrench_right6: np.ndarray,
        wrist_xyz: np.ndarray,
        *,
        dt: float | None = None,
        prev_delta_xyz: np.ndarray | None = None,
        priv_lat_m: float | None = None,
        priv_along_m: float | None = None,
        priv_lat_vec: np.ndarray | None = None,
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

        # Recovery-gated seek monitor (priv lat only; no force-enter change).
        rec_lat = float(getattr(cfg, "priv_recovery_lat_m", 0.0))
        if (
            rec_lat > 0.0
            and float(cfg.priv_seek_step_m) > 0.0
            and priv_lat_m is not None
        ):
            if float(priv_lat_m) > rec_lat:
                self._recovery_streak += 1
            else:
                self._recovery_streak = 0
            was_rec = bool(self._recovery_active)
            need = max(1, int(getattr(cfg, "priv_recovery_confirm", 15)))
            self._recovery_active = self._recovery_streak >= need
            if self._recovery_active and not was_rec:
                # Rebias spiral origin nearer current tip so spiral doesn't drift further.
                self._rebias_spiral = True
        else:
            self._recovery_streak = 0
            self._recovery_active = False

        if abs(fz_raw) > cfg.f_axial_max_n or f_lat > cfg.f_axial_max_n:
            rim_stuck = (
                cfg.priv_assist
                and priv_lat_m is not None
                and priv_along_m is not None
                and float(priv_along_m) > float(cfg.priv_enter_along_max_m)
                and float(priv_lat_m) <= float(cfg.priv_enter_lat_m) * 2.0
            )
            if rim_stuck and priv_lat_vec is not None and float(cfg.priv_seek_step_m) > 0.0:
                v = np.asarray(priv_lat_vec, dtype=np.float64).reshape(3)
                v = v - frame.approach_axis * float(np.dot(v, frame.approach_axis))
                vn = float(np.linalg.norm(v))
                if vn > 1e-9:
                    step = min(
                        float(cfg.priv_seek_step_m),
                        float(priv_lat_m) * 0.45,
                    )
                    lat = -v * (step / vn)
                    lat = self._clip_lat(frame, lat)
                    self._prev_wrist = wrist.copy()
                    return SearchStepResult(
                        False,
                        False,
                        "rim_recenter",
                        lat,
                        delta_left_xyz=-self._left_share() * lat,
                    )
            retreat = frame.axial_world(-self._push_sign * cfg.retreat_step_m)
            left = -self._left_share() * retreat
            left = left - frame.approach_axis * float(np.dot(left, frame.approach_axis))
            self._prev_wrist = wrist.copy()
            return SearchStepResult(
                False, False, "force_retreat", retreat, delta_left_xyz=left
            )

        if not self._on_surface:
            self._contact_steps += 1
            v_tool = self._v_tool(frame, wrist, dt=dt, prev_delta_xyz=prev_delta_xyz)
            admit_tool = np.zeros(3, dtype=np.float64)
            admit_tool[2] = self._axial_seek_tool_z(fz, float(v_tool[2]))
            delta = self._clip_norm(frame.to_world(admit_tool), cfg.max_admit_step_m)
            force_rise = abs(fz_raw) - self._min_abs_fz_seek
            contact_ok = self._contact_steps >= cfg.contact_min_seek_steps and (
                force_rise >= cfg.contact_rise_n or abs(fz_raw) >= cfg.contact_min_n
            )
            if contact_ok:
                self._on_surface = True
                self._peak_abs_fz = abs(fz_raw)
                self._abs_fz_ema = abs(fz_raw)
                self._hole_confirm = 0
                self._baseline_fz = fz
                self._f_push_target = float(cfg.push_force_n)
                self._spiral_origin = wrist.copy()
                self._rebias_spiral = True
                self._prev_wrist = wrist.copy()
                return SearchStepResult(False, False, "surface_contact", delta)
            if self._contact_steps >= cfg.contact_seek_max_steps:
                self._prev_wrist = wrist.copy()
                return SearchStepResult(True, False, "surface_seek_timeout", delta)
            self._prev_wrist = wrist.copy()
            return SearchStepResult(False, False, "seeking_surface", delta)

        # Privilege retunes thresholds; true enter needs seat (along drop), not lat alone.
        drop_need = float(cfg.hole_detect_fz_drop_n)
        priv_near = False
        priv_far = False
        along_ok = True
        if cfg.priv_assist and priv_lat_m is not None:
            priv_near = float(priv_lat_m) <= float(cfg.priv_enter_lat_m)
            priv_far = float(priv_lat_m) >= float(cfg.priv_far_lat_m)
            if priv_far:
                drop_need = drop_need * 4.0
            # Do NOT loosen drop_need when near — false enter was from soft force gate.
        if cfg.priv_assist and bool(cfg.priv_enter_require_along):
            if priv_along_m is None:
                along_ok = False
            else:
                along_ok = float(priv_along_m) <= float(cfg.priv_enter_along_max_m)

        alpha = float(np.clip(cfg.wrench_lpf_alpha, 0.05, 0.9))
        self._abs_fz_ema = (1.0 - alpha) * self._abs_fz_ema + alpha * abs(fz_raw)
        unload = max(self._peak_abs_fz - abs(fz), self._abs_fz_ema - abs(fz))
        fz_after_ok = abs(fz) <= float(
            getattr(cfg, "hole_fz_max_after_drop_n", cfg.hole_f_max_n)
        )
        fxy_ok = f_lat <= float(getattr(cfg, "hole_fxy_max_n", cfg.f_axial_max_n))
        force_ok = (
            unload >= drop_need
            and abs(fz) <= cfg.hole_f_max_n
            and fz_after_ok
            and fxy_ok
        )
        r_now = self._spiral_radius(self._theta)
        radius_ok = r_now >= float(cfg.spiral_try_min_radius_m)
        if bool(cfg.priv_enter_require_force):
            seat_force = force_ok
        else:
            # Prefer along seat; force drop is optional bonus.
            seat_force = True
        # Force enter requires spiral to have swept past min radius (reject face noise).
        # Priv along-seat path is not gated by radius (diagnostic only).
        hole_force = (
            self._steps >= cfg.min_search_steps
            and self._peak_abs_fz >= cfg.contact_min_n
            and seat_force
            and along_ok
            and ((not cfg.priv_assist) or priv_near or priv_lat_m is None)
        )
        if hole_force and force_ok and radius_ok:
            self._hole_confirm += 1
        elif hole_force and along_ok and priv_near and not force_ok:
            # Privileged along-seat path (diagnostic): tip dropped into mouth.
            self._hole_confirm += 1
        else:
            self._hole_confirm = 0
        if self._hole_confirm >= max(1, int(cfg.hole_detect_confirm)):
            if not (cfg.priv_assist and priv_far):
                if (not cfg.priv_assist) or priv_near or priv_lat_m is None:
                    # Privileged monitor reject: far tip ≠ real mouth (not seek).
                    reject_lat = float(getattr(cfg, "reject_hole_if_priv_lat_m", 0.0))
                    reject_far = (
                        force_ok
                        and radius_ok
                        and reject_lat > 0.0
                        and priv_lat_m is not None
                        and float(priv_lat_m) > reject_lat
                    )
                    if reject_far:
                        self._hole_confirm = 0
                    else:
                        reason = (
                            "hole_detected" if force_ok and radius_ok else "priv_along_seat"
                        )
                        self._prev_wrist = wrist.copy()
                        return SearchStepResult(True, True, reason, zero)

        # Ban lat-only / spiral-complete false enters (multi-agent + literature).
        if self._steps >= cfg.max_search_steps:
            self._prev_wrist = wrist.copy()
            return SearchStepResult(True, False, "search_timeout", zero)

        v_tool = self._v_tool(frame, wrist, dt=dt, prev_delta_xyz=prev_delta_xyz)
        return self._dual_spiral_step(
            frame,
            wh,
            wrist,
            v_tool,
            priv_lat_vec=priv_lat_vec,
            priv_lat_m=priv_lat_m,
        )
