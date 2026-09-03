"""Track-A TEC-slim extrinsic tip (+axis) state EKF (theory observer).

Estimates tip planar state, peg axis, and grasp coupling C under non-rigid grasp.
Does NOT assume tip = wrist + const offset (failed empirically and theoretically).

Cites / spirit (no GTSAM runtime dep):
- Kim et al. ICRA 2023 TEC — ContactMotion / extrinsic contact factors
  (refs/Tactile-Estimator-Controller)
- Bronars et al. ICRA 2024 TEXterity — continuous extrinsic pose (tip + orientation)
- Kim & Rodriguez ICRA 2022 Active Extrinsic Contact — wrench / contact geometry
- Doshi / pci.track_b_clep — wrist F/T lever-arm soft measurement
- Pfanne / pci.coupling_ekf — slip-aware C process

Design: docs/TRACK_A_THEORY_ESTIMATOR.md · docs/TRACK_A_TIP_AXIS_OBSERVER.md
Forbidden: tip_gt, tip_gt+noise, peg xpos in tip_hat; freeze/threshold patches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pci.track_b_clep import estimate_contact_from_wrench


def _unit(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(x))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return x / n


def _planar(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    n = _unit(normal)
    x = np.asarray(v, dtype=np.float64).reshape(3)
    return x - n * float(np.dot(x, n))


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


def _from2(xy: np.ndarray, origin: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(2)
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    return o + xy[0] * t1 + xy[1] * t2


@dataclass
class TheoryEstConfig:
    """Process / measurement noise (information form — not success thresholds)."""

    process_tip: float = 2e-6
    process_c: float = 2e-5
    process_axis: float = 5e-5  # tip+axis observer
    meas_contact_var: float = 2e-7  # residual-priv extrinsic contact
    meas_wrench_var: float = 4e-5  # soft Active Extrinsic / Doshi
    # TEC PoseDiff / TacGraph in-hand object tip soft prior (weaker than contact).
    # Wave-6: prevents ˆt free-run when extrinsic contact intermittent.
    meas_geom_tip_var: float = 1.5e-5
    meas_axis_finger_var: float = 8e-3  # finger proprio axis (eligible)
    meas_axis_wrist_var: float = 4e-2  # wrist approach soft prior
    meas_axis_contact_var: float = 1e-2  # intermittent extrinsic axis cue
    slip_dw_m: float = 0.0012
    slip_ratio: float = 0.25
    slip_P_c: float = 0.25
    slip_P_axis: float = 0.35
    alpha_floor: float = 0.15
    alpha_ceil: float = 1.2
    wrench_f_min_n: float = 0.02  # measurement validity only
    origin_refresh_along: bool = True
    # Soft PoseDiff prior from in-hand geom tip (GMHQP surrogate / tactile object pose).
    geom_tip_prior: bool = False


@dataclass
class TheoryTipEstimate:
    tip_hat: np.ndarray
    axis_hat: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )
    contact_n: int = 0
    contact_force_n: float = 0.0
    seated: bool = False
    source: str = "predict"
    tip_err_to_gt_m: float = float("nan")
    axis_err_to_gt_rad: float = float("nan")
    uncert_m: float = float("nan")
    uncert_axis_rad: float = float("nan")
    C: np.ndarray = field(default_factory=lambda: np.eye(2))
    meta: dict = field(default_factory=dict)


class ExtrinsicTipStateEKF:
    """TEC-slim EKF: tip planar + C; wrench + intermittent contact measurements."""

    def __init__(self, cfg: TheoryEstConfig | None = None) -> None:
        self.cfg = cfg or TheoryEstConfig()
        self._x = np.zeros(6, dtype=np.float64)
        self._P = np.eye(6, dtype=np.float64) * 0.02
        self._initialized = False
        self._wrist_prev: np.ndarray | None = None
        self._origin = np.zeros(3, dtype=np.float64)  # plane origin (along bookkeeping)
        self._along0 = 0.0
        self._last_dw2 = np.zeros(2, dtype=np.float64)
        self._last_tip2: np.ndarray | None = None
        self.just_slipped = False
        self._last_source = "init"
        # Wave-7 TEC joint: last contact / geom innovation + NIS (Kalman gate).
        self.last_contact_innov_m = float("nan")
        self.last_contact_nis = float("nan")
        self.last_geom_innov_m = float("nan")
        self.last_geom_nis = float("nan")
        self.had_contact_meas = False

    @property
    def initialized(self) -> bool:
        return bool(self._initialized)

    def _C(self) -> np.ndarray:
        return np.array(
            [
                [self._x[2], self._x[3]],
                [self._x[4], self._x[5]],
            ],
            dtype=np.float64,
        )

    def _clip_c(self) -> None:
        c = self._C()
        u, s, vt = np.linalg.svd(c)
        s = np.clip(s, self.cfg.alpha_floor, self.cfg.alpha_ceil)
        c = u @ np.diag(s) @ vt
        self._x[2:6] = np.array([c[0, 0], c[0, 1], c[1, 0], c[1, 1]])

    def reset(
        self,
        *,
        wrist_pos: np.ndarray,
        plane_n: np.ndarray,
        tip_seed: np.ndarray | None = None,
    ) -> TheoryTipEstimate:
        """Init tip state. Prefer tip_seed only if disclosed; else wrist planar prior."""
        wrist = np.asarray(wrist_pos, dtype=np.float64).reshape(3)
        n = _unit(plane_n)
        tip = (
            np.asarray(tip_seed, dtype=np.float64).reshape(3)
            if tip_seed is not None
            else wrist.copy()
        )
        self._along0 = float(np.dot(tip, n))
        self._origin = n * self._along0
        t1, t2 = _tangent_basis(n)
        t2p = _to2(tip, t1, t2)
        self._x = np.array(
            [t2p[0], t2p[1], 1.0, 0.0, 0.0, 1.0], dtype=np.float64
        )
        self._P = np.eye(6) * 0.02
        self._P[2:, 2:] *= 0.1
        self._initialized = True
        self._wrist_prev = wrist.copy()
        self._last_dw2[:] = 0.0
        self._last_tip2 = t2p.copy()
        self.just_slipped = False
        self._last_source = "seed" if tip_seed is not None else "wrist_prior"
        return self._pack(n, tip_gt=None, ncon=0, fsum=0.0, seated=False)

    def slip_reset(self) -> None:
        """Inflate C covariance; do not snap tip to wrist+offset."""
        self._x[2:6] = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._P[2:, 2:] = np.eye(4) * float(self.cfg.slip_P_c)
        self._last_dw2[:] = 0.0
        self.just_slipped = True

    def predict(self, dw_wrist3: np.ndarray, plane_n: np.ndarray) -> None:
        if not self._initialized:
            return
        t1, t2 = _tangent_basis(plane_n)
        dw2 = _to2(_planar(dw_wrist3, plane_n), t1, t2)
        c = self._C()
        # ContactMotion spirit: tip moves through C, not rigid Δw.
        self._x[0:2] = self._x[0:2] + c @ dw2
        f = np.eye(6)
        f[0:2, 2:6] = np.kron(dw2.reshape(1, 2), np.eye(2))
        q = np.diag(
            [self.cfg.process_tip, self.cfg.process_tip]
            + [self.cfg.process_c] * 4
        )
        self._P = f @ self._P @ f.T + q
        self._last_dw2 = dw2.copy()

    def _kalman_tip_update(
        self, z2: np.ndarray, meas_var: float
    ) -> tuple[float, float]:
        """Return (‖y‖, NIS=yᵀS⁻¹y). NIS used by TEC joint consistency gate."""
        h = np.zeros((2, 6))
        h[0, 0] = 1.0
        h[1, 1] = 1.0
        r = np.eye(2) * float(meas_var)
        y = z2 - self._x[0:2]
        innov = float(np.linalg.norm(y))
        s = h @ self._P @ h.T + r
        try:
            s_inv = np.linalg.inv(s)
            nis = float(y.T @ s_inv @ y)
        except np.linalg.LinAlgError:
            s_inv = np.eye(2) / max(float(meas_var), 1e-18)
            nis = float(y.T @ s_inv @ y)
        k = self._P @ h.T @ s_inv
        self._x = self._x + k @ y
        self._P = (np.eye(6) - k @ h) @ self._P
        self._clip_c()
        self._last_tip2 = z2.copy()
        return innov, nis

    def update_contact_point(
        self, tip_meas3: np.ndarray, plane_n: np.ndarray
    ) -> float:
        """Harder measurement: intermittent extrinsic contact (residual priv in sim)."""
        t1, t2 = _tangent_basis(plane_n)
        z = _to2(tip_meas3, t1, t2)
        if not self._initialized:
            return float("nan")
        innov, nis = self._kalman_tip_update(z, self.cfg.meas_contact_var)
        self.last_contact_innov_m = float(innov)
        self.last_contact_nis = float(nis)
        self.had_contact_meas = True
        return innov

    def update_geom_tip(
        self, tip_geom3: np.ndarray, plane_n: np.ndarray
    ) -> float:
        """Soft PoseDiff prior: in-hand object tip (geom / tactile object-pose surrogate).

        TEC FACTORS.md PoseDiff / DispDiff + TacGraph in-hand pose factors:
        ˆt must stay information-consistent with object tip kinematics; contact
        (lower meas_contact_var) still dominates when seated. Not a freeze switch.
        """
        if not self._initialized:
            return float("nan")
        t1, t2 = _tangent_basis(plane_n)
        z = _to2(tip_geom3, t1, t2)
        innov, nis = self._kalman_tip_update(z, self.cfg.meas_geom_tip_var)
        self.last_geom_innov_m = float(innov)
        self.last_geom_nis = float(nis)
        return innov

    def update_wrench(
        self,
        *,
        wrist_pos: np.ndarray,
        force_xyz: np.ndarray,
        torque_xyz: np.ndarray,
        plane_n: np.ndarray,
    ) -> float | None:
        """Soft measurement: Active Extrinsic / Doshi lever arm (eligible)."""
        if not self._initialized:
            return None
        tip_anchor = self.tip_world(plane_n)
        c_hat, meta = estimate_contact_from_wrench(
            site_xyz=wrist_pos,
            force_xyz=force_xyz,
            torque_xyz=torque_xyz,
            normal=plane_n,
            tip_anchor=tip_anchor,
        )
        if c_hat is None:
            return None
        f = np.asarray(force_xyz, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(f)) < float(self.cfg.wrench_f_min_n):
            return None
        t1, t2 = _tangent_basis(plane_n)
        z = _to2(c_hat, t1, t2)
        innov, _nis = self._kalman_tip_update(z, self.cfg.meas_wrench_var)
        # Slip check vs last contact motion prediction.
        if (
            self._last_tip2 is not None
            and float(np.linalg.norm(self._last_dw2)) > self.cfg.slip_dw_m
        ):
            dt_pred = self._C() @ self._last_dw2
            # After update, innov already applied; use residual scale vs predict.
            if innov < self.cfg.slip_ratio * float(np.linalg.norm(dt_pred) + 1e-9):
                # Low tip motion vs wrist → possible slip/stiction; inflate C.
                if float(np.linalg.norm(dt_pred)) > 2.0 * float(innov + 1e-9):
                    self.slip_reset()
        _ = meta
        return innov

    def tip_world(self, plane_n: np.ndarray) -> np.ndarray:
        n = _unit(plane_n)
        t1, t2 = _tangent_basis(n)
        tip = _from2(self._x[0:2], self._origin, t1, t2)
        # Restore along-normal bookkeeping.
        tip = tip - n * float(np.dot(tip, n)) + n * self._along0
        return tip

    def uncert_m(self) -> float:
        return float(np.sqrt(max(self._P[0, 0] + self._P[1, 1], 0.0)))

    def _pack(
        self,
        plane_n: np.ndarray,
        *,
        tip_gt: np.ndarray | None,
        ncon: int,
        fsum: float,
        seated: bool,
    ) -> TheoryTipEstimate:
        tip = self.tip_world(plane_n)
        err = float("nan")
        if tip_gt is not None:
            err = float(
                np.linalg.norm(
                    _planar(
                        tip - np.asarray(tip_gt, dtype=np.float64).reshape(3),
                        plane_n,
                    )
                )
            )
        return TheoryTipEstimate(
            tip_hat=tip.copy(),
            axis_hat=_unit(np.asarray(plane_n, dtype=np.float64)),
            contact_n=int(ncon),
            contact_force_n=float(fsum),
            seated=bool(seated),
            source=str(self._last_source),
            tip_err_to_gt_m=err,
            uncert_m=self.uncert_m(),
            C=self._C().copy(),
            meta={
                "tip_est_backend": "theory_ekf",
                "tip_est_source": self._last_source,
                "tip_est_uncert_m": self.uncert_m(),
                "tip_est_uses_peg_xpos": False,
                "tip_est_residual_privilege": (
                    "mujoco_peg_tray_contact_pos_optional+wrench_eligible"
                    + ("+geom_tip_posediff_prior" if self.cfg.geom_tip_prior else "")
                ),
                "tip_est_just_slipped": bool(self.just_slipped),
                "tip_est_C": self._C().tolist(),
                "tip_est_geom_tip_prior": bool(self.cfg.geom_tip_prior),
            },
        )

    def step(
        self,
        *,
        wrist_pos: np.ndarray,
        plane_n: np.ndarray,
        force_xyz: np.ndarray | None = None,
        torque_xyz: np.ndarray | None = None,
        contact_tip: np.ndarray | None = None,
        contact_force_n: float = 0.0,
        contact_n: int = 0,
        tip_gt: np.ndarray | None = None,
        geom_tip: np.ndarray | None = None,
    ) -> TheoryTipEstimate:
        """One filter step: predict → geom prior → contact → wrench."""
        wrist = np.asarray(wrist_pos, dtype=np.float64).reshape(3)
        n = _unit(plane_n)
        if not self._initialized:
            _seed = contact_tip if contact_tip is not None else geom_tip
            self.reset(wrist_pos=wrist, plane_n=n, tip_seed=_seed)
        self.just_slipped = False
        self.had_contact_meas = False
        self.last_contact_innov_m = float("nan")
        self.last_contact_nis = float("nan")
        if self._wrist_prev is not None:
            self.predict(wrist - self._wrist_prev, n)
        self._last_source = "predict"

        # Soft in-hand tip prior (PoseDiff) before hard extrinsic contact.
        if (
            bool(self.cfg.geom_tip_prior)
            and geom_tip is not None
        ):
            self.update_geom_tip(geom_tip, n)
            self._last_source = "geom_prior"

        seated = bool(contact_n > 0 and contact_tip is not None)
        if contact_tip is not None and contact_n > 0:
            self.update_contact_point(contact_tip, n)
            self._last_source = (
                "geom+contact" if self._last_source == "geom_prior" else "contact_meas"
            )
            if self.cfg.origin_refresh_along:
                self._along0 = float(
                    np.dot(np.asarray(contact_tip, dtype=np.float64).reshape(3), n)
                )
                self._origin = n * self._along0

        if force_xyz is not None and torque_xyz is not None:
            innov_w = self.update_wrench(
                wrist_pos=wrist,
                force_xyz=force_xyz,
                torque_xyz=torque_xyz,
                plane_n=n,
            )
            if innov_w is not None:
                if self._last_source in ("contact_meas", "geom+contact"):
                    self._last_source = f"{self._last_source}+wrench"
                elif self._last_source == "geom_prior":
                    self._last_source = "geom+wrench"
                else:
                    self._last_source = "wrench_meas"

        self._wrist_prev = wrist.copy()
        out = self._pack(
            n,
            tip_gt=tip_gt,
            ncon=int(contact_n),
            fsum=float(contact_force_n),
            seated=seated,
        )
        out.meta["tip_est_contact_innov_m"] = float(self.last_contact_innov_m)
        out.meta["tip_est_contact_nis"] = float(self.last_contact_nis)
        out.meta["tip_est_geom_innov_m"] = float(self.last_geom_innov_m)
        out.meta["tip_est_geom_nis"] = float(self.last_geom_nis)
        out.meta["tip_est_had_contact_meas"] = bool(self.had_contact_meas)
        return out


def _axis_to_tan(a: np.ndarray, t1: np.ndarray, t2: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Unit axis → planar tangent coords (a1,a2) with an = sqrt(1-‖a‖²) ≥ 0."""
    u = _unit(a)
    if float(np.dot(u, n)) < 0.0:
        u = -u
    return np.array([float(np.dot(u, t1)), float(np.dot(u, t2))], dtype=np.float64)


def _tan_to_axis(a2: np.ndarray, t1: np.ndarray, t2: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Planar tangent coords → unit axis (hemisphere toward +n)."""
    xy = np.asarray(a2, dtype=np.float64).reshape(2)
    r2 = float(xy[0] * xy[0] + xy[1] * xy[1])
    # Soft clamp outside unit disk (large tilt / filter overshoot).
    if r2 > 0.999:
        s = float(np.sqrt(0.999 / r2))
        xy = xy * s
        r2 = 0.999
    an = float(np.sqrt(max(1.0 - r2, 1e-12)))
    return _unit(an * n + xy[0] * t1 + xy[1] * t2)


def _axis_err_rad(a_hat: np.ndarray, a_gt: np.ndarray) -> float:
    u = _unit(a_hat)
    v = _unit(a_gt)
    if float(np.dot(u, v)) < 0.0:
        u = -u
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.arccos(c))


class ExtrinsicTipAxisEKF:
    """TEC/TEXterity-slim EKF: tip planar + peg axis + C.

    State x = [tx, ty, a1, a2, c11, c12, c21, c22]^T.
    Axis world: â = normalize(an·n + a1·t1 + a2·t2), an≥0 (toward tray normal).

    Measurements:
    - intermittent extrinsic contact tip z_c (residual priv in sim)
    - wrist wrench lever tip z_w (eligible; Active Extrinsic / Doshi)
    - finger proprio axis z_a = normalize(tip − grasp) (eligible FK)
    - wrist approach soft axis prior (eligible)
    - intermittent contact-line / normal cue when multi-point extrinsic (residual priv)

    Process: ContactMotion tip through C; axis follows wrist ΔR then noise;
    slip inflates P_C and P_a — never snap tip←wrist+o or axis←peg xpos.
    """

    def __init__(self, cfg: TheoryEstConfig | None = None) -> None:
        self.cfg = cfg or TheoryEstConfig()
        self._x = np.zeros(8, dtype=np.float64)
        self._P = np.eye(8, dtype=np.float64) * 0.02
        self._initialized = False
        self._wrist_prev: np.ndarray | None = None
        self._wrist_rot_prev: np.ndarray | None = None  # 3x3
        self._origin = np.zeros(3, dtype=np.float64)
        self._along0 = 0.0
        self._last_dw2 = np.zeros(2, dtype=np.float64)
        self._last_tip2: np.ndarray | None = None
        self.just_slipped = False
        self._last_source = "init"

    @property
    def initialized(self) -> bool:
        return bool(self._initialized)

    def _C(self) -> np.ndarray:
        return np.array(
            [
                [self._x[4], self._x[5]],
                [self._x[6], self._x[7]],
            ],
            dtype=np.float64,
        )

    def _clip_c(self) -> None:
        c = self._C()
        u, s, vt = np.linalg.svd(c)
        s = np.clip(s, self.cfg.alpha_floor, self.cfg.alpha_ceil)
        c = u @ np.diag(s) @ vt
        self._x[4:8] = np.array([c[0, 0], c[0, 1], c[1, 0], c[1, 1]])

    def _clip_axis(self) -> None:
        r2 = float(self._x[2] ** 2 + self._x[3] ** 2)
        if r2 > 0.999:
            s = float(np.sqrt(0.999 / r2))
            self._x[2] *= s
            self._x[3] *= s

    def tip_world(self, plane_n: np.ndarray) -> np.ndarray:
        n = _unit(plane_n)
        t1, t2 = _tangent_basis(n)
        tip = _from2(self._x[0:2], self._origin, t1, t2)
        tip = tip - n * float(np.dot(tip, n)) + n * self._along0
        return tip

    def axis_world(self, plane_n: np.ndarray) -> np.ndarray:
        n = _unit(plane_n)
        t1, t2 = _tangent_basis(n)
        return _tan_to_axis(self._x[2:4], t1, t2, n)

    def uncert_m(self) -> float:
        return float(np.sqrt(max(self._P[0, 0] + self._P[1, 1], 0.0)))

    def uncert_axis_rad(self) -> float:
        # Local tangent cov → approximate angular uncert.
        return float(np.sqrt(max(self._P[2, 2] + self._P[3, 3], 0.0)))

    def reset(
        self,
        *,
        wrist_pos: np.ndarray,
        plane_n: np.ndarray,
        tip_seed: np.ndarray | None = None,
        axis_seed: np.ndarray | None = None,
        wrist_rot: np.ndarray | None = None,
    ) -> TheoryTipEstimate:
        wrist = np.asarray(wrist_pos, dtype=np.float64).reshape(3)
        n = _unit(plane_n)
        tip = (
            np.asarray(tip_seed, dtype=np.float64).reshape(3)
            if tip_seed is not None
            else wrist.copy()
        )
        self._along0 = float(np.dot(tip, n))
        self._origin = n * self._along0
        t1, t2 = _tangent_basis(n)
        t2p = _to2(tip, t1, t2)
        a0 = (
            np.asarray(axis_seed, dtype=np.float64).reshape(3)
            if axis_seed is not None
            else n.copy()
        )
        a2 = _axis_to_tan(a0, t1, t2, n)
        self._x = np.array(
            [t2p[0], t2p[1], a2[0], a2[1], 1.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        )
        self._P = np.eye(8) * 0.02
        self._P[4:, 4:] *= 0.1
        self._P[2:4, 2:4] = np.eye(2) * 0.08
        self._initialized = True
        self._wrist_prev = wrist.copy()
        self._wrist_rot_prev = (
            np.asarray(wrist_rot, dtype=np.float64).reshape(3, 3).copy()
            if wrist_rot is not None
            else None
        )
        self._last_dw2[:] = 0.0
        self._last_tip2 = t2p.copy()
        self.just_slipped = False
        self._last_source = "seed" if tip_seed is not None else "wrist_prior"
        return self._pack(n, tip_gt=None, axis_gt=None, ncon=0, fsum=0.0, seated=False)

    def slip_reset(self) -> None:
        """Inflate C and axis cov; do not snap tip/axis to privileged GT."""
        self._x[4:8] = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._P[4:, 4:] = np.eye(4) * float(self.cfg.slip_P_c)
        self._P[2:4, 2:4] = np.eye(2) * float(self.cfg.slip_P_axis)
        self._last_dw2[:] = 0.0
        self.just_slipped = True

    def predict(
        self,
        dw_wrist3: np.ndarray,
        plane_n: np.ndarray,
        *,
        wrist_rot: np.ndarray | None = None,
    ) -> None:
        if not self._initialized:
            return
        n = _unit(plane_n)
        t1, t2 = _tangent_basis(n)
        dw2 = _to2(_planar(dw_wrist3, plane_n), t1, t2)
        c = self._C()
        # ContactMotion: tip moves through C, not rigid Δw.
        self._x[0:2] = self._x[0:2] + c @ dw2
        # Axis follows wrist rotation (TEXterity continuous pose / rigid prior),
        # with process noise; non-rigid slip handled via inflate, not hard snap.
        if wrist_rot is not None and self._wrist_rot_prev is not None:
            r_prev = self._wrist_rot_prev
            r_now = np.asarray(wrist_rot, dtype=np.float64).reshape(3, 3)
            r_delta = r_now @ r_prev.T
            a = self.axis_world(n)
            a_new = _unit(r_delta @ a)
            self._x[2:4] = _axis_to_tan(a_new, t1, t2, n)
            self._wrist_rot_prev = r_now.copy()
        elif wrist_rot is not None:
            self._wrist_rot_prev = (
                np.asarray(wrist_rot, dtype=np.float64).reshape(3, 3).copy()
            )
        f = np.eye(8)
        # ∂(t+C dw)/∂C ≈ kron(dw^T, I) on tip block (same as tip-only EKF).
        f[0:2, 4:8] = np.array(
            [
                [dw2[0], dw2[1], 0.0, 0.0],
                [0.0, 0.0, dw2[0], dw2[1]],
            ],
            dtype=np.float64,
        )
        q = np.diag(
            [self.cfg.process_tip, self.cfg.process_tip]
            + [self.cfg.process_axis, self.cfg.process_axis]
            + [self.cfg.process_c] * 4
        )
        self._P = f @ self._P @ f.T + q
        self._last_dw2 = dw2.copy()
        self._clip_axis()
        self._clip_c()

    def _kalman_tip_update(self, z2: np.ndarray, meas_var: float) -> float:
        h = np.zeros((2, 8))
        h[0, 0] = 1.0
        h[1, 1] = 1.0
        r = np.eye(2) * float(meas_var)
        y = z2 - self._x[0:2]
        innov = float(np.linalg.norm(y))
        s = h @ self._P @ h.T + r
        k = self._P @ h.T @ np.linalg.inv(s)
        self._x = self._x + k @ y
        self._P = (np.eye(8) - k @ h) @ self._P
        self._clip_c()
        self._clip_axis()
        self._last_tip2 = z2.copy()
        return innov

    def _kalman_axis_update(self, z_a2: np.ndarray, meas_var: float) -> float:
        h = np.zeros((2, 8))
        h[0, 2] = 1.0
        h[1, 3] = 1.0
        r = np.eye(2) * float(meas_var)
        y = z_a2 - self._x[2:4]
        innov = float(np.linalg.norm(y))
        s = h @ self._P @ h.T + r
        k = self._P @ h.T @ np.linalg.inv(s)
        self._x = self._x + k @ y
        self._P = (np.eye(8) - k @ h) @ self._P
        self._clip_axis()
        self._clip_c()
        return innov

    def update_contact_point(
        self, tip_meas3: np.ndarray, plane_n: np.ndarray
    ) -> float:
        t1, t2 = _tangent_basis(plane_n)
        z = _to2(tip_meas3, t1, t2)
        if not self._initialized:
            return float("nan")
        return self._kalman_tip_update(z, self.cfg.meas_contact_var)

    def update_wrench(
        self,
        *,
        wrist_pos: np.ndarray,
        force_xyz: np.ndarray,
        torque_xyz: np.ndarray,
        plane_n: np.ndarray,
    ) -> float | None:
        if not self._initialized:
            return None
        tip_anchor = self.tip_world(plane_n)
        c_hat, meta = estimate_contact_from_wrench(
            site_xyz=wrist_pos,
            force_xyz=force_xyz,
            torque_xyz=torque_xyz,
            normal=plane_n,
            tip_anchor=tip_anchor,
        )
        if c_hat is None:
            return None
        f = np.asarray(force_xyz, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(f)) < float(self.cfg.wrench_f_min_n):
            return None
        t1, t2 = _tangent_basis(plane_n)
        z = _to2(c_hat, t1, t2)
        innov = self._kalman_tip_update(z, self.cfg.meas_wrench_var)
        if (
            self._last_tip2 is not None
            and float(np.linalg.norm(self._last_dw2)) > self.cfg.slip_dw_m
        ):
            dt_pred = self._C() @ self._last_dw2
            if innov < self.cfg.slip_ratio * float(np.linalg.norm(dt_pred) + 1e-9):
                if float(np.linalg.norm(dt_pred)) > 2.0 * float(innov + 1e-9):
                    self.slip_reset()
        _ = meta
        return innov

    def update_axis_meas(
        self, axis_meas3: np.ndarray, plane_n: np.ndarray, meas_var: float
    ) -> float | None:
        if not self._initialized:
            return None
        n = _unit(plane_n)
        t1, t2 = _tangent_basis(n)
        z = _axis_to_tan(axis_meas3, t1, t2, n)
        return self._kalman_axis_update(z, meas_var)

    def _pack(
        self,
        plane_n: np.ndarray,
        *,
        tip_gt: np.ndarray | None,
        axis_gt: np.ndarray | None,
        ncon: int,
        fsum: float,
        seated: bool,
    ) -> TheoryTipEstimate:
        tip = self.tip_world(plane_n)
        axis = self.axis_world(plane_n)
        err = float("nan")
        if tip_gt is not None:
            err = float(
                np.linalg.norm(
                    _planar(
                        tip - np.asarray(tip_gt, dtype=np.float64).reshape(3),
                        plane_n,
                    )
                )
            )
        aerr = float("nan")
        if axis_gt is not None:
            aerr = _axis_err_rad(axis, np.asarray(axis_gt, dtype=np.float64))
        return TheoryTipEstimate(
            tip_hat=tip.copy(),
            axis_hat=axis.copy(),
            contact_n=int(ncon),
            contact_force_n=float(fsum),
            seated=bool(seated),
            source=str(self._last_source),
            tip_err_to_gt_m=err,
            axis_err_to_gt_rad=aerr,
            uncert_m=self.uncert_m(),
            uncert_axis_rad=self.uncert_axis_rad(),
            C=self._C().copy(),
            meta={
                "tip_est_backend": "tip_axis_ekf",
                "tip_est_source": self._last_source,
                "tip_est_uncert_m": self.uncert_m(),
                "tip_est_uncert_axis_rad": self.uncert_axis_rad(),
                "tip_est_uses_peg_xpos": False,
                "tip_est_residual_privilege": (
                    "mujoco_peg_tray_contact_pos_optional"
                    "+wrench_eligible"
                    "+finger_proprio_axis"
                    "+wrist_approach_prior"
                ),
                "tip_est_just_slipped": bool(self.just_slipped),
                "tip_est_C": self._C().tolist(),
                "tip_est_axis_hat": axis.tolist(),
            },
        )

    def step(
        self,
        *,
        wrist_pos: np.ndarray,
        plane_n: np.ndarray,
        force_xyz: np.ndarray | None = None,
        torque_xyz: np.ndarray | None = None,
        contact_tip: np.ndarray | None = None,
        contact_force_n: float = 0.0,
        contact_n: int = 0,
        contact_normal: np.ndarray | None = None,
        finger_grasp: np.ndarray | None = None,
        wrist_rot: np.ndarray | None = None,
        wrist_approach: np.ndarray | None = None,
        tip_gt: np.ndarray | None = None,
        axis_gt: np.ndarray | None = None,
    ) -> TheoryTipEstimate:
        """Predict → tip meas → axis meas (finger / wrist / contact cue)."""
        wrist = np.asarray(wrist_pos, dtype=np.float64).reshape(3)
        n = _unit(plane_n)
        if not self._initialized:
            a_seed = wrist_approach if wrist_approach is not None else n
            self.reset(
                wrist_pos=wrist,
                plane_n=n,
                tip_seed=contact_tip,
                axis_seed=a_seed,
                wrist_rot=wrist_rot,
            )
        self.just_slipped = False
        if self._wrist_prev is not None:
            self.predict(wrist - self._wrist_prev, n, wrist_rot=wrist_rot)
        elif wrist_rot is not None:
            self._wrist_rot_prev = (
                np.asarray(wrist_rot, dtype=np.float64).reshape(3, 3).copy()
            )
        self._last_source = "predict"
        axis_updated = False

        seated = bool(contact_n > 0 and contact_tip is not None)
        if contact_tip is not None and contact_n > 0:
            self.update_contact_point(contact_tip, n)
            self._last_source = "contact_meas"
            if self.cfg.origin_refresh_along:
                self._along0 = float(
                    np.dot(np.asarray(contact_tip, dtype=np.float64).reshape(3), n)
                )
                self._origin = n * self._along0
            # Soft axis cue: seated tip contact → prefer axis toward +n
            # (Active Extrinsic: extrinsic contact constrains orientation).
            if contact_normal is not None:
                # Contact normal ≈ tray; pull axis toward hole/tray normal softly.
                self.update_axis_meas(
                    contact_normal, n, self.cfg.meas_axis_contact_var
                )
                axis_updated = True
            else:
                self.update_axis_meas(n, n, self.cfg.meas_axis_contact_var * 2.0)
                axis_updated = True

        if force_xyz is not None and torque_xyz is not None:
            innov_w = self.update_wrench(
                wrist_pos=wrist,
                force_xyz=force_xyz,
                torque_xyz=torque_xyz,
                plane_n=n,
            )
            if innov_w is not None:
                if self._last_source == "contact_meas":
                    self._last_source = "contact+wrench"
                else:
                    self._last_source = "wrench_meas"

        # Finger proprio axis: tip free-end direction from grasp centroid (eligible).
        if finger_grasp is not None:
            tip_now = self.tip_world(n)
            g = np.asarray(finger_grasp, dtype=np.float64).reshape(3)
            d = tip_now - g
            if float(np.linalg.norm(d)) > 1e-4:
                # Peg axis ≈ grasp → tip (free end below fingers).
                a_finger = _unit(d)
                self.update_axis_meas(
                    a_finger, n, self.cfg.meas_axis_finger_var
                )
                axis_updated = True
                if "wrench" in self._last_source or "contact" in self._last_source:
                    self._last_source = self._last_source + "+finger_axis"
                else:
                    self._last_source = "finger_axis"

        # Soft wrist-approach prior (eligible; weak).
        if wrist_approach is not None:
            self.update_axis_meas(
                wrist_approach, n, self.cfg.meas_axis_wrist_var
            )
            if not axis_updated and self._last_source == "predict":
                self._last_source = "wrist_axis_prior"

        self._wrist_prev = wrist.copy()
        return self._pack(
            n,
            tip_gt=tip_gt,
            axis_gt=axis_gt,
            ncon=int(contact_n),
            fsum=float(contact_force_n),
            seated=seated,
        )


def is_track_a_theory_mode(mode: str) -> bool:
    m = str(mode or "").strip().lower()
    return m in (
        "track_a_theory_est",
        "track_a_theory_ekf",
        "theory_est_oracle",
        "tec_slim_oracle",
        "track_a_tip_axis",
        "track_a_tip_axis_ekf",
        "tip_axis_oracle",
        "tec_tip_axis_oracle",
    )


def is_track_a_tip_axis_mode(mode: str) -> bool:
    m = str(mode or "").strip().lower()
    return m in (
        "track_a_tip_axis",
        "track_a_tip_axis_ekf",
        "tip_axis_oracle",
        "tec_tip_axis_oracle",
    )
