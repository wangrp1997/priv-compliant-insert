"""Paper baselines for grasp-aware extrinsic tip tracking on spiral hole search.

References (ported concepts, not runtime deps):
- ConnTact SpiralToFindHole: swri-robotics/ConnTact (Apache-2.0)
- Tip servo offset: pci _tip_servo_lift_wrist_cmd
- Extrinsic estimation: sangwkim/Tactile-Estimator-Controller (factor-graph idea;
  here replaced by privileged online coupling in sim)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pci.fail_driven_hard_hqp import (
    HardHqpConfig,
    hard_hqp_wrist_delta,
    is_hard_hqp_mode,
)
from pci.tip_tracking_qp import solve_wrist_qp
from pci.track_b_clep import (
    ClepConfig,
    ClepState,
    clep_wrist_delta,
    is_clep_mode,
)
from pci.track_b_fasr import (
    FasrConfig,
    FasrState,
    fasr_wrist_delta,
    is_fasr_mode,
)
from pci.track_b_gmhqp import (
    GmhqpConfig,
    GmhqpState,
    gmhqp_wrist_delta,
    is_gmhqp_mode,
)
from pci.track_b_gmhqp_ac import (
    GmhqpAcConfig,
    GmhqpAcState,
    gmhqp_ac_wrist_delta,
    is_gmhqp_ac_mode,
)
from pci.track_b_gmhqp_compc import (
    GmhqpCompcConfig,
    GmhqpCompcState,
    gmhqp_compc_wrist_delta,
    is_gmhqp_compc_mode,
)
from pci.track_b_gmhqp_ftip import (
    GmhqpFtipConfig,
    GmhqpFtipState,
    gmhqp_ftip_wrist_delta,
    is_gmhqp_ftip_mode,
)
from pci.track_b_hybrid_gmhqp import (
    HybridGmhqpConfig,
    HybridGmhqpState,
    hybrid_gmhqp_wrist_delta,
    is_hybrid_gmhqp_mode,
)
from pci.track_b_oigs import (
    OigsConfig,
    OigsState,
    is_oigs_mode,
    oigs_wrist_delta,
)
from pci.track_b_phig import PhigConfig, is_phig_mode, phig_wrist_delta

if TYPE_CHECKING:
    from pci.coupling_ekf import CouplingEKF


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _planar(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    n = _unit(normal)
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v - n * float(np.dot(v, n))


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


def _from2(v2: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    v2 = np.asarray(v2, dtype=np.float64).reshape(2)
    return t1 * v2[0] + t2 * v2[1]


@dataclass
class TipCouplingEstimator:
    """Online wrist→tip planar coupling (scalar α ablation for priv_coupling)."""

    window: int = 24
    ema_offset: float = 0.88
    alpha_floor: float = 0.18
    alpha_ceil: float = 1.05
    _wrist_hist: list[np.ndarray] = field(default_factory=list)
    _tip_hist: list[np.ndarray] = field(default_factory=list)
    offset_wt: np.ndarray | None = None

    def reset(self, wrist: np.ndarray, tip: np.ndarray) -> None:
        w = np.asarray(wrist, dtype=np.float64).reshape(3)
        t = np.asarray(tip, dtype=np.float64).reshape(3)
        self._wrist_hist = [w.copy()]
        self._tip_hist = [t.copy()]
        self.offset_wt = w - t

    def update(self, wrist: np.ndarray, tip: np.ndarray, normal: np.ndarray | None = None) -> None:
        del normal
        w = np.asarray(wrist, dtype=np.float64).reshape(3)
        t = np.asarray(tip, dtype=np.float64).reshape(3)
        self._wrist_hist.append(w.copy())
        self._tip_hist.append(t.copy())
        if len(self._wrist_hist) > max(4, self.window):
            self._wrist_hist = self._wrist_hist[-self.window :]
            self._tip_hist = self._tip_hist[-self.window :]
        off = w - t
        if self.offset_wt is None:
            self.offset_wt = off.copy()
        else:
            a = float(self.ema_offset)
            self.offset_wt = a * self.offset_wt + (1.0 - a) * off

    def alpha_planar(self, normal: np.ndarray) -> float:
        if len(self._wrist_hist) < 6:
            return 1.0
        w0, w1 = self._wrist_hist[0], self._wrist_hist[-1]
        t0, t1 = self._tip_hist[0], self._tip_hist[-1]
        dw = _planar(w1 - w0, normal)
        dt = _planar(t1 - t0, normal)
        nw = float(np.linalg.norm(dw))
        nt = float(np.linalg.norm(dt))
        if nw < 1e-7:
            return self.alpha_floor
        return float(np.clip(nt / nw, self.alpha_floor, self.alpha_ceil))


@dataclass
class PlanarCouplingEstimator:
    """2×2 wrist→tip planar coupling C in contact tangent basis (t1, t2)."""

    window: int = 24
    c_reg: float = 0.15
    slip_reset_dw_m: float = 0.0012
    alpha_floor: float = 0.18
    alpha_ceil: float = 1.05
    _wrist_hist: list[np.ndarray] = field(default_factory=list)
    _tip_hist: list[np.ndarray] = field(default_factory=list)
    _C: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=np.float64))

    def reset(self, wrist: np.ndarray, tip: np.ndarray) -> None:
        w = np.asarray(wrist, dtype=np.float64).reshape(3)
        t = np.asarray(tip, dtype=np.float64).reshape(3)
        self._wrist_hist = [w.copy()]
        self._tip_hist = [t.copy()]
        self._C = np.eye(2, dtype=np.float64)

    def _pair_deltas(
        self, normal: np.ndarray, i0: int, i1: int
    ) -> tuple[np.ndarray, np.ndarray]:
        n = _unit(normal)
        t1, t2 = _tangent_basis(n)
        w0, w1 = self._wrist_hist[i0], self._wrist_hist[i1]
        t0, t1p = self._tip_hist[i0], self._tip_hist[i1]
        dw = _to2(_planar(w1 - w0, n), t1, t2)
        dt = _to2(_planar(t1p - t0, n), t1, t2)
        return dw, dt

    def _slip_reset(self) -> None:
        self._wrist_hist = self._wrist_hist[-3:]
        self._tip_hist = self._tip_hist[-3:]
        self._C = np.eye(2, dtype=np.float64)

    def _fit_c(self, normal: np.ndarray) -> None:
        if len(self._wrist_hist) < 3:
            return
        n = _unit(normal)
        t1, t2 = _tangent_basis(n)
        dw_rows: list[np.ndarray] = []
        dt_rows: list[np.ndarray] = []
        for i in range(1, len(self._wrist_hist)):
            w0, w1 = self._wrist_hist[i - 1], self._wrist_hist[i]
            t0, t1p = self._tip_hist[i - 1], self._tip_hist[i]
            dw = _to2(_planar(w1 - w0, n), t1, t2)
            dt = _to2(_planar(t1p - t0, n), t1, t2)
            if float(np.linalg.norm(dw)) > 1e-9:
                dw_rows.append(dw)
                dt_rows.append(dt)
        if len(dw_rows) < 2:
            return
        w_mat = np.stack(dw_rows, axis=0)
        d_mat = np.stack(dt_rows, axis=0)
        wtw = w_mat.T @ w_mat + self.c_reg * np.eye(2)
        wtd = w_mat.T @ d_mat
        try:
            c_t = np.linalg.solve(wtw, wtd)
        except np.linalg.LinAlgError:
            return
        self._C = c_t.T
        u, s, vt = np.linalg.svd(self._C)
        s = np.clip(s, self.alpha_floor, self.alpha_ceil)
        self._C = u @ np.diag(s) @ vt

    def update(self, wrist: np.ndarray, tip: np.ndarray, normal: np.ndarray) -> None:
        w = np.asarray(wrist, dtype=np.float64).reshape(3)
        t = np.asarray(tip, dtype=np.float64).reshape(3)
        self._wrist_hist.append(w.copy())
        self._tip_hist.append(t.copy())
        if len(self._wrist_hist) > max(4, self.window):
            self._wrist_hist = self._wrist_hist[-self.window :]
            self._tip_hist = self._tip_hist[-self.window :]
        if len(self._wrist_hist) < 2:
            return
        dw, dt = self._pair_deltas(normal, -2, -1)
        dw_n = float(np.linalg.norm(dw))
        dt_n = float(np.linalg.norm(dt))
        if dw_n > self.slip_reset_dw_m and dt_n < 0.25 * dw_n:
            self._slip_reset()
            return
        self._fit_c(normal)

    def invert_planar(
        self, dt_des: np.ndarray, normal: np.ndarray, max_step_m: float
    ) -> np.ndarray:
        """Map desired tip planar delta to wrist planar delta: dw = C^{-1} dt_des."""
        n = _unit(normal)
        t1, t2 = _tangent_basis(n)
        dt2 = _to2(_planar(dt_des, n), t1, t2)
        try:
            dw2 = np.linalg.solve(self._C + self.c_reg * np.eye(2), dt2)
        except np.linalg.LinAlgError:
            dw2 = dt2
        return clip_step(_from2(dw2, t1, t2), max_step_m)


def clip_step(delta: np.ndarray, max_m: float) -> np.ndarray:
    d = np.asarray(delta, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(d))
    if n > max_m > 0.0:
        return d * (max_m / n)
    return d


def project_to_contact_plane(
    hold_r: np.ndarray, *, anchor: np.ndarray, normal: np.ndarray
) -> np.ndarray:
    """Project hold position onto contact plane through anchor."""
    out = np.asarray(hold_r, dtype=np.float64).reshape(3).copy()
    n = _unit(normal)
    anchor = np.asarray(anchor, dtype=np.float64).reshape(3)
    depth = float(np.dot(out - anchor, n))
    return out - n * depth


def apply_along_hold_scalar(
    hold_r: np.ndarray,
    *,
    along_now_m: float,
    along0_m: float,
    hole_axis: np.ndarray,
    along_slack_m: float,
    max_correct_m: float = 0.0015,
    hard_along_m: float | None = None,
) -> np.ndarray:
    out = np.asarray(hold_r, dtype=np.float64).reshape(3).copy()
    limit = along0_m + along_slack_m
    if hard_along_m is not None:
        limit = min(limit, along0_m + hard_along_m)
    if along_now_m <= limit:
        return out
    hole_u = _unit(hole_axis)
    excess = along_now_m - limit
    corr = min(max_correct_m, 0.35 * excess)
    if hard_along_m is not None and excess > hard_along_m:
        corr = min(0.012, 0.65 * excess)
    return out - hole_u * corr


def surface_plane_lock(
    hold_r: np.ndarray,
    *,
    anchor: np.ndarray,
    normal: np.ndarray,
    max_lift_m: float,
    max_press_m: float,
) -> np.ndarray:
    out = np.asarray(hold_r, dtype=np.float64).reshape(3).copy()
    n = _unit(normal)
    anchor = np.asarray(anchor, dtype=np.float64).reshape(3)
    depth = float(np.dot(out - anchor, n))
    if depth > max_lift_m:
        out = out - n * (depth - max_lift_m)
    elif depth < -max_press_m:
        out = out - n * (depth + max_press_m)
    return out


def _is_planar_c_mode(mode: str) -> bool:
    return mode in ("planar_c", "planar_c_matrix", "s8", "ekf_qp", "s9")


def _is_oracle_tip_mode(mode: str) -> bool:
    """Privileged tip→mouth servo (diagnostic upper bound; not deployable).

    Includes ``priv_oracle_tip_obs`` (INVALID tip_gt+noise) and Track-A
    ``track_a_est_oracle`` / ``track_a_theory_est`` / ``track_a_tip_axis``
    (tip_hat [+axis_hat] + same Oracle-v2 law).
    """
    return mode in (
        "priv_oracle_tip_servo",
        "oracle_tip",
        "oracle",
        "priv_oracle_tip_obs",
        "track_a_est_oracle",
        "est_oracle_tip_servo",
        "est_oracle",
        "track_a_theory_est",
        "track_a_theory_ekf",
        "theory_est_oracle",
        "tec_slim_oracle",
        "track_a_tip_axis",
        "track_a_tip_axis_ekf",
        "tip_axis_oracle",
        "tec_tip_axis_oracle",
    )


def is_track_a_est_mode(mode: str) -> bool:
    """Path-A estimator tip_hat (no peg xpos / no tip_gt+noise in control)."""
    return mode in (
        "track_a_est_oracle",
        "est_oracle_tip_servo",
        "est_oracle",
        # TEC-slim theory EKF (docs/TRACK_A_THEORY_ESTIMATOR.md); wire separately.
        "track_a_theory_est",
        "track_a_theory_ekf",
        "theory_est_oracle",
        "tec_slim_oracle",
        "track_a_tip_axis",
        "track_a_tip_axis_ekf",
        "tip_axis_oracle",
        "tec_tip_axis_oracle",
    )


def _is_track_a_est_mode(mode: str) -> bool:
    return is_track_a_est_mode(mode)


def _is_ekf_qp_mode(mode: str) -> bool:
    return mode in ("ekf_qp", "s9")


def observe_tip_planar(
    tip_gt: np.ndarray,
    normal: np.ndarray,
    *,
    sigma_m: float,
    rng: np.random.Generator,
    lag_buf: list[np.ndarray] | None = None,
    lag_frames: int = 0,
) -> tuple[np.ndarray, dict]:
    """Sim proxy for tactile tip estimator (observability bridge).

    ``tip_hat = tip_lag + ε``, ``ε ~ N(0, σ² I)`` in the contact plane.
    Ground-truth tip must stay for logging / SUCCESS_STANDARD audit only —
    not for the control loop when this hook is active.
    """
    tip_use = np.asarray(tip_gt, dtype=np.float64).reshape(3).copy()
    lag_n = max(0, int(lag_frames))
    if lag_buf is not None and lag_n > 0:
        lag_buf.append(tip_use.copy())
        keep = lag_n + 1
        if len(lag_buf) > keep:
            del lag_buf[: len(lag_buf) - keep]
        # Index 0 is the oldest sample (= lag_n frames behind when buffer full).
        if len(lag_buf) > lag_n:
            tip_use = lag_buf[0].copy()
    sig = float(sigma_m)
    if sig > 0.0:
        t1, t2 = _tangent_basis(normal)
        eps = np.asarray(rng.normal(0.0, sig, size=2), dtype=np.float64)
        tip_use = tip_use + t1 * float(eps[0]) + t2 * float(eps[1])
    return tip_use, {
        "obs_sigma_m": float(sig),
        "obs_lag_frames": int(lag_n),
    }


def priv_oracle_mouth_target(
    *,
    tip: np.ndarray,
    hole_center: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    """Planar tip→mouth target from privileged tip + hole (Agent-D oracle)."""
    tip_p = np.asarray(tip, dtype=np.float64).reshape(3)
    hole = np.asarray(hole_center, dtype=np.float64).reshape(3)
    return tip_p + _planar(hole - tip_p, normal)


def spiral_force_gate(
    *,
    contact_resid_n: float,
    f_des_n: float,
    track: float,
    max_step_m: float,
    ax_step: float,
    grasp_slip_m: float = 0.0,
    planar_gate_n: float = 0.052,
    overload_n: float = 0.070,
    overload_planar_scale: float = 0.12,
    planar_ramp: float = 1.0,
    heavy_unload_m: float = 0.0015,
    grasp_slip_soft_m: float = 0.008,
    gate_axial: bool = True,
) -> tuple[float, float, float, dict]:
    """Scale planar chase when wrist residual overloads or peg slips in frozen grasp."""
    scale = 1.0
    if contact_resid_n > planar_gate_n:
        scale = float(overload_planar_scale)
    scale *= max(float(planar_ramp), 0.08)
    if grasp_slip_m > grasp_slip_soft_m:
        slip_ex = min(0.05, float(grasp_slip_m - grasp_slip_soft_m))
        scale *= max(0.05, 1.0 - 3.5 * slip_ex)
    if grasp_slip_m > 0.02:
        scale *= max(0.02, 0.05 * (0.02 / max(grasp_slip_m, 1e-6)))
    track_eff = float(track) * scale
    max_eff = float(max_step_m) * scale
    ax_out = float(ax_step)
    if gate_axial:
        # Only unload on heavy overload. Soft over-force keeps light probe press
        # so spiral can drop into the hole without priv lat.
        if contact_resid_n > overload_n:
            over = max(float(contact_resid_n) - float(f_des_n), 0.0)
            ax_out = min(
                ax_out,
                -min(
                    heavy_unload_m,
                    heavy_unload_m * max(1.0, over / max(float(f_des_n), 1e-6)),
                ),
            )
    return track_eff, max_eff, ax_out, {
        "force_scale": scale,
        "contact_resid_n": float(contact_resid_n),
        "grasp_slip_m": float(grasp_slip_m),
    }


@dataclass
class SpiralForceHoleDetectState:
    peak_abs_fz: float = 0.0
    abs_fz_ema: float = 0.0
    baseline_fz: float | None = None
    peak_contact_resid: float = 0.0
    hole_confirm: int = 0
    entered: bool = False


def update_spiral_force_hole_detect(
    st: SpiralForceHoleDetectState,
    *,
    fz_axial: float,
    f_lat: float,
    spiral_radius_m: float,
    step_idx: int,
    search_cfg: dict,
    contact_resid_n: float | None = None,
    tip_lat_m: float | None = None,
) -> bool:
    """Deployable F/T hole seat detect (mirrors compliant.search, no priv motion)."""
    fz = float(fz_axial)
    fz_raw = abs(fz)
    use_resid = bool(search_cfg.get("contact_use_residual", False)) and (
        contact_resid_n is not None
    )
    drop_need = float(search_cfg.get("hole_detect_fz_drop_n", 0.85))
    if use_resid:
        drop_need = float(
            search_cfg.get(
                "hole_detect_resid_drop_n",
                min(0.025, drop_need * 0.04),
            )
        )
    alpha = float(np.clip(search_cfg.get("wrench_lpf_alpha", 0.35), 0.05, 0.9))
    if st.baseline_fz is None:
        st.baseline_fz = float(fz)
    if st.abs_fz_ema <= 0.0:
        st.abs_fz_ema = fz_raw
    else:
        st.abs_fz_ema = (1.0 - alpha) * st.abs_fz_ema + alpha * fz_raw
    st.peak_abs_fz = max(st.peak_abs_fz, fz_raw)
    if use_resid:
        cr = max(float(contact_resid_n), 0.0)
        st.peak_contact_resid = max(st.peak_contact_resid, cr)
        unload = max(st.peak_contact_resid - cr, st.peak_abs_fz - abs(fz))
        contact_min = float(search_cfg.get("contact_min_n", 0.35)) * 0.08
        # Mouth probe: require a real rim/seat peak before residual drop counts.
        if float(spiral_radius_m) <= float(
            search_cfg.get("mouth_max_lat_m", 0.0045)
        ) * 1.5:
            contact_min = max(
                contact_min,
                float(search_cfg.get("mouth_probe_peak_min_n", 0.08)),
            )
        peak_ok = st.peak_contact_resid >= contact_min
    else:
        unload = max(st.peak_abs_fz - abs(fz), st.abs_fz_ema - abs(fz))
        contact_min = float(search_cfg.get("contact_min_n", 0.35))
        peak_ok = st.peak_abs_fz >= contact_min
    fz_after_ok = abs(fz) <= float(
        search_cfg.get("hole_fz_max_after_drop_n", search_cfg.get("hole_f_max_n", 6.0))
    )
    fxy_ok = float(f_lat) <= float(search_cfg.get("hole_fxy_max_n", 5.0))
    # Residual contact gate is the deployable signal; wrist Fxy often includes
    # grasp/bias and falsely blocks mouth-probe enter (ep05).
    if use_resid and bool(search_cfg.get("hole_detect_ignore_fxy_with_resid", True)):
        fxy_ok = True
    force_ok = unload >= drop_need and fz_after_ok and fxy_ok
    if not use_resid:
        force_ok = force_ok and abs(fz) <= float(search_cfg.get("hole_f_max_n", 6.0))
    radius_min_try = float(search_cfg.get("spiral_try_min_radius_m", 0.012))
    radius_mouth = float(search_cfg.get("mouth_max_lat_m", 0.0045))
    # Inward planned spiral starts at large r. Soft-contact latch leaves a high
    # residual peak; any later settle looks like "unload" and falsely fires hole
    # detect (ep01 i=9 lat~16mm, ep03 i=4 lat~48mm). Enter only near mouth radius.
    if bool(search_cfg.get("hole_detect_require_near_mouth", True)):
        mouth_scale = float(
            search_cfg.get("hole_detect_mouth_radius_scale", 2.0)
        )
        radius_ok = float(spiral_radius_m) <= radius_mouth * max(1.0, mouth_scale)
    else:
        # Legacy outward / try-radius gate.
        radius_ok = float(spiral_radius_m) >= radius_min_try or float(
            spiral_radius_m
        ) <= (radius_mouth * 1.5)
    # Tip must also be near mouth — commanded r alone is not enough when tip lags
    # (ep01: r_cmd=9mm but tip lat=37mm → false hole then insert flyaway).
    if tip_lat_m is not None and bool(
        search_cfg.get("hole_detect_require_tip_near_mouth", True)
    ):
        tip_scale = float(
            search_cfg.get("hole_detect_tip_mouth_scale", 2.0)
        )
        radius_ok = radius_ok and (
            float(tip_lat_m) <= radius_mouth * max(1.0, tip_scale)
        )
    min_steps = int(search_cfg.get("min_search_steps", 40))
    if float(spiral_radius_m) <= radius_mouth * 1.5:
        min_steps = min(min_steps, int(search_cfg.get("mouth_probe_min_steps", 8)))
    confirm_need = max(1, int(search_cfg.get("hole_detect_confirm", 4)))
    hole_force = step_idx >= min_steps and peak_ok and force_ok and radius_ok
    if hole_force:
        st.hole_confirm += 1
    else:
        st.hole_confirm = 0
    if st.hole_confirm >= confirm_need and not st.entered:
        st.entered = True
        return True
    return False


def baseline_hold_r(
    mode: str,
    *,
    target: np.ndarray,
    tip: np.ndarray,
    site_xyz: np.ndarray,
    offset_frozen: np.ndarray,
    coupling: TipCouplingEstimator | None,
    planar_coupling: PlanarCouplingEstimator | None = None,
    coupling_ekf: CouplingEKF | None = None,
    qp_lambda_reg: float = 0.12,
    spiral_n: np.ndarray,
    press_ax: np.ndarray,
    ax_step: float,
    track: float,
    max_step_m: float,
    along_now_m: float,
    along0_m: float,
    hole_axis: np.ndarray,
    along_slack_m: float,
    tip_anchor: np.ndarray | None = None,
    surface_lock_lift_m: float = 0.0,
    surface_lock_press_m: float = 0.0,
    hard_along_slack_m: float | None = None,
    contact_resid_n: float | None = None,
    f_des_n: float = 0.025,
    planar_gate_n: float = 0.052,
    overload_n: float = 0.070,
    overload_planar_scale: float = 0.12,
    planar_ramp: float = 1.0,
    heavy_unload_m: float = 0.0015,
    grasp_slip_m: float = 0.0,
    force_gate_enable: bool = True,
    force_gate_axial: bool = True,
    phig_cfg: PhigConfig | None = None,
    wrench_xyz: np.ndarray | None = None,
    wrench_tau_xyz: np.ndarray | None = None,
    clep_cfg: ClepConfig | None = None,
    clep_state: ClepState | None = None,
    fasr_cfg: FasrConfig | None = None,
    fasr_state: FasrState | None = None,
    oigs_cfg: OigsConfig | None = None,
    oigs_state: OigsState | None = None,
    gmhqp_cfg: GmhqpConfig | None = None,
    gmhqp_state: GmhqpState | None = None,
    gmhqp_ftip_cfg: GmhqpFtipConfig | None = None,
    gmhqp_ftip_state: GmhqpFtipState | None = None,
    gmhqp_compc_cfg: GmhqpCompcConfig | None = None,
    gmhqp_compc_state: GmhqpCompcState | None = None,
    gmhqp_ac_cfg: GmhqpAcConfig | None = None,
    gmhqp_ac_state: GmhqpAcState | None = None,
    hybrid_gmhqp_cfg: HybridGmhqpConfig | None = None,
    hybrid_gmhqp_state: HybridGmhqpState | None = None,
    hard_hqp_cfg: HardHqpConfig | None = None,
    axis_err_deg: float = 0.0,
    peg_axis: np.ndarray | None = None,
    r_cmd_m: float = 0.02,
    mouth_xyz: np.ndarray | None = None,
) -> tuple[np.ndarray, dict, float]:
    """Compute wrist hold position for paper baselines. Returns (hold_r, meta, ax_step_out)."""
    mode = str(mode).strip().lower()
    meta: dict = {"baseline": mode}
    target = np.asarray(target, dtype=np.float64).reshape(3)
    tip = np.asarray(tip, dtype=np.float64).reshape(3)
    site = np.asarray(site_xyz, dtype=np.float64).reshape(3)
    off0 = np.asarray(offset_frozen, dtype=np.float64).reshape(3)
    n = _unit(spiral_n)
    press = _unit(press_ax)
    ax_out = float(ax_step)
    paper_planar_c = _is_planar_c_mode(mode)
    oracle_tip = _is_oracle_tip_mode(mode)
    phig = is_phig_mode(mode)
    hard_hqp = is_hard_hqp_mode(mode)
    clep = is_clep_mode(mode)
    fasr = is_fasr_mode(mode)
    oigs = is_oigs_mode(mode)
    gmhqp_compc = is_gmhqp_compc_mode(mode)
    gmhqp_ac = is_gmhqp_ac_mode(mode)
    gmhqp_ftip = is_gmhqp_ftip_mode(mode)
    gmhqp = (
        is_gmhqp_mode(mode)
        and (not gmhqp_ftip)
        and (not gmhqp_compc)
        and (not gmhqp_ac)
    )
    hybrid_gmhqp = is_hybrid_gmhqp_mode(mode)
    track_use = float(track)
    max_use = float(max_step_m)

    if force_gate_enable and contact_resid_n is not None:
        slip_for_gate = float(grasp_slip_m)
        # Non-rigid grasp: tip lag looks like "slip"; crushing planar track
        # then freezes the spiral (ep01 tip_xy stuck, r never shrinks).
        # Oracle uses live tip feedback → do not gate planar chase on slip.
        # Track-B path modes: wrist/contact/object-impedance (no tip GT).
        if (
            bool(meta.get("ignore_grasp_slip_gate"))
            or paper_planar_c
            or oracle_tip
            or phig
            or hard_hqp
            or clep
            or fasr
            or oigs
            or gmhqp
            or gmhqp_ftip
            or gmhqp_compc
            or gmhqp_ac
            or hybrid_gmhqp
        ):
            slip_for_gate = 0.0
        track_use, max_use, ax_out, fg = spiral_force_gate(
            contact_resid_n=float(contact_resid_n),
            f_des_n=float(f_des_n),
            track=float(track),
            max_step_m=float(max_step_m),
            ax_step=ax_out,
            grasp_slip_m=slip_for_gate,
            planar_gate_n=float(planar_gate_n),
            overload_n=float(overload_n),
            overload_planar_scale=float(overload_planar_scale),
            planar_ramp=float(planar_ramp),
            heavy_unload_m=float(heavy_unload_m),
            gate_axial=bool(force_gate_axial),
        )
        meta.update(fg)

    err_tip = _planar(target - tip, n)
    en = float(np.linalg.norm(err_tip))

    along_limit = along0_m + along_slack_m
    if (
        paper_planar_c
        or oracle_tip
        or phig
        or hard_hqp
        or clep
        or fasr
        or oigs
        or gmhqp
        or gmhqp_ftip
        or gmhqp_compc
        or gmhqp_ac
        or hybrid_gmhqp
    ) and along_now_m > along_limit:
        ax_out = 0.0
    elif hard_along_slack_m is not None and along_now_m > along0_m + hard_along_slack_m:
        ax_out = min(ax_out, 0.0)

    if gmhqp_ac:
        # Track-B Adaptive-C: CouplingEKF C + uncertainty-weighted map on GMHQP.
        _st = gmhqp_ac_state if gmhqp_ac_state is not None else GmhqpAcState()
        hold_r, ac_meta, _st_out = gmhqp_ac_wrist_delta(
            site_xyz=site,
            tip_obj=tip,
            path_target=target,
            normal=n,
            press_ax=press,
            contact_resid_n=float(
                contact_resid_n if contact_resid_n is not None else 0.0
            ),
            axis_err_deg=float(axis_err_deg),
            peg_axis=peg_axis,
            hole_axis=hole_axis,
            r_cmd_m=float(r_cmd_m),
            ax_step=ax_out,
            state=_st,
            cfg=gmhqp_ac_cfg,
        )
        d_cmd = hold_r - site
        ax_comp = float(np.dot(d_cmd, press))
        planar = d_cmd - press * ax_comp
        planar = clip_step(planar * (track_use / max(float(track), 1e-6)), max_use)
        hold_r = site + planar + press * float(ac_meta.get("ax_step_m", ax_out))
        ax_out = float(ac_meta.get("ax_step_m", ax_out))
        meta.update(ac_meta)
        meta["tip_err_mm"] = en * 1000.0
        if gmhqp_ac_state is not None:
            gmhqp_ac_state.ekf = _st_out.ekf
            gmhqp_ac_state.C = np.asarray(_st_out.C, dtype=np.float64).reshape(2, 2)
            gmhqp_ac_state.level = str(_st_out.level)
            gmhqp_ac_state.meta = dict(_st_out.meta)
    elif gmhqp_compc:
        # Track-B CompC: contact horizon cost on GMHQP; tip_obj stays in-hand tip.
        _st = (
            gmhqp_compc_state
            if gmhqp_compc_state is not None
            else GmhqpCompcState()
        )
        hold_r, cc_meta, _st_out = gmhqp_compc_wrist_delta(
            site_xyz=site,
            tip_obj=tip,
            path_target=target,
            mouth_xyz=mouth_xyz if mouth_xyz is not None else target,
            normal=n,
            press_ax=press,
            contact_resid_n=float(
                contact_resid_n if contact_resid_n is not None else 0.0
            ),
            axis_err_deg=float(axis_err_deg),
            force_xyz=wrench_xyz,
            torque_xyz=wrench_tau_xyz,
            tip_anchor=tip_anchor,
            peg_axis=peg_axis,
            hole_axis=hole_axis,
            r_cmd_m=float(r_cmd_m),
            ax_step=ax_out,
            state=_st,
            cfg=gmhqp_compc_cfg,
        )
        d_cmd = hold_r - site
        ax_comp = float(np.dot(d_cmd, press))
        planar = d_cmd - press * ax_comp
        planar = clip_step(planar * (track_use / max(float(track), 1e-6)), max_use)
        hold_r = site + planar + press * float(cc_meta.get("ax_step_m", ax_out))
        ax_out = float(cc_meta.get("ax_step_m", ax_out))
        meta.update(cc_meta)
        meta["tip_err_mm"] = en * 1000.0
        if gmhqp_compc_state is not None:
            gmhqp_compc_state.gmhqp = _st_out.gmhqp
            gmhqp_compc_state.clep = _st_out.clep
            gmhqp_compc_state.level = str(_st_out.level)
            gmhqp_compc_state.meta = dict(_st_out.meta)
    elif gmhqp_ftip:
        # Track-B REUSE: FT extrinsic tip → GMHQP Escande tip-task.
        _st = (
            gmhqp_ftip_state
            if gmhqp_ftip_state is not None
            else GmhqpFtipState()
        )
        hold_r, ft_meta, _st_out = gmhqp_ftip_wrist_delta(
            site_xyz=site,
            tip_geom=tip,
            path_target=target,
            normal=n,
            press_ax=press,
            contact_resid_n=float(
                contact_resid_n if contact_resid_n is not None else 0.0
            ),
            axis_err_deg=float(axis_err_deg),
            force_xyz=wrench_xyz,
            torque_xyz=wrench_tau_xyz,
            tip_anchor=tip_anchor,
            peg_axis=peg_axis,
            hole_axis=hole_axis,
            r_cmd_m=float(r_cmd_m),
            ax_step=ax_out,
            state=_st,
            cfg=gmhqp_ftip_cfg,
        )
        d_cmd = hold_r - site
        ax_comp = float(np.dot(d_cmd, press))
        planar = d_cmd - press * ax_comp
        planar = clip_step(planar * (track_use / max(float(track), 1e-6)), max_use)
        hold_r = site + planar + press * float(ft_meta.get("ax_step_m", ax_out))
        ax_out = float(ft_meta.get("ax_step_m", ax_out))
        meta.update(ft_meta)
        meta["tip_err_mm"] = en * 1000.0
        if gmhqp_ftip_state is not None:
            gmhqp_ftip_state.gmhqp = _st_out.gmhqp
            gmhqp_ftip_state.clep = _st_out.clep
            gmhqp_ftip_state.tip_source = str(_st_out.tip_source)
            gmhqp_ftip_state.meta = dict(_st_out.meta)
    elif hybrid_gmhqp:
        # Track-B Hybrid: Escande tip_task ∨ planar_C ContactMotion fallback.
        _st = (
            hybrid_gmhqp_state
            if hybrid_gmhqp_state is not None
            else HybridGmhqpState()
        )
        hold_r, hyb_meta, _st_out = hybrid_gmhqp_wrist_delta(
            site_xyz=site,
            tip_obj=tip,
            path_target=target,
            normal=n,
            press_ax=press,
            contact_resid_n=float(
                contact_resid_n if contact_resid_n is not None else 0.0
            ),
            axis_err_deg=float(axis_err_deg),
            peg_axis=peg_axis,
            hole_axis=hole_axis,
            r_cmd_m=float(r_cmd_m),
            ax_step=ax_out,
            state=_st,
            cfg=hybrid_gmhqp_cfg,
        )
        d_cmd = hold_r - site
        ax_comp = float(np.dot(d_cmd, press))
        planar = d_cmd - press * ax_comp
        planar = clip_step(planar * (track_use / max(float(track), 1e-6)), max_use)
        hold_r = site + planar + press * float(hyb_meta.get("ax_step_m", ax_out))
        ax_out = float(hyb_meta.get("ax_step_m", ax_out))
        meta.update(hyb_meta)
        meta["tip_err_mm"] = en * 1000.0
        if hybrid_gmhqp_state is not None:
            hybrid_gmhqp_state.gmhqp = _st_out.gmhqp
            hybrid_gmhqp_state.task = str(_st_out.task)
            hybrid_gmhqp_state.tip_err_hist = list(_st_out.tip_err_hist)
            hybrid_gmhqp_state.sat_streak = int(_st_out.sat_streak)
            hybrid_gmhqp_state.wrist_hist = list(_st_out.wrist_hist)
            hybrid_gmhqp_state.tip_hist = list(_st_out.tip_hist)
            hybrid_gmhqp_state.C_cm = np.asarray(
                _st_out.C_cm, dtype=np.float64
            ).reshape(2, 2)
            hybrid_gmhqp_state.level = str(_st_out.level)
            hybrid_gmhqp_state.meta = dict(_st_out.meta)
    elif gmhqp:
        # Track-B GMHQP: tip-task error through planar C (not wrist TCP path).
        _st = gmhqp_state if gmhqp_state is not None else GmhqpState()
        hold_r, gmhqp_meta, _st_out = gmhqp_wrist_delta(
            site_xyz=site,
            tip_obj=tip,
            path_target=target,
            normal=n,
            press_ax=press,
            contact_resid_n=float(
                contact_resid_n if contact_resid_n is not None else 0.0
            ),
            axis_err_deg=float(axis_err_deg),
            peg_axis=peg_axis,
            hole_axis=hole_axis,
            r_cmd_m=float(r_cmd_m),
            ax_step=ax_out,
            state=_st,
            cfg=gmhqp_cfg,
        )
        d_cmd = hold_r - site
        ax_comp = float(np.dot(d_cmd, press))
        planar = d_cmd - press * ax_comp
        planar = clip_step(planar * (track_use / max(float(track), 1e-6)), max_use)
        hold_r = site + planar + press * float(gmhqp_meta.get("ax_step_m", ax_out))
        ax_out = float(gmhqp_meta.get("ax_step_m", ax_out))
        meta.update(gmhqp_meta)
        meta["tip_err_mm"] = en * 1000.0
        if gmhqp_state is not None:
            gmhqp_state.C = np.asarray(_st_out.C, dtype=np.float64).reshape(2, 2)
            gmhqp_state.last_wrist = (
                None
                if _st_out.last_wrist is None
                else np.asarray(_st_out.last_wrist, dtype=np.float64).reshape(3)
            )
            gmhqp_state.last_tip = (
                None
                if _st_out.last_tip is None
                else np.asarray(_st_out.last_tip, dtype=np.float64).reshape(3)
            )
            gmhqp_state.level = str(_st_out.level)
            gmhqp_state.meta = dict(_st_out.meta)
    elif oigs:
        # Track-B OIGS: known-path × Pfanne object-residual alpha (no tip GT).
        _rot = float(oigs_state.obj_rot_err_rad) if oigs_state is not None else 0.0
        _pos = float(oigs_state.obj_pos_err_m) if oigs_state is not None else 0.0
        hold_r, oigs_meta = oigs_wrist_delta(
            site_xyz=site,
            path_target=target,
            normal=n,
            press_ax=press,
            contact_resid_n=float(
                contact_resid_n if contact_resid_n is not None else 0.0
            ),
            r_cmd_m=float(r_cmd_m),
            ax_step=ax_out,
            obj_rot_err_rad=_rot,
            obj_pos_err_m=_pos,
            cfg=oigs_cfg,
        )
        d_cmd = hold_r - site
        ax_comp = float(np.dot(d_cmd, press))
        planar = d_cmd - press * ax_comp
        planar = clip_step(planar * (track_use / max(float(track), 1e-6)), max_use)
        hold_r = site + planar + press * float(oigs_meta.get("ax_step_m", ax_out))
        ax_out = float(oigs_meta.get("ax_step_m", ax_out))
        meta.update(oigs_meta)
        meta["tip_err_mm"] = en * 1000.0
    elif fasr:
        # Track-B FASR: known-path + F/T edge bias / center recenter (no tip GT).
        hold_r, fasr_meta, _fasr_st = fasr_wrist_delta(
            site_xyz=site,
            path_target=target,
            normal=n,
            press_ax=press,
            contact_resid_n=float(
                contact_resid_n if contact_resid_n is not None else 0.0
            ),
            r_cmd_m=float(r_cmd_m),
            ax_step=ax_out,
            force_xyz=wrench_xyz,
            state=fasr_state,
            cfg=fasr_cfg,
        )
        d_cmd = hold_r - site
        ax_comp = float(np.dot(d_cmd, press))
        planar = d_cmd - press * ax_comp
        planar = clip_step(planar * (track_use / max(float(track), 1e-6)), max_use)
        hold_r = site + planar + press * float(fasr_meta.get("ax_step_m", ax_out))
        ax_out = float(fasr_meta.get("ax_step_m", ax_out))
        meta.update(fasr_meta)
        meta["tip_err_mm"] = en * 1000.0
        if fasr_state is not None:
            fasr_state.center_offset = np.asarray(
                _fasr_st.center_offset, dtype=np.float64
            ).reshape(3)
            fasr_state.fn_ema = float(_fasr_st.fn_ema)
            fasr_state.meta = dict(_fasr_st.meta)
    elif clep:
        # Track-B CLEP: F/T contact proxy → known spiral (no tip GT).
        hold_r, clep_meta, _clep_st = clep_wrist_delta(
            site_xyz=site,
            path_target=target,
            normal=n,
            press_ax=press,
            contact_resid_n=float(
                contact_resid_n if contact_resid_n is not None else 0.0
            ),
            r_cmd_m=float(r_cmd_m),
            ax_step=ax_out,
            force_xyz=wrench_xyz,
            torque_xyz=wrench_tau_xyz,
            tip_anchor=tip_anchor,
            state=clep_state,
            cfg=clep_cfg,
        )
        d_cmd = hold_r - site
        ax_comp = float(np.dot(d_cmd, press))
        planar = d_cmd - press * ax_comp
        planar = clip_step(planar * (track_use / max(float(track), 1e-6)), max_use)
        hold_r = site + planar + press * float(clep_meta.get("ax_step_m", ax_out))
        ax_out = float(clep_meta.get("ax_step_m", ax_out))
        meta.update(clep_meta)
        meta["tip_err_mm"] = en * 1000.0
        if clep_state is not None and _clep_st.contact_ema is not None:
            clep_state.contact_ema = _clep_st.contact_ema
            clep_state.valid = bool(_clep_st.valid)
            clep_state.source = str(_clep_st.source)
            clep_state.meta = dict(_clep_st.meta)
    elif hard_hqp:
        # Track-B fail-driven: hard seat ≻ upright ≻ path (no tip GT).
        hold_r, hh_meta = hard_hqp_wrist_delta(
            site_xyz=site,
            path_target=target,
            normal=n,
            press_ax=press,
            contact_resid_n=float(contact_resid_n if contact_resid_n is not None else 0.0),
            axis_err_deg=float(axis_err_deg),
            peg_axis=peg_axis,
            hole_axis=hole_axis,
            r_cmd_m=float(r_cmd_m),
            ax_step=ax_out,
            cfg=hard_hqp_cfg,
        )
        d_cmd = hold_r - site
        ax_comp = float(np.dot(d_cmd, press))
        planar = d_cmd - press * ax_comp
        planar = clip_step(planar * (track_use / max(float(track), 1e-6)), max_use)
        hold_r = site + planar + press * float(hh_meta.get("ax_step_m", ax_out))
        ax_out = float(hh_meta.get("ax_step_m", ax_out))
        meta.update(hh_meta)
        meta["tip_err_mm"] = en * 1000.0
    elif phig:
        # Track-B: wrist follows known spiral waypoint (no tip error in law).
        # Tip argument is unused for Δw; kept for meta tip_err diagnostic only.
        hold_r, phig_meta = phig_wrist_delta(
            site_xyz=site,
            path_target=target,
            normal=n,
            press_ax=press,
            ax_step=ax_out,
            wrench_xyz=wrench_xyz,
            cfg=phig_cfg,
        )
        # Apply track/max from force gate via re-clip of planar part.
        d_cmd = hold_r - site
        ax_comp = float(np.dot(d_cmd, press))
        planar = d_cmd - press * ax_comp
        planar = clip_step(planar * (track_use / max(float(track), 1e-6)), max_use)
        hold_r = site + planar + press * ax_out
        meta.update(phig_meta)
        meta["tip_err_mm"] = en * 1000.0
        # Fall through to final clip below via early path marker.
    elif mode in (
        "conntact_tcp",
        "conntact_faithful",
        "franka_tcp",
        "franka_spiral",
        "psft_tcp",
        "psft_spiral",
    ):
        # Faithful ConnTact / Franka / PSFT: command TCP/wrist to spiral XY
        # (z via seeking), not tip-error servo. Rigid-EE assumption under grasp.
        err_tcp = _planar(target - site, n)
        d_w = clip_step(err_tcp * track_use, max_use)
        hold_r = site + d_w + press * ax_out
        if mode in ("franka_tcp", "franka_spiral"):
            meta["path"] = "franka_tcp"
        elif mode in ("psft_tcp", "psft_spiral"):
            meta["path"] = "psft_tcp"
        else:
            meta["path"] = "conntact_tcp"
        meta["tcp_err_mm"] = float(np.linalg.norm(err_tcp)) * 1000.0
    elif mode in ("conntact_wrist", "conntact", "a"):
        d_w = clip_step(err_tip * track_use, max_use)
        hold_r = site + d_w + press * ax_out
        meta["path"] = "wrist_spiral"
    elif mode in ("tip_servo_ff", "wrist_ff", "b"):
        hold_r = target + off0 + press * ax_out
        meta["path"] = "tip_servo_ff"
    elif mode in ("tip_track_live", "live", "b_prime"):
        d_w = clip_step(err_tip * track_use, max_use)
        hold_r = site + d_w + press * ax_out
        meta["path"] = "tip_track_live"
    elif oracle_tip:
        # Agent-D / Track-A: tip PD with C≈I toward mouth target.
        # Tip = GT (oracle), contact tip_hat (Track-A), or INVALID obs_noise.
        d_w = clip_step(err_tip * track_use, max_use)
        hold_r = site + d_w + press * ax_out
        if mode == "priv_oracle_tip_obs":
            meta["path"] = "priv_oracle_tip_obs"
        elif _is_track_a_est_mode(mode):
            # Theory EKF vs v0 contact tip_hat share Oracle-v2 law; path distinguishes.
            meta["path"] = (
                "track_a_tip_axis"
                if mode
                in (
                    "track_a_tip_axis",
                    "track_a_tip_axis_ekf",
                    "tip_axis_oracle",
                    "tec_tip_axis_oracle",
                )
                else (
                    "track_a_theory_est"
                    if mode
                    in (
                        "track_a_theory_est",
                        "track_a_theory_ekf",
                        "theory_est_oracle",
                        "tec_slim_oracle",
                    )
                    else "track_a_est_oracle"
                )
            )
        else:
            meta["path"] = "priv_oracle_tip_servo"
        meta["privileged"] = bool(not _is_track_a_est_mode(mode))
        meta["obs_noise"] = bool(mode == "priv_oracle_tip_obs")
        meta["track_a_tip_est"] = bool(_is_track_a_est_mode(mode))
    elif mode in ("priv_coupling", "coupling", "ours", "c"):
        assert coupling is not None
        coupling.update(site, tip, n)
        alpha = coupling.alpha_planar(n)
        d_w = clip_step(err_tip * track_use / max(alpha, 1e-6), max_use)
        meta["path"] = "priv_coupling"
        meta["alpha"] = alpha
        hold_r = site + d_w + press * ax_out
    elif _is_ekf_qp_mode(mode):
        assert coupling_ekf is not None
        coupling_ekf.step_filter(site, tip, n)
        hold_r = solve_wrist_qp(
            C=coupling_ekf.C_matrix,
            err_tip3=err_tip * track_use,
            site=site,
            normal=n,
            max_step_m=max_use,
            lambda_reg=qp_lambda_reg,
        )
        hold_r = hold_r + press * ax_out
        meta["path"] = "ekf_qp"
        meta["C_trace"] = float(np.trace(coupling_ekf.C_matrix))
        meta["ekf_just_slipped"] = int(bool(coupling_ekf.just_slipped))
        meta["ekf_tip_innov_mm"] = float(coupling_ekf.tip_innov_norm) * 1000.0
    elif paper_planar_c:
        assert planar_coupling is not None
        planar_coupling.update(site, tip, n)
        d_w = planar_coupling.invert_planar(err_tip * track_use, n, max_use)
        hold_r = site + d_w + press * ax_out
        meta["path"] = "planar_C"
        meta["C_trace"] = float(np.trace(planar_coupling._C))
    elif mode in ("planar_coupling", "s7"):
        assert coupling is not None
        coupling.update(site, tip, n)
        alpha = coupling.alpha_planar(n)
        d_w = clip_step(err_tip * track_use / max(alpha, 1e-6), max_use)
        meta["path"] = "planar_coupling_legacy"
        meta["alpha"] = alpha
        hold_r = site + d_w + press * ax_out
    else:
        hold_r = target + off0 + press * ax_out
        meta["path"] = "unknown"

    if (
        paper_planar_c
        or oracle_tip
        or phig
        or hard_hqp
        or clep
        or fasr
        or oigs
    ) and tip_anchor is not None:
        hold_r = project_to_contact_plane(hold_r, anchor=tip_anchor, normal=n)
    elif (
        not paper_planar_c
        and not oracle_tip
        and not phig
        and not hard_hqp
        and not clep
        and not fasr
        and not oigs
    ):
        hold_r = apply_along_hold_scalar(
            hold_r,
            along_now_m=along_now_m,
            along0_m=along0_m,
            hole_axis=hole_axis,
            along_slack_m=along_slack_m,
            max_correct_m=0.0015 if hard_along_slack_m is None else 0.012,
            hard_along_m=hard_along_slack_m,
        )
        if tip_anchor is not None and (surface_lock_lift_m > 0 or surface_lock_press_m > 0):
            hold_r = surface_plane_lock(
                hold_r,
                anchor=tip_anchor,
                normal=n,
                max_lift_m=surface_lock_lift_m,
                max_press_m=surface_lock_press_m,
            )

    d_cmd = hold_r - site
    dn = float(np.linalg.norm(d_cmd))
    if dn > max_use > 0.0:
        # Keep axial probe intact: clip planar only, then reinject ax_out.
        if abs(ax_out) > 1e-12:
            ax_comp = float(np.dot(d_cmd, press))
            planar = d_cmd - press * ax_comp
            pn = float(np.linalg.norm(planar))
            if pn > max_use > 0.0:
                planar = planar * (max_use / pn)
            hold_r = site + planar + press * float(ax_out)
        else:
            hold_r = site + d_cmd * (max_use / dn)
    elif abs(ax_out) > 1e-12:
        # Even without clip, enforce commanded axial (downstream may have zeroed it).
        ax_comp = float(np.dot(hold_r - site, press))
        hold_r = hold_r + press * (float(ax_out) - ax_comp)
    meta["tip_err_mm"] = en * 1000.0
    return hold_r, meta, ax_out


def conntact_hole_drop(
    tip_depth_mm: float,
    priv_fn_n: float,
    *,
    drop_depth_mm: float = 0.4,
    fn_drop_n: float = 0.015,
) -> bool:
    """ConnTact-style hole found: tip drops below surface / force dips."""
    return tip_depth_mm > drop_depth_mm or priv_fn_n < fn_drop_n


@dataclass
class TipHybridState:
    """Priority FSM for non-rigid tip spiral: SEAT → UPRIGHT → SPIRAL."""

    mode: str = "seat"
    seat_ok_streak: int = 0
    upright_ok_streak: int = 0
    # Tip-MSAR extensions (Agent-I) — superseded by Tip-STAR for deployable invent
    msar_mode: str = "spiral_track"  # mouth_seek | slip_reseat | spiral_track
    slip_reseat_left: int = 0
    # Tip-STAR: saturating track-aware recovery (no privileged mouth attractor)
    star_mode: str = "spiral_track"  # spiral_track | local_recover | grasp_restore | yield_mouth
    star_sat_streak: int = 0
    star_reseat_left: int = 0
    star_grasp_boosted: bool = False
    # Type-B latch: slip_peak only grows; must not re-arm reseat every frame.
    star_slip_armed_m: float = -1.0
    star_grasp_mean_abs_before: float = 0.0
    star_grasp_mean_abs_after: float = 0.0
    # Agent-D privileged oracle v2 (NOT deployable): slip reseat + mouth press.
    oracle_reseat_left: int = 0
    oracle_grasp_boosted: bool = False
    oracle_slip_armed_m: float = -1.0
    oracle_mouth_press_frames: int = 0


def tip_oracle_slip_trip(
    *,
    grasp_slip_m: float,
    tip_lat_m: float,
    slip_tau_m: float = 0.035,
    tip_lag_m: float = 0.008,
    mouth_m: float = 0.0045,
    slip_armed_m: float = -1.0,
) -> bool:
    """Oracle Type-B: large *new* slip, or tip stuck outside mouth under slip.

    Privileged diagnostic only — see PRIVILEGED_SENSING_DISCLOSURE / ORACLE_TIP_SERVO.
    """
    slip = float(grasp_slip_m)
    armed = float(slip_armed_m)
    if slip > float(slip_tau_m) and slip > armed + 1e-6:
        return True
    # Lag outside mouth with growing slip → reseat before chasing tip→mouth.
    if (
        slip > max(0.5 * float(slip_tau_m), 0.015)
        and float(tip_lat_m) > max(float(mouth_m) * 1.25, float(tip_lag_m))
        and slip > armed + 1e-6
    ):
        return True
    return False


def tip_msar_blend_target(
    *,
    tip: np.ndarray,
    spiral_target: np.ndarray,
    hole_center: np.ndarray,
    normal: np.ndarray,
    mouth_m: float = 0.0045,
    blend_m: float = 0.012,
) -> tuple[np.ndarray, float, str]:
    """Type A: blend spiral track error with tip→hole mouth attractor.

    α→1 when tip far from mouth so wrist pulls toward hole instead of
    chasing unreachable outer spiral waypoints.

    NOTE: hole attractor is privileged for deployable claims; prefer Tip-STAR.
    """
    tip_p = np.asarray(tip, dtype=np.float64).reshape(3)
    n = _unit(normal)
    e_mouth = _planar(np.asarray(hole_center, dtype=np.float64).reshape(3) - tip_p, n)
    rho = float(np.linalg.norm(e_mouth))
    alpha = float(np.clip((rho - float(mouth_m)) / max(float(blend_m), 1e-6), 0.0, 1.0))
    e_tip = _planar(np.asarray(spiral_target, dtype=np.float64).reshape(3) - tip_p, n)
    e = (1.0 - alpha) * e_tip + alpha * e_mouth
    target = tip_p + e
    mode = "mouth_seek" if alpha > 0.2 else "spiral_track"
    return target, alpha, mode


def tip_msar_slip_trip(
    *,
    grasp_slip_m: float,
    tip_err_m: float,
    slip_tau_m: float = 0.08,
    tip_err_tau_m: float = 0.012,
) -> bool:
    """Type B: slip or tip lag gate → freeze spiral advance + reseat."""
    return float(grasp_slip_m) > float(slip_tau_m) or float(tip_err_m) > float(
        tip_err_tau_m
    )


def tip_star_radial_lag(
    *,
    r_cmd_m: float,
    tip_rho_m: float,
    lag_margin_m: float = 0.003,
    spiral_direction: str = "inward",
) -> bool:
    """True when commanded ring and tip radius have separated past lag margin.

    - outward: planner expands ahead of tip → ``r_cmd > ρ_tip + δ``
    - inward: tip stuck outside while planner shrinks → ``ρ_tip > r_cmd + δ``

    S2 uses inward spirals; the outward-only gate left LOCAL_RECOVER at 0 frames
    on Type-A fails (v10 ep2/3/8, STAR/v11 smoke).
    """
    margin = float(lag_margin_m)
    if str(spiral_direction).lower() == "outward":
        return float(r_cmd_m) > (float(tip_rho_m) + margin)
    return float(tip_rho_m) > (float(r_cmd_m) + margin)


def tip_star_track_saturate(
    *,
    tip_err_m: float,
    r_cmd_m: float,
    tip_rho_m: float,
    tip_err_tau_m: float = 0.010,
    lag_margin_m: float = 0.003,
    spiral_direction: str = "inward",
) -> bool:
    """True when tip cannot catch a spiral ring that has already left tip radius.

    Anti-regression vs MSAR: healthy near-mouth tracks have |r_cmd − ρ_tip| ≲ δ
    (and small tip_err), so STAR stays idle (ep01/04/05/06 invariance).
    """
    if float(tip_err_m) <= float(tip_err_tau_m):
        return False
    return tip_star_radial_lag(
        r_cmd_m=r_cmd_m,
        tip_rho_m=tip_rho_m,
        lag_margin_m=lag_margin_m,
        spiral_direction=spiral_direction,
    )


def tip_star_yield_mouth(
    *,
    r_cmd_m: float,
    tip_rho_m: float,
    tip_err_m: float,
    mouth_m: float = 0.0045,
    tip_err_near_m: float = 0.008,
    force_mouth_cue: bool = False,
) -> bool:
    """Deployable mouth-local handoff: no hole PD; yield to SEAT / mouth-probe.

    Tip must already be near the mouth ring **and** tracking the waypoint.
    Do **not** yield solely because inward ``r_cmd`` shrunk to mouth while tip
    is still stuck at 8–14 mm (Type-A smoke ep03: yield 2k frames, planar 8mm).
    """
    if bool(force_mouth_cue):
        return True
    mouth = float(mouth_m)
    # Tip still outside mouth neighborhood → not a mouth handoff.
    if float(tip_rho_m) > mouth * 1.5:
        return False
    # Large tip_err means still saturating — stay on LOCAL_RECOVER, not yield.
    if float(tip_err_m) > float(tip_err_near_m):
        return False
    if float(r_cmd_m) <= mouth * 1.15:
        return True
    return float(tip_rho_m) <= mouth


def deploy_mouth_yield_cue(
    *,
    r_cmd_m: float,
    tip_rho_m: float,
    tip_err_m: float,
    mouth_m: float = 0.0045,
    tip_err_near_m: float = 0.004,
    contact_resid_n: float = 0.0,
    peak_contact_resid_n: float = 0.0,
    resid_dip_n: float = 0.05,
    seat_contact_n: float = 0.05,
    force_mouth_cue: bool = False,
    allow_resid_dip: bool = False,
) -> bool:
    """Deployable mouth yield (planner radius + F/T); no privileged hole pose.

    Strict tip-ring handoff: tip must already be inside mouth radius with small
    spiral tracking error. Residual-dip alone is off by default (premature
    yield at ρ≈mouth·1.5 aborted healthy Tip-Hybrid spirals).
    """
    mouth = float(mouth_m)
    # Hard: tip ring inside mouth (not 1.5× fringe).
    if float(tip_rho_m) > mouth * 1.05:
        return False
    if float(tip_err_m) > float(tip_err_near_m):
        return False
    if tip_star_yield_mouth(
        r_cmd_m=r_cmd_m,
        tip_rho_m=tip_rho_m,
        tip_err_m=tip_err_m,
        mouth_m=mouth_m,
        tip_err_near_m=tip_err_near_m,
        force_mouth_cue=force_mouth_cue,
    ):
        return True
    if not bool(allow_resid_dip):
        return False
    peak = float(peak_contact_resid_n)
    cur = float(contact_resid_n)
    if peak < float(seat_contact_n):
        return False
    return (peak - cur) >= float(resid_dip_n)


def tip_star_local_target(
    *,
    tip: np.ndarray,
    spiral_target: np.ndarray,
    spiral_center: np.ndarray,
    normal: np.ndarray,
    r_cmd_m: float | None = None,
    spiral_direction: str = "inward",
    mouth_m: float = 0.0045,
    shrink_step_m: float = 0.001,
) -> np.ndarray:
    """Deployable Type-A recover: phase/radius local servo, no hole attractor.

    Uses planner center (same Archimedean origin) only — never privileged mouth PD.

    - outward: pin radius to tip ring, keep spiral phase (catch phase on tip ρ)
    - inward: pull tip onto a ring at tip azimuth; ratchet ρ toward planner mouth
      even while ``ṙ`` is frozen (ep08: 1.6k recover frames stuck at ~10mm)
    """
    tip_p = np.asarray(tip, dtype=np.float64).reshape(3)
    c = np.asarray(spiral_center, dtype=np.float64).reshape(3)
    n = _unit(normal)
    tip_off = _planar(tip_p - c, n)
    tip_rho = float(np.linalg.norm(tip_off))
    if tip_rho < 1e-9:
        return tip_p.copy()
    if str(spiral_direction).lower() != "outward" and r_cmd_m is not None:
        # Inward: radial pull at tip azimuth. Ratchet toward mouth radius so freeze
        # of planned_i cannot leave tip parked on an outer ring.
        r_cmd = float(r_cmd_m)
        mouth = float(mouth_m)
        step = max(0.0, float(shrink_step_m))
        r_tgt = min(r_cmd, tip_rho)
        if step > 0.0 and tip_rho > mouth + 1e-9:
            r_tgt = min(r_tgt, max(mouth, tip_rho - step))
        r_tgt = max(mouth, min(r_tgt, tip_rho))
        return c + tip_off * (r_tgt / tip_rho)
    phase_off = _planar(np.asarray(spiral_target, dtype=np.float64).reshape(3) - c, n)
    phase_n = float(np.linalg.norm(phase_off))
    if phase_n < 1e-9:
        return tip_p.copy()
    return c + phase_off * (tip_rho / phase_n)


def tip_star_slip_trip(
    *,
    grasp_slip_m: float,
    tip_err_m: float,
    r_cmd_m: float,
    tip_rho_m: float,
    slip_tau_m: float = 0.12,
    tip_err_tau_m: float = 0.020,
    lag_margin_m: float = 0.003,
    spiral_direction: str = "inward",
    slip_armed_m: float = -1.0,
) -> bool:
    """Type B trip: large *new* slip, or tip_err only when spiral has left tip.

    Unlike MSAR tip_err-only trips, near-ring lag does not reseat.
    ``slip_armed_m`` latches the last armed peak so sticky ``slip_peak`` cannot
    re-arm GRASP_RESTORE every frame (v11 ep02: 5k+ grasp_restore, 0 local_recover).
    """
    slip = float(grasp_slip_m)
    armed = float(slip_armed_m)
    if slip > float(slip_tau_m) and slip > armed + 1e-6:
        return True
    if float(tip_err_m) <= float(tip_err_tau_m):
        return False
    return tip_star_radial_lag(
        r_cmd_m=r_cmd_m,
        tip_rho_m=tip_rho_m,
        lag_margin_m=lag_margin_m,
        spiral_direction=spiral_direction,
    )


def tip_slip_can_enter_restore(
    *,
    trip: bool,
    already_restoring: bool,
    last_trip_slip_m: float,
    slip_now_m: float,
    retrip_delta_m: float = 0.05,
) -> bool:
    """Edge/latch guard for Type-B GRASP_RESTORE.

    `grasp_slip_peak` is monotonic — without this, reseat_left is refreshed every
    frame and spiral never resumes. Re-trip only after additional slip growth.
    """
    if not bool(trip) or bool(already_restoring):
        return False
    last = float(last_trip_slip_m)
    now = float(slip_now_m)
    if last < 0.0:
        return True
    return now >= last + float(retrip_delta_m)


def tip_hybrid_select(
    st: TipHybridState,
    *,
    float_m: float,
    contact_n: float,
    axis_err_rad: float,
    seat_float_tol_m: float = 0.0015,
    seat_contact_n: float = 0.06,
    upright_soft_rad: float = 0.35,  # ~20 deg
    seat_confirm: int = 4,
    upright_confirm: int = 4,
    hard_seat: bool = False,
) -> str:
    """Craig/ConnTact hybrid: force seat first, upright second, spiral last.

    Hard SEAT only when floating / mouth-high-along (hard_seat).
    Low contact alone → soft reseat press in caller, still allow SPIRAL.
    """
    del seat_confirm, seat_contact_n, contact_n
    need_seat = bool(hard_seat) or float(float_m) > float(seat_float_tol_m)
    if need_seat:
        st.seat_ok_streak = 0
        st.upright_ok_streak = 0
        st.mode = "seat"
        return st.mode
    st.seat_ok_streak += 1
    if float(axis_err_rad) > float(upright_soft_rad):
        st.upright_ok_streak = 0
        st.mode = "upright"
        return st.mode
    st.upright_ok_streak += 1
    if st.upright_ok_streak < max(1, int(upright_confirm)):
        st.mode = "upright"
        return st.mode
    st.mode = "spiral"
    return st.mode


def tip_pivot_upright(
    *,
    site_xyz: np.ndarray,
    tip: np.ndarray,
    peg_axis: np.ndarray,
    hole_axis: np.ndarray,
    max_ang_rad: float = 0.025,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotate peg toward hole about tip (extrinsic contact pivot).

    Returns (Δwrist_pos, rotvec ω, remaining_axis_err_rad).
    Small-angle: p' = tip + R(p−tip) ⇒ Δp ≈ ω × (site − tip).
    Ref idea: Kim Active Extrinsic Contact / ConnTact Z-align before spiral.
    """
    site = np.asarray(site_xyz, dtype=np.float64).reshape(3)
    tip_p = np.asarray(tip, dtype=np.float64).reshape(3)
    peg = _unit(np.asarray(peg_axis, dtype=np.float64))
    hole = _unit(np.asarray(hole_axis, dtype=np.float64))
    if float(np.dot(peg, hole)) < 0.0:
        peg = -peg
    ax = np.cross(peg, hole)
    s = float(np.linalg.norm(ax))
    c = float(np.clip(np.dot(peg, hole), -1.0, 1.0))
    ang = float(np.arctan2(s, c))
    if ang < 1e-5 or s < 1e-12:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), ang
    ax = ax / s
    step = float(min(ang, max(0.0, float(max_ang_rad))))
    omega = ax * step
    d_pos = np.cross(omega, site - tip_p)
    return d_pos.astype(np.float64), omega.astype(np.float64), ang
