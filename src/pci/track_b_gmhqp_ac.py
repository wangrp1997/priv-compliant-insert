"""Track-B Adaptive-C GMHQP: CouplingEKF online C + uncertainty-weighted map.

Theory: docs/TRACK_B_ADAPTIVE_C_GMHQP.md
Orthogonal to COMPC (contact cost): this upgrades C estimation / C⁺, not L2 cost.

REUSE: CouplingEKF (Pfanne/bgf slip spirit) + Montana C⁺ + Escande HQP.
Forbidden: tip GT; per-ep patches; stacking COMPC in this mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pci.coupling_ekf import CouplingEKF
from pci.track_b_gmhqp import (
    _from2,
    _planar,
    _tangent_basis,
    _to2,
    _unit,
    clip_step,
)


@dataclass
class GmhqpAcConfig:
    """Fixed equation params (METHOD_GATE: no per-ep retune)."""

    k_tip: float = 1.4
    max_step_m: float = 0.012
    f_seat_n: float = 0.05
    seat_press_m: float = 0.00045
    upright_soft_deg: float = 16.0
    upright_step_m: float = 0.004
    mouth_r_m: float = 0.0045
    mouth_press_m: float = 0.0004
    # Uncertainty-aware Tikhonov on C⁺
    c_ridge0: float = 0.25
    c_ridge_p: float = 2.0
    c_ridge_innov: float = 8.0
    c_cond_max: float = 25.0
    # CouplingEKF process / slip
    ekf_process_tip: float = 1e-6
    ekf_process_c: float = 1e-5
    ekf_meas_var: float = 4e-7
    ekf_slip_ratio: float = 0.25
    ekf_slip_dw_m: float = 0.0012
    ekf_alpha_floor: float = 0.18
    ekf_alpha_ceil: float = 1.05


@dataclass
class GmhqpAcState:
    ekf: CouplingEKF | None = None
    level: str = "tip_path"
    meta: dict = field(default_factory=dict)
    C: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=np.float64))


def is_gmhqp_ac_mode(mode: str) -> bool:
    m = str(mode).strip().lower()
    return m in (
        "track_b_gmhqp_ac",
        "gmhqp_ac",
        "adaptive_c_gmhqp",
        "track_b_adaptive_c",
        "gmhqp_adaptive_c",
    )


def _make_ekf(cfg: GmhqpAcConfig) -> CouplingEKF:
    return CouplingEKF(
        process_tip=float(cfg.ekf_process_tip),
        process_c=float(cfg.ekf_process_c),
        meas_var=float(cfg.ekf_meas_var),
        slip_ratio=float(cfg.ekf_slip_ratio),
        slip_dw_m=float(cfg.ekf_slip_dw_m),
        alpha_floor=float(cfg.ekf_alpha_floor),
        alpha_ceil=float(cfg.ekf_alpha_ceil),
    )


def adaptive_ridge(
    *,
    p_c_trace: float,
    tip_innov_norm: float,
    cfg: GmhqpAcConfig,
) -> float:
    """λ = λ0 + λ_P Tr(P_C) + λ_i ‖y_tip‖ — general, not ep-gated."""
    return float(
        cfg.c_ridge0
        + float(cfg.c_ridge_p) * float(p_c_trace)
        + float(cfg.c_ridge_innov) * float(tip_innov_norm)
    )


def map_tip_to_wrist_adaptive(
    v_tip_planar: np.ndarray,
    *,
    C: np.ndarray,
    normal: np.ndarray,
    ridge: float,
    cond_max: float = 25.0,
) -> np.ndarray:
    """Δw_∥ = (CᵀC + λI)⁻¹ Cᵀ v_tip with adaptive λ."""
    t1, t2 = _tangent_basis(normal)
    vt = _to2(np.asarray(v_tip_planar, dtype=np.float64).reshape(3), t1, t2)
    Cm = np.asarray(C, dtype=np.float64).reshape(2, 2)
    try:
        cond = float(np.linalg.cond(Cm))
    except np.linalg.LinAlgError:
        cond = 1e9
    if (not np.isfinite(cond)) or cond > float(cond_max):
        Cm = np.eye(2, dtype=np.float64)
    lam = max(float(ridge), 1e-6)
    A = Cm.T @ Cm + lam * np.eye(2, dtype=np.float64)
    b = Cm.T @ vt
    try:
        dw2 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        dw2 = vt.copy()
    return _from2(dw2, t1, t2)


def update_adaptive_coupling(
    state: GmhqpAcState,
    *,
    wrist: np.ndarray,
    tip_obj: np.ndarray,
    normal: np.ndarray,
    cfg: GmhqpAcConfig | None = None,
) -> tuple[np.ndarray, float, float, bool]:
    """Step CouplingEKF; return (C, Tr(P_C), tip_innov, just_slipped)."""
    cfg = cfg or GmhqpAcConfig()
    if state.ekf is None:
        state.ekf = _make_ekf(cfg)
    ekf = state.ekf
    w = np.asarray(wrist, dtype=np.float64).reshape(3)
    t = np.asarray(tip_obj, dtype=np.float64).reshape(3)
    n = _unit(normal)
    if not ekf.initialized:
        ekf.reset(w, t, n)
    else:
        ekf.step_filter(w, t, n)
    C = ekf.C_matrix
    state.C = C.copy()
    return C, float(ekf.P_c_trace), float(ekf.tip_innov_norm), bool(ekf.just_slipped)


def gmhqp_ac_wrist_delta(
    *,
    site_xyz: np.ndarray,
    tip_obj: np.ndarray,
    path_target: np.ndarray,
    normal: np.ndarray,
    press_ax: np.ndarray,
    contact_resid_n: float,
    axis_err_deg: float,
    peg_axis: np.ndarray | None = None,
    hole_axis: np.ndarray | None = None,
    r_cmd_m: float = 0.02,
    ax_step: float = 0.0,
    state: GmhqpAcState | None = None,
    cfg: GmhqpAcConfig | None = None,
) -> tuple[np.ndarray, dict, GmhqpAcState]:
    """Escande HQP + CouplingEKF C + uncertainty-weighted C⁺ (no contact MPC)."""
    cfg = cfg or GmhqpAcConfig()
    st = state if state is not None else GmhqpAcState()
    site = np.asarray(site_xyz, dtype=np.float64).reshape(3)
    tip = np.asarray(tip_obj, dtype=np.float64).reshape(3)
    tgt = np.asarray(path_target, dtype=np.float64).reshape(3)
    n = _unit(normal)
    press = _unit(press_ax)

    C, p_tr, innov, slipped = update_adaptive_coupling(
        st, wrist=site, tip_obj=tip, normal=n, cfg=cfg
    )
    ridge = adaptive_ridge(p_c_trace=p_tr, tip_innov_norm=innov, cfg=cfg)

    e_tip = _planar(tgt - tip, n)
    tip_err = float(np.linalg.norm(e_tip))

    meta: dict = {
        "path": "track_b_gmhqp_ac",
        "privileged": False,
        "theory": "coupling_ekf_adaptive_C+escande_gmhqp",
        "tip_task_err_mm": tip_err * 1000.0,
        "C_fro": float(np.linalg.norm(C, ord="fro")),
        "P_c_trace": float(p_tr),
        "tip_innov_mm": float(innov) * 1000.0,
        "c_ridge_eff": float(ridge),
        "just_slipped": bool(slipped),
    }

    d_w = np.zeros(3, dtype=np.float64)
    ax = float(ax_step)

    if float(contact_resid_n) < float(cfg.f_seat_n):
        st.level = "seat"
        ax = max(ax, float(cfg.seat_press_m))
        meta["path_scale"] = 0.0
        meta["hqp_level"] = "seat"
    elif float(axis_err_deg) > float(cfg.upright_soft_deg):
        st.level = "upright"
        peg = _unit(peg_axis if peg_axis is not None else n)
        hole = _unit(hole_axis if hole_axis is not None else n)
        cross = np.cross(peg, hole)
        pivot = _planar(cross, n)
        pn = float(np.linalg.norm(pivot))
        if pn > 1e-9:
            d_w = pivot * (float(cfg.upright_step_m) / pn)
        meta["path_scale"] = 0.0
        meta["upright_residual_deg"] = float(axis_err_deg)
        meta["hqp_level"] = "upright"
    else:
        st.level = "mouth" if float(r_cmd_m) <= float(cfg.mouth_r_m) + 1e-9 else "tip_path"
        v_tip = e_tip * float(cfg.k_tip)
        d_w = map_tip_to_wrist_adaptive(
            v_tip,
            C=C,
            normal=n,
            ridge=ridge,
            cond_max=float(cfg.c_cond_max),
        )
        if st.level == "mouth":
            ax = max(ax, float(cfg.mouth_press_m))
        meta["path_scale"] = 1.0
        meta["hqp_level"] = st.level

    d_w = clip_step(_planar(d_w, n), float(cfg.max_step_m))
    hold = site + d_w + press * ax
    meta["ax_step_m"] = float(ax)
    meta["path_err_mm"] = tip_err * 1000.0
    st.level = str(meta["hqp_level"])
    st.meta = dict(meta)
    return hold, meta, st
