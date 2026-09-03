"""Track-B GMHQP: Grasp-Map Operational Space HQP (no Oracle tip_gt mode).

Theory: docs/TRACK_B_GRASP_MAP_HQP.md
Cites: Montana grasp map; Pfanne RA-L 2020 object impedance;
       GraspQP; Escande HQP (seat ≻ upright ≻ tip path).

Distinct from OIGS: path error is Π(p* − tip_obj), mapped through planar C
to wrist — not Π(p* − wrist)×α.

tip_obj = in-hand object tip (sim: peg tip geom as tactile/object-pose surrogate).
Forbidden: tip_gt+noise Oracle flag; FASR/PHIG/CLEP soups; per-ep patches.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _planar(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    n = _unit(normal)
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v - n * float(np.dot(v, n))


def clip_step(v: np.ndarray, max_m: float) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if max_m > 0.0 and n > max_m:
        return v * (max_m / n)
    return v


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
    return t1 * float(v2[0]) + t2 * float(v2[1])


@dataclass
class GmhqpConfig:
    """Equation parameters — fixed for smoke (not per-ep patches)."""

    k_tip: float = 1.4
    max_step_m: float = 0.012
    f_seat_n: float = 0.05
    seat_press_m: float = 0.00045
    upright_soft_deg: float = 16.0
    upright_step_m: float = 0.004
    mouth_r_m: float = 0.0045
    mouth_press_m: float = 0.0004
    # Planar coupling C: ridge toward I; EMA on finite-diff updates.
    c_ridge: float = 0.35
    c_ema: float = 0.85
    c_dw_min_m: float = 5e-5
    c_cond_max: float = 25.0


@dataclass
class GmhqpState:
    """Online planar C and last samples for Montana-style tip↔wrist map."""

    C: np.ndarray = field(
        default_factory=lambda: np.eye(2, dtype=np.float64)
    )
    last_wrist: np.ndarray | None = None
    last_tip: np.ndarray | None = None
    level: str = "tip_path"
    meta: dict = field(default_factory=dict)


def is_gmhqp_mode(mode: str) -> bool:
    m = str(mode).strip().lower()
    return m in (
        "track_b_gmhqp",
        "gmhqp",
        "grasp_map_hqp",
        "grasp_map_operational_hqp",
        "object_task_hqp",
        # Track-A wave-5: tip_hat → same GMHQP law (docs/TRACK_A_GMHQP_TIP_OBS.md)
        "track_a_gmhqp_tip_obs",
        "gmhqp_tip_obs",
        "track_a_gmhqp",
        "track_a_gmhqp_tip_fuse",
        "gmhqp_tip_fuse",
        # Wave-7: TEC joint est–ctrl → GMHQP tip task (gated ˆt)
        "track_a_tec_joint_gmhqp",
        "tec_joint_gmhqp",
        "tec_joint",
        "track_a_tec_joint",
        "gmhqp_tec_joint",
    )


def is_gmhqp_tip_obs_mode(mode: str) -> bool:
    """GMHQP control with TEC tip_hat (free-run or consistency-gated)."""
    m = str(mode).strip().lower()
    return m in (
        "track_a_gmhqp_tip_obs",
        "gmhqp_tip_obs",
        "track_a_gmhqp",
        # Wave-6: same law; ˆt = TEC-slim + PoseDiff geom soft prior
        "track_a_gmhqp_tip_fuse",
        "gmhqp_tip_fuse",
        # Wave-7: gated ˆt (TEC joint); still uses tip_ctrl path into GMHQP
        "track_a_tec_joint_gmhqp",
        "tec_joint_gmhqp",
        "tec_joint",
        "track_a_tec_joint",
        "gmhqp_tec_joint",
    )


def update_planar_coupling(
    state: GmhqpState,
    *,
    wrist: np.ndarray,
    tip_obj: np.ndarray,
    normal: np.ndarray,
    cfg: GmhqpConfig | None = None,
) -> np.ndarray:
    """Update planar C from Δtip ≈ C Δwrist (Montana / grasp kinematics proxy)."""
    cfg = cfg or GmhqpConfig()
    t1, t2 = _tangent_basis(normal)
    w = np.asarray(wrist, dtype=np.float64).reshape(3)
    t = np.asarray(tip_obj, dtype=np.float64).reshape(3)
    if state.last_wrist is not None and state.last_tip is not None:
        dw = _to2(_planar(w - state.last_wrist, normal), t1, t2)
        dt = _to2(_planar(t - state.last_tip, normal), t1, t2)
        ndw = float(np.linalg.norm(dw))
        if ndw >= float(cfg.c_dw_min_m):
            # Rank-1 update toward dt ≈ C dw, then ridge to I.
            C_prev = np.asarray(state.C, dtype=np.float64).reshape(2, 2)
            # Least-squares one-step: C_new dw = dt → outer product update.
            C_ls = np.outer(dt, dw) / max(ndw * ndw, 1e-18)
            C_blend = float(cfg.c_ema) * C_prev + (1.0 - float(cfg.c_ema)) * C_ls
            ridge = float(cfg.c_ridge)
            C_new = (1.0 - ridge) * C_blend + ridge * np.eye(2, dtype=np.float64)
            try:
                cond = float(np.linalg.cond(C_new))
            except np.linalg.LinAlgError:
                cond = 1e9
            if cond > float(cfg.c_cond_max) or not np.isfinite(cond):
                C_new = np.eye(2, dtype=np.float64)
            state.C = C_new
    state.last_wrist = w.copy()
    state.last_tip = t.copy()
    return np.asarray(state.C, dtype=np.float64).reshape(2, 2)


def map_tip_to_wrist(
    v_tip_planar: np.ndarray,
    *,
    C: np.ndarray,
    normal: np.ndarray,
    c_ridge: float = 0.35,
) -> np.ndarray:
    """Δw_∥ = C^{+} v_tip (ridge LS); returns 3D planar wrist delta."""
    t1, t2 = _tangent_basis(normal)
    vt = _to2(np.asarray(v_tip_planar, dtype=np.float64).reshape(3), t1, t2)
    Cm = np.asarray(C, dtype=np.float64).reshape(2, 2)
    # Solve C dw = vt with Tikhonov: (CᵀC + λI) dw = Cᵀ vt
    lam = float(c_ridge)
    A = Cm.T @ Cm + lam * np.eye(2, dtype=np.float64)
    b = Cm.T @ vt
    try:
        dw2 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        dw2 = vt.copy()
    return _from2(dw2, t1, t2)


def gmhqp_wrist_delta(
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
    state: GmhqpState | None = None,
    cfg: GmhqpConfig | None = None,
) -> tuple[np.ndarray, dict, GmhqpState]:
    """One-step Escande HQP with tip-task error mapped through C (no wrist-path chase).

    Level-0 SEAT / Level-1 UPRIGHT hard; Level-2 tip path via C⁺ K e_tip.
    """
    cfg = cfg or GmhqpConfig()
    st = state if state is not None else GmhqpState()
    site = np.asarray(site_xyz, dtype=np.float64).reshape(3)
    tip = np.asarray(tip_obj, dtype=np.float64).reshape(3)
    tgt = np.asarray(path_target, dtype=np.float64).reshape(3)
    n = _unit(normal)
    press = _unit(press_ax)

    C = update_planar_coupling(
        st, wrist=site, tip_obj=tip, normal=n, cfg=cfg
    )

    e_tip = _planar(tgt - tip, n)
    e_wrist = _planar(tgt - site, n)  # diagnostic only (OIGS would chase this)
    tip_err = float(np.linalg.norm(e_tip))
    wrist_err = float(np.linalg.norm(e_wrist))

    meta: dict = {
        "path": "track_b_gmhqp",
        "privileged": False,
        "theory": "montana_grasp_map+pfanne+escande_hqp",
        "tip_task_err_mm": tip_err * 1000.0,
        "wrist_path_err_mm": wrist_err * 1000.0,
        "C_fro": float(np.linalg.norm(C, ord="fro")),
    }

    d_w = np.zeros(3, dtype=np.float64)
    ax = float(ax_step)

    if float(contact_resid_n) < float(cfg.f_seat_n):
        st.level = "seat"
        ax = max(ax, float(cfg.seat_press_m))
        meta["path_scale"] = 0.0
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
    else:
        st.level = "mouth" if float(r_cmd_m) <= float(cfg.mouth_r_m) + 1e-9 else "tip_path"
        v_tip = e_tip * float(cfg.k_tip)
        d_w = map_tip_to_wrist(
            v_tip, C=C, normal=n, c_ridge=float(cfg.c_ridge)
        )
        if st.level == "mouth":
            ax = max(ax, float(cfg.mouth_press_m))
        meta["path_scale"] = 1.0

    d_w = clip_step(_planar(d_w, n), float(cfg.max_step_m))
    hold = site + d_w + press * ax
    meta["hqp_level"] = st.level
    meta["ax_step_m"] = float(ax)
    meta["path_err_mm"] = tip_err * 1000.0  # tip-frame (not wrist)
    st.level = str(meta["hqp_level"])
    st.meta = dict(meta)
    return hold, meta, st
