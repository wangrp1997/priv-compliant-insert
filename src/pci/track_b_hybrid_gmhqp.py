"""Track-B REUSE: Escande compose of GMHQP tip-task ⊕ S8 planar_C.

Theory: docs/TRACK_B_HYBRID_GMHQP.md §0 Search→reuse (not invent).
Cites: Escande HQP; Montana; Pfanne/ContactMotion; in-tree PlanarCouplingEstimator.

Healthy Level-2: GMHQP tip-through-C+.
Saturated F1: Escande-switch to S8/Type-A planar_C (window ContactMotion C).
No tip_gt; no FASR/OIGS restack; no parallel invented acronym-as-paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pci.track_b_gmhqp import (
    GmhqpConfig,
    GmhqpState,
    _from2,
    _planar,
    _tangent_basis,
    _to2,
    _unit,
    clip_step,
    map_tip_to_wrist,
    update_planar_coupling,
)


@dataclass
class HybridGmhqpConfig:
    """Equation parameters — fixed for smoke (METHOD_GATE: no per-ep soups)."""

    # Shared Escande / GMHQP tip-task
    k_tip: float = 1.4
    max_step_m: float = 0.012
    f_seat_n: float = 0.05
    seat_press_m: float = 0.00045
    upright_soft_deg: float = 16.0
    upright_step_m: float = 0.004
    mouth_r_m: float = 0.0045
    mouth_press_m: float = 0.0004
    c_ridge: float = 0.35
    c_ema: float = 0.85
    c_dw_min_m: float = 5e-5
    c_cond_max: float = 25.0
    # planar_C fallback (Type-A spirit / ContactMotion window C)
    k_path: float = 1.6
    c_window: int = 24
    c_reg: float = 0.15
    c_alpha_floor: float = 0.18
    c_alpha_ceil: float = 1.05
    c_slip_dw_m: float = 0.0012
    # Saturation → task switch (general F1; not per-ep)
    sat_err_m: float = 0.008
    sat_progress_m: float = 0.0005
    sat_frames: int = 40
    recover_err_m: float = 0.0055


@dataclass
class HybridGmhqpState:
    """GMHQP C + ContactMotion window C + Escande task mode."""

    gmhqp: GmhqpState = field(default_factory=GmhqpState)
    task: str = "tip_task"  # tip_task | planar_c_fb
    tip_err_hist: list[float] = field(default_factory=list)
    sat_streak: int = 0
    # ContactMotion window for planar_C fallback
    wrist_hist: list[np.ndarray] = field(default_factory=list)
    tip_hist: list[np.ndarray] = field(default_factory=list)
    C_cm: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=np.float64))
    level: str = "tip_path"
    meta: dict = field(default_factory=dict)


def is_hybrid_gmhqp_mode(mode: str) -> bool:
    m = str(mode).strip().lower()
    return m in (
        "track_b_hybrid_gmhqp",
        "hybrid_gmhqp",
        "gmhqp_hybrid",
        "tip_planar_c_hybrid_hqp",
        "escande_hybrid_gmhqp",
    )


def _gmhqp_cfg_from_hybrid(cfg: HybridGmhqpConfig) -> GmhqpConfig:
    return GmhqpConfig(
        k_tip=float(cfg.k_tip),
        max_step_m=float(cfg.max_step_m),
        f_seat_n=float(cfg.f_seat_n),
        seat_press_m=float(cfg.seat_press_m),
        upright_soft_deg=float(cfg.upright_soft_deg),
        upright_step_m=float(cfg.upright_step_m),
        mouth_r_m=float(cfg.mouth_r_m),
        mouth_press_m=float(cfg.mouth_press_m),
        c_ridge=float(cfg.c_ridge),
        c_ema=float(cfg.c_ema),
        c_dw_min_m=float(cfg.c_dw_min_m),
        c_cond_max=float(cfg.c_cond_max),
    )


def _update_contact_motion_c(
    state: HybridGmhqpState,
    *,
    wrist: np.ndarray,
    tip_obj: np.ndarray,
    normal: np.ndarray,
    cfg: HybridGmhqpConfig,
) -> np.ndarray:
    """Window LS ContactMotion: Δtip ≈ C_cm Δwrist (Pfanne / Type-A planar_C)."""
    w = np.asarray(wrist, dtype=np.float64).reshape(3)
    t = np.asarray(tip_obj, dtype=np.float64).reshape(3)
    state.wrist_hist.append(w.copy())
    state.tip_hist.append(t.copy())
    win = max(4, int(cfg.c_window))
    if len(state.wrist_hist) > win:
        state.wrist_hist = state.wrist_hist[-win:]
        state.tip_hist = state.tip_hist[-win:]
    if len(state.wrist_hist) < 3:
        return np.asarray(state.C_cm, dtype=np.float64).reshape(2, 2)

    n = _unit(normal)
    t1, t2 = _tangent_basis(n)
    # Slip-ish: large wrist Δ, tip barely moves → reset C toward I.
    dw_last = _to2(_planar(state.wrist_hist[-1] - state.wrist_hist[-2], n), t1, t2)
    dt_last = _to2(_planar(state.tip_hist[-1] - state.tip_hist[-2], n), t1, t2)
    dw_n = float(np.linalg.norm(dw_last))
    dt_n = float(np.linalg.norm(dt_last))
    if dw_n > float(cfg.c_slip_dw_m) and dt_n < 0.25 * dw_n:
        state.wrist_hist = state.wrist_hist[-3:]
        state.tip_hist = state.tip_hist[-3:]
        state.C_cm = np.eye(2, dtype=np.float64)
        return state.C_cm.copy()

    dw_rows: list[np.ndarray] = []
    dt_rows: list[np.ndarray] = []
    for i in range(1, len(state.wrist_hist)):
        dw = _to2(_planar(state.wrist_hist[i] - state.wrist_hist[i - 1], n), t1, t2)
        dt = _to2(_planar(state.tip_hist[i] - state.tip_hist[i - 1], n), t1, t2)
        if float(np.linalg.norm(dw)) > 1e-9:
            dw_rows.append(dw)
            dt_rows.append(dt)
    if len(dw_rows) < 2:
        return np.asarray(state.C_cm, dtype=np.float64).reshape(2, 2)
    w_mat = np.stack(dw_rows, axis=0)
    d_mat = np.stack(dt_rows, axis=0)
    reg = float(cfg.c_reg)
    try:
        c_t = np.linalg.solve(w_mat.T @ w_mat + reg * np.eye(2), w_mat.T @ d_mat)
    except np.linalg.LinAlgError:
        return np.asarray(state.C_cm, dtype=np.float64).reshape(2, 2)
    C = c_t.T
    u, s, vt = np.linalg.svd(C)
    s = np.clip(s, float(cfg.c_alpha_floor), float(cfg.c_alpha_ceil))
    state.C_cm = u @ np.diag(s) @ vt
    return np.asarray(state.C_cm, dtype=np.float64).reshape(2, 2)


def _invert_planar_c(
    dt_des: np.ndarray,
    *,
    C: np.ndarray,
    normal: np.ndarray,
    c_reg: float,
    max_step_m: float,
) -> np.ndarray:
    """dw = C^{-1} dt_des (Type-A planar_C)."""
    t1, t2 = _tangent_basis(normal)
    dt2 = _to2(_planar(dt_des, normal), t1, t2)
    Cm = np.asarray(C, dtype=np.float64).reshape(2, 2)
    try:
        dw2 = np.linalg.solve(Cm + float(c_reg) * np.eye(2), dt2)
    except np.linalg.LinAlgError:
        dw2 = dt2
    return clip_step(_from2(dw2, t1, t2), float(max_step_m))


def _update_task_mode(
    state: HybridGmhqpState,
    tip_err: float,
    cfg: HybridGmhqpConfig,
) -> str:
    """Escande Level-2 task switch: tip residual saturate → planar_c_fb."""
    state.tip_err_hist.append(float(tip_err))
    # Keep enough history for progress window (~sat_frames).
    keep = max(int(cfg.sat_frames) + 2, 8)
    if len(state.tip_err_hist) > keep:
        state.tip_err_hist = state.tip_err_hist[-keep:]

    if state.task == "planar_c_fb":
        if tip_err < float(cfg.recover_err_m):
            state.task = "tip_task"
            state.sat_streak = 0
        return state.task

    # tip_task: detect F1 plateau
    if tip_err > float(cfg.sat_err_m) and len(state.tip_err_hist) >= int(cfg.sat_frames):
        e_now = float(state.tip_err_hist[-1])
        e_old = float(state.tip_err_hist[-int(cfg.sat_frames)])
        progress = e_old - e_now  # positive = improving
        if progress < float(cfg.sat_progress_m):
            state.sat_streak += 1
        else:
            state.sat_streak = 0
        if state.sat_streak >= int(cfg.sat_frames):
            state.task = "planar_c_fb"
            state.sat_streak = 0
    else:
        state.sat_streak = 0
    return state.task


def hybrid_gmhqp_wrist_delta(
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
    state: HybridGmhqpState | None = None,
    cfg: HybridGmhqpConfig | None = None,
) -> tuple[np.ndarray, dict, HybridGmhqpState]:
    """One-step Escande HQP with hybrid Level-2 task (tip_task | planar_c_fb)."""
    cfg = cfg or HybridGmhqpConfig()
    st = state if state is not None else HybridGmhqpState()
    site = np.asarray(site_xyz, dtype=np.float64).reshape(3)
    tip = np.asarray(tip_obj, dtype=np.float64).reshape(3)
    tgt = np.asarray(path_target, dtype=np.float64).reshape(3)
    n = _unit(normal)
    press = _unit(press_ax)

    # Always refresh both C estimators (ContactMotion + GMHQP fd).
    gcfg = _gmhqp_cfg_from_hybrid(cfg)
    C_fd = update_planar_coupling(
        st.gmhqp, wrist=site, tip_obj=tip, normal=n, cfg=gcfg
    )
    C_cm = _update_contact_motion_c(
        st, wrist=site, tip_obj=tip, normal=n, cfg=cfg
    )

    e_tip = _planar(tgt - tip, n)
    e_wrist = _planar(tgt - site, n)
    tip_err = float(np.linalg.norm(e_tip))
    wrist_err = float(np.linalg.norm(e_wrist))
    task = _update_task_mode(st, tip_err, cfg)

    meta: dict = {
        "path": "track_b_hybrid_gmhqp",
        "privileged": False,
        "theory": "escande_hybrid_tip_task+planar_C_ContactMotion",
        "hybrid_task": task,
        "tip_task_err_mm": tip_err * 1000.0,
        "wrist_path_err_mm": wrist_err * 1000.0,
        "C_fd_fro": float(np.linalg.norm(C_fd, ord="fro")),
        "C_cm_fro": float(np.linalg.norm(C_cm, ord="fro")),
        "sat_streak": int(st.sat_streak),
    }

    d_w = np.zeros(3, dtype=np.float64)
    ax = float(ax_step)

    if float(contact_resid_n) < float(cfg.f_seat_n):
        st.level = "seat"
        ax = max(ax, float(cfg.seat_press_m))
        meta["path_scale"] = 0.0
        meta["hybrid_task"] = task  # seat still freezes path
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
        if task == "planar_c_fb":
            st.level = "planar_c_fb"
            # Type-A planar_C: dw = C_cm^{-1} K_path e_tip
            d_w = _invert_planar_c(
                e_tip * float(cfg.k_path),
                C=C_cm,
                normal=n,
                c_reg=float(cfg.c_reg),
                max_step_m=float(cfg.max_step_m),
            )
            meta["path_scale"] = 1.0
            meta["fallback"] = "planar_c_contact_motion"
        else:
            st.level = "mouth" if float(r_cmd_m) <= float(cfg.mouth_r_m) + 1e-9 else "tip_path"
            v_tip = e_tip * float(cfg.k_tip)
            d_w = map_tip_to_wrist(
                v_tip, C=C_fd, normal=n, c_ridge=float(cfg.c_ridge)
            )
            meta["path_scale"] = 1.0
            meta["fallback"] = "none"
        if float(r_cmd_m) <= float(cfg.mouth_r_m) + 1e-9:
            ax = max(ax, float(cfg.mouth_press_m))
            if st.level == "tip_path":
                st.level = "mouth"

    d_w = clip_step(_planar(d_w, n), float(cfg.max_step_m))
    hold = site + d_w + press * ax
    meta["hqp_level"] = st.level
    meta["ax_step_m"] = float(ax)
    meta["path_err_mm"] = tip_err * 1000.0
    st.level = str(meta["hqp_level"])
    st.meta = dict(meta)
    return hold, meta, st
