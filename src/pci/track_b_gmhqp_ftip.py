"""Track-B REUSE: GMHQP tip-task with F/T extrinsic contact tip (no tip GT).

Theory: docs/TRACK_B_GMHQP_FTIP.md · Phase-0: docs/WAVE6_TRACK_B_PRIV.md
Cites: Doshi F/T contact; Kim Active Extrinsic; Contact Occupancy spirit;
       Montana + Escande GMHQP (in-tree).

Wave-5 CLEP alone chased ĉ→path without C⁺/Escande → 0/4.
Wave-5 Hybrid switched on proxy tip_err (already ~3 mm) → 0/4.
This compose: tip_obj := FT contact ĉ (CLEP estimator) → same GMHQP law.

Forbidden: tip_gt+noise; FASR/OIGS/PHIG restack; per-ep soups.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pci.track_b_clep import (
    ClepConfig,
    ClepState,
    clep_update_state,
)
from pci.track_b_gmhqp import (
    GmhqpConfig,
    GmhqpState,
    gmhqp_wrist_delta,
)


@dataclass
class GmhqpFtipConfig:
    """Fixed equation params (METHOD_GATE: no per-ep retune)."""

    # GMHQP
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
    # FT contact tip (CLEP estimator reuse)
    ft_ema: float = 0.55
    ft_f_min_n: float = 0.02
    ft_max_lever_m: float = 0.08
    # Soft blend toward geom when FT invalid streak
    geom_fallback: bool = True


@dataclass
class GmhqpFtipState:
    gmhqp: GmhqpState = field(default_factory=GmhqpState)
    clep: ClepState = field(default_factory=ClepState)
    tip_source: str = "geom"
    meta: dict = field(default_factory=dict)


def is_gmhqp_ftip_mode(mode: str) -> bool:
    m = str(mode).strip().lower()
    return m in (
        "track_b_gmhqp_ftip",
        "gmhqp_ftip",
        "gmhqp_ft_contact",
        "gmhqp_ft_tip",
        "grasp_map_hqp_ftip",
    )


def _gmhqp_cfg(cfg: GmhqpFtipConfig) -> GmhqpConfig:
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


def _clep_cfg(cfg: GmhqpFtipConfig) -> ClepConfig:
    return ClepConfig(
        ema=float(cfg.ft_ema),
        f_min_n=float(cfg.ft_f_min_n),
        max_contact_offset_m=float(cfg.ft_max_lever_m),
        f_seat_n=float(cfg.f_seat_n),
        seat_press_m=float(cfg.seat_press_m),
        mouth_r_m=float(cfg.mouth_r_m),
        mouth_press_m=float(cfg.mouth_press_m),
        max_step_m=float(cfg.max_step_m),
    )


def resolve_ft_tip(
    *,
    tip_geom: np.ndarray,
    site_xyz: np.ndarray,
    force_xyz: np.ndarray | None,
    torque_xyz: np.ndarray | None,
    normal: np.ndarray,
    tip_anchor: np.ndarray | None,
    state: GmhqpFtipState,
    cfg: GmhqpFtipConfig,
) -> tuple[np.ndarray, str, dict]:
    """Return tip used in GMHQP tip-task: FT ĉ if valid else geom."""
    geom = np.asarray(tip_geom, dtype=np.float64).reshape(3)
    info: dict = {"ft_valid": False}
    if force_xyz is None or torque_xyz is None:
        state.tip_source = "geom_no_wrench"
        return geom, state.tip_source, info

    state.clep = clep_update_state(
        state.clep,
        site_xyz=site_xyz,
        force_xyz=force_xyz,
        torque_xyz=torque_xyz,
        normal=normal,
        tip_anchor=tip_anchor if tip_anchor is not None else geom,
        cfg=_clep_cfg(cfg),
    )
    info.update(dict(state.clep.meta))
    info["ft_valid"] = bool(state.clep.valid)
    if state.clep.valid and state.clep.contact_ema is not None:
        tip = np.asarray(state.clep.contact_ema, dtype=np.float64).reshape(3)
        state.tip_source = "ft_contact"
        return tip, state.tip_source, info
    if bool(cfg.geom_fallback):
        state.tip_source = "geom_fallback"
        return geom, state.tip_source, info
    # Keep last FT if any
    if state.clep.contact_ema is not None:
        state.tip_source = "ft_stale"
        return (
            np.asarray(state.clep.contact_ema, dtype=np.float64).reshape(3),
            state.tip_source,
            info,
        )
    state.tip_source = "geom_fallback"
    return geom, state.tip_source, info


def gmhqp_ftip_wrist_delta(
    *,
    site_xyz: np.ndarray,
    tip_geom: np.ndarray,
    path_target: np.ndarray,
    normal: np.ndarray,
    press_ax: np.ndarray,
    contact_resid_n: float,
    axis_err_deg: float,
    force_xyz: np.ndarray | None = None,
    torque_xyz: np.ndarray | None = None,
    tip_anchor: np.ndarray | None = None,
    peg_axis: np.ndarray | None = None,
    hole_axis: np.ndarray | None = None,
    r_cmd_m: float = 0.02,
    ax_step: float = 0.0,
    state: GmhqpFtipState | None = None,
    cfg: GmhqpFtipConfig | None = None,
) -> tuple[np.ndarray, dict, GmhqpFtipState]:
    """GMHQP Escande step with tip_obj from FT extrinsic contact (REUSE)."""
    cfg = cfg or GmhqpFtipConfig()
    st = state if state is not None else GmhqpFtipState()
    tip_task, src, ft_info = resolve_ft_tip(
        tip_geom=tip_geom,
        site_xyz=site_xyz,
        force_xyz=force_xyz,
        torque_xyz=torque_xyz,
        normal=normal,
        tip_anchor=tip_anchor,
        state=st,
        cfg=cfg,
    )
    hold, meta, gst = gmhqp_wrist_delta(
        site_xyz=site_xyz,
        tip_obj=tip_task,
        path_target=path_target,
        normal=normal,
        press_ax=press_ax,
        contact_resid_n=contact_resid_n,
        axis_err_deg=axis_err_deg,
        peg_axis=peg_axis,
        hole_axis=hole_axis,
        r_cmd_m=r_cmd_m,
        ax_step=ax_step,
        state=st.gmhqp,
        cfg=_gmhqp_cfg(cfg),
    )
    st.gmhqp = gst
    meta = dict(meta)
    meta["path"] = "track_b_gmhqp_ftip"
    meta["theory"] = "doshi_ft_contact+montana_gmhqp_escande"
    meta["tip_source"] = src
    meta["privileged"] = False
    meta["ft_lever_mm"] = float(ft_info.get("lever_mm", 0.0) or 0.0)
    meta["ft_valid"] = bool(ft_info.get("ft_valid", False))
    # Diagnostic: geom tip error vs FT tip error
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    nn = float(np.linalg.norm(n))
    if nn > 1e-12:
        n = n / nn
    tgt = np.asarray(path_target, dtype=np.float64).reshape(3)
    geom = np.asarray(tip_geom, dtype=np.float64).reshape(3)
    e_g = tgt - geom
    e_g = e_g - n * float(np.dot(e_g, n))
    e_f = tgt - tip_task
    e_f = e_f - n * float(np.dot(e_f, n))
    meta["geom_tip_err_mm"] = float(np.linalg.norm(e_g)) * 1000.0
    meta["ft_tip_err_mm"] = float(np.linalg.norm(e_f)) * 1000.0
    st.meta = dict(meta)
    return hold, meta, st
