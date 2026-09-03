"""Track-B: Contact-optimized short-horizon MPC layered on GMHQP.

General law (docs/GENERAL_ALGO_MANDATE.md + TRACK_B_CONTACT_OPT_ON_GMHQP.md):
  tip track p* + seat + upright + force-enter under C≠I for any rollout.
  No episode IDs, fail-subset, or keep-set branches in the controller.

REUSE: Kim Active Extrinsic (contact→mouth residual);
       LeTac-MPC / TACTIC spirit (path residual + seat force over horizon);
       Montana C⁺ + Escande HQP (keep GMHQP tip-task structure).

Critical vs FTIP: tip_obj stays in-hand geom tip; FT ĉ enters cost only.
Forbidden: tip_gt; tip_obj:=ĉ; FASR/per-ep soups; ep-gated logic.
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
    _planar,
    _unit,
    clip_step,
    map_tip_to_wrist,
    update_planar_coupling,
)


@dataclass
class GmhqpCompcConfig:
    """Fixed equation params (METHOD_GATE: no per-ep retune)."""

    # GMHQP / Escande
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
    # Contact MPC (LeTac / Active Extrinsic spirit)
    horizon: int = 4
    w_tip: float = 1.0
    w_contact: float = 1.2
    w_u: float = 0.08
    # False-closure length scale: α=σ²/(σ²+‖e_t‖²)
    sigma_false_m: float = 0.005
    # Seat force soft track (axial), Escande still hard-gates planar
    f_des_n: float = 0.08
    k_force_m_per_n: float = 0.0025
    force_press_cap_m: float = 0.00035
    # FT contact estimate (Doshi / Active Extrinsic lever)
    ft_ema: float = 0.55
    ft_f_min_n: float = 0.02
    ft_max_lever_m: float = 0.08
    # Cap contact-driven planar step (anti tip_err explosion)
    contact_step_cap_m: float = 0.008


@dataclass
class GmhqpCompcState:
    gmhqp: GmhqpState = field(default_factory=GmhqpState)
    clep: ClepState = field(default_factory=ClepState)
    level: str = "tip_path"
    meta: dict = field(default_factory=dict)


def is_gmhqp_compc_mode(mode: str) -> bool:
    m = str(mode).strip().lower()
    return m in (
        "track_b_gmhqp_compc",
        "gmhqp_compc",
        "gmhqp_contact_opt",
        "track_b_contact_opt_gmhqp",
        "contact_opt_on_gmhqp",
    )


def _gmhqp_cfg(cfg: GmhqpCompcConfig) -> GmhqpConfig:
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


def _clep_cfg(cfg: GmhqpCompcConfig) -> ClepConfig:
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


def residual_contact_weight(tip_err_m: float, sigma_m: float) -> float:
    """State-dependent contact weight α=σ²/(σ²+‖e_t‖²) — general, not ep-gated.

    Large tip residual → α→0 (pure tip task / GMHQP spirit).
    Small tip residual → α→1 (contact→mouth term can contribute).
    """
    s2 = float(sigma_m) * float(sigma_m)
    e2 = float(tip_err_m) * float(tip_err_m)
    return float(s2 / (s2 + e2 + 1e-18))


def contact_horizon_u(
    *,
    e_tip: np.ndarray,
    e_contact: np.ndarray | None,
    contact_valid: bool,
    cfg: GmhqpCompcConfig,
) -> tuple[np.ndarray, dict]:
    """Closed-form constant-control short-horizon QP in the planar tip frame.

    Dynamics: tip_k = tip_0 + k u, ĉ_k = ĉ_0 + k u (rigid extrinsic translate).
    Cost: tip residual + α-gated contact→mouth + control regularizer.
    """
    et = np.asarray(e_tip, dtype=np.float64).reshape(3)
    tip_err = float(np.linalg.norm(et))
    alpha = residual_contact_weight(tip_err, float(cfg.sigma_false_m))
    H = max(1, int(cfg.horizon))
    wt = float(cfg.w_tip)
    wu = float(cfg.w_u)
    info: dict = {
        "alpha_contact": alpha,
        "horizon": H,
        "contact_valid": bool(contact_valid),
        "tip_err_mm": tip_err * 1000.0,
    }

    if H == 1:
        s1, s2 = 0.0, 0.0  # unused; one-step: J = wt‖e-u‖² + ...
        # One-step closed form: u = (wt et + wc α ec) / (wt + wc α + wu)
        wc_eff = float(cfg.w_contact) * alpha if contact_valid else 0.0
        denom = wt + wc_eff + wu
        rhs = wt * et
        if contact_valid and e_contact is not None and wc_eff > 0.0:
            ec = np.asarray(e_contact, dtype=np.float64).reshape(3)
            rhs = rhs + wc_eff * ec
            info["contact_err_mm"] = float(np.linalg.norm(ec)) * 1000.0
        else:
            info["contact_err_mm"] = 0.0
        u = rhs / max(denom, 1e-12)
        info["wc_eff"] = wc_eff
        return u, info

    # Σ_{k=0}^{H-1} k = H(H-1)/2 ; Σ k² = H(H-1)(2H-1)/6
    # For k starting at 0, residual at step k is (e - k u); k=0 term has no u in residual
    # but still counts state cost. Gradient uses k=1..H-1 effectively via S1,S2.
    s1 = 0.5 * H * (H - 1)
    s2 = H * (H - 1) * (2 * H - 1) / 6.0
    wc_eff = float(cfg.w_contact) * alpha if contact_valid else 0.0
    info["wc_eff"] = wc_eff
    coeff = wt * s2 + wc_eff * s2 + H * wu
    rhs = wt * s1 * et
    if contact_valid and e_contact is not None and wc_eff > 0.0:
        ec = np.asarray(e_contact, dtype=np.float64).reshape(3)
        rhs = rhs + wc_eff * s1 * ec
        info["contact_err_mm"] = float(np.linalg.norm(ec)) * 1000.0
    else:
        info["contact_err_mm"] = 0.0
    if coeff < 1e-12 or s1 < 1e-12:
        # H=1 edge already handled; tiny H→ fall back to tip PD direction
        u = et.copy()
    else:
        u = rhs / coeff
    return u, info


def gmhqp_compc_wrist_delta(
    *,
    site_xyz: np.ndarray,
    tip_obj: np.ndarray,
    path_target: np.ndarray,
    mouth_xyz: np.ndarray | None,
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
    state: GmhqpCompcState | None = None,
    cfg: GmhqpCompcConfig | None = None,
) -> tuple[np.ndarray, dict, GmhqpCompcState]:
    """Escande HQP + contact-horizon tip residual; tip_obj never replaced by FT ĉ."""
    cfg = cfg or GmhqpCompcConfig()
    st = state if state is not None else GmhqpCompcState()
    site = np.asarray(site_xyz, dtype=np.float64).reshape(3)
    tip = np.asarray(tip_obj, dtype=np.float64).reshape(3)
    tgt = np.asarray(path_target, dtype=np.float64).reshape(3)
    n = _unit(normal)
    press = _unit(press_ax)
    mouth = (
        np.asarray(mouth_xyz, dtype=np.float64).reshape(3)
        if mouth_xyz is not None
        else tgt.copy()
    )

    gcfg = _gmhqp_cfg(cfg)
    C = update_planar_coupling(
        st.gmhqp, wrist=site, tip_obj=tip, normal=n, cfg=gcfg
    )

    e_tip = _planar(tgt - tip, n)
    tip_err = float(np.linalg.norm(e_tip))

    # FT contact estimate — cost only (Active Extrinsic)
    contact_valid = False
    e_c: np.ndarray | None = None
    c_hat = None
    if force_xyz is not None and torque_xyz is not None:
        st.clep = clep_update_state(
            st.clep,
            site_xyz=site,
            force_xyz=force_xyz,
            torque_xyz=torque_xyz,
            normal=n,
            tip_anchor=tip_anchor if tip_anchor is not None else tip,
            cfg=_clep_cfg(cfg),
        )
        contact_valid = bool(st.clep.valid) and st.clep.contact_ema is not None
        if contact_valid:
            c_hat = np.asarray(st.clep.contact_ema, dtype=np.float64).reshape(3)
            e_c = _planar(mouth - c_hat, n)

    meta: dict = {
        "path": "track_b_gmhqp_compc",
        "privileged": False,
        "theory": "active_extrinsic+letac_mpc_on_gmhqp",
        "tip_task_err_mm": tip_err * 1000.0,
        "C_fro": float(np.linalg.norm(C, ord="fro")),
        "ft_valid": bool(contact_valid),
    }

    d_w = np.zeros(3, dtype=np.float64)
    ax = float(ax_step)

    if float(contact_resid_n) < float(cfg.f_seat_n):
        st.level = "seat"
        ax = max(ax, float(cfg.seat_press_m))
        meta["path_scale"] = 0.0
        meta["hqp_level"] = "seat"
        meta["mpc_active"] = False
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
        meta["mpc_active"] = False
    else:
        st.level = "mouth" if float(r_cmd_m) <= float(cfg.mouth_r_m) + 1e-9 else "tip_path"
        u_star, mpc_info = contact_horizon_u(
            e_tip=e_tip,
            e_contact=e_c,
            contact_valid=contact_valid,
            cfg=cfg,
        )
        meta.update(mpc_info)
        meta["mpc_active"] = True
        # Cap contact-driven excess vs pure tip PD (anti FTIP tip_err blow-up)
        u_tip = e_tip.copy()
        u_delta = u_star - u_tip
        u_delta = clip_step(u_delta, float(cfg.contact_step_cap_m))
        u_use = u_tip + u_delta
        v_tip = u_use * float(cfg.k_tip)
        d_w = map_tip_to_wrist(
            v_tip, C=C, normal=n, c_ridge=float(cfg.c_ridge)
        )
        # Soft seat-force track (LeTac spirit); Escande already handled hard seat
        f_err = float(cfg.f_des_n) - float(contact_resid_n)
        ax_f = float(cfg.k_force_m_per_n) * f_err
        ax_f = float(np.clip(ax_f, -float(cfg.force_press_cap_m), float(cfg.force_press_cap_m)))
        ax = max(ax, 0.0) + ax_f
        if st.level == "mouth":
            ax = max(ax, float(cfg.mouth_press_m))
        meta["path_scale"] = 1.0
        meta["hqp_level"] = st.level
        meta["force_ax_m"] = float(ax_f)
        if c_hat is not None:
            meta["contact_to_mouth_mm"] = float(
                np.linalg.norm(_planar(mouth - c_hat, n))
            ) * 1000.0

    d_w = clip_step(_planar(d_w, n), float(cfg.max_step_m))
    hold = site + d_w + press * ax
    meta["ax_step_m"] = float(ax)
    meta["path_err_mm"] = tip_err * 1000.0
    st.level = str(meta["hqp_level"])
    st.meta = dict(meta)
    return hold, meta, st
