"""Track-B CLEP: Contact-Line Extrinsic Pivot (no tip GT).

When the peg tip is seated, extrinsic contact ≈ tip on the tray plane.
Estimate contact from wrist F/T (Doshi / Kim Active Extrinsic spirit) and
command the wrist so the contact proxy tracks the known spiral waypoint.

Design: docs/TRACK_B_CLEP.md
Cites: Kim & Rodriguez ICRA 2022 (arXiv:2110.03555); Doshi F/T contact;
       Kim TEC / TEXterity (factor-graph heavy — not ported here).

Forbidden: tip_gt, tip_gt+noise, peg xpos in the control law.
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


@dataclass
class ClepConfig:
    k_contact: float = 1.8  # track estimated contact → path
    k_wrist_fallback: float = 0.6  # when contact estimate invalid
    max_step_m: float = 0.012
    ema: float = 0.55
    f_min_n: float = 0.02  # need some normal force for estimate
    f_seat_n: float = 0.05
    seat_press_m: float = 0.00045
    seat_path_scale: float = 0.15
    mouth_r_m: float = 0.0045
    mouth_press_m: float = 0.00045
    f_mouth_press_n: float = 0.08
    max_contact_offset_m: float = 0.08  # reject absurd lever arms
    tang_gate_n: float = 0.55  # high tangential → shrink planar (hold contact mode)
    tang_path_scale: float = 0.35


@dataclass
class ClepState:
    contact_ema: np.ndarray | None = None
    valid: bool = False
    source: str = "none"
    meta: dict = field(default_factory=dict)


def is_clep_mode(mode: str) -> bool:
    m = str(mode).strip().lower()
    return m in (
        "track_b_clep",
        "clep",
        "contact_line_extrinsic_pivot",
        "active_extrinsic_clep",
    )


def estimate_contact_from_wrench(
    *,
    site_xyz: np.ndarray,
    force_xyz: np.ndarray,
    torque_xyz: np.ndarray,
    normal: np.ndarray,
    tip_anchor: np.ndarray | None = None,
    cfg: ClepConfig | None = None,
) -> tuple[np.ndarray | None, dict]:
    """Estimate planar extrinsic contact from wrist wrench (no tip GT).

    Prefer planar lever-arm ``r = (n × τ) / (n·F)`` when seated (dominant normal).
    Fallback: point-contact ``r = (F × τ) / ||F||²``.
    """
    cfg = cfg or ClepConfig()
    site = np.asarray(site_xyz, dtype=np.float64).reshape(3)
    f = np.asarray(force_xyz, dtype=np.float64).reshape(3)
    tau = np.asarray(torque_xyz, dtype=np.float64).reshape(3)
    n = _unit(normal)
    meta: dict = {"est": "none"}

    fn = float(np.dot(f, n))
    f_norm = float(np.linalg.norm(f))
    if f_norm < float(cfg.f_min_n):
        return None, {**meta, "reason": "low_force", "f_norm": f_norm}

    r = None
    if abs(fn) >= float(cfg.f_min_n):
        # Planar support: τ ≈ r × (fn n) ⇒ r_⊥ = (n × τ) / fn
        r = np.cross(n, tau) / (fn + 1e-12 * np.sign(fn if fn != 0 else 1.0))
        r = _planar(r, n)
        meta["est"] = "planar_lever"
        meta["fn_n"] = fn
    else:
        r = np.cross(f, tau) / (f_norm * f_norm + 1e-12)
        r = _planar(r, n)
        meta["est"] = "point_contact"
        meta["f_norm"] = f_norm

    rn = float(np.linalg.norm(r))
    meta["lever_mm"] = rn * 1000.0
    if rn > float(cfg.max_contact_offset_m):
        return None, {**meta, "reason": "lever_cap"}

    c = site + r
    anchor = tip_anchor if tip_anchor is not None else site
    # Project contact onto tray plane through anchor.
    c = c - n * float(np.dot(c - np.asarray(anchor, dtype=np.float64).reshape(3), n))
    meta["ok"] = True
    return c, meta


def clep_update_state(
    state: ClepState,
    *,
    site_xyz: np.ndarray,
    force_xyz: np.ndarray,
    torque_xyz: np.ndarray,
    normal: np.ndarray,
    tip_anchor: np.ndarray | None = None,
    cfg: ClepConfig | None = None,
) -> ClepState:
    cfg = cfg or ClepConfig()
    c_raw, em = estimate_contact_from_wrench(
        site_xyz=site_xyz,
        force_xyz=force_xyz,
        torque_xyz=torque_xyz,
        normal=normal,
        tip_anchor=tip_anchor,
        cfg=cfg,
    )
    st = state
    st.meta = dict(em)
    if c_raw is None:
        st.valid = False
        st.source = str(em.get("reason", "none"))
        return st
    alpha = float(np.clip(cfg.ema, 0.0, 1.0))
    if st.contact_ema is None:
        st.contact_ema = np.asarray(c_raw, dtype=np.float64).reshape(3).copy()
    else:
        st.contact_ema = (1.0 - alpha) * st.contact_ema + alpha * np.asarray(
            c_raw, dtype=np.float64
        ).reshape(3)
    st.valid = True
    st.source = str(em.get("est", "ok"))
    return st


def clep_wrist_delta(
    *,
    site_xyz: np.ndarray,
    path_target: np.ndarray,
    normal: np.ndarray,
    press_ax: np.ndarray,
    contact_resid_n: float,
    r_cmd_m: float,
    ax_step: float = 0.0,
    force_xyz: np.ndarray | None = None,
    torque_xyz: np.ndarray | None = None,
    tip_anchor: np.ndarray | None = None,
    state: ClepState | None = None,
    cfg: ClepConfig | None = None,
) -> tuple[np.ndarray, dict, ClepState]:
    """One-step CLEP wrist command: contact-proxy → known path (no tip GT)."""
    cfg = cfg or ClepConfig()
    st = state or ClepState()
    site = np.asarray(site_xyz, dtype=np.float64).reshape(3)
    tgt = np.asarray(path_target, dtype=np.float64).reshape(3)
    n = _unit(normal)
    press = _unit(press_ax)
    ax = float(ax_step)

    meta: dict = {
        "path": "track_b_clep",
        "privileged": False,
        "clep_valid": False,
    }

    # SEAT hard-ish: if under-contact, press and shrink path.
    seat = float(contact_resid_n) < float(cfg.f_seat_n)
    if seat:
        ax = max(ax, float(cfg.seat_press_m))
        meta["clep_level"] = "seat"

    if force_xyz is not None and torque_xyz is not None:
        st = clep_update_state(
            st,
            site_xyz=site,
            force_xyz=force_xyz,
            torque_xyz=torque_xyz,
            normal=n,
            tip_anchor=tip_anchor,
            cfg=cfg,
        )
        meta.update({f"clep_{k}": v for k, v in st.meta.items()})
        meta["clep_source"] = st.source

    path_scale = 1.0
    if seat:
        path_scale = float(cfg.seat_path_scale)

    # Tangential overload → hold contact mode (Kim active regulation spirit).
    if force_xyz is not None:
        f = np.asarray(force_xyz, dtype=np.float64).reshape(3)
        ft = float(np.linalg.norm(_planar(f, n)))
        meta["clep_ft_n"] = ft
        if ft > float(cfg.tang_gate_n):
            path_scale = min(path_scale, float(cfg.tang_path_scale))
            meta["clep_tang_gate"] = True

    if st.valid and st.contact_ema is not None:
        c_hat = np.asarray(st.contact_ema, dtype=np.float64).reshape(3)
        e = _planar(tgt - c_hat, n)
        d_w = e * float(cfg.k_contact) * path_scale
        meta["clep_valid"] = True
        meta["clep_track"] = "contact"
        meta["clep_err_mm"] = float(np.linalg.norm(e)) * 1000.0
        # Wrist must move with contact error (rigid offset assumption locally).
    else:
        e = _planar(tgt - site, n)
        d_w = e * float(cfg.k_wrist_fallback) * path_scale
        meta["clep_track"] = "wrist_fallback"
        meta["clep_err_mm"] = float(np.linalg.norm(e)) * 1000.0

    if float(r_cmd_m) <= float(cfg.mouth_r_m) + 1e-9 and not seat:
        if float(contact_resid_n) >= float(cfg.f_mouth_press_n):
            ax = max(ax, float(cfg.mouth_press_m))
            meta["clep_level"] = "mouth"
        elif "clep_level" not in meta:
            meta["clep_level"] = "path"
    elif "clep_level" not in meta:
        meta["clep_level"] = "path"

    d_w = clip_step(d_w, float(cfg.max_step_m))
    hold = site + d_w + press * ax
    meta["ax_step_m"] = float(ax)
    meta["path_scale"] = float(path_scale)
    return hold, meta, st
