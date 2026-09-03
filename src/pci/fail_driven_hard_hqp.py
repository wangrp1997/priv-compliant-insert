"""Fail-driven Track-B: Escande-style **hard** HQP (seat ≻ upright ≻ path).

Taxonomy: docs/FAIL_PRIV_TAXONOMY.md · Design: docs/FAIL_DRIVEN_HARD_HQP.md

Unlike PHIG soft impedance weights and tip_tracking_qp ridge LS, this cascade
treats SEAT and UPRIGHT as hard levels: PATH only runs in the residual freedom
after seat/upright residuals are driven below tolerances.

Track-B: uses known spiral waypoint + wrist F/T + FK axis — **no tip GT**.
Cite: Escande, Mansard, Wieber IJRR 2014 (HQP hierarchy).
"""
from __future__ import annotations

from dataclasses import dataclass

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
class HardHqpConfig:
    f_seat_n: float = 0.05
    seat_press_m: float = 0.00045
    upright_soft_deg: float = 16.0
    upright_step_m: float = 0.004  # tip-pivot planar proxy magnitude
    path_k: float = 1.4
    max_step_m: float = 0.012
    mouth_r_m: float = 0.0045
    mouth_press_m: float = 0.00045
    # When seat/upright hard levels active, scale path to this fraction (≈0).
    path_when_hard: float = 0.0


@dataclass
class HardHqpState:
    level: str = "path"  # seat | upright | path | mouth
    seat_active: bool = False
    upright_active: bool = False


def is_hard_hqp_mode(mode: str) -> bool:
    m = str(mode).strip().lower()
    return m in (
        "fail_driven_hard_hqp",
        "hard_hqp",
        "track_b_hard_hqp",
        "escande_hard_hqp",
    )


def hard_hqp_select_level(
    *,
    contact_resid_n: float,
    axis_err_deg: float,
    r_cmd_m: float,
    cfg: HardHqpConfig | None = None,
) -> HardHqpState:
    """Strict priority: seat → upright → (mouth if near) → path."""
    cfg = cfg or HardHqpConfig()
    st = HardHqpState()
    if float(contact_resid_n) < float(cfg.f_seat_n):
        st.level = "seat"
        st.seat_active = True
        return st
    if float(axis_err_deg) > float(cfg.upright_soft_deg):
        st.level = "upright"
        st.upright_active = True
        return st
    if float(r_cmd_m) <= float(cfg.mouth_r_m) + 1e-9:
        st.level = "mouth"
        return st
    st.level = "path"
    return st


def hard_hqp_wrist_delta(
    *,
    site_xyz: np.ndarray,
    path_target: np.ndarray,
    normal: np.ndarray,
    press_ax: np.ndarray,
    contact_resid_n: float,
    axis_err_deg: float,
    peg_axis: np.ndarray | None = None,
    hole_axis: np.ndarray | None = None,
    r_cmd_m: float = 0.02,
    ax_step: float = 0.0,
    cfg: HardHqpConfig | None = None,
) -> tuple[np.ndarray, dict]:
    """One-step hard hierarchical wrist command (no tip pose in law).

    Level-0 SEAT: axial press until f_n ≥ f_seat (path frozen).
    Level-1 UPRIGHT: planar tip-pivot proxy toward hole axis (path frozen).
    Level-2 PATH / MOUTH: known spiral waypoint + optional mouth press.
    """
    cfg = cfg or HardHqpConfig()
    site = np.asarray(site_xyz, dtype=np.float64).reshape(3)
    tgt = np.asarray(path_target, dtype=np.float64).reshape(3)
    n = _unit(normal)
    press = _unit(press_ax)
    st = hard_hqp_select_level(
        contact_resid_n=float(contact_resid_n),
        axis_err_deg=float(axis_err_deg),
        r_cmd_m=float(r_cmd_m),
        cfg=cfg,
    )

    meta: dict = {
        "path": "fail_driven_hard_hqp",
        "privileged": False,
        "hqp_level": st.level,
        "seat_active": bool(st.seat_active),
        "upright_active": bool(st.upright_active),
    }

    d_w = np.zeros(3, dtype=np.float64)
    ax = float(ax_step)

    if st.level == "seat":
        ax = max(ax, float(cfg.seat_press_m))
        # Hard: zero planar path while reseating.
        meta["path_scale"] = 0.0
    elif st.level == "upright":
        # Tip-pivot proxy without tip GT: rotate peg toward hole in plane
        # by commanding a small planar Δw orthogonal to hole axis error.
        peg = _unit(peg_axis if peg_axis is not None else n)
        hole = _unit(hole_axis if hole_axis is not None else n)
        # Desired: reduce peg×hole cross in plane (Escande L1 residual).
        cross = np.cross(peg, hole)
        pivot = _planar(cross, n)
        pn = float(np.linalg.norm(pivot))
        if pn > 1e-9:
            d_w = pivot * (float(cfg.upright_step_m) / pn)
        meta["path_scale"] = float(cfg.path_when_hard)
        meta["upright_residual_deg"] = float(axis_err_deg)
    else:
        e_path = _planar(tgt - site, n)
        scale = 1.0 if st.level == "path" else 0.35
        d_w = e_path * float(cfg.path_k) * scale
        if st.level == "mouth":
            ax = max(ax, float(cfg.mouth_press_m))
        meta["path_scale"] = float(scale)
        meta["path_err_mm"] = float(np.linalg.norm(e_path)) * 1000.0

    d_w = clip_step(d_w, float(cfg.max_step_m))
    hold = site + d_w + press * ax
    meta["ax_step_m"] = float(ax)
    return hold, meta
