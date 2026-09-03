"""Track-B FASR: Force-Asymmetry Spiral Recenter (no tip GT).

Diagnosis: docs/TRACK_B_PRIV_DIAG.md — dominant residual F1
(`priv_planar_min` 6–12 mm with mouth False on Type-A fails; CLEP/PHIG
made planar worse; HardHQP cut slip but ep10 planar still 14.7 mm).

Idea (Park/Tang PiH force-search spirit): keep known-path wrist follow,
add planar bias opposite tangential wrench (edge push → slide toward hole),
and EMA-shift a spiral-center offset so the planned path recenters without
tip XY in the law. Mouth cue: normal-force drop near small r_cmd → press.

Forbidden: tip_gt, tip_gt+noise, peg xpos in control.
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
class FasrConfig:
    k_path: float = 1.5
    k_force: float = 0.004  # m per N of planar tangential force
    f_edge_n: float = 0.08  # arm force-bias above this |f_t|
    f_seat_n: float = 0.05
    seat_press_m: float = 0.00045
    seat_path_scale: float = 0.25
    max_step_m: float = 0.012
    # Spiral-center EMA recenter (accumulates in FasrState)
    recenter_ema: float = 0.08
    recenter_step_m: float = 0.00035
    recenter_cap_m: float = 0.012
    # Mouth: Fz drop while r_cmd small
    mouth_r_m: float = 0.006
    f_mouth_drop_n: float = 0.035  # below this seated → mouth press
    mouth_press_m: float = 0.0005
    mouth_path_scale: float = 0.4


@dataclass
class FasrState:
    center_offset: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    fn_ema: float = 0.0
    meta: dict = field(default_factory=dict)


def is_fasr_mode(mode: str) -> bool:
    m = str(mode).strip().lower()
    return m in (
        "track_b_fasr",
        "fasr",
        "force_asymmetry_spiral_recenter",
        "force_guided_radial_seek",
    )


def fasr_wrist_delta(
    *,
    site_xyz: np.ndarray,
    path_target: np.ndarray,
    normal: np.ndarray,
    press_ax: np.ndarray,
    contact_resid_n: float,
    r_cmd_m: float,
    ax_step: float = 0.0,
    force_xyz: np.ndarray | None = None,
    state: FasrState | None = None,
    cfg: FasrConfig | None = None,
) -> tuple[np.ndarray, dict, FasrState]:
    """One-step FASR wrist command + updated recenter state (no tip GT)."""
    cfg = cfg or FasrConfig()
    st = state or FasrState()
    site = np.asarray(site_xyz, dtype=np.float64).reshape(3)
    tgt = np.asarray(path_target, dtype=np.float64).reshape(3)
    n = _unit(normal)
    press = _unit(press_ax)
    off = _planar(np.asarray(st.center_offset, dtype=np.float64).reshape(3), n)
    tgt_eff = tgt + off

    meta: dict = {
        "path": "track_b_fasr",
        "privileged": False,
        "fasr_mode": "path",
    }

    f = (
        np.asarray(force_xyz, dtype=np.float64).reshape(3)
        if force_xyz is not None
        else np.zeros(3)
    )
    fn = float(np.dot(f, n))
    # Prefer residual magnitude as seat proxy when wrench frame is noisy.
    seat_proxy = max(abs(fn), float(contact_resid_n))
    st.fn_ema = 0.85 * float(st.fn_ema) + 0.15 * seat_proxy

    f_t = _planar(f, n)
    ft_n = float(np.linalg.norm(f_t))
    meta["ft_n"] = ft_n
    meta["fn_proxy"] = seat_proxy
    meta["center_off_mm"] = float(np.linalg.norm(off)) * 1000.0

    ax = float(ax_step)
    path_scale = 1.0
    d_bias = np.zeros(3, dtype=np.float64)

    if seat_proxy < float(cfg.f_seat_n):
        meta["fasr_mode"] = "seat"
        ax = max(ax, float(cfg.seat_press_m))
        path_scale = float(cfg.seat_path_scale)
    elif (
        float(r_cmd_m) <= float(cfg.mouth_r_m) + 1e-9
        and seat_proxy < float(cfg.f_mouth_drop_n)
    ):
        # Classic spiral Fz-drop over mouth → press, shrink planar chase.
        meta["fasr_mode"] = "mouth_drop"
        ax = max(ax, float(cfg.mouth_press_m))
        path_scale = float(cfg.mouth_path_scale)
    elif float(r_cmd_m) <= float(cfg.mouth_r_m) + 1e-9:
        meta["fasr_mode"] = "mouth"
        ax = max(ax, float(cfg.mouth_press_m) * 0.7)
        path_scale = 0.7

    # Edge bias: reaction force points away from free space / hole → step opposite.
    if ft_n >= float(cfg.f_edge_n) and meta["fasr_mode"] != "seat":
        d_bias = -f_t * (float(cfg.k_force) / max(ft_n, 1e-9)) * ft_n
        # Accumulate spiral-center recenter (EMA), capped.
        step = -_unit(f_t) * float(cfg.recenter_step_m)
        off = off + float(cfg.recenter_ema) * step
        off = clip_step(off, float(cfg.recenter_cap_m))
        st.center_offset = off
        tgt_eff = tgt + off
        meta["fasr_bias"] = True
        meta["center_off_mm"] = float(np.linalg.norm(off)) * 1000.0
    else:
        meta["fasr_bias"] = False

    e_path = _planar(tgt_eff - site, n)
    d_w = e_path * float(cfg.k_path) * float(path_scale) + d_bias
    d_w = clip_step(d_w, float(cfg.max_step_m))
    hold = site + d_w + press * ax

    meta["path_scale"] = float(path_scale)
    meta["path_err_mm"] = float(np.linalg.norm(e_path)) * 1000.0
    meta["ax_step_m"] = float(ax)
    st.meta = dict(meta)
    return hold, meta, st
