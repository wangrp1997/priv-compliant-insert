"""Track-B PHIG: Path-constrained Hybrid Impedance + Grasp co-regulation.

Deployable candidate toward Oracle-v2 rate **without** tip GT / tip_hat in the loop.

Theory: docs/TRACK_B_ALTERNATIVES.md
Cites (concepts): Raibert&Craig hybrid force/position; Hogan impedance;
Escande HQP priorities (soft weights); GraspQP friction-cone grasp;
Oracle-v2 mouth press / slip reseat (F/T + proprio only).

Forbidden: tip_gt, tip_gt+noise, privileged hole attractor as deployable claim.
SUCCESS_STANDARD unchanged.
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
class PhigState:
    """PHIG sub-FSM on top of Tip-Hybrid SEAT→UPRIGHT→SPIRAL."""

    mode: str = "path_follow"  # path_follow | mouth_press | slip_reseat | enter_hold
    reseat_left: int = 0
    mouth_press_frames: int = 0
    slip_armed_m: float = -1.0
    grasp_boosted: bool = False


@dataclass
class PhigConfig:
    k_path: float = 1.6
    k_tang_admit: float = 0.00025  # m/N toward lower tangential residual
    max_step_m: float = 0.012
    f_seat_n: float = 0.05
    f_mouth_press_n: float = 0.08
    mouth_r_m: float = 0.0045
    mouth_press_m: float = 0.00045
    mouth_hard_unload_n: float = 0.90
    slip_tau_m: float = 0.12
    reseat_frames: int = 40
    retrip_delta_m: float = 0.05
    grasp_scale: float = 1.55
    hold_enter_max_axis_deg: float = 25.0


def phig_near_mouth(
    *,
    r_cmd_m: float,
    contact_resid_n: float,
    force_hole_cue: bool,
    mouth_r_m: float,
    f_mouth_press_n: float,
) -> bool:
    """Mouth gate without tip lat GT: planner radius + F/T."""
    if float(r_cmd_m) > float(mouth_r_m) + 1e-9:
        return False
    if bool(force_hole_cue):
        return True
    return float(contact_resid_n) >= float(f_mouth_press_n)


def phig_slip_trip(
    *,
    grasp_slip_m: float,
    slip_tau_m: float,
    slip_armed_m: float = -1.0,
    retrip_delta_m: float = 0.05,
) -> bool:
    """Proprio slip trip (no tip_lag privilege).

    Requires a *new* rise past ``slip_armed_m + retrip_delta`` so slow peak
    creep does not freeze spiral advance every frame (Oracle/STAR pattern).
    """
    slip = float(grasp_slip_m)
    armed = float(slip_armed_m)
    if slip <= float(slip_tau_m):
        return False
    if armed < 0.0:
        return True
    return slip > armed + float(retrip_delta_m) + 1e-9


def phig_wrist_delta(
    *,
    site_xyz: np.ndarray,
    path_target: np.ndarray,
    normal: np.ndarray,
    press_ax: np.ndarray,
    ax_step: float,
    wrench_xyz: np.ndarray | None = None,
    cfg: PhigConfig | None = None,
) -> tuple[np.ndarray, dict]:
    """One-step wrist command: path impedance + tangential admit + axial press.

    path_target = known spiral waypoint (world). Does **not** use tip pose.
    """
    cfg = cfg or PhigConfig()
    site = np.asarray(site_xyz, dtype=np.float64).reshape(3)
    tgt = np.asarray(path_target, dtype=np.float64).reshape(3)
    n = _unit(normal)
    press = _unit(press_ax)

    e_path = _planar(tgt - site, n)
    d_w = e_path * float(cfg.k_path)

    meta: dict = {"path": "track_b_phig", "privileged": False}
    if wrench_xyz is not None:
        f = np.asarray(wrench_xyz, dtype=np.float64).reshape(3)
        f_t = _planar(f, n)
        d_w = d_w + f_t * float(cfg.k_tang_admit)
        meta["ft_tang_n"] = float(np.linalg.norm(f_t))
        meta["fn_n"] = float(np.dot(f, n))

    d_w = clip_step(d_w, float(cfg.max_step_m))
    hold = site + d_w + press * float(ax_step)
    meta["path_err_mm"] = float(np.linalg.norm(e_path)) * 1000.0
    meta["ax_step_m"] = float(ax_step)
    return hold, meta


def phig_select_mode(
    state: PhigState,
    *,
    hybrid_mode: str,
    r_cmd_m: float,
    contact_resid_n: float,
    force_hole_cue: bool,
    grasp_slip_m: float,
    axis_err_deg: float,
    cfg: PhigConfig | None = None,
) -> PhigState:
    """Update PHIG submode; Tip-Hybrid still owns SEAT/UPRIGHT priority."""
    cfg = cfg or PhigConfig()
    st = state
    hyb = str(hybrid_mode).lower()

    if int(st.reseat_left) > 0:
        st.mode = "slip_reseat"
        st.reseat_left = max(0, int(st.reseat_left) - 1)
        return st

    if hyb in ("seat", "upright"):
        st.mode = "path_follow" if hyb == "upright" else "path_follow"
        # Hybrid owns motion; keep label idle under seat/upright.
        return st

    if phig_slip_trip(
        grasp_slip_m=float(grasp_slip_m),
        slip_tau_m=float(cfg.slip_tau_m),
        slip_armed_m=float(st.slip_armed_m),
        retrip_delta_m=float(cfg.retrip_delta_m),
    ):
        st.mode = "slip_reseat"
        st.slip_armed_m = float(grasp_slip_m)
        st.reseat_left = max(int(st.reseat_left), int(cfg.reseat_frames))
        return st

    near = phig_near_mouth(
        r_cmd_m=float(r_cmd_m),
        contact_resid_n=float(contact_resid_n),
        force_hole_cue=bool(force_hole_cue),
        mouth_r_m=float(cfg.mouth_r_m),
        f_mouth_press_n=float(cfg.f_mouth_press_n),
    )
    if near:
        if float(axis_err_deg) > float(cfg.hold_enter_max_axis_deg):
            st.mode = "enter_hold"
        else:
            st.mode = "mouth_press"
            st.mouth_press_frames = int(st.mouth_press_frames) + 1
        return st

    st.mode = "path_follow"
    return st


def phig_axial_override(
    *,
    mode: str,
    ax_step: float,
    cfg: PhigConfig | None = None,
) -> float:
    """Stronger axial press at mouth; freeze planar handled by caller."""
    cfg = cfg or PhigConfig()
    if str(mode) == "mouth_press":
        return max(float(ax_step), float(cfg.mouth_press_m))
    if str(mode) == "slip_reseat":
        return max(float(ax_step), 0.8 * float(cfg.mouth_press_m))
    return float(ax_step)


def is_phig_mode(mode: str) -> bool:
    m = str(mode).strip().lower()
    return m in ("track_b_phig", "phig", "track_b", "path_hybrid_grasp")
