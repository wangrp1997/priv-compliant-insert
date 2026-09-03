"""Track-B OIGS: Object Impedance Grasp Stiffening (no tip GT).

Theory: docs/TRACK_B_OIGS.md
Cites: Pfanne et al. RA-L 2020 (object impedance + friction QP);
       GraspQP (refs/graspqp) — in-tree priv_grasp_opt.

Wrist follows known spiral only to the extent object-in-hand error is small
(C≈I validity). Grasp QP runs separately (sim_runner pose-hold).

Forbidden: tip_gt, tip_gt+noise, peg tip XY in the wrist law.
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
class OigsConfig:
    """Equation parameters (Pfanne / path under C≈I) — not per-ep patches."""

    k_path: float = 1.4
    max_step_m: float = 0.012
    # Rigidifying gate: alpha = clip(1 - ||e||/e_rigid, 0, 1)
    e_rigid_rad: float = 0.12  # ~7° object rot → alpha→0
    e_rigid_pos_m: float = 0.008
    f_seat_n: float = 0.05
    seat_press_m: float = 0.00045
    mouth_r_m: float = 0.0045
    mouth_press_m: float = 0.0004
    # Minimum path scale when object error large (never thrash to exact 0 forever)
    alpha_floor: float = 0.05


@dataclass
class OigsState:
    """Filled each spiral step from PrivGraspOpt residuals."""

    obj_rot_err_rad: float = 0.0
    obj_pos_err_m: float = 0.0
    meta: dict = field(default_factory=dict)


def is_oigs_mode(mode: str) -> bool:
    m = str(mode).strip().lower()
    return m in (
        "track_b_oigs",
        "oigs",
        "object_impedance_grasp_stiff",
        "pfanne_grasp_stiff",
    )


def oigs_path_alpha(
    *,
    obj_rot_err_rad: float,
    obj_pos_err_m: float = 0.0,
    cfg: OigsConfig | None = None,
) -> float:
    """C≈I validity weight from object-in-hand impedance residual."""
    cfg = cfg or OigsConfig()
    r = abs(float(obj_rot_err_rad)) / max(float(cfg.e_rigid_rad), 1e-9)
    p = abs(float(obj_pos_err_m)) / max(float(cfg.e_rigid_pos_m), 1e-9)
    raw = 1.0 - max(r, p)
    return float(np.clip(raw, float(cfg.alpha_floor), 1.0))


def oigs_wrist_delta(
    *,
    site_xyz: np.ndarray,
    path_target: np.ndarray,
    normal: np.ndarray,
    press_ax: np.ndarray,
    contact_resid_n: float,
    r_cmd_m: float,
    ax_step: float = 0.0,
    obj_rot_err_rad: float = 0.0,
    obj_pos_err_m: float = 0.0,
    cfg: OigsConfig | None = None,
) -> tuple[np.ndarray, dict]:
    """Known-path wrist step scaled by object-impedance residual (no tip GT)."""
    cfg = cfg or OigsConfig()
    site = np.asarray(site_xyz, dtype=np.float64).reshape(3)
    tgt = np.asarray(path_target, dtype=np.float64).reshape(3)
    n = _unit(normal)
    press = _unit(press_ax)

    alpha = oigs_path_alpha(
        obj_rot_err_rad=float(obj_rot_err_rad),
        obj_pos_err_m=float(obj_pos_err_m),
        cfg=cfg,
    )
    meta: dict = {
        "path": "track_b_oigs",
        "privileged": False,
        "oigs_alpha": float(alpha),
        "oigs_obj_rot_err_rad": float(obj_rot_err_rad),
        "oigs_obj_pos_err_m": float(obj_pos_err_m),
        "theory": "pfanne_object_impedance+graspqp",
    }

    ax = float(ax_step)
    if float(contact_resid_n) < float(cfg.f_seat_n):
        ax = max(ax, float(cfg.seat_press_m))
        alpha = min(alpha, 0.2)
        meta["oigs_mode"] = "seat_reseat"
    elif float(r_cmd_m) <= float(cfg.mouth_r_m) + 1e-9 and alpha > 0.5:
        ax = max(ax, float(cfg.mouth_press_m))
        meta["oigs_mode"] = "mouth"
    else:
        meta["oigs_mode"] = "path_stiff"

    e_path = _planar(tgt - site, n)
    d_w = e_path * float(cfg.k_path) * float(alpha)
    d_w = clip_step(d_w, float(cfg.max_step_m))
    hold = site + d_w + press * ax
    meta["path_err_mm"] = float(np.linalg.norm(e_path)) * 1000.0
    meta["ax_step_m"] = float(ax)
    meta["path_scale"] = float(alpha)
    return hold, meta
