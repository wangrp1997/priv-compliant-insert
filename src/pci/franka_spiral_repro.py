"""Faithful Franka peg-in-hole force-feedback spiral (avinash246813579/franka-peg-in-hole).

Source of truth: ``refs/franka-peg-in-hole/scripted_expert_peg_v1.py``
``ScriptedExpertPegV1`` INSERT / DESCEND jam spiral (lines ~83–90, 243–282).

This module is the paper-facing **faithful reproduction** of the published
open-source jam spiral. Sim embedding may scale the force threshold and replace
absolute world XY with an on-tray tangent basis; those adaptations must be
listed in ``docs/FRANKA_SPIRAL_FAITHFUL_REPRO.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# scripted_expert_peg_v1.py class constants (INSERT force-feedback spiral)
DEFAULT_FORCE_THRESHOLD_N = 2.0
DEFAULT_SPIRAL_R_BASE_M = 0.0002
DEFAULT_SPIRAL_R_GROWTH_M = 0.0002
DEFAULT_SPIRAL_R_MAX_M = 0.015
DEFAULT_SPIRAL_OMEGA_RAD = 0.4
DEFAULT_INSERT_Z_RAMP_PER_STEP_M = 0.0005
DEFAULT_JAM_LIFT_ABOVE_HOLE_M = 0.040  # target peg-center z = hole_top + 0.040
DEFAULT_HEIGHT_DROP_M = 0.0004  # PCI enter marker (ConnTact-style); not in Franka ref


@dataclass(frozen=True)
class FrankaSpiralParams:
    force_threshold_n: float = DEFAULT_FORCE_THRESHOLD_N
    r_base_m: float = DEFAULT_SPIRAL_R_BASE_M
    r_growth_m: float = DEFAULT_SPIRAL_R_GROWTH_M
    r_max_m: float = DEFAULT_SPIRAL_R_MAX_M
    omega_rad: float = DEFAULT_SPIRAL_OMEGA_RAD
    jam_lift_above_hole_m: float = DEFAULT_JAM_LIFT_ABOVE_HOLE_M
    insert_z_ramp_per_step_m: float = DEFAULT_INSERT_Z_RAMP_PER_STEP_M
    height_drop_m: float = DEFAULT_HEIGHT_DROP_M


@dataclass
class FrankaSpiralState:
    """Mutable jam spiral counter (grows only while jammed; never resets mid-search)."""

    spiral_step: int = 0

    def reset(self) -> None:
        self.spiral_step = 0


def spiral_radius_theta(
    spiral_step: int,
    *,
    r_base_m: float = DEFAULT_SPIRAL_R_BASE_M,
    r_growth_m: float = DEFAULT_SPIRAL_R_GROWTH_M,
    r_max_m: float = DEFAULT_SPIRAL_R_MAX_M,
    omega_rad: float = DEFAULT_SPIRAL_OMEGA_RAD,
) -> tuple[float, float]:
    """Exact ``r`` / ``theta`` from ScriptedExpertPegV1 jam spiral.

    ``r = clamp(R_BASE + R_GROWTH * spiral_step, max=R_MAX)``
    ``theta = OMEGA * spiral_step``
    """
    step = float(max(int(spiral_step), 0))
    r = min(float(r_base_m) + float(r_growth_m) * step, float(r_max_m))
    theta = float(omega_rad) * step
    return float(r), float(theta)


def spiral_xy_offset(
    spiral_step: int,
    *,
    params: FrankaSpiralParams | None = None,
) -> tuple[float, float, float, float]:
    """Planar offset (dx, dy) plus (r, theta) for ``spiral_step > 0`` semantics."""
    p = params or FrankaSpiralParams()
    r, theta = spiral_radius_theta(
        spiral_step,
        r_base_m=p.r_base_m,
        r_growth_m=p.r_growth_m,
        r_max_m=p.r_max_m,
        omega_rad=p.omega_rad,
    )
    return r * math.cos(theta), r * math.sin(theta), r, theta


def is_jammed(
    *,
    upward_force_n: float,
    above_hole: bool,
    force_threshold_n: float = DEFAULT_FORCE_THRESHOLD_N,
) -> bool:
    """Exact jam predicate: ``force_high & above_hole`` (search phases only at call site)."""
    return bool(above_hole) and float(upward_force_n) > float(force_threshold_n)


def update_spiral_step(state: FrankaSpiralState, jammed: bool) -> int:
    """Grow counter only while actively jammed; preserve progress across clearances."""
    if jammed:
        state.spiral_step = int(state.spiral_step) + 1
    return int(state.spiral_step)


def spiral_target_on_plane(
    state: FrankaSpiralState,
    *,
    center: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    normal: np.ndarray,
    upward_force_n: float,
    above_hole: bool,
    hole_top_along_m: float | None = None,
    tip_along_m: float | None = None,
    params: FrankaSpiralParams | None = None,
) -> tuple[np.ndarray, dict]:
    """Map Franka jam spiral into tray-plane basis.

    Ref applies spiral XY once ``spiral_step > 0`` (ever jammed) and overrides Z
    to ``hole_top + 0.040`` while actively jammed. We keep seeking / Z ownership
    at the call site when not jammed; when jammed we add a lift along ``normal``.
    """
    p = params or FrankaSpiralParams()
    jammed = is_jammed(
        upward_force_n=upward_force_n,
        above_hole=above_hole,
        force_threshold_n=p.force_threshold_n,
    )
    step = update_spiral_step(state, jammed)
    ever = step > 0
    c = np.asarray(center, dtype=np.float64).reshape(3)
    e1 = np.asarray(t1, dtype=np.float64).reshape(3)
    e2 = np.asarray(t2, dtype=np.float64).reshape(3)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    nn = float(np.linalg.norm(n))
    if nn > 1e-12:
        n = n / nn
    target = c.copy()
    r = 0.0
    theta = 0.0
    if ever:
        dx, dy, r, theta = spiral_xy_offset(step, params=p)
        target = c + dx * e1 + dy * e2
    lift_m = 0.0
    if jammed:
        lift_m = float(p.jam_lift_above_hole_m)
        # Ref sets absolute peg-center z; here lift along contact normal from center.
        target = target + n * lift_m
        if hole_top_along_m is not None and tip_along_m is not None:
            # Optional along-space clamp diagnostic (not used for control math).
            pass
    meta = {
        "jammed": bool(jammed),
        "ever_jammed": bool(ever),
        "spiral_step": int(step),
        "r_m": float(r),
        "theta_rad": float(theta),
        "lift_m": float(lift_m),
        "force_n": float(upward_force_n),
        "above_hole": bool(above_hole),
    }
    return target, meta


def height_drop_enter(
    height_now: float,
    surface_height: float,
    *,
    drop_m: float = DEFAULT_HEIGHT_DROP_M,
) -> bool:
    """PCI enter marker (same numeric form as ConnTact); Franka ref uses insert depth."""
    return float(height_now) <= float(surface_height) - float(drop_m)


def assert_matches_ref_sample() -> None:
    """Numeric lock against scripted_expert_peg_v1.py constants."""
    for step in (0, 1, 10, 100):
        r, th = spiral_radius_theta(step)
        expect_r = min(0.0002 + 0.0002 * float(step), 0.015)
        expect_th = 0.4 * float(step)
        assert abs(r - expect_r) < 1e-12
        assert abs(th - expect_th) < 1e-12
    st = FrankaSpiralState()
    assert is_jammed(upward_force_n=2.1, above_hole=True) is True
    assert is_jammed(upward_force_n=1.9, above_hole=True) is False
    assert is_jammed(upward_force_n=3.0, above_hole=False) is False
    update_spiral_step(st, True)
    update_spiral_step(st, False)
    update_spiral_step(st, True)
    assert st.spiral_step == 2  # grows only while jammed; does not reset
    dx, dy, r, th = spiral_xy_offset(2)
    assert abs(dx - r * math.cos(th)) < 1e-12
    assert abs(dy - r * math.sin(th)) < 1e-12
    assert height_drop_enter(0.9995, 1.0) is True
    assert height_drop_enter(0.9997, 1.0) is False
