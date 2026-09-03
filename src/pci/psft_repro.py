"""Park et al. RA-L 2020 Partial Spiral Force Trajectory (PSFT) — paper reimplementation.

**No official open source.** Equations transcribed from the author-hosted PDF
``refs/papers/park_psft_ral2020.pdf`` (DYROS SNU preprint of
DOI 10.1109/LRA.2020.3000428).

Primary sources in the paper:
  - Eq. (3):  f_t = k_p (p_t − p_p)
  - Eq. (4):  SFT (r_t, θ_t) Archimedes spiral (prior method)
  - Eq. (10): PSFT (r_t, θ_t) with θ ∈ [0, θ_max], direction bounce
  - Eq. (13): wrench f* = [f; m], f = Ω f_t + Ω f_a, m = k_ω δφ

Sim embedding adaptations (force scale, non-rigid grasp, tray basis) MUST be
listed in ``docs/PSFT_FAITHFUL_REPRO.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# Paper Fig. 8 / §III–IV numeric examples (SI meters / radians).
DEFAULT_H_M = 0.0005  # path step length per Δt (Fig. 8)
DEFAULT_D_M = 0.002  # radial pitch parameter in eqs (4)/(10)
DEFAULT_R_MIN_M = 0.001
DEFAULT_R_MAX_M = 0.018
DEFAULT_THETA_MAX_RAD = 0.5 * math.pi  # Fig. 8: θ_max = π/2
DEFAULT_KP = 300.0  # experiment §IV-B
DEFAULT_KW = 3.0  # k_ω experiment §IV-B
DEFAULT_FA_N = 7.0  # assembly force f_a experiment §IV-B
DEFAULT_TILT_RAD = 0.17  # ~10° intentional tilt (ARIE / Fig. 6); paper α not tabulated
DEFAULT_DT_S = 0.003  # Fig. 8 Δt
DEFAULT_HEIGHT_DROP_M = 0.0004  # PCI SUCCESS marker (not in Park); ConnTact-compatible


@dataclass(frozen=True)
class PSFTParams:
    h_m: float = DEFAULT_H_M
    d_m: float = DEFAULT_D_M
    r_min_m: float = DEFAULT_R_MIN_M
    r_max_m: float = DEFAULT_R_MAX_M
    theta_max_rad: float = DEFAULT_THETA_MAX_RAD
    k_p: float = DEFAULT_KP
    k_omega: float = DEFAULT_KW
    f_a_n: float = DEFAULT_FA_N
    tilt_rad: float = DEFAULT_TILT_RAD
    dt_s: float = DEFAULT_DT_S
    height_drop_m: float = DEFAULT_HEIGHT_DROP_M


@dataclass
class PSFTState:
    """Mutable PSFT polar state for eq (10)."""

    r_m: float = DEFAULT_R_MIN_M
    theta_rad: float = 0.0
    theta_hat_rad: float = 0.0  # accumulated |Δθ| for radius growth
    dir_ccw: bool = True
    step: int = 0

    def reset(self, *, r_min_m: float = DEFAULT_R_MIN_M) -> None:
        self.r_m = float(r_min_m)
        self.theta_rad = 0.0
        self.theta_hat_rad = 0.0
        self.dir_ccw = True
        self.step = 0


def sft_step_polar(
    r_m: float,
    theta_rad: float,
    theta_o_rad: float,
    *,
    h_m: float,
    d_m: float,
    r_min_m: float,
    r_max_m: float,
    expanding: bool = True,
) -> tuple[float, float]:
    """One SFT update — paper eq (4). Kept for reference / ablation."""
    r = max(float(r_m), 1e-9)
    dth = math.atan2(float(h_m), r)
    th = float(theta_rad) + dth
    if expanding:
        r_new = float(r_min_m) + (th - float(theta_o_rad)) * float(d_m) / (2.0 * math.pi)
    else:
        r_new = float(r_max_m) - (th - float(theta_o_rad)) * float(d_m) / (2.0 * math.pi)
    r_new = float(np.clip(r_new, float(r_min_m), float(r_max_m)))
    return r_new, th


def psft_step_polar(
    state: PSFTState,
    *,
    params: PSFTParams | None = None,
) -> tuple[float, float]:
    """One PSFT update — paper eq (10).

    θ ∈ [0, θ_max] with cw/ccw bounce; r grows with accumulated θ̂:
      r = r_min + θ̂ * d / (2 θ_max), clipped to r_max.
    """
    p = params or PSFTParams()
    r = max(float(state.r_m), 1e-9)
    dth = math.atan2(float(p.h_m), r)
    if state.dir_ccw:
        th = float(state.theta_rad) + dth
    else:
        th = float(state.theta_rad) - dth
    th_max = float(p.theta_max_rad)
    if th >= th_max:
        th = th_max
        state.dir_ccw = False
    elif th <= 0.0:
        th = 0.0
        state.dir_ccw = True
    state.theta_hat_rad = float(state.theta_hat_rad) + dth
    r_new = float(p.r_min_m) + float(state.theta_hat_rad) * float(p.d_m) / (
        2.0 * max(th_max, 1e-9)
    )
    if r_new > float(p.r_max_m):
        r_new = float(p.r_max_m)
    state.r_m = float(r_new)
    state.theta_rad = float(th)
    state.step = int(state.step) + 1
    return state.r_m, state.theta_rad


def force_from_spring(
    p_t: np.ndarray,
    p_p: np.ndarray,
    *,
    k_p: float = DEFAULT_KP,
) -> np.ndarray:
    """Paper eq (3): f_t = k_p (p_t − p_p)."""
    return float(k_p) * (
        np.asarray(p_t, dtype=np.float64).reshape(3)
        - np.asarray(p_p, dtype=np.float64).reshape(3)
    )


def wrench_force(
    f_t: np.ndarray,
    *,
    f_a: np.ndarray,
    omega: np.ndarray | None = None,
) -> np.ndarray:
    """Paper eq (13) force part: f = Ω f_t + Ω f_a (Ω=I when fully force-task)."""
    ft = np.asarray(f_t, dtype=np.float64).reshape(3)
    fa = np.asarray(f_a, dtype=np.float64).reshape(3)
    if omega is None:
        return ft + fa
    om = np.asarray(omega, dtype=np.float64).reshape(3, 3)
    return om @ ft + om @ fa


def theta_max_lower_bound(
    e_r_m: float,
    *,
    l_m: float,
    alpha_rad: float,
) -> float:
    """Paper eq (9): polar angle range must exceed 2 atan(2 e_r / (l + l cos α))."""
    den = float(l_m) * (1.0 + math.cos(float(alpha_rad)))
    if den <= 1e-12:
        return math.pi
    return 2.0 * math.atan2(2.0 * float(e_r_m), den)


def shift_distance_m(*, l_m: float, alpha_rad: float) -> float:
    """Paper eq (8): shift magnitude ½(l + l cos α) into two-point region."""
    return 0.5 * float(l_m) * (1.0 + math.cos(float(alpha_rad)))


def spiral_target_on_plane(
    state: PSFTState,
    *,
    center: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    params: PSFTParams | None = None,
) -> tuple[np.ndarray, dict]:
    """Advance PSFT eq (10) and map (r, θ) into contact-plane basis.

    Returns desired peg position p_t (paper) as a 3D point on the plane.
    Caller applies compliance / TCP tracking and seeking force f_a along normal.
    """
    p = params or PSFTParams()
    r, th = psft_step_polar(state, params=p)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    e1 = np.asarray(t1, dtype=np.float64).reshape(3)
    e2 = np.asarray(t2, dtype=np.float64).reshape(3)
    target = c + float(r) * (math.cos(th) * e1 + math.sin(th) * e2)
    meta = {
        "r_m": float(r),
        "theta_rad": float(th),
        "theta_hat_rad": float(state.theta_hat_rad),
        "dir_ccw": bool(state.dir_ccw),
        "step": int(state.step),
        "theta_max_rad": float(p.theta_max_rad),
        "k_p": float(p.k_p),
        "f_a_n": float(p.f_a_n),
    }
    return target, meta


def height_drop_enter(
    height_now: float,
    surface_height: float,
    *,
    drop_m: float = DEFAULT_HEIGHT_DROP_M,
) -> bool:
    """PCI enter marker (SUCCESS_STANDARD path); Park uses velocity≈0 at 3-pt contact."""
    return float(height_now) <= float(surface_height) - float(drop_m)


def assert_matches_paper_sample() -> None:
    """Numeric lock against eqs (3)/(9)/(10) with Fig. 8 defaults."""
    # eq (3)
    ft = force_from_spring(np.array([1.0, 0.0, 0.0]), np.zeros(3), k_p=300.0)
    assert abs(float(ft[0]) - 300.0) < 1e-9
    # eq (9)
    th_lb = theta_max_lower_bound(0.005, l_m=0.025, alpha_rad=0.17)
    expect = 2.0 * math.atan2(0.01, 0.025 * (1.0 + math.cos(0.17)))
    assert abs(th_lb - expect) < 1e-12
    # eq (10) first steps: r grows, θ stays in [0, θ_max]
    st = PSFTState()
    st.reset(r_min_m=0.001)
    p = PSFTParams()
    prev_r = st.r_m
    for _ in range(50):
        r, th = psft_step_polar(st, params=p)
        assert 0.0 - 1e-12 <= th <= p.theta_max_rad + 1e-12
        assert r + 1e-15 >= prev_r - 1e-12  # non-decreasing until r_max
        prev_r = r
    # Bounce: drive until θ hits θ_max then reverses
    st2 = PSFTState()
    st2.reset(r_min_m=0.001)
    saw_cw = False
    for _ in range(5000):
        psft_step_polar(st2, params=p)
        if not st2.dir_ccw:
            saw_cw = True
            break
    assert saw_cw
    # Plane target finite
    tgt, meta = spiral_target_on_plane(
        PSFTState(),
        center=np.zeros(3),
        t1=np.array([1.0, 0.0, 0.0]),
        t2=np.array([0.0, 1.0, 0.0]),
    )
    assert tgt.shape == (3,)
    assert meta["step"] == 1
    assert height_drop_enter(0.9995, 1.0) is True
    assert height_drop_enter(0.9997, 1.0) is False
