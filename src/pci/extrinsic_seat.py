"""Extrinsic / tip-contact seat observation (shared A/B SEAT gate).

Gap (priv): START_FLOAT — wrist |resid| can look "seated" from grasp while tip
is still off tray (along0≈120–165mm vs seated ≈101mm). Absolute wrist |n·F|
(~6N gravity/bias) also false-fires extrinsic seat (AC ep07 smoke).

Math object: observe true tip–tray contact before enabling planar search
(hybrid SEAT ≻ position; Raibert/Craig 1981; ConnTact seeking).

Track-A: peg–tray contact cluster seated flag (TEC / ContactTipEstimator spirit).
Track-B: **residual** wrist force along tray/hole normal (+ optional CLEP)
         (Kim Active Extrinsic ICRA 2022; Doshi F/T lever) — no tip GT.
         Pass ``F_resid = hole_u * (fz - fz_base)``, not absolute wrench.

Forbidden: ep-specific switches; using tip_gt as the seat bit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pci.track_b_clep import ClepConfig, estimate_contact_from_wrench


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


@dataclass
class ExtrinsicSeatConfig:
    # Threshold on |n·F_resid| (contact residual), NOT absolute wrist force.
    f_n_seat: float = 0.05
    require_contact_est: bool = True
    clep: ClepConfig | None = None


@dataclass
class SeatVerdict:
    seated: bool
    source: str
    fn_n: float = 0.0
    f_norm: float = 0.0
    tip_contact_n: int = 0
    tip_force_n: float = 0.0
    meta: dict | None = None


def seat_from_tip_contact(
    *,
    ncon: int,
    force_sum_n: float,
    seat_force_n: float = 0.015,
) -> SeatVerdict:
    """Track-A: TEC-style peg–tray contact seat (no tip XY GT)."""
    ok = int(ncon) > 0 and float(force_sum_n) >= float(seat_force_n)
    return SeatVerdict(
        seated=bool(ok),
        source="tip_contact",
        tip_contact_n=int(ncon),
        tip_force_n=float(force_sum_n),
        fn_n=float(force_sum_n),
    )


def seat_from_extrinsic_wrench(
    *,
    force_xyz: np.ndarray,
    torque_xyz: np.ndarray,
    normal: np.ndarray,
    site_xyz: np.ndarray | None = None,
    tip_anchor: np.ndarray | None = None,
    cfg: ExtrinsicSeatConfig | None = None,
) -> SeatVerdict:
    """Track-B: seat iff **residual** normal wrench seats (+ optional CLEP).

    ``force_xyz`` must be residual contact force (e.g. hole_u*(fz-fz_base)),
    not absolute sensor wrench (gravity/bias false seat).
    """
    cfg = cfg or ExtrinsicSeatConfig()
    n = _unit(normal)
    f = np.asarray(force_xyz, dtype=np.float64).reshape(3)
    tau = np.asarray(torque_xyz, dtype=np.float64).reshape(3)
    fn = float(np.dot(f, n))
    f_norm = float(np.linalg.norm(f))
    meta: dict = {"fn_n": fn, "f_norm": f_norm, "residual": True}
    if abs(fn) < float(cfg.f_n_seat):
        return SeatVerdict(
            seated=False,
            source="extrinsic_low_fn",
            fn_n=fn,
            f_norm=f_norm,
            meta=meta,
        )
    if cfg.require_contact_est and site_xyz is not None:
        clep_cfg = cfg.clep or ClepConfig(f_min_n=min(0.02, float(cfg.f_n_seat)))
        _c, em = estimate_contact_from_wrench(
            site_xyz=site_xyz,
            force_xyz=f,
            torque_xyz=tau,
            normal=n,
            tip_anchor=tip_anchor,
            cfg=clep_cfg,
        )
        meta["clep"] = em
        if _c is None:
            return SeatVerdict(
                seated=False,
                source="extrinsic_no_contact_est",
                fn_n=fn,
                f_norm=f_norm,
                meta=meta,
            )
    return SeatVerdict(
        seated=True,
        source="extrinsic_resid",
        fn_n=fn,
        f_norm=f_norm,
        meta=meta,
    )


def resolve_seat(
    *,
    mode: str,
    force_xyz: np.ndarray | None = None,
    torque_xyz: np.ndarray | None = None,
    normal: np.ndarray | None = None,
    site_xyz: np.ndarray | None = None,
    tip_anchor: np.ndarray | None = None,
    tip_ncon: int | None = None,
    tip_force_n: float | None = None,
    tip_seat_force_n: float = 0.015,
    extrinsic_cfg: ExtrinsicSeatConfig | None = None,
) -> SeatVerdict:
    """mode: tip_contact | extrinsic | auto (tip if ncon given else extrinsic)."""
    m = str(mode or "auto").strip().lower()
    if m in ("tip_contact", "tec", "track_a") or (
        m == "auto" and tip_ncon is not None
    ):
        if tip_ncon is None:
            return SeatVerdict(seated=False, source="tip_contact_missing")
        return seat_from_tip_contact(
            ncon=int(tip_ncon),
            force_sum_n=float(tip_force_n or 0.0),
            seat_force_n=float(tip_seat_force_n),
        )
    if force_xyz is None or normal is None or torque_xyz is None:
        return SeatVerdict(seated=False, source="extrinsic_missing_wrench")
    return seat_from_extrinsic_wrench(
        force_xyz=force_xyz,
        torque_xyz=torque_xyz,
        normal=normal,
        site_xyz=site_xyz,
        tip_anchor=tip_anchor,
        cfg=extrinsic_cfg,
    )
