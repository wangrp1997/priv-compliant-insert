"""Privileged peg–hole geometry (thin wrapper over dexjoco hybrid_insert)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reach_insert_rl.env.full_obs import privileged_full_features


@dataclass(frozen=True, slots=True)
class InsertFeatures:
    """Geometry at one control step (sim privileged)."""

    tip_pos: np.ndarray
    socket_pos: np.ndarray
    hole_axis: np.ndarray
    peg_axis: np.ndarray
    lateral_m: float
    along_m: float
    tip_socket_dist_m: float
    axis_error_rad: float


def features_from_raw(raw, *, target_along_m: float = 0.02) -> InsertFeatures:
    feat = privileged_full_features(raw, target_along_m=target_along_m)
    return InsertFeatures(
        tip_pos=np.asarray(feat["tip"], dtype=np.float64),
        socket_pos=np.asarray(feat["socket"], dtype=np.float64),
        hole_axis=np.asarray(feat["hole"], dtype=np.float64),
        peg_axis=np.asarray(feat["peg_axis"], dtype=np.float64),
        lateral_m=float(feat["lat_err"]),
        along_m=float(feat["along"]),
        tip_socket_dist_m=float(feat["tip_dist"]),
        axis_error_rad=float(feat["axis_err"]),
    )


def pbvs_standoff_gate_ok(
    feat: InsertFeatures,
    *,
    ang_gate_rad: float,
    standoff_m: float,
    standoff_tol_m: float,
) -> bool:
    """Standoff + axis only (allow lateral offset from object pose error)."""
    if feat.axis_error_rad > ang_gate_rad:
        return False
    lo = standoff_m - standoff_tol_m
    hi = standoff_m + standoff_tol_m
    return lo <= feat.tip_socket_dist_m <= hi


def coarse_align_gate_ok(
    feat: InsertFeatures,
    *,
    lat_gate_m: float,
    ang_gate_rad: float,
    standoff_m: float,
    standoff_tol_m: float,
) -> bool:
    """PBVS coarse done: near hole mouth standoff, not seated in hole."""
    if feat.lateral_m > lat_gate_m:
        return False
    if feat.axis_error_rad > ang_gate_rad:
        return False
    lo = standoff_m - standoff_tol_m
    hi = standoff_m + standoff_tol_m
    return lo <= feat.tip_socket_dist_m <= hi


def approach_gate_ok(
    feat: InsertFeatures,
    *,
    lat_gate_m: float,
    ang_gate_rad: float,
    standoff_lo_m: float,
    standoff_hi_m: float,
) -> bool:
    """True when tip is at hole mouth: aligned and at standoff distance."""
    if feat.lateral_m > lat_gate_m:
        return False
    if feat.axis_error_rad > ang_gate_rad:
        return False
    return standoff_lo_m <= feat.tip_socket_dist_m <= standoff_hi_m
