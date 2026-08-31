"""PBVS perception bias: privileged lateral + hole-axis tilt (eval/setup only).

Biases only what hybrid *sees*; physical hole pose unchanged. Control loop
still does not read tip/lat after A→B freeze.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pci.wrist import hole_task_basis


@dataclass(frozen=True, slots=True)
class SocketBiasConfig:
    lat_std_m: float = 0.016
    along_std_m: float = 0.0
    min_lat_m: float = 0.014
    max_lat_m: float = 0.022
    max_resamples: int = 8
    seed: int | None = 0
    # Perceived hole-axis tilt vs true (rad); PBVS aligns to tilted hole.
    axis_tilt_std_rad: float = 0.12
    axis_tilt_min_rad: float = 0.08
    axis_tilt_max_rad: float = 0.18


def sample_socket_bias_world(
    hole_axis_world: np.ndarray,
    *,
    cfg: SocketBiasConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """孔平面内横向标定误差；幅度强制落在 [min_lat, max_lat]。"""
    t1, t2, _axis = hole_task_basis(hole_axis_world)
    gen = rng if rng is not None else np.random.default_rng(cfg.seed)
    lo = float(cfg.min_lat_m)
    hi = float(getattr(cfg, "max_lat_m", max(lo, 0.012)))
    if hi < lo:
        hi = lo

    for _ in range(max(1, int(cfg.max_resamples))):
        d1 = float(gen.normal(0.0, cfg.lat_std_m))
        d2 = float(gen.normal(0.0, cfg.lat_std_m))
        lat = float(np.hypot(d1, d2))
        if lat < 1e-9:
            continue
        if lo <= lat <= hi:
            return t1 * d1 + t2 * d2
        if lat > 0:
            scale = float(np.clip(lat, lo, hi) / lat)
            return t1 * (d1 * scale) + t2 * (d2 * scale)

    theta = float(gen.uniform(0.0, 2.0 * np.pi))
    lat = float(gen.uniform(lo, hi))
    return t1 * (lat * np.cos(theta)) + t2 * (lat * np.sin(theta))


def sample_hole_axis_tilt(
    hole_axis_world: np.ndarray,
    *,
    cfg: SocketBiasConfig,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """Return rotation matrix R (3x3) and tilt angle: perceived = R @ true_axis."""
    t1, t2, axis = hole_task_basis(hole_axis_world)
    gen = rng if rng is not None else np.random.default_rng(cfg.seed)
    lo = float(cfg.axis_tilt_min_rad)
    hi = float(cfg.axis_tilt_max_rad)
    if hi <= 0.0 and float(cfg.axis_tilt_std_rad) <= 0.0:
        return np.eye(3, dtype=np.float64), 0.0
    if hi < lo:
        hi = lo
    ang = abs(float(gen.normal(0.0, max(cfg.axis_tilt_std_rad, 1e-6))))
    ang = float(np.clip(ang, lo, hi)) if hi > 0.0 else float(ang)
    if ang < 1e-9:
        return np.eye(3, dtype=np.float64), 0.0
    phi = float(gen.uniform(0.0, 2.0 * np.pi))
    pivot = t1 * np.cos(phi) + t2 * np.sin(phi)
    pivot = pivot / (np.linalg.norm(pivot) + 1e-12)
    # Rodrigues: rotate true axis around pivot by ang.
    k = pivot
    c = float(np.cos(ang))
    s = float(np.sin(ang))
    K = np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ],
        dtype=np.float64,
    )
    rot = np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)
    return rot, ang


def install_socket_bias(controller, offset_world: np.ndarray) -> None:
    """Monkey-patch hybrid controller: PBVS 对准偏置孔位，物理孔位不变。"""
    orig = controller._socket_pos
    off = np.asarray(offset_world, dtype=np.float64).reshape(3)

    def _biased(raw_env) -> np.ndarray:
        return orig(raw_env) + off

    controller._socket_pos = _biased  # type: ignore[method-assign]
    controller._pci_socket_bias_world = off  # noqa: SLF001


def install_hole_axis_tilt(controller, rot_world: np.ndarray) -> None:
    """Monkey-patch hybrid: PBVS sees tilted hole axis (pose error)."""
    orig = controller._hole_axis
    rot = np.asarray(rot_world, dtype=np.float64).reshape(3, 3)

    def _biased(raw_env) -> np.ndarray:
        axis = np.asarray(orig(raw_env), dtype=np.float64).reshape(3)
        out = rot @ axis
        n = float(np.linalg.norm(out))
        return out / (n + 1e-12)

    controller._hole_axis = _biased  # type: ignore[method-assign]
    controller._pci_hole_axis_tilt_rot = rot  # noqa: SLF001


def disable_hybrid_release(controller) -> None:
    """贴面验证：禁止 hybrid 因偏置几何误判 seated 而松手。"""
    controller._should_release_grasp = lambda _raw: False  # type: ignore[method-assign]


def bias_meta(offset_world: np.ndarray, *, axis_tilt_rad: float = 0.0) -> dict[str, float]:
    off = np.asarray(offset_world, dtype=np.float64).reshape(3)
    return {
        "socket_bias_lat_m": float(np.linalg.norm(off)),
        "socket_bias_x_m": float(off[0]),
        "socket_bias_y_m": float(off[1]),
        "socket_bias_z_m": float(off[2]),
        "axis_tilt_rad": float(axis_tilt_rad),
        "axis_tilt_deg": float(np.degrees(axis_tilt_rad)),
    }
