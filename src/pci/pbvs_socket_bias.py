"""PBVS 孔位标定误差：privileged 采样偏移，只改 hybrid 感知的 socket 位置。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pci.wrist import hole_task_basis


@dataclass(frozen=True, slots=True)
class SocketBiasConfig:
    lat_std_m: float = 0.010
    along_std_m: float = 0.0
    min_lat_m: float = 0.008
    max_lat_m: float = 0.014
    max_resamples: int = 8
    seed: int | None = 0


def sample_socket_bias_world(
    hole_axis_world: np.ndarray,
    *,
    cfg: SocketBiasConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """孔平面内横向标定误差；幅度强制落在 [min_lat, max_lat]，保证偏孔贴面、插不进。"""
    t1, t2, _axis = hole_task_basis(hole_axis_world)
    gen = rng if rng is not None else np.random.default_rng(cfg.seed)
    lo = float(cfg.min_lat_m)
    hi = float(getattr(cfg, "max_lat_m", max(lo, 0.012)))
    if hi < lo:
        hi = lo

    # 方向均匀；幅度优先截断高斯，失败则均匀落到区间内（永不返回过小偏置）。
    for _ in range(max(1, int(cfg.max_resamples))):
        d1 = float(gen.normal(0.0, cfg.lat_std_m))
        d2 = float(gen.normal(0.0, cfg.lat_std_m))
        lat = float(np.hypot(d1, d2))
        if lat < 1e-9:
            continue
        if lo <= lat <= hi:
            return t1 * d1 + t2 * d2
        # 方向保留，幅度夹到区间（比直接丢弃更稳）
        if lat > 0:
            scale = float(np.clip(lat, lo, hi) / lat)
            return t1 * (d1 * scale) + t2 * (d2 * scale)

    theta = float(gen.uniform(0.0, 2.0 * np.pi))
    lat = float(gen.uniform(lo, hi))
    return t1 * (lat * np.cos(theta)) + t2 * (lat * np.sin(theta))


def install_socket_bias(controller, offset_world: np.ndarray) -> None:
    """Monkey-patch hybrid controller: PBVS 对准偏置孔位，物理孔位不变。"""
    orig = controller._socket_pos
    off = np.asarray(offset_world, dtype=np.float64).reshape(3)

    def _biased(raw_env) -> np.ndarray:
        return orig(raw_env) + off

    controller._socket_pos = _biased  # type: ignore[method-assign]
    controller._pci_socket_bias_world = off  # noqa: SLF001


def disable_hybrid_release(controller) -> None:
    """贴面验证：禁止 hybrid 因偏置几何误判 seated 而松手。"""
    controller._should_release_grasp = lambda _raw: False  # type: ignore[method-assign]


def bias_meta(offset_world: np.ndarray) -> dict[str, float]:
    off = np.asarray(offset_world, dtype=np.float64).reshape(3)
    return {
        "socket_bias_lat_m": float(np.linalg.norm(off)),
        "socket_bias_x_m": float(off[0]),
        "socket_bias_y_m": float(off[1]),
        "socket_bias_z_m": float(off[2]),
    }
