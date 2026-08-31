"""socket bias must miss hole; axis tilt samples in configured band."""

from __future__ import annotations

import numpy as np

from pci.pbvs_socket_bias import (
    SocketBiasConfig,
    sample_hole_axis_tilt,
    sample_socket_bias_world,
)


def test_sample_socket_bias_enforces_min_max():
    cfg = SocketBiasConfig(lat_std_m=0.012, min_lat_m=0.008, max_lat_m=0.014, seed=0)
    hole = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    rng = np.random.default_rng(0)
    for _ in range(40):
        off = sample_socket_bias_world(hole, cfg=cfg, rng=rng)
        lat = float(np.linalg.norm(off))
        assert 0.008 - 1e-9 <= lat <= 0.014 + 1e-9
        assert abs(float(off @ hole)) < 1e-9


def test_sample_hole_axis_tilt_band():
    cfg = SocketBiasConfig(
        axis_tilt_std_rad=0.06,
        axis_tilt_min_rad=0.03,
        axis_tilt_max_rad=0.08,
        seed=1,
    )
    hole = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    rng = np.random.default_rng(1)
    for _ in range(30):
        rot, ang = sample_hole_axis_tilt(hole, cfg=cfg, rng=rng)
        assert 0.03 - 1e-9 <= ang <= 0.08 + 1e-9
        tilted = rot @ hole
        tilted = tilted / (np.linalg.norm(tilted) + 1e-12)
        cos = float(np.clip(tilted @ hole, -1.0, 1.0))
        assert abs(float(np.arccos(cos)) - ang) < 1e-5
