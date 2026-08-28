"""socket bias must always miss the hole enough to press on rim."""

from __future__ import annotations

import numpy as np

from pci.pbvs_socket_bias import SocketBiasConfig, sample_socket_bias_world


def test_sample_socket_bias_enforces_min_max():
    cfg = SocketBiasConfig(lat_std_m=0.010, min_lat_m=0.008, max_lat_m=0.014, seed=0)
    hole = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    rng = np.random.default_rng(0)
    for _ in range(40):
        off = sample_socket_bias_world(hole, cfg=cfg, rng=rng)
        lat = float(np.linalg.norm(off))
        assert 0.008 - 1e-9 <= lat <= 0.014 + 1e-9
        # 应在孔平面内（与 hole 正交）
        assert abs(float(off @ hole)) < 1e-9
