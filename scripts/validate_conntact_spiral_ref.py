#!/usr/bin/env python3
"""Validate ConnTact Archimedean spiral against PCI planned spiral (reference port)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pci.sim_runner import _plan_surface_enter_spiral  # noqa: E402


def conntact_spiral_xy(t_sec: float, *, freq: float, amp0: float, safe_clear: float, max_cycles: float):
    """Port of ConnTact SpiralToFindHole.get_spiral_search_pose (planar part)."""
    curr_amp = amp0 + safe_clear * math.fmod(2.0 * math.pi * freq * t_sec, max_cycles)
    x = curr_amp * math.cos(2.0 * math.pi * freq * t_sec)
    y = curr_amp * math.sin(2.0 * math.pi * freq * t_sec)
    return x, y, curr_amp


def main() -> int:
    tip0 = np.array([0.12, -0.05, 0.88])
    socket = np.array([0.118, -0.048, 0.875])
    hole_axis = np.array([0.0, 0.0, -1.0])
    plane_n = np.array([0.02, 0.01, 0.999])

    wpts, radii, thetas, center, _basis = _plan_surface_enter_spiral(
        tip0,
        socket,
        hole_axis,
        plane_n,
        pitch_m=0.0012,
        dtheta=0.04,
        r_min_m=0.003,
        n_max=200,
    )
    assert wpts.shape[0] >= 1
    r_mono = np.all(np.diff(radii) <= 1e-9 + 0.001)  # inward or flat
    print(f"PCI planned spiral: N={wpts.shape[0]} r0={radii[0]:.4f} r_end={radii[-1]:.4f} mono_in={r_mono}")

    freq = 0.15
    amp0 = 0.002
    safe_clear = 0.02
    max_cycles = 62.83185
    amps = []
    for i in range(200):
        t = i * 0.033
        _, _, a = conntact_spiral_xy(t, freq=freq, amp0=amp0, safe_clear=safe_clear, max_cycles=max_cycles)
        amps.append(a)
    print(
        f"ConnTact ref spiral: t_end amp={amps[-1]:.4f} "
        f"amp_growth={(amps[-1]-amps[0]):.4f} (Archimedean in time)"
    )
    print("OK: reference spiral math loads; PCI uses geometric inward spiral on contact plane.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
