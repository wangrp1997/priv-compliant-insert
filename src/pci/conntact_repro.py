"""Faithful ConnTact SpiralToFindHole math (swri-robotics/ConnTact, Apache-2.0).

Source of truth: ``refs/ConnTact/src/conntact/spiral_search.py``
``SpiralToFindHole.get_spiral_search_pose`` / ``exit_conditions``.

This module is the paper-facing **faithful reproduction** of the published
open-source algorithm. Sim embedding may change units (force scale) and
replace absolute world XY with an on-tray tangent basis; those adaptations
must be listed in ``docs/CONNTACT_FAITHFUL_REPRO.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ConnTact connfig_peg_10mm.yaml defaults
DEFAULT_FREQUENCY_HZ = 0.15
DEFAULT_MIN_AMPLITUDE_M = 0.002
DEFAULT_MAX_CYCLES = 62.83
# objects.dimensions.safe_clearance is 0.02 mm in yaml comments but code does
# ``safe_clearance/100`` then uses meters in spiral amp — industrial board uses
# clearance in mm/100 → meters-scale pitch. Our yaml port uses meters directly
# via ``surface_conntact_safe_clearance_m`` (default 0.00002 matches /100 of 0.02mm
# is tiny; industrial effective pitch is ~safe_clearance after /100 of mm?).
# From spiral_search.py line 170: ``safe_clearance/100`` with yaml 0.02 → 0.0002 m.
DEFAULT_SAFE_CLEARANCE_M = 0.02 / 100.0  # matches SpiralToFindHole.__init__
DEFAULT_HEIGHT_DROP_M = 0.0004  # exit_conditions: surface_height - 0.0004
DEFAULT_SEEKING_FORCE_N = 7.0  # seeking_force = [0,0,-7]


@dataclass(frozen=True)
class ConnTactSpiralParams:
    frequency_hz: float = DEFAULT_FREQUENCY_HZ
    min_amplitude_m: float = DEFAULT_MIN_AMPLITUDE_M
    max_cycles: float = DEFAULT_MAX_CYCLES
    safe_clearance_m: float = DEFAULT_SAFE_CLEARANCE_M
    height_drop_m: float = DEFAULT_HEIGHT_DROP_M


def spiral_amp_xy(
    t_sec: float,
    *,
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    min_amplitude_m: float = DEFAULT_MIN_AMPLITUDE_M,
    max_cycles: float = DEFAULT_MAX_CYCLES,
    safe_clearance_m: float = DEFAULT_SAFE_CLEARANCE_M,
) -> tuple[float, float, float]:
    """Exact planar part of ``SpiralToFindHole.get_spiral_search_pose``.

    Returns (x, y, curr_amp) in meters about the hole XY origin.
    """
    t = float(t_sec)
    freq = float(frequency_hz)
    curr_amp = float(min_amplitude_m) + float(safe_clearance_m) * math.fmod(
        2.0 * math.pi * freq * t, float(max_cycles)
    )
    ang = 2.0 * math.pi * freq * t
    x = curr_amp * math.cos(ang)
    y = curr_amp * math.sin(ang)
    return x, y, curr_amp


def spiral_target_on_plane(
    t_sec: float,
    *,
    center: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    params: ConnTactSpiralParams | None = None,
) -> tuple[np.ndarray, float]:
    """Map ConnTact (x,y) into the contact-plane basis (sim tray may be tilted).

    ConnTact uses absolute world XY + current Z. We keep Z out of the pose
    (caller holds current height / seeking force owns normal).
    """
    p = params or ConnTactSpiralParams()
    x, y, amp = spiral_amp_xy(
        t_sec,
        frequency_hz=p.frequency_hz,
        min_amplitude_m=p.min_amplitude_m,
        max_cycles=p.max_cycles,
        safe_clearance_m=p.safe_clearance_m,
    )
    c = np.asarray(center, dtype=np.float64).reshape(3)
    e1 = np.asarray(t1, dtype=np.float64).reshape(3)
    e2 = np.asarray(t2, dtype=np.float64).reshape(3)
    return c + x * e1 + y * e2, float(amp)


def height_drop_enter(
    height_now: float,
    surface_height: float,
    *,
    drop_m: float = DEFAULT_HEIGHT_DROP_M,
) -> bool:
    """Exact ``exit_conditions`` of SpiralToFindHole (height ≤ surface − 0.4 mm)."""
    return float(height_now) <= float(surface_height) - float(drop_m)


def assert_matches_ref_sample() -> None:
    """Numeric lock against refs formula at a few times."""
    # Hand-expanded from spiral_search.py with connfig defaults.
    for t in (0.0, 1.0 / 0.15, 3.333):
        x, y, a = spiral_amp_xy(t)
        curr_amp = 0.002 + (0.02 / 100.0) * math.fmod(
            2.0 * math.pi * 0.15 * t, 62.83
        )
        ang = 2.0 * math.pi * 0.15 * t
        assert abs(a - curr_amp) < 1e-12
        assert abs(x - curr_amp * math.cos(ang)) < 1e-12
        assert abs(y - curr_amp * math.sin(ang)) < 1e-12
    assert height_drop_enter(0.9995, 1.0) is True
    assert height_drop_enter(0.9997, 1.0) is False
