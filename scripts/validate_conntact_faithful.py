#!/usr/bin/env python3
"""Numeric lock: pci.conntact_repro vs refs/ConnTact SpiralToFindHole formula."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pci.conntact_repro import (  # noqa: E402
    assert_matches_ref_sample,
    height_drop_enter,
    spiral_amp_xy,
)


def main() -> int:
    assert_matches_ref_sample()
    # Spot-check vs hand expansion of refs/ConnTact spiral_search.py
    freq, amp0, clear, mx = 0.15, 0.002, 0.02 / 100.0, 62.83
    for t in (0.0, 2.5, 10.0):
        x, y, a = spiral_amp_xy(
            t,
            frequency_hz=freq,
            min_amplitude_m=amp0,
            max_cycles=mx,
            safe_clearance_m=clear,
        )
        a_ref = amp0 + clear * math.fmod(2.0 * math.pi * freq * t, mx)
        ang = 2.0 * math.pi * freq * t
        assert abs(a - a_ref) < 1e-12
        assert abs(x - a_ref * math.cos(ang)) < 1e-12
        assert abs(y - a_ref * math.sin(ang)) < 1e-12
    assert height_drop_enter(1.0 - 0.0004, 1.0) is True
    assert height_drop_enter(1.0 - 0.00039, 1.0) is False
    print("OK: ConnTact SpiralToFindHole formula matches refs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
