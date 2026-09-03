#!/usr/bin/env python3
"""Unit smoke: TEC PoseDiff geom prior → tip_hat → GMHQP (no MuJoCo).

Search: docs/TRACK_A_GMHQP_TIP_FUSE.md — REUSE Kim TEC PoseDiff + GMHQP.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pci.tip_theory_estimator import ExtrinsicTipStateEKF, TheoryEstConfig
from pci.track_b_gmhqp import is_gmhqp_mode, is_gmhqp_tip_obs_mode


def main() -> int:
    assert is_gmhqp_tip_obs_mode("track_a_gmhqp_tip_fuse")
    assert is_gmhqp_mode("track_a_gmhqp_tip_fuse")

    n = np.array([0.0, 0.0, 1.0])
    wrist0 = np.array([0.12, 0.0, 0.05])
    geom = np.array([0.10, 0.0, 0.05])
    contact = np.array([0.095, 0.01, 0.05])

    # Without geom prior: free-run predict drifts away from geom under large dw.
    ekf_free = ExtrinsicTipStateEKF(TheoryEstConfig(geom_tip_prior=False))
    ekf_free.reset(wrist_pos=wrist0, plane_n=n, tip_seed=geom)
    w = wrist0.copy()
    for _ in range(40):
        w = w + np.array([0.002, 0.001, 0.0])
        ekf_free.step(
            wrist_pos=w,
            plane_n=n,
            force_xyz=None,
            torque_xyz=None,
            contact_tip=None,
            contact_n=0,
            geom_tip=geom,
        )
    err_free = float(np.linalg.norm(ekf_free.tip_world(n)[:2] - geom[:2]))

    # With geom prior: ˆt stays near geom when no extrinsic contact.
    ekf_pri = ExtrinsicTipStateEKF(
        TheoryEstConfig(geom_tip_prior=True, meas_geom_tip_var=1.5e-5)
    )
    ekf_pri.reset(wrist_pos=wrist0, plane_n=n, tip_seed=geom)
    w = wrist0.copy()
    for _ in range(40):
        w = w + np.array([0.002, 0.001, 0.0])
        ekf_pri.step(
            wrist_pos=w,
            plane_n=n,
            force_xyz=None,
            torque_xyz=None,
            contact_tip=None,
            contact_n=0,
            geom_tip=geom,
        )
    err_pri = float(np.linalg.norm(ekf_pri.tip_world(n)[:2] - geom[:2]))
    assert err_pri < 0.005, f"geom prior should anchor tip_hat, got {err_pri}"
    assert err_free > err_pri + 0.01, (
        f"prior should beat free-run: free={err_free} pri={err_pri}"
    )

    # Contact (low var) still pulls ˆt toward extrinsic locus vs soft geom.
    ekf_c = ExtrinsicTipStateEKF(
        TheoryEstConfig(geom_tip_prior=True, meas_geom_tip_var=1.5e-5)
    )
    ekf_c.reset(wrist_pos=wrist0, plane_n=n, tip_seed=geom)
    for _ in range(8):
        ekf_c.step(
            wrist_pos=wrist0,
            plane_n=n,
            contact_tip=contact,
            contact_n=1,
            contact_force_n=0.1,
            geom_tip=geom,
        )
    tip_c = ekf_c.tip_world(n)
    d_c = float(np.linalg.norm(tip_c[:2] - contact[:2]))
    d_g = float(np.linalg.norm(tip_c[:2] - geom[:2]))
    assert d_c < d_g, f"contact should dominate soft geom: dc={d_c} dg={d_g}"

    cfg = yaml.safe_load(
        (REPO / "configs/scheme_l3/S2_trackA_gmhqp_tip_fuse.yaml").read_text(
            encoding="utf-8"
        )
    )
    a = cfg["approach"]
    assert a["surface_tip_baseline"] == "track_a_gmhqp_tip_fuse"
    assert a["surface_tip_theory_geom_prior"] is True
    assert a["surface_tip_gmhqp_use_tip_obs"] is True
    assert a.get("surface_tip_oracle_mouth") is False
    print(
        f"smoke_track_a_gmhqp_tip_fuse: OK "
        f"(free={err_free*1e3:.1f}mm pri={err_pri*1e3:.1f}mm contact_pull)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
