#!/usr/bin/env python3
"""Unit smoke: TEC joint gated tip → GMHQP (no MuJoCo).

General law: docs/TRACK_A_TEC_JOINT_ON_GMHQP.md — REUSE Kim TEC + GMHQP.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pci.tip_theory_estimator import ExtrinsicTipStateEKF, TheoryEstConfig, TheoryTipEstimate
from pci.track_a_tec_joint import (
    TecJointConfig,
    TecJointState,
    is_tec_joint_mode,
    select_tip_for_gmhqp,
)
from pci.track_b_gmhqp import is_gmhqp_mode, is_gmhqp_tip_obs_mode


def main() -> int:
    assert is_tec_joint_mode("track_a_tec_joint_gmhqp")
    assert is_gmhqp_mode("track_a_tec_joint_gmhqp")
    assert is_gmhqp_tip_obs_mode("track_a_tec_joint_gmhqp")

    n = np.array([0.0, 0.0, 1.0])
    tip_geom = np.array([0.10, 0.0, 0.05])
    tip_hat_far = np.array([0.18, 0.05, 0.05])  # large PoseDiff disagree
    tip_hat_near = np.array([0.101, 0.001, 0.05])

    cfg = TheoryEstConfig(geom_tip_prior=True)
    ekf = ExtrinsicTipStateEKF(cfg)
    ekf.reset(wrist_pos=np.array([0.12, 0.0, 0.05]), plane_n=n, tip_seed=tip_geom)

    # No contact → must hold geom (estimability fail).
    ekf.had_contact_meas = False
    ekf.last_contact_nis = 0.1
    est = TheoryTipEstimate(tip_hat=tip_hat_near, uncert_m=0.001, seated=False)
    st = TecJointState()
    out, st = select_tip_for_gmhqp(
        tip_hat=tip_hat_near,
        tip_geom=tip_geom,
        plane_n=n,
        est=est,
        ekf=ekf,
        joint_cfg=TecJointConfig(),
        state=st,
    )
    assert np.allclose(out, tip_geom), "no_contact must hold geom"
    assert st.last_reason == "no_contact"

    # Contact + good NIS + near geom → use ˆt.
    ekf.had_contact_meas = True
    ekf.last_contact_nis = 1.0
    ekf.just_slipped = False
    out, st = select_tip_for_gmhqp(
        tip_hat=tip_hat_near,
        tip_geom=tip_geom,
        plane_n=n,
        est=est,
        ekf=ekf,
        joint_cfg=TecJointConfig(),
        state=st,
    )
    assert np.allclose(out, tip_hat_near), "consistent ˆt must pass"
    assert st.last_use_tip_hat

    # Contact but geom disagree → hold geom.
    out, st = select_tip_for_gmhqp(
        tip_hat=tip_hat_far,
        tip_geom=tip_geom,
        plane_n=n,
        est=est,
        ekf=ekf,
        joint_cfg=TecJointConfig(),
        state=st,
    )
    assert np.allclose(out, tip_geom), "PoseDiff disagree must hold geom"
    assert st.last_reason == "geom_disagree"

    yml = yaml.safe_load(
        (REPO / "configs/scheme_l3/S2_trackA_tec_joint_gmhqp.yaml").read_text(
            encoding="utf-8"
        )
    )
    a = yml["approach"]
    assert a["surface_tip_baseline"] == "track_a_tec_joint_gmhqp"
    assert a["surface_tip_tec_joint_enable"] is True
    assert a["surface_tip_theory_geom_prior"] is True
    print("smoke_track_a_tec_joint_gmhqp: OK (general TEC gate)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
