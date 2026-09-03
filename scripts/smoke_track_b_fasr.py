#!/usr/bin/env python3
"""Unit smoke for Track-B FASR (no MuJoCo).

Checks: force bias ignores tip GT; center offset accumulates; SUCCESS knobs locked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pci.tip_tracking_baselines import baseline_hold_r
from pci.track_b_fasr import (
    FasrConfig,
    FasrState,
    fasr_wrist_delta,
    is_fasr_mode,
)


def main() -> int:
    n = np.array([0.0, 0.0, 1.0])
    site = np.array([0.10, 0.00, 0.05])
    tip = np.array([0.50, 0.50, 0.05])  # absurd tip — must not drive law
    path = np.array([0.08, 0.00, 0.05])
    press = -n
    # Tangential edge force +x → bias should push −x (toward hole)
    force = np.array([0.20, 0.0, 0.10])

    st = FasrState()
    hold, meta, st2 = fasr_wrist_delta(
        site_xyz=site,
        path_target=path,
        normal=n,
        press_ax=press,
        contact_resid_n=0.12,
        r_cmd_m=0.012,
        ax_step=0.0001,
        force_xyz=force,
        state=st,
        cfg=FasrConfig(f_edge_n=0.05, k_force=0.01),
    )
    assert meta["path"] == "track_b_fasr"
    assert meta["privileged"] is False
    assert meta.get("fasr_bias") is True
    assert float(np.linalg.norm(st2.center_offset)) > 0.0
    # Hold must move from site toward −x bias and/or path
    d = hold - site
    assert float(d[0]) < 0.0 or float(np.linalg.norm(d[:2])) > 0.0

    assert is_fasr_mode("track_b_fasr")
    hold2, meta2, ax = baseline_hold_r(
        "track_b_fasr",
        target=path,
        tip=tip,
        site_xyz=site,
        offset_frozen=np.zeros(3),
        coupling=None,
        spiral_n=n,
        press_ax=press,
        ax_step=0.0001,
        track=1.0,
        max_step_m=0.012,
        along_now_m=0.10,
        along0_m=0.10,
        hole_axis=n,
        along_slack_m=0.01,
        contact_resid_n=0.12,
        wrench_xyz=force,
        fasr_cfg=FasrConfig(f_edge_n=0.05),
        fasr_state=FasrState(),
        r_cmd_m=0.012,
        force_gate_enable=False,
    )
    assert meta2["path"] == "track_b_fasr"
    assert float(np.linalg.norm(hold2 - tip)) > 0.1  # not chasing absurd tip

    cfg = yaml.safe_load(
        (REPO / "configs/scheme_l3/S2_trackB_fasr.yaml").read_text()
    )
    a = cfg["approach"]
    assert a["surface_tip_baseline"] == "track_b_fasr"
    assert a["surface_planned_priv_enter_max_axis_err_deg"] == 25.0
    assert a["surface_planned_priv_enter_max_along_m"] == 0.100
    assert a.get("surface_tip_fasr_enable") is True

    print("smoke_track_b_fasr: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
