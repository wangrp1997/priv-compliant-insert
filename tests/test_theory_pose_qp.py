"""Contract: theory_pose_qp pose-hold path (privileged_diagnostic only)."""

from __future__ import annotations

from pathlib import Path

import yaml

from pci.sim_runner import _experiment_tag, _phase_a_qp_enable, _priv_in_control, _theory_pose_qp


REPO = Path(__file__).resolve().parents[1]


def test_theory_pose_qp_flag() -> None:
    assert _theory_pose_qp({"theory_pose_qp": True}) is True
    assert _theory_pose_qp({"compliant": {"latch_freeze": True}}) is True
    assert _theory_pose_qp({"compliant": {"search": {"latch_freeze": True}}}) is True
    assert _theory_pose_qp({"compliant": {"search": {}}}) is False


def test_theory_yaml_contract() -> None:
    path = REPO / "configs" / "theory_pose_qp.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg.get("theory_pose_qp") is True
    s = cfg["compliant"]["search"]
    a = cfg["approach"]
    assert s["priv_in_control"] is True
    assert s["priv_assist"] is False
    assert s["priv_enter_require_force"] is True
    assert float(s["priv_recovery_lat_m"]) == 0.020
    assert float(s["near_hole_left_share"]) == 0.0
    assert s["mouth_hold_stop"] is True
    assert float(s["search_left_admit_scale"]) == 0.06
    assert float(a["pbvs_lambda_z"]) == 0.08
    assert cfg["compliant"]["fingers"]["enable"] is False
    assert cfg["compliant"]["priv_grasp_opt"]["enable"] is True
    assert float(cfg["compliant"]["priv_grasp_opt"]["friction_mu"]) == 0.45
    assert cfg["compliant"]["latch_freeze"] is True
    assert _priv_in_control(cfg) is True
    assert _experiment_tag(cfg) == "privileged_diagnostic"


def _sim_runner_source() -> str:
    return (REPO / "src" / "pci" / "sim_runner.py").read_text(encoding="utf-8")


def test_phase_a_qp_enable_flag() -> None:
    assert _phase_a_qp_enable({"compliant": {"search": {"phase_a_qp_enable": False}}}) is False
    assert _phase_a_qp_enable({"compliant": {"search": {"phase_a_qp_enable": True}}}) is True
    assert _phase_a_qp_enable({"compliant": {"search": {}}}) is True


def test_theory_r25_softland_knobs() -> None:
    path = REPO / "configs" / "theory_pose_qp.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    a = cfg["approach"]
    s = cfg["compliant"]["search"]
    assert float(a["pbvs_lambda_z"]) == 0.08
    assert float(a["surface_force_soft_delta_n"]) == 0.28
    assert float(a["surface_force_delta_n"]) == 0.55
    assert int(a["surface_force_confirm_frames"]) == 4
    assert float(a["surface_contact_unload_m"]) == 0.0045
    assert float(a["surface_settle_unload_m"]) == 0.003
    assert int(a["surface_hold_frames"]) == 12
    assert float(a["soft_latch_max_tilt_deg"]) == 8.0
    assert float(a["deliver_re_latch_max_tilt_deg"]) == 6.0
    assert float(a["surface_align_left_admit_scale"]) == 0.22
    assert float(a["surface_soft_left_admit_scale"]) == 0.35
    assert float(a["surface_hold_left_admit_scale"]) == 0.10
    assert float(a["hover_assist_lambda_z_cap"]) == 0.06
    assert s["settle_soft_continue_search"] is False
    assert float(s["settle_open_search_max_tilt_deg"]) == 8.0
    assert float(cfg["compliant"]["priv_grasp_opt"]["friction_mu"]) == 0.45


def test_phase_a_qp_armed_without_early_latch() -> None:
    src = _sim_runner_source()
    assert "Phase-A QP controller exists even if early_latch is off" in src
    assert "ALIGN left_admit armed" in src
    assert "hover_assist_lambda_z_cap" in src
    assert "damp Z" in src


def test_deliver_re_latch_in_runner() -> None:
    src = _sim_runner_source()
    assert "deliver re-latch" in src
    assert "_phase_a_qp_enable" in src


def test_insert_entry_skips_relatch_on_theory_path() -> None:
    src = _sim_runner_source()
    assert "keep surface latch (no INSERT re-lock)" in src
    assert "not _theory_pose_qp(cfg)" in src


def test_priv_grasp_delta_not_blocked_by_hold_xy() -> None:
    src = _sim_runner_source()
    marker = "fingers.enable / priv_grasp_opt: Δq from step 0"
    assert marker in src
    block = src[src.index(marker) : src.index(marker) + 280]
    assert "use_priv_grasp" in block


def test_surface_deliver_latches_priv_geom() -> None:
    src = _sim_runner_source()
    assert 'meta["latch_priv_geom"]' in src
    assert "copy_priv_grasp_geom" in src


def test_search_recovery_gated_seek_not_always_on() -> None:
    src = _sim_runner_source()
    assert "Recovery-gated mode: seek only inside search.py" in src
    assert "recovery_gate_on" in src
    search_src = (REPO / "src" / "pci" / "compliant" / "search.py").read_text(
        encoding="utf-8"
    )
    assert "_recovery_active" in search_src


def test_tip_resync_keeps_spiral_theta() -> None:
    src = _sim_runner_source()
    assert "_rebias_keep_theta" in src
    search_src = (REPO / "src" / "pci" / "compliant" / "search.py").read_text(
        encoding="utf-8"
    )
    assert "_rebias_keep_theta" in search_src


def test_theory_r4_force_enter_knobs() -> None:
    path = REPO / "configs" / "theory_pose_qp.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    s = cfg["compliant"]["search"]
    assert float(s["hole_detect_fz_drop_n"]) == 0.85
    assert int(s["hole_detect_confirm"]) == 4
    assert float(s["reject_hole_if_priv_lat_m"]) == 0.025
    assert s["priv_assist"] is False
    assert s["settle_soft_continue_search"] is False
    assert float(s.get("settle_open_search_max_tilt_deg", 20.0)) == 8.0
    assert float(cfg["compliant"]["priv_grasp_opt"]["friction_mu"]) == 0.45


def test_search_priv_gated_on_assist() -> None:
    src = _sim_runner_source()
    assert "search_priv = bool(priv_ctrl) and priv_assist" in src
    assert "priv_lat_for_search" in src


def test_priv_straighten_before_search() -> None:
    src = _sim_runner_source()
    assert "priv straighten before SEARCH" in src
    assert "settle_straighten_max_tilt_deg" in src
