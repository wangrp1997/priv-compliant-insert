"""Contract: priv_in_control=false must not drive actions from tip/lat/along/hole_axis."""

from __future__ import annotations

from pathlib import Path

from pci.sim_runner import _compliance_label, _experiment_tag, _priv_in_control


def test_priv_in_control_defaults_false_without_keys() -> None:
    assert _priv_in_control({"compliant": {"search": {}}}) is False


def test_priv_in_control_explicit_false() -> None:
    cfg = {"compliant": {"search": {"priv_in_control": False, "priv_assist": True}}}
    assert _priv_in_control(cfg) is False


def test_priv_in_control_true() -> None:
    cfg = {"compliant": {"search": {"priv_in_control": True}}}
    assert _priv_in_control(cfg) is True


def test_priv_assist_fallback_when_no_priv_in_control_key() -> None:
    assert _priv_in_control({"compliant": {"search": {"priv_assist": True}}}) is True
    assert _priv_in_control({"compliant": {"search": {"priv_assist": False}}}) is False


def test_sensor_tags_when_not_in_control() -> None:
    cfg = {"compliant": {"search": {"priv_in_control": False}}}
    assert _experiment_tag(cfg) == "sensor_control_priv_monitor"
    assert _compliance_label(cfg) == "sensor_control_priv_monitor"


def test_sensor_yaml_priv_in_control_false() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "sensor_compliant_priv_monitor.yaml"
    text = path.read_text(encoding="utf-8")
    assert "priv_in_control: false" in text


def _sim_runner_source() -> str:
    return (
        Path(__file__).resolve().parents[1] / "src" / "pci" / "sim_runner.py"
    ).read_text(encoding="utf-8")


def test_right_fz_uses_wrist_sensors_only() -> None:
    src = _sim_runner_source()
    start = src.index("def _right_fz_sensor")
    end = src.index("\ndef ", start + 1)
    body = src[start:end]
    assert "wrench_in_hole_frame" not in body
    assert "features_from_raw" not in body
    assert "read_wrist_wrench" in body


def test_tip_resync_gated_by_priv_ctrl() -> None:
    src = _sim_runner_source()
    marker = "# tip_resync uses privileged tip motion"
    assert marker in src
    block = src[src.index(marker) : src.index(marker) + 600]
    assert "priv_ctrl" in block
    assert "tip_resync_window" in block


def test_axial_clamp_and_tip_soft_require_priv_ctrl() -> None:
    src = _sim_runner_source()
    soft_marker = "# Soft tip: priv may seek/unload"
    soft_block = src[src.index(soft_marker) : src.index(soft_marker) + 900]
    assert "priv_ctrl" in soft_block
    assert "seek_m > 0.0" in soft_block
    clamp_marker = "# Privileged along gate only"
    clamp_block = src[src.index(clamp_marker) : src.index(clamp_marker) + 400]
    assert "priv_ctrl" in clamp_block
    assert "near_bottom_axial_drift_cap_m" in clamp_block
