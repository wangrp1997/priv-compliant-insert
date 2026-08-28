"""DexJoCo sim: PBVS coarse align → noise → force contact → compliant insert."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_insert.config import HybridInsertConfig
from hybrid_insert.controller import _Phase
from hybrid_insert.integration import EvalHybridInsert, get_raw_env
from interaction_retarget.sim.replay import rotvec_dual_arm_to_policy
from reach_insert_rl.env.full_obs import current_action44
from reach_insert_rl.env.handoff_env import InsertHandoffEnv, load_manifest_entries

from pci.approach_noise import ApproachNoiseConfig
from pci.compliant.fingers import FingerCompliantConfig
from pci.compliant.insert import CompliantInsertConfig
from pci.compliant.search import CompliantSearchConfig
from pci.features import features_from_raw, pbvs_standoff_gate_ok
from pci.pbvs_socket_bias import (
    SocketBiasConfig,
    bias_meta,
    disable_hybrid_release,
    install_socket_bias,
    sample_socket_bias_world,
)
from pci.pipeline import InsertPipeline, PipelineConfig, PipelinePhase
from pci.sensors import read_right_finger_force12, read_wrist_wrench_local
from pci.task_frame import TaskFrame
from pci.ego_video import EgoVideoRecorder
from pci.wrist import apply_tip_delta44, wrench_in_hole_frame


def _hybrid_config(cfg: dict) -> HybridInsertConfig:
    a = cfg.get("approach", {})
    names = {f.name for f in HybridInsertConfig.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in a.items() if k in names}
    return HybridInsertConfig(**kwargs)


def _sim_dt(cfg: dict) -> float:
    return 1.0 / float(cfg.get("sim", {}).get("fps", 30))


def _search_config(cfg: dict) -> CompliantSearchConfig:
    s = cfg.get("compliant", {}).get("search", {})
    merged = {**s, "dt": s.get("dt", _sim_dt(cfg))}
    names = {f.name for f in fields(CompliantSearchConfig)}
    return CompliantSearchConfig(**{k: v for k, v in merged.items() if k in names})


def _insert_config(cfg: dict) -> CompliantInsertConfig:
    ins = cfg.get("compliant", {}).get("insert", {})
    rel = cfg.get("compliant", {}).get("release", {})
    merged = {
        **ins,
        "dt": ins.get("dt", _sim_dt(cfg)),
        "max_release_steps": rel.get("max_release_steps", 80),
        "open_rate": rel.get("open_rate", 0.15),
    }
    names = {f.name for f in fields(CompliantInsertConfig)}
    return CompliantInsertConfig(**{k: v for k, v in merged.items() if k in names})


def _finger_config(cfg: dict) -> FingerCompliantConfig:
    f = cfg.get("compliant", {}).get("fingers", {})
    names = {f.name for f in fields(FingerCompliantConfig)}
    return FingerCompliantConfig(**{k: v for k, v in f.items() if k in names})


def _pipeline_config(cfg: dict) -> PipelineConfig:
    p = cfg.get("compliant", {}).get("pipeline", {})
    dt = _sim_dt(cfg)
    return PipelineConfig(
        max_jam_recoveries=int(p.get("max_jam_recoveries", 5)),
        default_dt=float(p.get("default_dt", dt)),
    )


def actual_action44_from_sites(raw) -> np.ndarray:
    """用 EE site 真值构造 44d（不是 mocap 命令），接触停用，避免钉穿入目标。"""
    from scipy.spatial.transform import Rotation as R

    from interaction_retarget.skill_replay.insert import dual_arm23_to_action44

    data = raw._data
    out = []
    for side, site_id, dof_ids in (
        ("right", int(raw._site_right_id), raw._allegro_dof_right_ids),
        ("left", int(raw._site_left_id), raw._allegro_dof_left_ids),
    ):
        pos = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
        mat = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
        quat_xyzw = R.from_matrix(mat).as_quat()
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64
        )
        hand = np.asarray(data.qpos[np.asarray(dof_ids, dtype=int)], dtype=np.float64).copy()
        if hand.shape[0] != 16:
            # fallback ctrl
            ctrl_ids = np.asarray(raw._allegro_ctrl_ids, dtype=int)
            hand = (
                np.asarray(data.ctrl[ctrl_ids[:16]], dtype=np.float64)
                if side == "right"
                else np.asarray(data.ctrl[ctrl_ids[16:32]], dtype=np.float64)
            )
        out.append(np.concatenate([pos, quat_wxyz, hand], axis=0))
    # dual_arm23_to_action44(left, right)
    return dual_arm23_to_action44(out[1], out[0]).astype(np.float64)


def step_action44(
    gym_env,
    action44: np.ndarray,
    *,
    ego_recorder: EgoVideoRecorder | None = None,
) -> None:
    action46 = rotvec_dual_arm_to_policy(np.asarray(action44, dtype=np.float64))
    out = gym_env.step(action46)
    if ego_recorder is not None:
        from pci.ego_video import ego_from_gym_obs

        frame = ego_from_gym_obs(out[0])
        if frame is not None:
            ego_recorder.write_rgb(frame)


def force_pci_handoff(hybrid: EvalHybridInsert, gym_env, action44: np.ndarray) -> None:
    if not hybrid.enabled or hybrid.controller is None or hybrid.controller.active:
        return
    hybrid.controller._activate(  # noqa: SLF001
        np.asarray(action44, dtype=np.float64).reshape(44),
        get_raw_env(gym_env),
    )
    print("pci: hybrid handoff -> PBVS coarse ALIGN (no INSERT)", flush=True)


def _socket_bias_cfg(cfg: dict) -> SocketBiasConfig:
    b = cfg.get("approach", {}).get("socket_bias", {})
    return SocketBiasConfig(
        lat_std_m=float(b.get("lat_std_m", 0.010)),
        along_std_m=float(b.get("along_std_m", 0.0)),
        min_lat_m=float(b.get("min_lat_m", 0.008)),
        max_lat_m=float(b.get("max_lat_m", 0.014)),
        max_resamples=int(b.get("max_resamples", 8)),
        seed=b.get("seed"),
    )


def run_pbvs_coarse_align(
    env: InsertHandoffEnv,
    hybrid: EvalHybridInsert,
    gym_env,
    *,
    cfg: dict,
    ego_recorder: EgoVideoRecorder | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[int, str, dict[str, float]]:
    """Privileged socket bias → hybrid PBVS standoff（真孔位不变，禁止 INSERT）。"""
    raw = env.unwrapped
    a_cfg = cfg.get("approach", {})
    bias_cfg = _socket_bias_cfg(cfg)
    feat0 = features_from_raw(raw)
    gen = rng if rng is not None else np.random.default_rng(bias_cfg.seed)
    socket_off = sample_socket_bias_world(feat0.hole_axis, cfg=bias_cfg, rng=gen)
    setup_meta = bias_meta(socket_off)

    if hybrid.controller is not None and not hybrid.controller.active:
        install_socket_bias(hybrid.controller, socket_off)
        print(
            "pci: PBVS socket bias "
            f"|off|={np.linalg.norm(socket_off)*1000:.1f}mm "
            f"true_lat={feat0.lateral_m*1000:.1f}mm",
            flush=True,
        )
        force_pci_handoff(hybrid, gym_env, current_action44(raw))

    max_align = int(a_cfg.get("max_align_steps", 800))
    standoff = float(a_cfg.get("pbvs_standoff_m", 0.055))
    standoff_tol = float(a_cfg.get("standoff_tol_m", 0.012))
    ang_gate = float(a_cfg.get("angle_tol_rad", 0.14))
    align_steps = 0

    for _ in range(max_align):
        action44 = current_action44(raw)
        hybrid.observe(gym_env, action44)
        merged = hybrid.merge(gym_env, action44)
        step_action44(gym_env, merged, ego_recorder=ego_recorder)
        align_steps += 1
        if hybrid.controller is None:
            break
        feat = features_from_raw(raw)
        if hybrid.controller._phase == _Phase.INSERT:  # noqa: SLF001
            hybrid.controller._phase = _Phase.ALIGN  # noqa: SLF001
            hybrid.controller._insert_steps = 0  # noqa: SLF001
            hybrid.controller._insert_align_streak = 0  # noqa: SLF001
            if pbvs_standoff_gate_ok(
                feat, ang_gate_rad=ang_gate, standoff_m=standoff, standoff_tol_m=standoff_tol
            ):
                hybrid.controller._deactivate()  # noqa: SLF001
                break
            continue
        if pbvs_standoff_gate_ok(
            feat, ang_gate_rad=ang_gate, standoff_m=standoff, standoff_tol_m=standoff_tol
        ):
            print(
                "pci: standoff (true geom) "
                f"tip={feat.tip_socket_dist_m*1000:.1f}mm "
                f"lat={feat.lateral_m*1000:.1f}mm along={feat.along_m*1000:.1f}mm",
                flush=True,
            )
            hybrid.controller._deactivate()  # noqa: SLF001
            break
        if hybrid.controller.phase_name == "RELEASE":
            break

    if hybrid.controller is not None and hybrid.controller.active:
        hybrid.controller._deactivate()  # noqa: SLF001

    outcome = env._labeler.compute(raw)
    if not outcome.peg_ok:
        return align_steps, "peg_lost_after_align", setup_meta
    if not outcome.tray_ok:
        return align_steps, "tray_lost_after_align", setup_meta

    feat_after = features_from_raw(raw)
    setup_meta["b_lat_mm"] = feat_after.lateral_m * 1000
    setup_meta["b_along_mm"] = feat_after.along_m * 1000
    return align_steps, "align_ok", setup_meta


def _final_geom_meta(feat, outcome, ctrl) -> dict[str, Any]:
    return {
        "entered_insert": False,
        "dual_arm": True,
        "left_share_xy": float(getattr(getattr(ctrl, "config", None), "left_share_xy", 0.0)),
        "final_lat_mm": feat.lateral_m * 1000,
        "final_along_mm": feat.along_m * 1000,
        "final_tip_mm": feat.tip_socket_dist_m * 1000,
        "final_axis_err_deg": float(np.degrees(feat.axis_error_rad)),
        "insert_ok": bool(getattr(outcome, "insert_ok", False)),
    }


def _hold_for_video(
    gym_env,
    raw,
    *,
    frames: int,
    ego_recorder: EgoVideoRecorder | None,
) -> int:
    """失败/贴面后多拍几帧，方便肉眼看。"""
    hold44 = current_action44(raw)
    for _ in range(max(0, frames)):
        step_action44(gym_env, hold44, ego_recorder=ego_recorder)
    return max(0, frames)


def _right_fz_hole(raw) -> float:
    feat = features_from_raw(raw)
    wr = read_wrist_wrench_local(raw)
    return float(wrench_in_hole_frame(wr[0], feat.hole_axis)[2])


def run_pbvs_biased_surface_press(
    env: InsertHandoffEnv,
    hybrid: EvalHybridInsert,
    gym_env,
    *,
    cfg: dict,
    ego_recorder: EgoVideoRecorder | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[int, str, dict[str, Any]]:
    """偏置孔 + 双臂 ALIGN；轻触力门控：一接触就冻左/关拧轴，绝不顶歪 tray。"""
    raw = env.unwrapped
    a_cfg = cfg.get("approach", {})
    bias_cfg = _socket_bias_cfg(cfg)
    feat0 = features_from_raw(raw)
    gen = rng if rng is not None else np.random.default_rng(bias_cfg.seed)
    socket_off = sample_socket_bias_world(feat0.hole_axis, cfg=bias_cfg, rng=gen)
    meta: dict[str, Any] = bias_meta(socket_off)

    if hybrid.controller is None:
        return 0, "no_controller", meta

    ctrl = hybrid.controller
    left_share0 = float(a_cfg.get("left_share_xy", 0.30))
    left_rot0 = float(a_cfg.get("left_share_rot", 0.20))
    lambda_z0 = float(a_cfg.get("pbvs_lambda_z", 0.22))
    lambda_rot0 = float(a_cfg.get("pbvs_lambda_rot", 0.12))
    ctrl.config.freeze_left_arm_at_handoff = False
    ctrl.config.left_share_xy = left_share0
    ctrl.config.left_share_rot = left_rot0
    ctrl.config.insert_align_confirm_frames = 99999
    ctrl.config.pbvs_lambda_xy = float(a_cfg.get("pbvs_lambda_xy", 0.50))
    ctrl.config.pbvs_lambda_z = lambda_z0
    ctrl.config.pbvs_lambda_rot = lambda_rot0
    ctrl.config.max_left_wrist_step_m = float(a_cfg.get("max_left_wrist_step_m", 0.002))
    disable_hybrid_release(ctrl)
    install_socket_bias(ctrl, socket_off)
    print(
        "pci: light force-gated surface "
        f"bias={np.linalg.norm(socket_off)*1000:.1f}mm "
        f"left_share={left_share0:.2f}",
        flush=True,
    )
    if not ctrl.active:
        force_pci_handoff(hybrid, gym_env, current_action44(raw))
    # 保存左手原始抓取，后续只从此放大一次（避免反复 scale 叠乘）
    left_hand0 = None
    if getattr(ctrl, "_hold_l_hand", None) is not None:
        left_hand0 = np.asarray(ctrl._hold_l_hand, dtype=np.float64).reshape(16).copy()
        s0 = float(a_cfg.get("surface_left_grasp_scale_init", 1.15))
        # 与右手 lock 一致：取更大闭合（代数更大）
        h = np.maximum(left_hand0, left_hand0 * s0)
        ctrl._hold_l_hand = np.clip(h, -1.5, 1.5)  # noqa: SLF001
        print(f"pci: left grasp init scale={s0:.2f}", flush=True)

    min_lat = float(a_cfg.get("surface_min_lat_m", 0.006))
    max_align = int(a_cfg.get("max_align_steps", 800))
    hold_frames = int(a_cfg.get("surface_hold_frames", 40))
    fail_hold = int(a_cfg.get("surface_fail_hold_frames", 45))
    tray_lost_need = int(a_cfg.get("surface_tray_lost_frames", 20))
    baseline_n = int(a_cfg.get("surface_force_baseline_frames", 40))
    soft_thresh = float(a_cfg.get("surface_force_soft_delta_n", 1.0))
    force_thresh = float(a_cfg.get("surface_force_delta_n", 1.6))
    force_confirm = int(a_cfg.get("surface_force_confirm_frames", 1))
    force_arm_after = int(a_cfg.get("surface_force_arm_after_steps", 60))
    grasp_scale = float(a_cfg.get("surface_left_grasp_scale", 1.35))
    retract_m = float(a_cfg.get("surface_unload_retract_m", 0.003))

    steps = 0
    tray_lost_streak = 0
    fz_hist: list[float] = []
    fz_baseline: float | None = None
    contact_streak = 0
    soft_latched = False
    reason = "align_timeout"

    def _tighten_left_grasp(scale: float) -> None:
        if left_hand0 is None:
            return
        h = np.maximum(left_hand0, left_hand0 * float(scale))
        ctrl._hold_l_hand = np.clip(h, -1.5, 1.5)  # noqa: SLF001
        print(
            f"pci: left grasp tighten scale={scale:.2f} "
            f"mean|q|={np.mean(np.abs(ctrl._hold_l_hand)):.3f}",
            flush=True,
        )

    def _hold_pose_only(*, frames: int) -> int:
        """钉住 EE site 真实位姿（同步 mocap=site），不再发穿入命令。"""
        hold = actual_action44_from_sites(raw)
        if getattr(ctrl, "_hold_l_hand", None) is not None:
            hold = hold.copy()
            hold[28:44] = np.asarray(ctrl._hold_l_hand, dtype=np.float64).reshape(16)
        for _ in range(max(0, frames)):
            if _ % 5 == 0:
                hold = actual_action44_from_sites(raw)
                if getattr(ctrl, "_hold_l_hand", None) is not None:
                    hold = hold.copy()
                    hold[28:44] = np.asarray(ctrl._hold_l_hand, dtype=np.float64).reshape(16)
            step_action44(gym_env, hold, ego_recorder=ego_recorder)
        return max(0, frames)

    def _fail(why: str, feat, outcome) -> tuple[int, str, dict[str, Any]]:
        nonlocal steps
        if ctrl.active:
            ctrl._deactivate()  # noqa: SLF001
        steps += _hold_pose_only(frames=min(fail_hold, 15))
        feat2 = features_from_raw(raw)
        meta["align_lat_mm"] = feat.lateral_m * 1000
        meta["align_along_mm"] = feat.along_m * 1000
        meta.update(_final_geom_meta(feat2, outcome, ctrl))
        meta["fz_baseline"] = fz_baseline
        meta["fz_last"] = _right_fz_hole(raw)
        print(
            f"pci: fail={why} "
            f"lat={feat2.lateral_m*1000:.1f}mm along={feat2.along_m*1000:.1f}mm "
            f"Fz={meta['fz_last']:+.2f}N",
            flush=True,
        )
        return steps, why, meta

    def _deliver_surface(feat, outcome, *, fz: float, d_fz: float) -> tuple[int, str, dict[str, Any]]:
        nonlocal steps
        _tighten_left_grasp(grasp_scale)
        if ctrl.active:
            ctrl._deactivate()  # noqa: SLF001
        sync = actual_action44_from_sites(raw)
        cmd = current_action44(raw)
        drift = float(np.linalg.norm(sync[0:3] - cmd[0:3]) * 1000)
        print(
            "pci: CONTACT — sync site→cmd & STOP "
            f"Fz={fz:+.2f}N residual={d_fz:+.2f}N |r|={abs(d_fz):.2f}N "
            f"drift={drift:.1f}mm "
            f"lat={feat.lateral_m*1000:.1f}mm along={feat.along_m*1000:.1f}mm",
            flush=True,
        )
        hold_n = int(a_cfg.get("surface_hold_frames", 18))
        steps += _hold_pose_only(frames=hold_n)
        feat_f = features_from_raw(raw)
        outcome_f = env._labeler.compute(raw)
        hole_u = feat_f.hole_axis / (np.linalg.norm(feat_f.hole_axis) + 1e-12)
        meta["align_lat_mm"] = feat.lateral_m * 1000
        meta["align_along_mm"] = feat.along_m * 1000
        meta.update(_final_geom_meta(feat_f, outcome_f, ctrl))
        meta.update(
            {
                "force_gated": True,
                "force_gate": "ema_residual",
                "soft_latched": soft_latched,
                "motion_stopped_on_contact": True,
                "synced_to_site": True,
                "cmd_site_drift_mm": drift,
                "left_grasp_tightened": True,
                "fz_baseline": fz_baseline,
                "fz_contact": fz,
                "fz_delta": d_fz,
                "fz_delta_abs": abs(d_fz),
                "hold_frames": hold_n,
                "final_hole_tilt_deg": float(
                    np.degrees(
                        np.arccos(
                            np.clip(abs(float(hole_u @ np.array([0.0, 0.0, 1.0]))), -1.0, 1.0)
                        )
                    )
                ),
            }
        )
        why = "surface_press"
        if outcome_f.insert_ok:
            why = "inserted_despite_bias"
        elif not outcome_f.tray_ok:
            why = "surface_then_tray_lost"
        elif feat_f.lateral_m < min_lat * 0.5:
            why = "surface_press_but_bad_geom"
        return steps, why, meta

    for _ in range(max_align):
        action44 = current_action44(raw)
        hybrid.observe(gym_env, action44)
        merged = hybrid.merge(gym_env, action44)
        if getattr(ctrl, "_hold_l_hand", None) is not None:
            merged = np.asarray(merged, dtype=np.float64).copy()
            merged[28:44] = np.asarray(ctrl._hold_l_hand, dtype=np.float64).reshape(16)
        step_action44(gym_env, merged, ego_recorder=ego_recorder)
        steps += 1

        if ctrl._phase == _Phase.INSERT:  # noqa: SLF001
            ctrl._phase = _Phase.ALIGN  # noqa: SLF001
            ctrl._insert_align_streak = 0  # noqa: SLF001

        feat = features_from_raw(raw)
        outcome = env._labeler.compute(raw)
        fz = _right_fz_hole(raw)
        fz_hist.append(fz)

        if not outcome.peg_ok:
            return _fail("peg_lost", feat, outcome)
        if not outcome.tray_ok:
            tray_lost_streak += 1
            if tray_lost_streak >= tray_lost_need:
                return _fail("tray_lost", feat, outcome)
        else:
            tray_lost_streak = 0

        if outcome.insert_ok:
            return _fail("inserted_despite_bias", feat, outcome)

        # 慢 EMA 残差：先算 residual，再更新 EMA（接触前不冻结）
        if fz_baseline is None:
            fz_baseline = float(fz)
            d_fz = 0.0
        else:
            d_fz = float(fz - fz_baseline)
            alpha = 1.0 / float(max(baseline_n, 1))
            fz_baseline = (1.0 - alpha) * float(fz_baseline) + alpha * float(fz)

        if steps == force_arm_after:
            print(f"pci: Fz EMA residual armed base={fz_baseline:+.2f}N", flush=True)

        if steps >= force_arm_after:
            contact_mag = abs(d_fz)
            if (not soft_latched) and contact_mag >= soft_thresh:
                soft_latched = True
                _tighten_left_grasp(grasp_scale)
                ctrl.config.pbvs_lambda_z = min(ctrl.config.pbvs_lambda_z, 0.08)
                ctrl.config.pbvs_lambda_rot = min(ctrl.config.pbvs_lambda_rot, 0.05)
                print(
                    f"pci: soft-contact latch |residual|={contact_mag:.2f}N "
                    f"(r={d_fz:+.2f}) — tighten left, soften Z/rot",
                    flush=True,
                )

            if contact_mag >= force_thresh:
                contact_streak += 1
            else:
                contact_streak = 0
            if contact_streak >= force_confirm:
                return _deliver_surface(feat, outcome, fz=fz, d_fz=d_fz)

    if ctrl.active:
        ctrl._deactivate()  # noqa: SLF001
    feat_a = features_from_raw(raw)
    meta["align_lat_mm"] = feat_a.lateral_m * 1000
    meta["align_along_mm"] = feat_a.along_m * 1000
    meta["fz_baseline"] = fz_baseline
    meta["fz_last"] = _right_fz_hole(raw)
    steps += _hold_for_video(
        gym_env, raw, frames=fail_hold, ego_recorder=ego_recorder
    )
    meta.update(_final_geom_meta(features_from_raw(raw), env._labeler.compute(raw), ctrl))
    return steps, reason, meta


def _approach_noise_cfg(cfg: dict) -> ApproachNoiseConfig:
    n = cfg.get("approach", {}).get("noise", {})
    return ApproachNoiseConfig(
        lat_std_m=float(n.get("lat_std_m", 0.008)),
        along_std_m=float(n.get("along_std_m", 0.003)),
        rot_std_rad=float(n.get("rot_std_rad", 0.04)),
        seed=n.get("seed"),
    )


def run_pci_episode(
    env: InsertHandoffEnv,
    *,
    cfg: dict,
    hybrid: EvalHybridInsert,
    pipeline: InsertPipeline,
    force_labeler,
    max_align_steps: int | None = None,
    max_control_steps: int | None = None,
    ego_recorder: EgoVideoRecorder | None = None,
) -> dict[str, Any]:
    raw = env.unwrapped
    gym_env = env._env
    if max_align_steps is not None:
        cfg.setdefault("approach", {})["max_align_steps"] = int(max_align_steps)
    max_ctrl = (
        int(max_control_steps)
        if max_control_steps is not None
        else int(cfg.get("sim", {}).get("max_control_steps", 1200))
    )

    align_steps, align_reason, noise_meta = run_pbvs_coarse_align(
        env, hybrid, gym_env, cfg=cfg, ego_recorder=ego_recorder
    )
    if align_reason != "align_ok":
        feat = features_from_raw(raw)
        return {
            "success": False,
            "insert_ok": False,
            "fail_reason": align_reason,
            "align_phase": "ALIGN",
            "align_steps": align_steps,
            "control_steps": 0,
            "final_phase": "APPROACH",
            "final_tip_dist_m": feat.tip_socket_dist_m,
            "eval_only": True,
            "hybrid_summary": hybrid.episode_summary(),
            "traj_tail": [],
            "phase_a_mode": "pbvs_coarse_align",
            "approach_noise": noise_meta,
        }

    action44 = current_action44(raw)
    feat_b = features_from_raw(raw)
    task_frame = TaskFrame.from_hole_axis(
        action44[0:3],
        feat_b.hole_axis,
        peg_axis_world=feat_b.peg_axis,
    )
    wrench = read_wrist_wrench_local(raw)
    finger12 = read_right_finger_force12(raw, force_labeler)
    hold_fingers = action44[6:22].copy()
    pipeline.begin_compliant(
        task_frame,
        wrench[0],
        finger12,
        hold_fingers,
        action44[0:3],
        along_at_b_m=feat_b.along_m,
    )

    traj: list[dict[str, Any]] = []
    control_steps = 0
    insert_ok = False
    fail_reason = "max_control_steps"
    final_phase = PipelinePhase.COMPLIANT_SEARCH.name

    for _ in range(max_ctrl):
        outcome = env._labeler.compute(raw)
        insert_ok = bool(outcome.insert_ok)
        action44 = current_action44(raw)
        wrench = read_wrist_wrench_local(raw)
        finger12 = read_right_finger_force12(raw, force_labeler)
        pr = pipeline.step(
            wrench[0],
            action44[0:3],
            finger12,
            insert_ok_eval=insert_ok,
            dt=_sim_dt(cfg),
        )
        final_phase = pr.phase.name

        delta = np.asarray(pr.delta_xyz, dtype=np.float64).reshape(3)
        max_step = float(cfg.get("sim", {}).get("max_pos_step_m", 0.004))
        dn = float(np.linalg.norm(delta))
        if dn > max_step:
            delta = delta * (max_step / dn)

        action44 = apply_tip_delta44(action44, delta)
        action44[6:22] = action44[6:22] + pr.delta_hand16
        if pr.finger_open > 0.0:
            action44[6:22] = hold_fingers * (1.0 - pr.finger_open)
        step_action44(gym_env, action44, ego_recorder=ego_recorder)
        control_steps += 1

        traj.append(
            {
                "phase": final_phase,
                "reason": pr.reason,
                "insert_ok": insert_ok,
                "peg_ok": bool(outcome.peg_ok),
                "wrist_fz": float(
                    task_frame.wrench_tool(wrench[0])[2]
                ),
            }
        )

        if pr.done:
            fail_reason = "" if pr.success or insert_ok else pr.reason
            break
        if insert_ok and pr.phase.name == "DONE":
            fail_reason = ""
            break

    feat_eval = features_from_raw(raw)
    success = bool(insert_ok)
    return {
        "success": success,
        "insert_ok": insert_ok,
        "fail_reason": fail_reason,
        "align_phase": "ALIGN",
        "align_steps": align_steps,
        "control_steps": control_steps,
        "final_phase": final_phase,
        "final_tip_dist_m": feat_eval.tip_socket_dist_m,
        "final_lat_m": feat_eval.lateral_m,
        "final_along_m": feat_eval.along_m,
        "eval_only": True,
        "hybrid_summary": hybrid.episode_summary(),
        "traj_tail": traj[-5:],
        "phase_b_sensors": "wrist_ft+fingertip_force+ frozen_tool_frame",
        "phase_a_mode": "pbvs_coarse_align",
        "approach_noise": noise_meta,
    }


def build_env_and_controllers(
    cfg: dict,
    *,
    sidecar_dir: Path | None = None,
    episode_indices: list[int] | None = None,
) -> tuple[InsertHandoffEnv, EvalHybridInsert, InsertPipeline, Any]:
    from dexquery.data.finger_contact_forces import FingerForceLabeler

    entries = load_manifest_entries(sidecar_dir, episode_indices)
    env = InsertHandoffEnv(entries, sidecar_dir=sidecar_dir, use_force=False)
    raw = env.unwrapped
    force_labeler = FingerForceLabeler(raw)
    force_labeler.reset_reference(raw)
    hybrid = EvalHybridInsert(
        task="bimanual_assembly",
        enabled=True,
        config=_hybrid_config(cfg),
    )
    pipeline = InsertPipeline(
        search_config=_search_config(cfg),
        insert_config=_insert_config(cfg),
        finger_config=_finger_config(cfg),
        pipeline_config=_pipeline_config(cfg),
    )
    return env, hybrid, pipeline, force_labeler


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
