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
from pci.compliant.insert import CompliantInsertConfig, CompliantInsertController
from pci.compliant.left_tray_follow import LeftTrayFollowConfig, LeftTrayFollowController
from pci.compliant.left_wrist_admit import LeftWristAdmitConfig, LeftWristAdmitController
from pci.compliant.priv_grasp_opt import PrivGraspOptConfig, PrivGraspOptController
from pci.compliant.search import CompliantSearchConfig, CompliantSearchController
from pci.features import features_from_raw, pbvs_standoff_gate_ok
from pci.priv_geom import PrivGraspGeom, copy_priv_grasp_geom, read_priv_grasp_geom
from pci.conntact_repro import (
    ConnTactSpiralParams,
    height_drop_enter,
    spiral_target_on_plane,
)
from pci.franka_spiral_repro import (
    FrankaSpiralParams,
    FrankaSpiralState,
    height_drop_enter as franka_height_drop_enter,
    spiral_target_on_plane as franka_spiral_target_on_plane,
)
from pci.psft_repro import (
    PSFTParams,
    PSFTState,
    height_drop_enter as psft_height_drop_enter,
    spiral_target_on_plane as psft_spiral_target_on_plane,
)
from pci.coupling_ekf import CouplingEKF
from pci.tip_contact_estimator import ContactTipEstimator
from pci.tip_theory_estimator import (
    ExtrinsicTipAxisEKF,
    ExtrinsicTipStateEKF,
    TheoryEstConfig,
    is_track_a_theory_mode,
    is_track_a_tip_axis_mode,
)
from pci.fail_driven_hard_hqp import HardHqpConfig
from pci.track_b_clep import ClepConfig, ClepState
from pci.extrinsic_seat import ExtrinsicSeatConfig, resolve_seat
from pci.track_b_fasr import FasrConfig, FasrState
from pci.track_a_tec_joint import (
    TecJointConfig,
    TecJointState,
    is_tec_joint_mode,
    select_tip_for_gmhqp,
)
from pci.track_b_gmhqp import (
    GmhqpConfig,
    GmhqpState,
    is_gmhqp_mode,
    is_gmhqp_tip_obs_mode,
)
from pci.track_b_gmhqp_ftip import GmhqpFtipConfig, GmhqpFtipState
from pci.track_b_gmhqp_compc import GmhqpCompcConfig, GmhqpCompcState
from pci.track_b_gmhqp_ac import GmhqpAcConfig, GmhqpAcState
from pci.track_b_hybrid_gmhqp import HybridGmhqpConfig, HybridGmhqpState
from pci.track_b_oigs import OigsConfig, OigsState
from pci.track_b_phig import (
    PhigConfig,
    PhigState,
    phig_axial_override,
    phig_near_mouth,
    phig_select_mode,
)
from pci.tip_tracking_baselines import (
    PlanarCouplingEstimator,
    SpiralForceHoleDetectState,
    TipCouplingEstimator,
    TipHybridState,
    baseline_hold_r,
    is_track_a_est_mode,
    observe_tip_planar,
    priv_oracle_mouth_target,
    tip_hybrid_select,
    tip_msar_blend_target,
    tip_msar_slip_trip,
    tip_oracle_slip_trip,
    tip_pivot_upright,
    tip_slip_can_enter_restore,
    tip_star_local_target,
    tip_star_slip_trip,
    tip_star_track_saturate,
    tip_star_yield_mouth,
    deploy_mouth_yield_cue,
    update_spiral_force_hole_detect,
)
from pci.pbvs_socket_bias import (
    SocketBiasConfig,
    bias_meta,
    disable_hybrid_release,
    install_hole_axis_tilt,
    install_socket_bias,
    sample_hole_axis_tilt,
    sample_socket_bias_world,
)
from pci.pipeline import InsertPipeline, PipelineConfig, PipelinePhase
from pci.sensors import (
    read_dual_finger_force24,
    read_right_finger_force12,
    read_wrist_wrench_local,
    read_wrist_wrench_world,
)
from pci.task_frame import TaskFrame
from pci.ego_video import EgoVideoRecorder
from pci.friction_ablation import apply_friction_ablation
from pci.wrist import apply_dual_wrist_delta44, hole_task_basis, wrench_in_hole_frame


def _peg_in_right_wrist(geom: PrivGraspGeom) -> np.ndarray:
    """Peg COM in right-wrist frame (privileged grasp retention)."""
    return np.asarray(
        geom.right_wrist_rot.T @ (geom.peg_pos - geom.right_wrist_pos),
        dtype=np.float64,
    ).reshape(3)



def _plan_surface_enter_spiral(
    tip0: np.ndarray,
    socket_pos: np.ndarray,
    hole_axis: np.ndarray,
    plane_n: np.ndarray,
    *,
    pitch_m: float,
    dtheta: float,
    r_min_m: float,
    n_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Privileged on-surface Archimedean spiral into the hole.

    Waypoints lie on the contact plane (point tip0, normal plane_n), centered
    at the hole-axis point projected onto that plane. Radius shrinks from the
    start tip offset down to r_min so the path is guaranteed to reach the mouth.
    Returns (waypoints[N,3], radii[N], thetas[N], center[3], (t1,t2,n)).
    """
    tip0 = np.asarray(tip0, dtype=np.float64).reshape(3)
    n = np.asarray(plane_n, dtype=np.float64).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    t1, t2, n = hole_task_basis(n)
    c_hole = _hole_axis_point_at_tip(tip0, socket_pos, hole_axis)
    # Project hole center onto the contact plane through tip0.
    center = c_hole - n * float(np.dot(c_hole - tip0, n))
    off0 = tip0 - center
    off0 = off0 - n * float(np.dot(off0, n))
    r0 = float(np.linalg.norm(off0))
    if r0 < 1e-9:
        # Already at center: tiny circle then sink conceptually at mouth.
        r0 = max(float(r_min_m), 0.001)
        theta0 = 0.0
    else:
        theta0 = float(np.arctan2(float(off0 @ t2), float(off0 @ t1)))
    pitch = max(float(pitch_m), 1e-6)
    dth = max(float(dtheta), 1e-4)
    r_min = max(float(r_min_m), 0.0005)
    pts: list[np.ndarray] = []
    rs: list[float] = []
    ths: list[float] = []
    theta = theta0
    # Start at current tip radius (or slightly outside), spiral inward.
    r = max(r0, r_min)
    for _i in range(max(1, int(n_max))):
        p = center + r * (np.cos(theta) * t1 + np.sin(theta) * t2)
        pts.append(p)
        rs.append(r)
        ths.append(theta)
        if r <= r_min + 1e-9:
            break
        theta = theta + dth
        turn = (theta - theta0) / (2.0 * np.pi)
        r = max(r_min, r0 - pitch * turn)
    wps = np.stack(pts, axis=0)
    return (
        wps,
        np.asarray(rs, dtype=np.float64),
        np.asarray(ths, dtype=np.float64),
        center,
        np.stack([t1, t2, n], axis=0),
    )


def _priv_planned_hole_diag(
    feat,
    *,
    along0_m: float,
    mouth_lat_m: float,
    enter_lat_m: float,
    enter_along_min_m: float,
) -> tuple[bool, bool, float]:
    """Privileged mouth/enter flags for planned-spiral (diagnostic only)."""
    lat = float(feat.lateral_m)
    along = float(feat.along_m)
    along_enter_m = max(0.0, float(along0_m) - along)
    at_mouth = lat <= float(mouth_lat_m)
    entered = lat <= float(enter_lat_m) and along_enter_m >= float(enter_along_min_m)
    return at_mouth, entered, along_enter_m


def _priv_tip_surface_depth_m(
    tip: np.ndarray, anchor: np.ndarray, normal: np.ndarray
) -> float:
    """Signed tip depth from privileged contact plane (anchor, normal). + = lifted off."""
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    a = np.asarray(anchor, dtype=np.float64).reshape(3)
    t = np.asarray(tip, dtype=np.float64).reshape(3)
    return float(np.dot(t - a, n))


def _priv_clamp_hold_on_surface(
    hold_r: np.ndarray,
    *,
    anchor: np.ndarray,
    normal: np.ndarray,
    max_lift_m: float,
) -> np.ndarray:
    """Privileged: do not command wrist target above the contact plane."""
    out = np.asarray(hold_r, dtype=np.float64).reshape(3).copy()
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    a = np.asarray(anchor, dtype=np.float64).reshape(3)
    lift = float(np.dot(out - a, n))
    if lift > float(max_lift_m):
        out = out - n * (lift - float(max_lift_m))
    return out


def _priv_planned_surface_ax_step(
    a_cfg: dict,
    *,
    contact_sp: float,
    f_des_step: float,
    depth_err: float,
    planar_err_m: float,
    kp: float,
    max_step: float,
) -> float:
    """ConnTact-style axial seeking on the contact plane.

    Reference (swri-robotics/ConnTact SpiralToFindHole):
      - command a constant seeking wrench into the surface
      - spiral pose keeps z = current (no position lift-off)
      - hole enter = tip drops below surface

    We approximate seeking with tiny position steps around f_des.
    Never issue large negative ax (that was ep03 continuous lift-off).
    If tip is still on/into the plane, overload only zeros press — no flyaway.
    """
    f_tol = float(a_cfg.get("surface_planned_force_ok_tol_n", 0.008))
    lift_tol = float(a_cfg.get("surface_planned_surface_lift_tol_m", 0.0006))
    reseat_m = float(a_cfg.get("surface_planned_surface_reseat_m", 0.00012))
    seek_max = float(a_cfg.get("surface_planned_seek_step_m", 0.00012))
    probe_m = float(a_cfg.get("surface_planned_probe_press_m", 0.00006))
    # Cap unload per step (ConnTact does not position-retract off the tray).
    unload_cap = float(
        a_cfg.get("surface_planned_seek_unload_cap_m", seek_max)
    )
    unload_cap = max(1e-5, min(unload_cap, seek_max, max_step))
    contact = float(contact_sp)
    f_des = float(f_des_step)
    on_surface = float(depth_err) <= float(lift_tol)

    if float(depth_err) > lift_tol:
        # Floated off plane — reseat (press), never keep unloading.
        return max(
            float(min(reseat_m, seek_max) * min(depth_err / max(lift_tol, 1e-6), 1.5)),
            probe_m,
        )

    err_f = f_des - contact
    if err_f > f_tol:
        # Underforce: seek/probe into surface.
        return float(
            max(np.clip(kp * err_f, 0.0, min(seek_max, max_step)), probe_m)
        )
    if err_f < -f_tol:
        # ConnTact: stay on surface — no large lift. But ep01 dug to 2–4N
        # residual; allow a tiny unload when heavily overloaded.
        if on_surface:
            heavy_n = float(
                a_cfg.get("surface_planned_heavy_resid_n", max(0.45, f_des * 8.0))
            )
            if contact >= heavy_n:
                return -float(min(unload_cap, seek_max * 0.5))
            return 0.0
        return 0.0
    # In band: tiny probe (hole-drop sense).
    return float(min(max(probe_m * 0.5, 0.0), seek_max, max_step))


def _priv_peg_tray_contact_force_n(raw) -> tuple[float, int]:
    """Privileged: sum |F| of MuJoCo contacts between peg and tray/socket geoms."""
    import mujoco

    model = raw._model
    data = raw._data
    peg_body = int(model.body("industreal_round_peg_8mm").id)
    tray_body = int(model.body("industreal_tray_insert_round_peg_8mm").id)
    peg_geoms = {
        g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == peg_body
    }
    # tray subtree geoms
    tray_bodies = {tray_body}
    changed = True
    while changed:
        changed = False
        for bid in range(model.nbody):
            if int(model.body_parentid[bid]) in tray_bodies and bid not in tray_bodies:
                tray_bodies.add(bid)
                changed = True
    tray_geoms = {
        g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in tray_bodies
    }
    total = 0.0
    n = 0
    force = __import__("numpy").zeros(6, dtype=__import__("numpy").float64)
    for i in range(int(data.ncon)):
        g1, g2 = int(data.contact[i].geom1), int(data.contact[i].geom2)
        pair = {g1, g2}
        if peg_geoms & pair and tray_geoms & pair:
            mujoco.mj_contactForce(model, data, i, force)
            total += float(__import__("numpy").linalg.norm(force[:3]))
            n += 1
    return float(total), int(n)


def _hole_axis_point_at_tip(
    tip_pos: np.ndarray, socket_pos: np.ndarray, hole_axis: np.ndarray
) -> np.ndarray:
    """Hole-axis point coplanar with tip (socket + axis * along)."""
    ax = np.asarray(hole_axis, dtype=np.float64).reshape(3)
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    s = np.asarray(socket_pos, dtype=np.float64).reshape(3)
    t = np.asarray(tip_pos, dtype=np.float64).reshape(3)
    return s + ax * float(np.dot(t - s, ax))


def _tip_lat_offset_xy(
    tip_pos: np.ndarray, center: np.ndarray, hole_axis: np.ndarray
) -> np.ndarray:
    ax = np.asarray(hole_axis, dtype=np.float64).reshape(3)
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    d = np.asarray(tip_pos, dtype=np.float64).reshape(3) - np.asarray(
        center, dtype=np.float64
    ).reshape(3)
    return d - ax * float(np.dot(d, ax))


def _priv_lat_inward_delta(
    tip_pos: np.ndarray,
    center: np.ndarray,
    hole_axis: np.ndarray,
    peg_axis: np.ndarray,
    *,
    step_m: float,
    axis_ff_gain: float = 0.0,
) -> np.ndarray:
    """One XY step toward hole axis (optional peg/hole axis-error feedforward)."""
    off_xy = _tip_lat_offset_xy(tip_pos, center, hole_axis)
    dist = float(np.linalg.norm(off_xy))
    if dist < 1e-9:
        return np.zeros(3, dtype=np.float64)
    hole_u = np.asarray(hole_axis, dtype=np.float64).reshape(3)
    hole_u = hole_u / (np.linalg.norm(hole_u) + 1e-12)
    u = off_xy / dist
    ff = float(axis_ff_gain)
    if ff > 0.0:
        peg_u = np.asarray(peg_axis, dtype=np.float64).reshape(3)
        peg_u = peg_u / (np.linalg.norm(peg_u) + 1e-12)
        tilt_ax = np.cross(hole_u, peg_u)
        tilt_ax = tilt_ax - hole_u * float(np.dot(tilt_ax, hole_u))
        tn = float(np.linalg.norm(tilt_ax))
        if tn > 1e-9:
            tilt_ax = tilt_ax / tn
            ang = ff * float(np.linalg.norm(np.cross(hole_u, peg_u)))
            u = u + np.cross(tilt_ax, u) * ang
            u = u - hole_u * float(np.dot(u, hole_u))
            un = float(np.linalg.norm(u))
            if un > 1e-9:
                u = u / un
    step = min(float(step_m), dist)
    return -u * step


def _tip_hole_pose_err(
    tip_pos: np.ndarray,
    center: np.ndarray,
    hole_axis: np.ndarray,
    peg_axis: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Return (lat_m, axis_err_rad, planar unit from tip toward hole axis)."""
    off_xy = _tip_lat_offset_xy(tip_pos, center, hole_axis)
    lat = float(np.linalg.norm(off_xy))
    hole_u = np.asarray(hole_axis, dtype=np.float64).reshape(3)
    hole_u = hole_u / (np.linalg.norm(hole_u) + 1e-12)
    peg_u = np.asarray(peg_axis, dtype=np.float64).reshape(3)
    peg_u = peg_u / (np.linalg.norm(peg_u) + 1e-12)
    axis_err = float(np.linalg.norm(np.cross(hole_u, peg_u)))
    if lat > 1e-9:
        toward = -off_xy / lat
    else:
        toward = np.zeros(3, dtype=np.float64)
    return lat, axis_err, toward


def _tip_stick_recovery(
    *,
    tip_pos: np.ndarray,
    center: np.ndarray,
    hole_axis: np.ndarray,
    peg_axis: np.ndarray,
    press_ax: np.ndarray,
    contact_n: float,
    f_des: float,
    stuck: bool,
    stick_streak: int,
    lift_gain_m_per_rad: float,
    unload_gain_n_per_rad: float,
    lat_lift_gain: float,
    max_lift_m: float,
    max_unload_n: float,
    hop_every: int,
    hop_lift_m: float,
) -> tuple[np.ndarray, float, bool]:
    """Legacy helper kept for call sites; prefer stick FSM in search loop."""
    if not stuck:
        return np.zeros(3, dtype=np.float64), float(f_des), False
    lat, axis_err, toward = _tip_hole_pose_err(
        tip_pos, center, hole_axis, peg_axis
    )
    lift_m = float(
        np.clip(
            lift_gain_m_per_rad * axis_err + lat_lift_gain * lat,
            0.0,
            max_lift_m,
        )
    )
    escalate = 1.0 + 0.02 * float(min(stick_streak, 100))
    lift_m = min(max_lift_m * 1.5, lift_m * escalate)
    hop = bool(hop_every > 0 and stick_streak > 0 and stick_streak % hop_every == 0)
    if hop:
        lift_m = max(lift_m, float(hop_lift_m))
    d_lift = (-press_ax) * lift_m
    if hop and float(np.linalg.norm(toward)) > 1e-9:
        d_lift = d_lift + toward * min(0.003, 0.3 * lat)
    unload = float(
        np.clip(unload_gain_n_per_rad * axis_err * escalate, 0.0, max_unload_n)
    )
    f_cmd = max(0.02, float(f_des) - unload)
    if contact_n > f_des + 0.05 or hop:
        f_cmd = min(f_cmd, max(0.02, 0.4 * float(f_des)))
    return d_lift, f_cmd, stick_streak >= 15


def _tip_stick_fsm_delta(
    *,
    phase: str,
    tip_pos: np.ndarray,
    center: np.ndarray,
    hole_axis: np.ndarray,
    peg_axis: np.ndarray,
    press_ax: np.ndarray,
    contact_n: float,
    f_des: float,
    phase_i: int,
    lift_frames: int,
    slide_frames: int,
    slide_step_m: float,
    clear_lift_m: float,
    clear_contact_n: float,
    mouth_lat_m: float,
    lat_enter_m: float,
    press_frames: int = 16,
) -> tuple[np.ndarray, float, str, bool]:
    """ConnTact-style surface seek: brief unload → force-on-surface slide to hole.

    Default path keeps seeking force while moving XY toward the hole (like
    ConnTact SpiralToFindHole seeking_force), instead of flying unloaded.
    Optional hop_mode (short slide_frames) still does discrete lift/press cycles.

    Returns (d_tip, f_cmd, next_phase, use_live_offset).
    """
    del press_frames  # reserved; press/retouch timing below
    lat, axis_err, toward = _tip_hole_pose_err(
        tip_pos, center, hole_axis, peg_axis
    )
    # Mild stiction break only — ConnTact stays in contact for search.
    clear_m = float(np.clip(clear_lift_m, 0.0008, 0.003))
    hop_mode = float(slide_step_m) <= 0.006 and int(slide_frames) <= 12
    seek_f = max(0.05, float(f_des))
    if phase == "lift":
        d = (-press_ax) * max(0.0006, clear_m / max(1, lift_frames))
        f_cmd = 0.02
        # Don't demand full loss of contact; brief unload then slide on surface.
        if phase_i >= max(4, lift_frames // 2) or contact_n <= max(
            clear_contact_n, 0.25 * seek_f
        ):
            return d, f_cmd, "slide", True
        if phase_i >= lift_frames:
            return d, f_cmd, "slide", True
        return d, f_cmd, "lift", True
    if phase == "slide":
        if hop_mode:
            hop = float(np.clip(slide_step_m, 0.002, 0.008))
            step = hop / float(max(3, int(slide_frames)))
            step = min(step, max(lat * 0.9, 0.0008))
            tip_goal = tip_pos + toward * step * max(3, int(slide_frames))
            tip_goal = tip_goal + (-press_ax) * 0.0004
            d = tip_goal - tip_pos
            dn = float(np.linalg.norm(d))
            cap = max(hop, 0.004)
            if dn > cap > 0.0:
                d = d * (cap / dn)
            f_cmd = 0.02
            if phase_i + 1 >= max(3, int(slide_frames)):
                return d, f_cmd, "press", False
            return d, f_cmd, "slide", False
        # Surface-compliant seek: planar toward hole + tiny press (ConnTact).
        tip_goal = np.asarray(center, dtype=np.float64).reshape(3).copy()
        tip_goal = tip_goal + press_ax * 0.00015
        d = tip_goal - tip_pos
        # Prefer planar motion; allow small axial press from force loop.
        d_ax = float(np.dot(d, press_ax))
        d = d - press_ax * d_ax
        dn = float(np.linalg.norm(d))
        cap = max(float(slide_step_m), 0.006)
        if dn > cap > 0.0:
            d = d * (cap / dn)
        f_cmd = seek_f
        near = lat <= max(float(mouth_lat_m) * 1.25, 0.0055)
        improved = lat <= float(lat_enter_m) - 0.004
        if near:
            return d, f_cmd, "mouth", False
        if phase_i >= slide_frames:
            if lat > float(mouth_lat_m) * 1.25:
                return d, f_cmd, "lift" if (not improved) else "slide", False
            return d, f_cmd, "mouth", False
        return d, f_cmd, "slide", False
    # press / retouch
    if hop_mode and phase in ("press", "retouch"):
        d = press_ax * 0.00035
        if lat > float(mouth_lat_m):
            d = d + toward * min(0.0005, 0.2 * lat)
        f_cmd = float(f_des)
        seated = contact_n >= max(0.35 * float(f_des), clear_contact_n * 3.0)
        n_press = 14
        if phase_i + 1 >= n_press or (seated and phase_i >= 6):
            near = lat <= max(float(mouth_lat_m) * 1.25, 0.0055)
            if near:
                return d, f_cmd, "mouth", False
            return d, f_cmd, "lift", False
        return d, f_cmd, "press", False
    step = min(0.4 * float(slide_step_m), max(lat * 0.2, 0.001))
    d = toward * step + press_ax * 0.00012
    f_cmd = float(f_des)
    if phase_i >= 12:
        if lat > float(mouth_lat_m) * 1.25:
            return d, f_cmd, "slide", False
        return d, f_cmd, "mouth", False
    return d, f_cmd, "retouch", False



def _mouth_wiggle_delta(
    *,
    step_idx: int,
    tip_pos: np.ndarray,
    center: np.ndarray,
    hole_axis: np.ndarray,
    peg_axis: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    press_ax: np.ndarray,
    mouth_xy_m: float,
    mouth_wx: float,
    mouth_wy: float,
    mouth_phi: float,
    mouth_rot_rad: float,
    mouth_lat_step_m: float,
    mouth_axis_ff: float,
    mouth_ax_step_m: float,
    jam_mode: bool,
) -> tuple[np.ndarray, float, float]:
    """Peg mouth primitive: inward lat + Lissajous dither + light axial + optional jam unload."""
    ph = float(step_idx)
    d_in = _priv_lat_inward_delta(
        tip_pos,
        center,
        hole_axis,
        peg_axis,
        step_m=mouth_lat_step_m,
        axis_ff_gain=mouth_axis_ff,
    )
    d_wig = mouth_xy_m * (
        np.sin(mouth_wx * ph) * t1 + np.sin(mouth_wy * ph + mouth_phi) * t2
    )
    d_ax = press_ax * (0.0 if jam_mode else mouth_ax_step_m)
    d_rot = np.zeros(3, dtype=np.float64)
    if mouth_rot_rad > 0.0:
        d_rot = mouth_rot_rad * (
            np.sin(mouth_wx * ph * 0.7) * t1
            + np.sin(mouth_wy * ph * 0.5 + mouth_phi) * t2
        )
    d = d_in + d_wig + d_ax + d_rot
    if jam_mode:
        d = d - press_ax * mouth_ax_step_m * 2.5 + d_wig * 0.25
    return d, float(np.linalg.norm(d_in)), float(np.linalg.norm(d_wig))


def _priv_tip_spiral_gate(surface_meta: dict, cfg: dict) -> dict[str, Any]:
    """Privileged demo gate: tip lift + tip spiral + surface force + grasp hold.

    Grasp hold: peg-in-right-wrist slip vs latch must stay under threshold.
    Does not loosen grasp commands.
    """
    a = cfg.get("approach", {})
    lift_need = float(a.get("priv_gate_tip_lift_m", 0.0015))
    xy_need = float(a.get("priv_gate_tip_xy_mm", 6.0))
    f_lo = float(a.get("priv_gate_force_lo_n", 0.05))
    f_hi = float(a.get("priv_gate_force_hi_n", 0.45))
    grasp_need = float(a.get("priv_gate_grasp_slip_m", 0.008))
    tip_up = float(surface_meta.get("surface_tip_up_m") or 0.0)
    along_rise = float(surface_meta.get("surface_tip_along_rise_m") or 0.0)
    tip_xy = float(surface_meta.get("spiral_tip_xy_peak_mm") or 0.0)
    grasp_slip = float(surface_meta.get("grasp_slip_peak_m") or 0.0)
    grasp_latched = bool(surface_meta.get("grasp_latch_ok", False))
    tr = surface_meta.get("force_trace") or []
    sp = [
        x
        for x in tr
        if str(x.get("phase"))
        in (
            "spiral",
            "lissajous",
            "priv_lat",
            "priv_lat_lissajous",
            "mouth_wiggle",
            "planned_spiral",
            "force_insert",
        )
    ]
    if sp:
        mean_r = float(sum(float(x.get("resid_r", 0.0)) for x in sp) / len(sp))
    else:
        mean_r = float("nan")
    lift_ok = (tip_up >= lift_need) or (along_rise >= lift_need)
    spiral_ok = tip_xy >= xy_need or bool(
        surface_meta.get("mouth_wiggle_ok")
    )
    force_ok = bool(sp) and (f_lo <= mean_r <= f_hi)
    # Stick unstick spends many low-force frames; if tip moved in-plane enough,
    # accept slightly lower mean residual.
    if (
        (not force_ok)
        and tip_xy >= xy_need
        and bool(sp)
        and mean_r < f_lo
        and mean_r >= 0.01
        and tip_xy >= 12.0
    ):
        force_ok = True
    # After stick→mouth tip servo, residual mean can be tiny; tip_xy proves motion.
    if (
        (not force_ok)
        and tip_xy >= 12.0
        and bool(sp)
        and bool(surface_meta.get("stick_reached_mouth") or surface_meta.get("mouth_wiggle_ok"))
        and mean_r < f_lo
    ):
        force_ok = True
    grasp_ok = bool(grasp_latched) and (grasp_slip <= grasp_need)
    ok = bool(lift_ok and spiral_ok and force_ok and grasp_ok)
    reasons: list[str] = []
    if not lift_ok:
        reasons.append(
            f"tip_lift_fail up={tip_up*1e3:.2f}mm alongΔ={along_rise*1e3:.2f}mm "
            f"need>={lift_need*1e3:.1f}mm"
        )
    if not spiral_ok:
        reasons.append(f"tip_spiral_fail xy={tip_xy:.2f}mm need>={xy_need:.1f}mm")
    if not force_ok:
        reasons.append(
            f"surface_force_fail mean|r|={mean_r:.3f}N need∈[{f_lo:.2f},{f_hi:.2f}]N"
        )
    if not grasp_ok:
        reasons.append(
            f"grasp_slip_fail peak={grasp_slip*1e3:.2f}mm "
            f"need≤{grasp_need*1e3:.1f}mm latch={int(grasp_latched)}"
        )
    return {
        "ok": ok,
        "lift_ok": lift_ok,
        "spiral_ok": spiral_ok,
        "force_ok": force_ok,
        "grasp_ok": grasp_ok,
        "tip_up_m": tip_up,
        "along_rise_m": along_rise,
        "tip_xy_peak_mm": tip_xy,
        "spiral_mean_resid_n": mean_r,
        "grasp_slip_peak_m": grasp_slip,
        "reasons": reasons,
    }


def _tip_servo_lift_wrist_cmd(
    tip_target: np.ndarray,
    site_xyz: np.ndarray,
    max_wrist_step_m: float,
    offset_wt: np.ndarray,
) -> np.ndarray:
    """Scheme D: wrist := tip_target + frozen (wrist0-tip0) offset; clip step.

    Offset must be frozen at lift start. Recomputing live (wrist-tip) while the
    tip sticks on the surface turns the lift into a lateral chase and tip_up
    collapses after a brief peak (privileged_diagnostic).
    """
    tip_target = np.asarray(tip_target, dtype=np.float64).reshape(3)
    site_xyz = np.asarray(site_xyz, dtype=np.float64).reshape(3)
    offset_wt = np.asarray(offset_wt, dtype=np.float64).reshape(3)
    hold_r = tip_target + offset_wt
    d_cmd = hold_r - site_xyz
    dn = float(np.linalg.norm(d_cmd))
    max_cmd = float(max_wrist_step_m)
    if dn > max_cmd > 0.0:
        hold_r = site_xyz + d_cmd * (max_cmd / dn)
    return hold_r


def _priv_in_control(cfg: dict) -> bool:
    """True only when privileged geom may drive actions (diagnostic)."""
    s = cfg.get("compliant", {}).get("search", {})
    if "priv_in_control" in s:
        return bool(s.get("priv_in_control"))
    return bool(s.get("priv_assist", False))


def _rel_rot_err_rad(latch: PrivGraspGeom, geom: PrivGraspGeom) -> float:
    """Peg–tray relative rotation error (rad) vs frozen latch."""
    from scipy.spatial.transform import Rotation as R

    r_rel0 = latch.tray_rot.T @ latch.peg_rot
    r_rel = geom.tray_rot.T @ geom.peg_rot
    return float(np.linalg.norm(R.from_matrix(r_rel0.T @ r_rel).as_rotvec()))


def _theory_pose_qp(cfg: dict) -> bool:
    """Pose-hold theory path: latch peg–tray at surface; no INSERT re-lock."""
    if bool(cfg.get("theory_pose_qp", False)):
        return True
    c = cfg.get("compliant", {})
    if bool(c.get("latch_freeze", False)):
        return True
    return bool(c.get("search", {}).get("latch_freeze", False))


def _phase_a_qp_enable(cfg: dict) -> bool:
    """Phase-A / pre-search grasp QP (surface); SEARCH/INSERT uses priv_grasp separately."""
    search = cfg.get("compliant", {}).get("search", {})
    if "phase_a_qp_enable" in search:
        return bool(search["phase_a_qp_enable"])
    return True


def _experiment_tag(cfg: dict) -> str:
    if not _priv_in_control(cfg):
        return "sensor_control_priv_monitor"
    # Theory pose-hold is privileged_diagnostic (not sensor-deployable).
    if _theory_pose_qp(cfg):
        return "privileged_diagnostic"
    if bool(cfg.get("compliant", {}).get("search", {}).get("mouth_hold_stop", False)):
        return "privileged_diagnostic_mouth"
    return "privileged_diagnostic"


def _compliance_label(cfg: dict) -> str:
    if not _priv_in_control(cfg):
        return "sensor_control_priv_monitor"
    return "force_tactile"


def _right_wrist_site_pose(raw) -> tuple[np.ndarray, np.ndarray]:
    site_id = int(raw._site_right_id)
    pos = np.asarray(raw._data.site_xpos[site_id], dtype=np.float64).copy()
    rot = np.asarray(raw._data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()
    return pos, rot


def _task_frame_from_wrist(raw, origin_world: np.ndarray | None = None) -> TaskFrame:
    pos, rot = _right_wrist_site_pose(raw)
    if origin_world is not None:
        pos = np.asarray(origin_world, dtype=np.float64).reshape(3).copy()
    return TaskFrame.from_wrist_site(pos, rot)


def _wrist_approach_axis(raw) -> np.ndarray:
    _, rot = _right_wrist_site_pose(raw)
    ax = rot[:, 2].copy()
    ax /= np.linalg.norm(ax) + 1e-12
    return ax


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


def _run_planned_spiral_axial_insert(
    env: InsertHandoffEnv,
    gym_env,
    *,
    cfg: dict,
    ctrl,
    ego_recorder: EgoVideoRecorder | None,
    force_trace: list,
    t_force: int,
    dt_f: float,
    spiral_reason: str,
    force_hole_entered: bool,
    spiral_lat_min_m: float,
) -> tuple[int, int, dict[str, Any]]:
    """Phase-B axial admittance after planned_spiral (F/T only, fingers frozen)."""
    a_cfg = cfg.get("approach", {})
    if not bool(a_cfg.get("surface_spiral_force_insert", False)):
        return 0, t_force, {}
    require_hole = bool(a_cfg.get("surface_spiral_force_insert_require_hole", True))
    deployable_only = bool(a_cfg.get("surface_spiral_force_insert_deployable_only", True))
    min_lat = float(a_cfg.get("surface_spiral_force_insert_min_lat_m", 0.015))
    allow_after = bool(a_cfg.get("surface_spiral_force_insert_after_spiral", False))
    lat_ok = float(spiral_lat_min_m) <= min_lat
    if deployable_only and not force_hole_entered:
        return 0, t_force, {
            "force_insert_skipped": True,
            "force_insert_skip_reason": "deployable_requires_force_hole",
        }
    if require_hole and not force_hole_entered:
        return 0, t_force, {
            "force_insert_skipped": True,
            "force_insert_skip_reason": "no_force_hole",
        }
    if (not force_hole_entered) and not (allow_after and lat_ok):
        return 0, t_force, {
            "force_insert_skipped": True,
            "force_insert_skip_reason": "lat_or_hole_gate",
            "force_insert_spiral_lat_min_mm": float(spiral_lat_min_m) * 1000.0,
        }

    raw = env.unwrapped
    insert_ctrl = CompliantInsertController(_insert_config(cfg))
    hold = current_action44(raw)
    if getattr(ctrl, "_hold_l_hand", None) is not None:
        hold = hold.copy()
        hold[28:44] = np.asarray(ctrl._hold_l_hand, dtype=np.float64).reshape(16)
    if getattr(ctrl, "_hold_l_arm", None) is not None:
        arm_h = np.asarray(ctrl._hold_l_arm, dtype=np.float64).reshape(22)
        hold[22:28] = arm_h[0:6].copy()
        hold[28:44] = arm_h[6:22].copy()

    feat0 = features_from_raw(raw)
    wr0 = read_wrist_wrench_world(raw)
    use_wrist_frame = bool(a_cfg.get("surface_spiral_force_insert_wrist_frame", True))
    if use_wrist_frame:
        pos0, rot0 = _right_wrist_site_pose(raw)
        task_frame = TaskFrame.from_wrist_site(hold[0:3], rot0)
    else:
        task_frame = TaskFrame.from_hole_axis(
            hold[0:3],
            feat0.hole_axis,
            peg_axis_world=feat0.peg_axis,
        )
    insert_ctrl.reset(task_frame, wr0[0], hold[0:3])

    max_steps = int(
        a_cfg.get(
            "surface_spiral_force_insert_steps",
            cfg.get("compliant", {}).get("insert", {}).get("max_insert_steps", 800),
        )
    )
    peg_grace = int(a_cfg.get("surface_spiral_force_insert_peg_grace_frames", 24))
    dt = _sim_dt(cfg)
    insert_steps = 0
    insert_reason = "run"
    insert_ok = False
    prev_delta: np.ndarray | None = None
    peg_lost_streak = 0
    tray_lost_streak = 0
    pure_axial = bool(a_cfg.get("surface_spiral_force_insert_pure_axial", True))
    max_ax_step = float(a_cfg.get("surface_spiral_force_insert_max_ax_step_m", 0.00035))
    print(
        f"pci: force-insert start reason={spiral_reason} "
        f"hole={int(force_hole_entered)} lat_min={spiral_lat_min_m*1e3:.1f}mm "
        f"max={max_steps}",
        flush=True,
    )
    for _ in range(max_steps):
        feat = features_from_raw(raw)
        out = env._labeler.compute(raw)
        wr = read_wrist_wrench_world(raw)
        hold_r = hold[0:3].copy()
        if use_wrist_frame:
            _pos_i, rot_i = _right_wrist_site_pose(raw)
            task_frame = TaskFrame.from_wrist_site(hold_r, rot_i)
        else:
            task_frame = TaskFrame.from_hole_axis(
                hold_r, feat.hole_axis, peg_axis_world=feat.peg_axis
            )
        pr = insert_ctrl.step(
            task_frame,
            wr[0],
            hold_r,
            release_trigger=False,
            dt=dt,
            prev_delta_xyz=prev_delta,
        )
        prev_delta = np.asarray(pr.delta_xyz, dtype=np.float64).reshape(3)
        hold = hold.copy()
        if pure_axial:
            approach = task_frame.approach_axis
            ax_step = max(float(np.dot(prev_delta, approach)), 0.0)
            ax_step = min(ax_step, max_ax_step)
            hold[0:3] = hold[0:3] + approach * ax_step
        else:
            hold[0:3] = hold[0:3] + prev_delta
        if pr.delta_left_xyz is not None:
            d_left = np.asarray(pr.delta_left_xyz, dtype=np.float64).reshape(3)
            hold[22:25] = hold[22:25] + d_left
        if getattr(ctrl, "_hold_l_hand", None) is not None:
            hold[28:44] = np.asarray(ctrl._hold_l_hand, dtype=np.float64).reshape(16)
        step_action44(gym_env, hold, ego_recorder=ego_recorder)
        insert_steps += 1
        wh = task_frame.wrench_tool(wr[0])
        wh_l = task_frame.wrench_tool(wr[1])
        f_des_ins = float(
            a_cfg.get(
                "surface_planned_insert_f_des_n",
                a_cfg.get("surface_const_force_des_n", 0.05),
            )
        )
        force_trace.append(
            {
                "t": float(t_force) * dt_f,
                "step": float(t_force),
                "phase": "force_insert",
                "resid_r": float(np.linalg.norm(wr[0][:3])),
                "resid_l": float(np.linalg.norm(wr[1][:3])),
                "fz_r": float(wh[2]),
                "fz_l": float(wh_l[2]),
                "f_des": float(f_des_ins),
                "lat_mm": float(feat.lateral_m) * 1000.0,
                "along_mm": float(feat.along_m) * 1000.0,
                "insert_reason": str(pr.reason),
                "force_hole_entered": int(force_hole_entered),
                "peg_ok": int(out.peg_ok),
                "tray_ok": int(out.tray_ok),
                "peg_contact_n": int(out.peg_contact_count),
            }
        )
        t_force += 1
        insert_ok = bool(out.insert_ok)
        if insert_ok:
            insert_reason = "insert_ok"
            break
        if pr.done:
            insert_reason = str(pr.reason)
            break
        if out.peg_ok:
            peg_lost_streak = 0
        else:
            peg_lost_streak += 1
        if out.tray_ok:
            tray_lost_streak = 0
        else:
            tray_lost_streak += 1
        if peg_lost_streak >= peg_grace and not out.peg_ok:
            insert_reason = "peg_contact_lost"
            break
        if tray_lost_streak >= peg_grace and not out.tray_ok:
            insert_reason = "tray_contact_lost"
            break

    print(
        f"pci: force-insert done steps={insert_steps} reason={insert_reason} "
        f"insert_ok={int(insert_ok)} along={feat.along_m*1e3:.1f}mm "
        f"lat={feat.lateral_m*1e3:.1f}mm",
        flush=True,
    )
    return insert_steps, t_force, {
        "force_insert_steps": int(insert_steps),
        "force_insert_reason": str(insert_reason),
        "force_insert_ok": bool(insert_ok),
        "insert_ok": bool(insert_ok),
        "force_hole_entered": bool(force_hole_entered),
    }


def _finger_config(cfg: dict) -> FingerCompliantConfig:
    f = cfg.get("compliant", {}).get("fingers", {})
    names = {f.name for f in fields(FingerCompliantConfig)}
    return FingerCompliantConfig(**{k: v for k, v in f.items() if k in names})


def _priv_grasp_config(cfg: dict) -> PrivGraspOptConfig:
    p = cfg.get("compliant", {}).get("priv_grasp_opt", {})
    names = {f.name for f in fields(PrivGraspOptConfig)}
    return PrivGraspOptConfig(**{k: v for k, v in p.items() if k in names})


def _left_wrist_admit_config(cfg: dict) -> LeftWristAdmitConfig:
    p = cfg.get("compliant", {}).get("left_wrist_admit", {})
    names = {f.name for f in fields(LeftWristAdmitConfig)}
    return LeftWristAdmitConfig(**{k: v for k, v in p.items() if k in names})


def _left_tray_follow_config(cfg: dict) -> LeftTrayFollowConfig:
    p = cfg.get("compliant", {}).get("left_tray_follow", {})
    names = {f.name for f in fields(LeftTrayFollowConfig)}
    return LeftTrayFollowConfig(**{k: v for k, v in p.items() if k in names})


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
        lat_std_m=float(b.get("lat_std_m", 0.016)),
        along_std_m=float(b.get("along_std_m", 0.0)),
        min_lat_m=float(b.get("min_lat_m", 0.014)),
        max_lat_m=float(b.get("max_lat_m", 0.022)),
        max_resamples=int(b.get("max_resamples", 8)),
        seed=b.get("seed"),
        axis_tilt_std_rad=float(b.get("axis_tilt_std_rad", 0.12)),
        axis_tilt_min_rad=float(b.get("axis_tilt_min_rad", 0.08)),
        axis_tilt_max_rad=float(b.get("axis_tilt_max_rad", 0.18)),
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
    tilt_rot, tilt_ang = sample_hole_axis_tilt(feat0.hole_axis, cfg=bias_cfg, rng=gen)
    setup_meta = bias_meta(socket_off, axis_tilt_rad=tilt_ang)

    if hybrid.controller is not None and not hybrid.controller.active:
        install_socket_bias(hybrid.controller, socket_off)
        install_hole_axis_tilt(hybrid.controller, tilt_rot)
        print(
            "pci: PBVS socket bias "
            f"|off|={np.linalg.norm(socket_off)*1000:.1f}mm "
            f"tilt={np.degrees(tilt_ang):.1f}deg "
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


def _right_fz_sensor(raw, task_frame: TaskFrame | None = None) -> float:
    """Right wrist axial Fz from sensors only (no privileged hole projection)."""
    if task_frame is not None:
        wr = read_wrist_wrench_world(raw)
        return float(task_frame.wrench_tool(wr[0])[2])
    local = read_wrist_wrench_local(raw)
    return float(local[0, 2])


def _right_fz_hole(raw) -> float:
    """Phase-A contact Fz projected on privileged hole axis (approach only)."""
    feat = features_from_raw(raw)
    wr = read_wrist_wrench_world(raw)
    return float(wrench_in_hole_frame(wr[0], feat.hole_axis)[2])


def run_pbvs_biased_surface_press(
    env: InsertHandoffEnv,
    hybrid: EvalHybridInsert,
    gym_env,
    *,
    cfg: dict,
    ego_recorder: EgoVideoRecorder | None = None,
    rng: np.random.Generator | None = None,
    force_labeler=None,
) -> tuple[int, str, dict[str, Any]]:
    """偏置孔 + 双臂 ALIGN；轻触力门控：一接触就冻左/关拧轴，绝不顶歪 tray。"""
    raw = env.unwrapped
    a_cfg = cfg.get("approach", {})
    bias_cfg = _socket_bias_cfg(cfg)
    # Optional privileged_diagnostic: scale peg/tray geom friction at episode start.
    friction_meta = apply_friction_ablation(raw, cfg)
    feat0 = features_from_raw(raw)
    gen = rng if rng is not None else np.random.default_rng(bias_cfg.seed)
    socket_off = sample_socket_bias_world(feat0.hole_axis, cfg=bias_cfg, rng=gen)
    tilt_rot, tilt_ang = sample_hole_axis_tilt(feat0.hole_axis, cfg=bias_cfg, rng=gen)
    meta: dict[str, Any] = bias_meta(socket_off, axis_tilt_rad=tilt_ang)
    if friction_meta.get("enabled"):
        meta["friction_ablation"] = friction_meta

    if hybrid.controller is None:
        return 0, "no_controller", meta

    ctrl = hybrid.controller
    # Surface approach: freeze left so tray is not wrenched while right seeks contact.
    left_share0 = float(a_cfg.get("surface_left_share_xy", a_cfg.get("left_share_xy", 0.0)))
    left_rot0 = float(a_cfg.get("surface_left_share_rot", a_cfg.get("left_share_rot", 0.0)))
    freeze_left = bool(a_cfg.get("surface_freeze_left", True))
    lambda_z0 = float(a_cfg.get("pbvs_lambda_z", 0.22))
    lambda_rot0 = float(a_cfg.get("pbvs_lambda_rot", 0.12))
    ctrl.config.freeze_left_arm_at_handoff = freeze_left
    ctrl.config.left_share_xy = left_share0
    ctrl.config.left_share_rot = left_rot0
    ctrl.config.insert_align_confirm_frames = 99999
    ctrl.config.pbvs_lambda_xy = float(a_cfg.get("pbvs_lambda_xy", 0.50))
    ctrl.config.pbvs_lambda_z = lambda_z0
    ctrl.config.pbvs_lambda_rot = lambda_rot0
    ctrl.config.max_left_wrist_step_m = float(a_cfg.get("max_left_wrist_step_m", 0.002))
    disable_hybrid_release(ctrl)
    install_socket_bias(ctrl, socket_off)
    install_hole_axis_tilt(ctrl, tilt_rot)
    print(
        "pci: light force-gated surface "
        f"bias={np.linalg.norm(socket_off)*1000:.1f}mm "
        f"tilt={np.degrees(tilt_ang):.1f}deg "
        f"left_share={left_share0:.2f} freeze_left={freeze_left}",
        flush=True,
    )
    if not ctrl.active:
        force_pci_handoff(hybrid, gym_env, current_action44(raw))
    # If already active, still pin left wrist+hand so tray cannot be dragged.
    if freeze_left and getattr(ctrl, "_hold_l_arm", None) is None:
        act0 = current_action44(raw)
        site0 = actual_action44_from_sites(raw)
        # Prefer site-synced left arm pose (true EE), keep tightened hand if present.
        left_arm = site0[22:44].copy()
        if getattr(ctrl, "_hold_l_hand", None) is not None:
            left_arm[6:22] = np.asarray(ctrl._hold_l_hand, dtype=np.float64).reshape(16)
        ctrl._hold_l_arm = left_arm  # noqa: SLF001
        print("pci: pin left arm to site (surface freeze)", flush=True)
    # 保存左手原始抓取，后续只从此放大一次（避免反复 scale 叠乘）
    left_hand0 = None
    freeze_left_hand = bool(a_cfg.get("surface_freeze_left_hand", True))
    phase_a_left_qp_fingers = bool(a_cfg.get("surface_phase_a_left_qp_fingers", False))
    # Tip→hole search pose-hold: priv_grasp QP on fingers while wrist tip-servos.
    # Default off — P1 baseline keeps locked hold_*_hand during tip search.
    tip_search_pose_hold = bool(a_cfg.get("surface_tip_search_pose_hold", False))
    if getattr(ctrl, "_hold_l_hand", None) is not None:
        left_hand0 = np.asarray(ctrl._hold_l_hand, dtype=np.float64).reshape(16).copy()
        s0 = float(a_cfg.get("surface_left_grasp_scale_init", 1.15))
        if freeze_left_hand or abs(s0 - 1.0) < 1e-9:
            # Keep handoff finger pose; do not re-squeeze.
            ctrl._hold_l_hand = left_hand0.copy()  # noqa: SLF001
            print("pci: left grasp FROZEN at handoff (scale=1.0)", flush=True)
        else:
            # 与右手 lock 一致：取更大闭合（代数更大）
            h = np.maximum(left_hand0, left_hand0 * s0)
            ctrl._hold_l_hand = np.clip(h, -1.5, 1.5)  # noqa: SLF001
            print(f"pci: left grasp init scale={s0:.2f}", flush=True)
        if getattr(ctrl, "_hold_l_arm", None) is not None:
            arm = np.asarray(ctrl._hold_l_arm, dtype=np.float64).reshape(22).copy()
            arm[6:22] = ctrl._hold_l_hand
            ctrl._hold_l_arm = arm  # noqa: SLF001

    # Snapshot left arm pin after handoff freeze (never rewrite under admit).
    left_arm_handoff = None
    if freeze_left and getattr(ctrl, "_hold_l_arm", None) is not None:
        left_arm_handoff = np.asarray(ctrl._hold_l_arm, dtype=np.float64).reshape(22).copy()
        meta["left_arm_handoff"] = left_arm_handoff.tolist()
        # finger lock ≠ whole-arm freeze (scheme A wrist admit needs this split)
        meta["handoff_left_locked"] = bool(freeze_left_hand)
        meta["handoff_wrist_frozen"] = bool(freeze_left)

    # Theory pose-hold: early latch + Phase-A grasp QP (before ALIGN wrenches tray).
    from scipy.spatial.transform import Rotation as R

    use_pose_qp = _theory_pose_qp(cfg) or bool(
        cfg.get("compliant", {}).get("priv_grasp_opt", {}).get("enable", False)
    )
    use_priv_grasp_a = bool(
        cfg.get("compliant", {}).get("priv_grasp_opt", {}).get("enable", False)
    )
    phase_a_qp: PrivGraspOptController | None = None
    phase_a_qp_ready = False
    phase_a_rel_peak = 0.0
    _geom_entry = read_priv_grasp_geom(raw)
    tray_R_entry = _geom_entry.tray_rot.copy()
    peg_R_entry = _geom_entry.peg_rot.copy()
    meta["tray_R_handoff"] = tray_R_entry.tolist()
    meta["handoff_left_locked"] = bool(freeze_left_hand)
    meta["handoff_wrist_frozen"] = bool(freeze_left)
    early_latch_deg = float(a_cfg.get("early_latch_max_tilt_deg", 8.0))
    early_latch_enable = bool(a_cfg.get("early_latch_enable", True))
    deliver_relatch_deg = float(a_cfg.get("deliver_re_latch_max_tilt_deg", 12.0))
    deliver_relatch_enable = bool(a_cfg.get("deliver_re_latch_enable", True))
    soft_latch_max_tilt_deg = float(a_cfg.get("soft_latch_max_tilt_deg", 15.0))
    soft_latch_max_along_m = float(a_cfg.get("surface_soft_latch_max_along_m", 0.120))
    soft_contact_relatch = bool(a_cfg.get("soft_contact_relatch", False))
    best_latch_tilt_deg = 999.0
    best_latch_snap: PrivGraspGeom | None = None

    def _tray_tilt_entry_deg() -> float:
        R_now = read_priv_grasp_geom(raw).tray_rot
        return float(
            np.linalg.norm(R.from_matrix(tray_R_entry.T @ R_now).as_rotvec()) * 180.0 / np.pi
        )

    def _peg_tilt_entry_deg() -> float:
        R_now = read_priv_grasp_geom(raw).peg_rot
        return float(
            np.linalg.norm(R.from_matrix(peg_R_entry.T @ R_now).as_rotvec()) * 180.0 / np.pi
        )

    def _rel_rot_vs_latch_rad() -> float:
        latch = meta.get("latch_priv_geom")
        if not isinstance(latch, PrivGraspGeom):
            return 0.0
        return float(_rel_rot_err_rad(latch, read_priv_grasp_geom(raw)))

    def _maybe_latch_priv_geom(*, tag: str) -> PrivGraspGeom | None:
        if isinstance(meta.get("latch_priv_geom"), PrivGraspGeom):
            return meta["latch_priv_geom"]
        snap = copy_priv_grasp_geom(read_priv_grasp_geom(raw))
        meta["latch_priv_geom"] = snap
        meta["latch_priv_geom_ok"] = True
        meta[f"latch_{tag}"] = True
        print(f"pci: latch priv geom ({tag})", flush=True)
        return snap

    def _phase_a_hand_refs() -> tuple[np.ndarray, np.ndarray]:
        hold_r = np.asarray(ctrl._hold_r_hand, dtype=np.float64).reshape(16).copy() if getattr(
            ctrl, "_hold_r_hand", None
        ) is not None else current_action44(raw)[6:22].copy()
        hold_l = (
            np.asarray(ctrl._hold_l_hand, dtype=np.float64).reshape(16).copy()
            if getattr(ctrl, "_hold_l_hand", None) is not None
            else current_action44(raw)[28:44].copy()
        )
        return hold_r, hold_l

    def _ensure_phase_a_qp(latch: PrivGraspGeom) -> None:
        nonlocal phase_a_qp, phase_a_qp_ready
        if not use_priv_grasp_a or not _phase_a_qp_enable(cfg):
            return
        if phase_a_qp is None:
            phase_a_qp = PrivGraspOptController(_priv_grasp_config(cfg))
        if phase_a_qp_ready:
            return
        hold_r, hold_l = _phase_a_hand_refs()
        phase_a_qp.reset(latch, hold_r, hold_l)
        phase_a_qp_ready = True
        print("pci: Phase-A grasp QP armed on latch", flush=True)

    def _reset_phase_a_qp(latch: PrivGraspGeom) -> None:
        nonlocal phase_a_qp, phase_a_qp_ready
        if not use_priv_grasp_a or not _phase_a_qp_enable(cfg):
            return
        if phase_a_qp is None:
            phase_a_qp = PrivGraspOptController(_priv_grasp_config(cfg))
        hold_r, hold_l = _phase_a_hand_refs()
        phase_a_qp.reset(latch, hold_r, hold_l)
        phase_a_qp_ready = True
        print("pci: Phase-A grasp QP re-armed on re-latch", flush=True)

    # r24: Phase-A QP controller exists even if early_latch is off (arm on first latch).
    if use_pose_qp and use_priv_grasp_a and _phase_a_qp_enable(cfg):
        phase_a_qp = PrivGraspOptController(_priv_grasp_config(cfg))
    if use_pose_qp and _phase_a_qp_enable(cfg) and early_latch_enable:
        out_entry = env._labeler.compute(raw)
        tilt_e = _tray_tilt_entry_deg()
        if bool(out_entry.peg_ok) and bool(out_entry.tray_ok) and tilt_e < early_latch_deg:
            early = copy_priv_grasp_geom(read_priv_grasp_geom(raw))
            meta["early_latch_priv_geom"] = early
            meta["latch_priv_geom"] = early
            meta["latch_priv_geom_ok"] = True
            meta["early_latch_tilt_deg"] = float(tilt_e)
            print(
                f"pci: early latch priv geom tilt={tilt_e:.1f}deg<{early_latch_deg:.1f}",
                flush=True,
            )
            _ensure_phase_a_qp(early)

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
    grasp_init_scale = float(a_cfg.get("surface_left_grasp_scale_init", 1.15))
    grasp_ramp_enable = bool(a_cfg.get("surface_grasp_ramp_enable", False))
    grasp_ramp_tilt_deg = float(a_cfg.get("surface_grasp_ramp_tilt_deg", 6.0))
    grasp_ramp_rel_rad = float(a_cfg.get("surface_grasp_ramp_rel_rot_rad", 0.10))
    grasp_ramp_step = float(a_cfg.get("surface_grasp_ramp_step", 0.05))
    grasp_ramp_max = float(a_cfg.get("surface_grasp_ramp_max_scale", 1.65))
    grasp_ramp_interval = int(a_cfg.get("surface_grasp_ramp_interval_steps", 4))
    grasp_ramp_admit_boost = float(a_cfg.get("surface_grasp_ramp_admit_boost", 0.04))
    priv_diag_interval = int(a_cfg.get("surface_priv_diag_interval_steps", 40))
    # surface_unload_retract_m applied in _deliver_surface

    steps = 0
    grasp_ramp_scale = float(grasp_init_scale)
    last_grasp_ramp_step = -9999
    soft_admit_boost = 0.0
    phase_a_tilt_peak = 0.0
    phase_a_peg_tilt_peak = 0.0
    phase_a_rel_contact_peak = 0.0
    tray_lost_streak = 0
    fz_hist: list[float] = []
    fz_baseline: float | None = None
    contact_streak = 0
    soft_latched = False
    wrist_pre_hist: list[np.ndarray] = []  # right wrist before soft latch (path reverse)
    soft_admit: LeftWristAdmitController | None = None
    soft_admit_scale = float(a_cfg.get("surface_soft_left_admit_scale", 0.0))
    hold_admit_scale = float(
        a_cfg.get("surface_hold_left_admit_scale", soft_admit_scale)
    )
    align_admit_scale = float(a_cfg.get("surface_align_left_admit_scale", 0.0))
    align_admit: LeftWristAdmitController | None = None
    if align_admit_scale > 1e-9 and use_pose_qp:
        align_admit = LeftWristAdmitController(_left_wrist_admit_config(cfg))
        align_admit.reset(read_wrist_wrench_world(raw)[1])
        align_admit.config.enable = True
        print(f"pci: ALIGN left_admit armed scale={align_admit_scale:.2f}", flush=True)
    reason = "align_timeout"

    def _tighten_left_grasp(
        scale: float, *, bypass_freeze: bool = False
    ) -> dict[str, float | bool]:
        """Raise left hand hold toward scale·q0.

        Handoff normally freezes fingers (`surface_freeze_left_hand`). Tip-STAR /
        Type-B slip restore may pass bypass_freeze=True so tighten is real.
        Returns meta for audit: whether cmd/scale actually changed.
        """
        nonlocal grasp_ramp_scale
        out: dict[str, float | bool] = {
            "ok": False,
            "bypassed_freeze": bool(bypass_freeze and freeze_left_hand),
            "scale": float(scale),
            "mean_abs_before": 0.0,
            "mean_abs_after": 0.0,
        }
        if left_hand0 is None:
            return out
        before = (
            float(np.mean(np.abs(ctrl._hold_l_hand)))
            if getattr(ctrl, "_hold_l_hand", None) is not None
            else 0.0
        )
        out["mean_abs_before"] = before
        if freeze_left_hand and not bypass_freeze:
            # Handoff lock: ignore squeeze/ramp requests.
            grasp_ramp_scale = 1.0
            out["mean_abs_after"] = before
            return out
        h = np.maximum(left_hand0, left_hand0 * float(scale))
        ctrl._hold_l_hand = np.clip(h, -1.5, 1.5)  # noqa: SLF001
        grasp_ramp_scale = float(scale)
        if getattr(ctrl, "_hold_l_arm", None) is not None:
            arm = np.asarray(ctrl._hold_l_arm, dtype=np.float64).reshape(22).copy()
            arm[6:22] = ctrl._hold_l_hand
            ctrl._hold_l_arm = arm  # noqa: SLF001
        after = float(np.mean(np.abs(ctrl._hold_l_hand)))
        out["mean_abs_after"] = after
        # Real change only — scale>1 alone is not enough (re-trip may be no-op).
        out["ok"] = bool(after > before + 1e-6)
        print(
            f"pci: left grasp tighten scale={scale:.2f} "
            f"mean|q|={before:.3f}->{after:.3f}"
            f"{' (star bypass freeze)' if bypass_freeze and freeze_left_hand else ''}",
            flush=True,
        )
        return out

    def _maybe_ramp_left_grasp_for_tilt(*, tag: str) -> None:
        """Tilt/rel_rot-gated left grasp ramp (privileged_diagnostic)."""
        nonlocal last_grasp_ramp_step, soft_admit_boost, phase_a_tilt_peak, phase_a_peg_tilt_peak, phase_a_rel_contact_peak
        if not grasp_ramp_enable or left_hand0 is None:
            return
        tilt = _tray_tilt_entry_deg()
        peg_t = _peg_tilt_entry_deg()
        rel = _rel_rot_vs_latch_rad()
        need = tilt > grasp_ramp_tilt_deg or peg_t > grasp_ramp_tilt_deg or rel > grasp_ramp_rel_rad
        if not need:
            return
        if steps - last_grasp_ramp_step < grasp_ramp_interval:
            return
        new_scale = min(grasp_ramp_scale + grasp_ramp_step, grasp_ramp_max)
        if new_scale <= grasp_ramp_scale + 1e-6:
            return
        last_grasp_ramp_step = steps
        _tighten_left_grasp(new_scale)
        soft_admit_boost = min(
            soft_admit_boost + grasp_ramp_admit_boost,
            float(a_cfg.get("surface_grasp_ramp_admit_boost_cap", 0.12)),
        )
        meta["grasp_ramp_events"] = int(meta.get("grasp_ramp_events", 0)) + 1
        meta["grasp_ramp_peak_scale"] = float(new_scale)
        print(
            f"pci: tilt-ramp [{tag}] grasp→{new_scale:.2f} "
            f"tray={tilt:.1f}deg peg={peg_t:.1f}deg rel={rel:.3f}rad "
            f"admit_boost={soft_admit_boost:.2f}",
            flush=True,
        )

    def _log_priv_pose_diag(*, tag: str) -> None:
        if not use_pose_qp or priv_diag_interval <= 0 or steps % priv_diag_interval != 0:
            return
        tilt = _tray_tilt_entry_deg()
        peg_t = _peg_tilt_entry_deg()
        rel = _rel_rot_vs_latch_rad()
        print(
            f"pci: priv-diag [{tag}] step={steps} "
            f"tray_tilt={tilt:.1f}deg peg_tilt={peg_t:.1f}deg rel_rot={rel:.3f}rad "
            f"grasp_scale={grasp_ramp_scale:.2f} admit={soft_admit_scale + soft_admit_boost:.2f}",
            flush=True,
        )

    def _hold_pose_only(*, frames: int, resync: bool = False) -> int:
        """钉住 EE：默认冻结 CONTACT 时的 site 位姿，避免 re-sync 追着 tip 侧滑。"""
        hold = actual_action44_from_sites(raw)
        if getattr(ctrl, "_hold_l_hand", None) is not None:
            hold = hold.copy()
            hold[28:44] = np.asarray(ctrl._hold_l_hand, dtype=np.float64).reshape(16)
        for i in range(max(0, frames)):
            if resync and i % 5 == 0:
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

    def _deliver_geom_ok(feat) -> bool:
        along_abs = abs(float(feat.along_m))
        lat_abs = abs(float(feat.lateral_m))
        tip_now = float(feat.tip_socket_dist_m)
        tilt_now = _tray_tilt_entry_deg()
        max_along = float(a_cfg.get("surface_soft_latch_max_along_m", 0.120))
        max_tip = float(a_cfg.get("surface_deliver_max_tip_m", 0.130))
        max_lat = float(a_cfg.get("surface_deliver_max_lat_m", 0.028))
        ok = (
            along_abs <= max_along
            and tip_now <= max_tip
            and lat_abs <= max_lat
            and tilt_now <= soft_latch_max_tilt_deg
        )
        if not ok:
            print(
                f"pci: BLOCK deliver along={along_abs*1e3:.0f} lat={lat_abs*1e3:.0f} "
                f"tip={tip_now*1e3:.0f} tilt={tilt_now:.1f} — keep aligning",
                flush=True,
            )
        return ok

    def _deliver_surface(feat, outcome, *, fz: float, d_fz: float) -> tuple[int, str, dict[str, Any]]:
        nonlocal steps, phase_a_rel_peak, phase_a_tilt_peak, phase_a_peg_tilt_peak
        if grasp_ramp_scale < grasp_scale - 1e-6:
            _tighten_left_grasp(grasp_scale)
        r_scale = float(a_cfg.get("surface_right_grasp_scale", 1.0))
        if r_scale > 1.0:
            # Tighten right fingers so tip tracks wrist during spiral.
            act = actual_action44_from_sites(raw)
            hand = np.clip(np.maximum(act[6:22], act[6:22] * r_scale), -1.5, 1.5)
            act = act.copy()
            act[6:22] = hand
            step_action44(gym_env, act, ego_recorder=ego_recorder)
            print(f"pci: right grasp tighten scale={r_scale:.2f}", flush=True)
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
        # Micro-unload along hole at frozen XY to drop contact force before hold.
        hole_u = feat.hole_axis / (np.linalg.norm(feat.hole_axis) + 1e-12)
        unload_m = float(a_cfg.get("surface_contact_unload_m", 0.001))
        hold = sync.copy()
        if unload_m > 0.0:
            hold[0:3] = hold[0:3] + hole_u * unload_m
        if getattr(ctrl, "_hold_l_hand", None) is not None:
            hold[28:44] = np.asarray(ctrl._hold_l_hand, dtype=np.float64).reshape(16)
        hold_n = int(a_cfg.get("surface_hold_frames", 8))
        hold_l_wrist = hold[22:28].copy()
        hold_r_hand = hold[6:22].copy()
        hold_l_hand = hold[28:44].copy()
        latch_snap = meta.get("latch_priv_geom")
        if (
            best_latch_snap is not None
            and best_latch_tilt_deg < _tray_tilt_entry_deg() - 0.5
        ):
            latch_snap = best_latch_snap
            meta["min_tilt_latch_deg"] = float(best_latch_tilt_deg)
            meta["latch_priv_geom"] = latch_snap
            meta["latch_priv_geom_ok"] = True
            print(
                f"pci: deliver min-tilt latch tilt={best_latch_tilt_deg:.1f}deg",
                flush=True,
            )
        if not isinstance(latch_snap, PrivGraspGeom) and use_pose_qp:
            latch_snap = _maybe_latch_priv_geom(tag="contact_deliver")
        if isinstance(latch_snap, PrivGraspGeom) and _phase_a_qp_enable(cfg):
            _ensure_phase_a_qp(latch_snap)
        deliver_admit_scale = hold_admit_scale
        deliver_admit: LeftWristAdmitController | None = None
        if deliver_admit_scale > 1e-9 and use_pose_qp:
            deliver_admit = LeftWristAdmitController(_left_wrist_admit_config(cfg))
            deliver_admit.reset(read_wrist_wrench_world(raw)[1])
            deliver_admit.config.enable = True
        geom_contact = read_priv_grasp_geom(raw)
        if isinstance(latch_snap, PrivGraspGeom):
            meta["rel_rot_at_contact_rad"] = float(
                _rel_rot_err_rad(latch_snap, geom_contact)
            )
        for _hi in range(max(0, hold_n)):
            cmd = hold.copy()
            cmd[22:28] = hold_l_wrist
            cmd[6:22] = hold_r_hand
            cmd[28:44] = hold_l_hand
            _maybe_ramp_left_grasp_for_tilt(tag="hold")
            if getattr(ctrl, "_hold_l_hand", None) is not None:
                cmd[28:44] = np.asarray(ctrl._hold_l_hand, dtype=np.float64).reshape(16)
            if (
                _phase_a_qp_enable(cfg)
                and phase_a_qp is not None
                and phase_a_qp_ready
                and force_labeler is not None
            ):
                geom_h = read_priv_grasp_geom(raw)
                f12, lf12 = read_dual_finger_force24(raw, force_labeler)
                po = phase_a_qp.step(geom_h, f12, lf12)
                phase_a_rel_peak = max(phase_a_rel_peak, float(po.rel_rot_err_rad))
                cmd[6:22] = hold_r_hand + po.delta_right_hand16
                if phase_a_left_qp_fingers and not freeze_left_hand:
                    cmd[28:44] = np.maximum(
                        hold_l_hand + po.delta_left_hand16, hold_l_hand * 0.98
                    )
                else:
                    cmd[28:44] = hold_l_hand
            if deliver_admit is not None:
                wr = read_wrist_wrench_world(raw)
                d_left = (
                    deliver_admit.step(wr[1], approach_axis=-hole_u) * deliver_admit_scale
                )
                hold_l_wrist[0:3] = hold_l_wrist[0:3] + d_left
                cmd[22:28] = hold_l_wrist
            step_action44(gym_env, cmd, ego_recorder=ego_recorder)
            phase_a_tilt_peak = max(phase_a_tilt_peak, _tray_tilt_entry_deg())
            phase_a_peg_tilt_peak = max(phase_a_peg_tilt_peak, _peg_tilt_entry_deg())
        steps += hold_n
        meta["phase_a_rel_rot_peak_rad"] = float(phase_a_rel_peak)
        meta["phase_a_tray_tilt_peak_deg"] = float(phase_a_tilt_peak)
        meta["phase_a_peg_tilt_peak_deg"] = float(phase_a_peg_tilt_peak)
        meta["grasp_ramp_peak_scale"] = float(
            meta.get("grasp_ramp_peak_scale", grasp_ramp_scale)
        )
        meta["soft_admit_peak_scale"] = float(soft_admit_scale + soft_admit_boost)
        feat_f = features_from_raw(raw)
        print(
            f"pci: CONTACT hold done lat={feat_f.lateral_m*1000:.1f}mm "
            f"along={feat_f.along_m*1000:.1f}mm (was {feat.lateral_m*1000:.1f})",
            flush=True,
        )
        outcome_f = env._labeler.compute(raw)
        hole_u = feat_f.hole_axis / (np.linalg.norm(feat_f.hole_axis) + 1e-12)
        meta["align_lat_mm"] = feat_f.lateral_m * 1000
        meta["align_along_mm"] = feat_f.along_m * 1000
        meta.update(_final_geom_meta(feat_f, outcome_f, ctrl))
        meta.update(
            {
                "force_gated": True,
                "force_gate": "ema_residual",
                "soft_latched": soft_latched,
                "motion_stopped_on_contact": True,
                "synced_to_site": True,
                "cmd_site_drift_mm": drift,
                "unload_retract_m": 0.0,
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
        # Keep early/soft latch; refresh at deliver if none yet.
        if not isinstance(meta.get("latch_priv_geom"), PrivGraspGeom):
            meta["latch_priv_geom"] = copy_priv_grasp_geom(read_priv_grasp_geom(raw))
        meta["latch_priv_geom_ok"] = True
        # Prefer flattest latch; do not re-latch tipped contact (locks tip pose).
        if (
            deliver_relatch_enable
            and use_pose_qp
            and bool(outcome_f.peg_ok)
            and bool(outcome_f.tray_ok)
        ):
            tilt_deliver = _tray_tilt_entry_deg()
            best_t = float(best_latch_tilt_deg) if best_latch_snap is not None else 999.0
            if tilt_deliver + 0.5 < best_t and tilt_deliver < deliver_relatch_deg:
                snap_d = copy_priv_grasp_geom(read_priv_grasp_geom(raw))
                meta["latch_priv_geom"] = snap_d
                meta["latch_priv_geom_ok"] = True
                meta["deliver_re_latch"] = True
                meta["deliver_re_latch_tilt_deg"] = float(tilt_deliver)
                if _phase_a_qp_enable(cfg) and phase_a_qp is not None:
                    _reset_phase_a_qp(snap_d)
                print(
                    f"pci: deliver re-latch tilt={tilt_deliver:.1f}deg"
                    f"<{deliver_relatch_deg:.1f} "
                    f"rel={_rel_rot_err_rad(snap_d, read_priv_grasp_geom(raw)):.3f}rad",
                    flush=True,
                )
            elif best_latch_snap is not None:
                meta["latch_priv_geom"] = best_latch_snap
                meta["latch_priv_geom_ok"] = True
                meta["min_tilt_latch_deg"] = float(best_latch_tilt_deg)
                if _phase_a_qp_enable(cfg) and phase_a_qp is not None:
                    _reset_phase_a_qp(best_latch_snap)
                print(
                    f"pci: keep min-tilt latch {best_latch_tilt_deg:.1f}deg "
                    f"(skip tipped re-latch {tilt_deliver:.1f}deg)",
                    flush=True,
                )
        return steps, why, meta

    for _ in range(max_align):
        action44 = current_action44(raw)
        hybrid.observe(gym_env, action44)
        merged = hybrid.merge(gym_env, action44)
        if getattr(ctrl, "_hold_l_hand", None) is not None:
            merged = np.asarray(merged, dtype=np.float64).copy()
            merged[28:44] = np.asarray(ctrl._hold_l_hand, dtype=np.float64).reshape(16)
        if (
            _phase_a_qp_enable(cfg)
            and phase_a_qp is not None
            and phase_a_qp_ready
            and force_labeler is not None
        ):
            merged = np.asarray(merged, dtype=np.float64).copy()
            hold_r = merged[6:22].copy()
            hold_l = merged[28:44].copy()
            geom_a = read_priv_grasp_geom(raw)
            f12, lf12 = read_dual_finger_force24(raw, force_labeler)
            po = phase_a_qp.step(geom_a, f12, lf12)
            phase_a_rel_peak = max(phase_a_rel_peak, float(po.rel_rot_err_rad))
            merged[6:22] = hold_r + po.delta_right_hand16
            if phase_a_left_qp_fingers and not freeze_left_hand:
                merged[28:44] = np.maximum(hold_l + po.delta_left_hand16, hold_l * 0.98)
            else:
                merged[28:44] = hold_l
        if soft_latched and soft_admit is not None and soft_admit_scale > 1e-9:
            merged = np.asarray(merged, dtype=np.float64).copy()
            feat_sc = features_from_raw(raw)
            hole_u_sc = feat_sc.hole_axis / (np.linalg.norm(feat_sc.hole_axis) + 1e-12)
            admit_eff = soft_admit_scale + soft_admit_boost
            d_left = (
                soft_admit.step(read_wrist_wrench_world(raw)[1], approach_axis=-hole_u_sc)
                * admit_eff
            )
            merged[22:25] = merged[22:25] + d_left
        elif (
            (not soft_latched)
            and align_admit is not None
            and align_admit_scale > 1e-9
            and not freeze_left  # whole-arm freeze blocks wrist yield
        ):
            # ALIGN wrist admit (scheme A): finger may stay frozen.
            merged = np.asarray(merged, dtype=np.float64).copy()
            feat_al = features_from_raw(raw)
            hole_u_al = feat_al.hole_axis / (np.linalg.norm(feat_al.hole_axis) + 1e-12)
            d_left = (
                align_admit.step(read_wrist_wrench_world(raw)[1], approach_axis=-hole_u_al)
                * align_admit_scale
            )
            # Tilt gate: stop yielding if tray already tipped vs handoff.
            tilt_ad = _tray_tilt_entry_deg()
            if tilt_ad > float(a_cfg.get("surface_admit_tilt_gate_deg", 3.0)):
                d_left = d_left * 0.0
            merged[22:25] = merged[22:25] + d_left
            if getattr(ctrl, "_hold_l_arm", None) is not None:
                arm = np.asarray(ctrl._hold_l_arm, dtype=np.float64).reshape(22).copy()
                arm[0:3] = arm[0:3] + d_left
                ctrl._hold_l_arm = arm  # noqa: SLF001
        if freeze_left and left_arm_handoff is not None:
            # Whole-arm freeze: re-pin wrist+hand every step.
            ctrl._hold_l_arm = left_arm_handoff.copy()  # noqa: SLF001
            ctrl._hold_l_hand = left_arm_handoff[6:22].copy()  # noqa: SLF001
            merged = np.asarray(merged, dtype=np.float64).copy()
            merged[22:44] = left_arm_handoff
        elif freeze_left_hand and left_arm_handoff is not None:
            # Finger-only freeze: keep handoff fingers, allow wrist admit drift.
            ctrl._hold_l_hand = left_arm_handoff[6:22].copy()  # noqa: SLF001
            merged = np.asarray(merged, dtype=np.float64).copy()
            merged[28:44] = left_arm_handoff[6:22]
            if getattr(ctrl, "_hold_l_arm", None) is not None:
                arm = np.asarray(ctrl._hold_l_arm, dtype=np.float64).reshape(22).copy()
                arm[6:22] = left_arm_handoff[6:22]
                ctrl._hold_l_arm = arm  # noqa: SLF001
        if use_pose_qp:
            out_sc2 = env._labeler.compute(raw)
            if bool(out_sc2.peg_ok) and bool(out_sc2.tray_ok):
                phase_a_tilt_peak = max(phase_a_tilt_peak, _tray_tilt_entry_deg())
                phase_a_peg_tilt_peak = max(phase_a_peg_tilt_peak, _peg_tilt_entry_deg())
                phase_a_rel_contact_peak = max(phase_a_rel_contact_peak, _rel_rot_vs_latch_rad())
                _maybe_ramp_left_grasp_for_tilt(tag="align")
        if soft_latched and use_pose_qp:
            out_sc = env._labeler.compute(raw)
            if bool(out_sc.peg_ok) and bool(out_sc.tray_ok):
                tilt_sc = _tray_tilt_entry_deg()
                peg_sc = _peg_tilt_entry_deg()
                phase_a_tilt_peak = max(phase_a_tilt_peak, tilt_sc)
                phase_a_peg_tilt_peak = max(phase_a_peg_tilt_peak, peg_sc)
                phase_a_rel_contact_peak = max(phase_a_rel_contact_peak, _rel_rot_vs_latch_rad())
                if tilt_sc < best_latch_tilt_deg:
                    best_latch_tilt_deg = float(tilt_sc)
                    best_latch_snap = copy_priv_grasp_geom(read_priv_grasp_geom(raw))
                _maybe_ramp_left_grasp_for_tilt(tag="soft-contact")
        _log_priv_pose_diag(tag="align")
        step_action44(gym_env, merged, ego_recorder=ego_recorder)
        steps += 1
        if not soft_latched:
            site_pre = actual_action44_from_sites(raw)
            wrist_pre_hist.append(
                np.asarray(site_pre[0:3], dtype=np.float64).reshape(3).copy()
            )
            if len(wrist_pre_hist) > 240:
                del wrist_pre_hist[0 : len(wrist_pre_hist) - 240]

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
            # Re-arm residual: slow EMA tracked approach so |d_fz|≈0 at arm time.
            fz_baseline = float(fz)
            d_fz = 0.0
            print(f"pci: Fz residual re-armed base={fz_baseline:+.2f}N", flush=True)

        if steps >= force_arm_after:
            contact_mag = abs(d_fz)
            # Near surface: damp XY if already centered; keep Z until contact.
            if feat.along_m < 0.110 and not soft_latched:
                if feat.lateral_m <= float(a_cfg.get("surface_min_lat_m", 0.008)) * 2.5:
                    ctrl.config.pbvs_lambda_xy = min(ctrl.config.pbvs_lambda_xy, 0.05)

            standoff_tgt = float(a_cfg.get("pbvs_standoff_m", 0.055))
            standoff_tol = float(a_cfg.get("standoff_tol_m", 0.020))
            tip_d = float(feat.tip_socket_dist_m)

            # Hover assist: lat OK but still above mouth — drop standoff, unlock Z (capped).
            if (
                (not soft_latched)
                and steps >= force_arm_after + 80
                and feat.lateral_m < 0.028
                and tip_d > standoff_tgt + 0.010
            ):
                hover_z_cap = float(a_cfg.get("hover_assist_lambda_z_cap", 0.06))
                ctrl.config.pbvs_standoff_m = min(
                    float(ctrl.config.pbvs_standoff_m), 0.040
                )
                ctrl.config.pbvs_lambda_z = min(
                    max(float(ctrl.config.pbvs_lambda_z), hover_z_cap * 0.5),
                    hover_z_cap,
                )
                # Hybrid blocks Z while do_lateral; widen pos_tol so small lat doesn't freeze Z.
                ctrl.config.pos_tol_m = max(
                    float(getattr(ctrl.config, "pos_tol_m", 0.006)),
                    float(feat.lateral_m) * 2.5 + 0.004,
                )
                if steps % 60 == 0:
                    print(
                        f"pci: hover-assist standoff→{ctrl.config.pbvs_standoff_m*1e3:.0f}mm "
                        f"λz→{ctrl.config.pbvs_lambda_z:.3f} "
                        f"tip={tip_d*1e3:.1f} lat={feat.lateral_m*1e3:.1f}",
                        flush=True,
                    )

            # Geom fallback: at mouth, lat OK, FT flat (never crosses soft gate).
            if (
                (not soft_latched)
                and steps >= force_arm_after + 100
                and feat.lateral_m
                < float(a_cfg.get("surface_min_lat_m", 0.008)) * 2.5
                and abs(tip_d - standoff_tgt) <= standoff_tol
            ):
                print(
                    f"pci: geom-standoff deliver tip={tip_d*1e3:.1f}mm "
                    f"lat={feat.lateral_m*1e3:.1f}mm |residual|={contact_mag:.2f}N",
                    flush=True,
                )
                if _deliver_geom_ok(feat):
                    if bool(a_cfg.get("stop_after_surface", False)):
                        # Soft-land demo must not walk into mouth via deliver.
                        contact_streak = 0
                    else:
                        return _deliver_surface(feat, outcome, fz=fz, d_fz=d_fz)

            if (not soft_latched) and contact_mag >= soft_thresh:
                tilt_soft_pre = _tray_tilt_entry_deg()
                near_mouth = float(feat.along_m) <= soft_latch_max_along_m
                if not near_mouth or tilt_soft_pre > soft_latch_max_tilt_deg:
                    # r25: 已歪则减速/卸压，禁止继续硬 λz 怼盘。
                    ctrl.config.pbvs_lambda_z = min(
                        float(ctrl.config.pbvs_lambda_z),
                        float(a_cfg.get("soft_contact_lambda_z", 0.015)),
                    )
                    ctrl.config.pbvs_lambda_xy = min(ctrl.config.pbvs_lambda_xy, 0.02)
                    if steps % 40 == 0:
                        print(
                            f"pci: soft-contact skip |r|={contact_mag:.2f}N "
                            f"along={feat.along_m*1e3:.0f}mm tilt={tilt_soft_pre:.1f}deg "
                            f"— damp Z (need along≤{soft_latch_max_along_m*1e3:.0f} "
                            f"tilt≤{soft_latch_max_tilt_deg:.1f})",
                            flush=True,
                        )
                else:
                    soft_latched = True
                    if grasp_ramp_scale < grasp_scale - 1e-6:
                        _tighten_left_grasp(grasp_scale)
                    # Freeze left share; optionally freeze whole arm (scheme B) or only fingers (A).
                    ctrl.config.left_share_xy = 0.0
                    ctrl.config.left_share_rot = 0.0
                    if freeze_left:
                        ctrl.config.freeze_left_arm_at_handoff = True
                    # Freeze right XY too: keep ALIGN lat (~2mm) while soft-Z lands.
                    # Scheme B tip-abort may later zero λ entirely.
                    ctrl.config.pbvs_lambda_xy = 0.0
                    soft_z = float(a_cfg.get("soft_contact_lambda_z", 0.015))
                    ctrl.config.pbvs_lambda_z = min(ctrl.config.pbvs_lambda_z, soft_z)
                    soft_rot = float(a_cfg.get("soft_contact_lambda_rot", 0.0))
                    ctrl.config.pbvs_lambda_rot = soft_rot
                    if getattr(ctrl, "_hold_l_arm", None) is None:
                        site_l = actual_action44_from_sites(raw)[22:44].copy()
                        if getattr(ctrl, "_hold_l_hand", None) is not None:
                            site_l[6:22] = np.asarray(ctrl._hold_l_hand, dtype=np.float64)
                        ctrl._hold_l_arm = site_l  # noqa: SLF001
                    if soft_admit_scale > 1e-9 and use_pose_qp:
                        soft_admit = LeftWristAdmitController(_left_wrist_admit_config(cfg))
                        soft_admit.reset(read_wrist_wrench_world(raw)[1])
                        soft_admit.config.enable = True
                    print(
                        f"pci: soft-contact latch |residual|={contact_mag:.2f}N "
                        f"(r={d_fz:+.2f}) — freeze left+rightXY, soft Z "
                        f"lat={feat.lateral_m*1000:.1f}mm "
                        f"left_admit={soft_admit_scale:.2f}",
                        flush=True,
                    )
                    # Demo: soft-land → wrist-FT bleed to light F → hold → spiral.
                    if bool(a_cfg.get("stop_after_surface", False)):
                        if ctrl.active:
                            ctrl._deactivate()  # noqa: SLF001
                        hold = actual_action44_from_sites(raw)
                        if getattr(ctrl, "_hold_l_hand", None) is not None:
                            hold = hold.copy()
                            hold[28:44] = np.asarray(
                                ctrl._hold_l_hand, dtype=np.float64
                            ).reshape(16)
                        if getattr(ctrl, "_hold_l_arm", None) is not None:
                            hold = hold.copy()
                            arm_h = np.asarray(
                                ctrl._hold_l_arm, dtype=np.float64
                            ).reshape(22)
                            hold[22:28] = arm_h[0:6]
                            hold[28:44] = arm_h[6:22]
                        feat_c0 = features_from_raw(raw)
                        hole_u = feat_c0.hole_axis / (
                            np.linalg.norm(feat_c0.hole_axis) + 1e-12
                        )
                        press_ax = -hole_u  # insert / into-surface
                        # Reconstruct pre-contact residual baseline from this-frame d_fz
                        # (EMA fz_baseline already mixed in contact after the residual calc).
                        fz_base_c = float(fz) - float(d_fz)
                        wr0 = read_wrist_wrench_world(raw)
                        left_fz0 = float(
                            wrench_in_hole_frame(wr0[1], feat_c0.hole_axis)[2]
                        )
                        # Left free estimate: peel off shared contact at latch.
                        left_fz_base = float(left_fz0) - float(d_fz) * float(
                            a_cfg.get("surface_const_force_left_share", 0.25)
                        )
                        f_des = float(a_cfg.get("surface_const_force_des_n", 0.15))
                        bleed_tol = float(
                            a_cfg.get("surface_force_bleed_tol_n", 0.04)
                        )
                        n_bleed = int(
                            a_cfg.get("surface_force_bleed_max_frames", 90)
                        )
                        bleed_step = float(
                            a_cfg.get("surface_force_bleed_step_m", 0.0004)
                        )
                        bleed_confirm = int(
                            a_cfg.get("surface_force_bleed_confirm", 6)
                        )
                        n_cf = int(a_cfg.get("surface_const_force_frames", 30))
                        kp = float(a_cfg.get("surface_const_force_kp_m_per_n", 0.00025))
                        max_step = float(
                            a_cfg.get("surface_const_force_max_step_m", 0.00012)
                        )
                        unload_boost = float(
                            a_cfg.get("surface_force_unload_boost", 2.5)
                        )
                        left_share_cf = float(
                            a_cfg.get("surface_const_force_left_share", 0.25)
                        )
                        max_sink = float(
                            a_cfg.get("surface_const_force_max_sink_m", 0.004)
                        )
                        min_lat_cf = float(
                            a_cfg.get(
                                "surface_const_force_min_lat_m",
                                a_cfg.get("surface_min_lat_m", 0.008),
                            )
                        )
                        max_tilt_cf = float(
                            a_cfg.get("surface_const_force_max_tilt_deg", 8.0)
                        )
                        along0_cf = float(feat_c0.along_m)
                        hold_r = hold[0:3].copy()
                        hold_l = hold[22:28].copy()
                        hold_r_hand = hold[6:22].copy()
                        hold_l_hand = hold[28:44].copy()
                        # Optional firm-up right grasp at soft latch (scale>=1).
                        r_grasp0 = float(a_cfg.get("surface_right_grasp_scale", 1.0))
                        if r_grasp0 > 1.0 + 1e-9:
                            hold_r_hand = np.clip(
                                np.maximum(hold_r_hand, hold_r_hand * r_grasp0),
                                -1.5,
                                1.5,
                            )
                            for _tg in range(4):
                                cmd = hold.copy()
                                cmd[0:3] = hold_r
                                cmd[6:22] = hold_r_hand
                                cmd[22:28] = hold_l
                                cmd[28:44] = hold_l_hand
                                step_action44(
                                    gym_env, cmd, ego_recorder=ego_recorder
                                )
                            steps += 4
                            print(
                                f"pci: soft-latch right grasp tighten "
                                f"scale={r_grasp0:.2f}",
                                flush=True,
                            )
                        resid_peak = abs(float(d_fz))
                        left_resid_peak = abs(float(left_fz0) - left_fz_base)
                        contact0 = abs(float(fz) - fz_base_c)
                        force_trace: list[dict[str, float | str]] = []
                        t_force = 0
                        dt_f = float(_sim_dt(cfg))
                        # Latch peg-in-right-wrist; slip > gate → grasp lost (abort).
                        geom_g0 = read_priv_grasp_geom(raw)
                        peg_in_r0 = _peg_in_right_wrist(geom_g0)
                        grasp_slip_peak = 0.0
                        grasp_slip_lim = float(
                            a_cfg.get("priv_gate_grasp_slip_m", 0.008)
                        )
                        meta["grasp_latch_ok"] = True
                        meta["grasp_slip_peak_m"] = 0.0

                        def _update_grasp_slip() -> float:
                            nonlocal grasp_slip_peak
                            g_now = read_priv_grasp_geom(raw)
                            slip = float(
                                np.linalg.norm(
                                    _peg_in_right_wrist(g_now) - peg_in_r0
                                )
                            )
                            grasp_slip_peak = max(grasp_slip_peak, slip)
                            meta["grasp_slip_peak_m"] = float(grasp_slip_peak)
                            return slip

                        def _grasp_lost() -> bool:
                            slip = _update_grasp_slip()
                            return bool(slip > grasp_slip_lim)

                        def _wrist_contact() -> tuple[float, float, float, float]:
                            wr_i = read_wrist_wrench_world(raw)
                            fz_r = float(
                                wrench_in_hole_frame(wr_i[0], hole_u)[2]
                            )
                            fz_l = float(
                                wrench_in_hole_frame(wr_i[1], hole_u)[2]
                            )
                            return (
                                abs(fz_r - fz_base_c),
                                abs(fz_l - left_fz_base),
                                fz_r,
                                fz_l,
                            )

                        def _log_force(phase: str) -> tuple[float, float, float, float]:
                            nonlocal t_force
                            contact, left_c, fz_r, fz_l = _wrist_contact()
                            feat_l = features_from_raw(raw)
                            force_trace.append(
                                {
                                    "t": float(t_force) * dt_f,
                                    "step": float(t_force),
                                    "phase": str(phase),
                                    "resid_r": float(contact),
                                    "resid_l": float(left_c),
                                    "fz_r": float(fz_r),
                                    "fz_l": float(fz_l),
                                    "lat_mm": float(feat_l.lateral_m) * 1000.0,
                                    "along_mm": float(feat_l.along_m) * 1000.0,
                                    "f_des": float(f_des),
                                }
                            )
                            t_force += 1
                            return contact, left_c, fz_r, fz_l

                        def _apply_axial(
                            step_m: float, left_share: float | None = None
                        ) -> None:
                            nonlocal hold_r, hold_l
                            share = (
                                float(left_share_cf)
                                if left_share is None
                                else float(left_share)
                            )
                            hold_r = hold_r + press_ax * step_m
                            hold_l[0:3] = hold_l[0:3] + press_ax * (
                                step_m * share
                            )
                            cmd = hold.copy()
                            cmd[0:3] = hold_r
                            cmd[6:22] = hold_r_hand
                            cmd[22:28] = hold_l
                            cmd[28:44] = hold_l_hand
                            step_action44(gym_env, cmd, ego_recorder=ego_recorder)

                        print(
                            f"pci: wrist bleed-down latch |r|={contact0:.2f}N "
                            f"→ f_des={f_des:.2f}N (continuous FT) "
                            f"left0={abs(left_fz0-left_fz_base):.2f}N "
                            f"lat0={feat_c0.lateral_m*1e3:.1f}mm",
                            flush=True,
                        )
                        _log_force("latch")

                        # --- Phase 1: unload spike until wrist residual ≈ f_des ---
                        bleed_ok = False
                        bleed_used = 0
                        ok_streak = 0
                        for bleed_used in range(1, max(0, n_bleed) + 1):
                            contact, left_c, fz_r, fz_l = _log_force("bleed")
                            resid_peak = max(resid_peak, contact)
                            left_resid_peak = max(left_resid_peak, left_c)
                            feat_i = features_from_raw(raw)
                            tilt_i = _tray_tilt_entry_deg()
                            phase_a_tilt_peak = max(phase_a_tilt_peak, tilt_i)
                            phase_a_peg_tilt_peak = max(
                                phase_a_peg_tilt_peak, _peg_tilt_entry_deg()
                            )
                            out_i = env._labeler.compute(raw)
                            if (not out_i.tray_ok) or tilt_i > max_tilt_cf:
                                print(
                                    f"pci: bleed abort tilt={tilt_i:.1f}deg "
                                    f"tray_ok={int(bool(out_i.tray_ok))} j={bleed_used}",
                                    flush=True,
                                )
                                break
                            if contact <= f_des + bleed_tol:
                                ok_streak += 1
                                step_b = float(
                                    np.clip(kp * (f_des - contact), -max_step, max_step)
                                )
                                if ok_streak >= bleed_confirm:
                                    bleed_ok = True
                                    print(
                                        f"pci: bleed OK j={bleed_used} "
                                        f"|r|={contact:.2f}N left|r|={left_c:.2f}N "
                                        f"FzR={fz_r:+.2f} FzL={fz_l:+.2f}",
                                        flush=True,
                                    )
                                    if abs(step_b) > 1e-9:
                                        _apply_axial(step_b)
                                    break
                            else:
                                ok_streak = 0
                                # Over target: always unload (watch wrist every step).
                                step_b = -float(bleed_step)
                            sink = along0_cf - float(feat_i.along_m)
                            if float(feat_i.lateral_m) < min_lat_cf or sink >= max_sink:
                                # Still allow unload away from mouth.
                                step_b = min(0.0, step_b)
                            _apply_axial(step_b)
                            if bleed_used % 10 == 0 or bleed_used == 1:
                                print(
                                    f"pci: bleed j={bleed_used} |r|={contact:.2f}N "
                                    f"left|r|={left_c:.2f}N "
                                    f"FzR={fz_r:+.2f}N FzL={fz_l:+.2f}N "
                                    f"step={step_b*1e3:.2f}mm "
                                    f"lat={feat_i.lateral_m*1e3:.1f}mm "
                                    f"tilt={tilt_i:.1f}deg",
                                    flush=True,
                                )
                        steps += bleed_used
                        meta["force_bleed_ok"] = bool(bleed_ok)
                        meta["force_bleed_frames"] = int(bleed_used)

                        # --- Path reverse: retreat along recorded approach wrist path.
                        # User ask: 贴面后原路返回一点 (not tip-chase / not left yield).
                        path_rev = bool(a_cfg.get("surface_path_reverse_enable", False))
                        path_rev_done = False
                        if path_rev:
                            backoff = int(
                                a_cfg.get("surface_path_reverse_backoff_frames", 30)
                            )
                            rev_need = float(
                                a_cfg.get(
                                    "surface_path_reverse_tip_m",
                                    a_cfg.get("surface_tip_lift_m", 0.002),
                                )
                            )
                            rev_frames = int(
                                a_cfg.get("surface_path_reverse_frames", 60)
                            )
                            rev_step = float(
                                a_cfg.get("surface_path_reverse_step_m", 0.00025)
                            )
                            site_now0 = actual_action44_from_sites(raw)
                            hold_r = site_now0[0:3].copy()
                            hold_l = site_now0[22:28].copy()
                            hold_r_hand = site_now0[6:22].copy()
                            # Firm-up right grasp before peel/retreat (scale>=1 only).
                            lift_r_scale_pr = float(
                                a_cfg.get("surface_lift_right_grasp_scale", 1.0)
                            )
                            if lift_r_scale_pr < 1.0 - 1e-9:
                                raise ValueError(
                                    "surface_lift_right_grasp_scale<1 forbidden "
                                    f"(got {lift_r_scale_pr:.3f})"
                                )
                            if lift_r_scale_pr > 1.0 + 1e-9:
                                hold_r_hand = np.clip(
                                    np.maximum(
                                        hold_r_hand, hold_r_hand * lift_r_scale_pr
                                    ),
                                    -1.5,
                                    1.5,
                                )
                                for _tg in range(5):
                                    cmd = hold.copy()
                                    cmd[0:3] = hold_r
                                    cmd[6:22] = hold_r_hand
                                    cmd[22:28] = hold_l
                                    cmd[28:44] = hold_l_hand
                                    step_action44(
                                        gym_env, cmd, ego_recorder=ego_recorder
                                    )
                                    _log_force("path_reverse_tighten")
                                steps += 5
                                print(
                                    f"pci: path-reverse tighten right grasp "
                                    f"scale={lift_r_scale_pr:.2f}",
                                    flush=True,
                                )
                            # Large retreat along hole opening (out of surface).
                            # History only sets lateral preference; distance is rev_dist.
                            rev_dist = float(
                                a_cfg.get("surface_path_reverse_dist_m", 0.04)
                            )
                            rev_tgt = hold_r + hole_u * rev_dist
                            if wrist_pre_hist:
                                idx = max(0, len(wrist_pre_hist) - 1 - max(0, backoff))
                                hist_p = np.asarray(
                                    wrist_pre_hist[idx], dtype=np.float64
                                ).reshape(3)
                                # Keep hist XY around hole, but force at least rev_dist out.
                                lat = hist_p - hole_u * float(np.dot(hist_p - hold_r, hole_u))
                                rev_tgt = hold_r + hole_u * rev_dist
                                # blend a bit of historical lateral offset
                                lat_off = lat - hold_r
                                lat_off = lat_off - hole_u * float(np.dot(lat_off, hole_u))
                                rev_tgt = rev_tgt + 0.35 * lat_off
                            feat_lift0 = features_from_raw(raw)
                            along_lift0 = float(feat_lift0.along_m)
                            tip0 = np.asarray(
                                feat_lift0.tip_pos, dtype=np.float64
                            ).reshape(3)
                            print(
                                f"pci: path-reverse retreat "
                                f"dist={rev_dist*1e3:.0f}mm backoff={backoff} "
                                f"hist={len(wrist_pre_hist)} "
                                f"need_tip≥{rev_need*1e3:.1f}mm "
                                f"step={rev_step*1e3:.2f}mm frames≤{rev_frames} "
                                f"keep_f={f_des:.2f}N left_share=0 (firm grasp)",
                                flush=True,
                            )
                            lift_used = 0
                            tip_up = 0.0
                            along_rise = 0.0
                            tip_d_xy = 0.0
                            keep_f = bool(
                                a_cfg.get("surface_path_reverse_keep_force", True)
                            )
                            for lift_used in range(1, max(0, rev_frames) + 1):
                                d = rev_tgt - hold_r
                                dn = float(np.linalg.norm(d))
                                if dn < 1e-6:
                                    break
                                step_v = d * (min(rev_step, dn) / dn)
                                hold_r = hold_r + step_v
                                # Keep light contact while peeling tip up.
                                if keep_f:
                                    contact_pre, _, _, _ = _wrist_contact()
                                    err_f = f_des - float(contact_pre)
                                    ax_corr = float(
                                        np.clip(kp * err_f, -max_step, max_step)
                                    )
                                    if err_f < 0.0:
                                        ax_corr = float(
                                            np.clip(
                                                kp * unload_boost * err_f,
                                                -max_step * unload_boost,
                                                max_step,
                                            )
                                        )
                                    hold_r = hold_r + press_ax * ax_corr
                                cmd = hold.copy()
                                cmd[0:3] = hold_r
                                cmd[6:22] = hold_r_hand
                                cmd[22:28] = hold_l  # left frozen
                                cmd[28:44] = hold_l_hand
                                step_action44(
                                    gym_env, cmd, ego_recorder=ego_recorder
                                )
                                contact, left_c, fz_r, fz_l = _log_force("path_reverse")
                                resid_peak = max(resid_peak, contact)
                                left_resid_peak = max(left_resid_peak, left_c)
                                if _grasp_lost():
                                    meta["grasp_abort_phase"] = "path_reverse"
                                    print(
                                        f"pci: path-reverse ABORT grasp_slip="
                                        f"{grasp_slip_peak*1e3:.1f}mm j={lift_used}",
                                        flush=True,
                                    )
                                    break
                                feat_l = features_from_raw(raw)
                                tip_now = np.asarray(
                                    feat_l.tip_pos, dtype=np.float64
                                ).reshape(3)
                                d_tip = tip_now - tip0
                                tip_up = float(np.dot(d_tip, hole_u))
                                along_rise = float(feat_l.along_m) - along_lift0
                                tip_d_xy = float(
                                    np.linalg.norm(
                                        d_tip - hole_u * float(np.dot(d_tip, hole_u))
                                    )
                                )
                                if tip_up >= rev_need or along_rise >= rev_need:
                                    print(
                                        f"pci: path-reverse done j={lift_used} "
                                        f"tip_up={tip_up*1e3:.1f}mm "
                                        f"Δalong={along_rise*1e3:.1f}mm "
                                        f"|r|={contact:.2f}N "
                                        f"slip={grasp_slip_peak*1e3:.1f}mm",
                                        flush=True,
                                    )
                                    break
                                if lift_used % 10 == 0 or lift_used == 1:
                                    print(
                                        f"pci: path-reverse j={lift_used} "
                                        f"tip_up={tip_up*1e3:.1f}mm "
                                        f"|r|={contact:.2f}N "
                                        f"slip={grasp_slip_peak*1e3:.1f}mm "
                                        f"remain={dn*1e3:.1f}mm",
                                        flush=True,
                                    )
                            steps += int(lift_used)
                            meta["surface_tip_up_m"] = float(tip_up)
                            meta["surface_tip_along_rise_m"] = float(along_rise)
                            meta["surface_light_lift_m"] = float(
                                abs(float(np.dot(hold_r - site_now0[0:3], hole_u)))
                            )
                            meta["path_reverse"] = True
                            meta["path_reverse_backoff_frames"] = int(backoff)
                            path_rev_done = True
                            # Path-reverse often lifts tip off tray (START_FLOAT).
                            # Hard seat: press-only reseat before any further search.
                            if bool(
                                a_cfg.get(
                                    "surface_path_reverse_reseat_enable", True
                                )
                            ):
                                pr_n = int(
                                    a_cfg.get(
                                        "surface_path_reverse_reseat_frames", 120
                                    )
                                )
                                pr_f = float(
                                    a_cfg.get(
                                        "surface_path_reverse_reseat_f_des_n",
                                        max(float(f_des), 0.12),
                                    )
                                )
                                print(
                                    f"pci: path-reverse reseat frames≤{pr_n} "
                                    f"f_des={pr_f:.2f}N (no planar)",
                                    flush=True,
                                )
                                for _prs in range(max(0, pr_n)):
                                    site_now = actual_action44_from_sites(raw)
                                    contact, left_c, fz_r, fz_l = _wrist_contact()
                                    if float(contact) >= max(
                                        float(
                                            a_cfg.get(
                                                "surface_planned_seat_contact_n",
                                                0.08,
                                            )
                                        ),
                                        0.05,
                                    ):
                                        if _prs >= 10:
                                            print(
                                                f"pci: path-reverse reseat OK "
                                                f"j={_prs} |r|={contact:.2f}N",
                                                flush=True,
                                            )
                                            break
                                    err_f = pr_f - float(contact)
                                    ax_step = float(
                                        np.clip(kp * err_f, -max_step, max_step)
                                    )
                                    if err_f > 0.0:
                                        ax_step = max(ax_step, 5e-5)
                                    cmd = hold.copy()
                                    cmd[0:3] = site_now[0:3] + press_ax * ax_step
                                    cmd[6:22] = hold_r_hand
                                    cmd[22:28] = hold_l
                                    cmd[28:44] = hold_l_hand
                                    step_action44(
                                        gym_env, cmd, ego_recorder=ego_recorder
                                    )
                                    _log_force("path_reverse_reseat")
                                    steps += 1
                                meta["path_reverse_reseat"] = True

                        # --- Phase 1a: break tip stiction (deep unload + tip-plane dither).
                        # Static friction sticks tip after soft latch; axial bleed alone
                        # may leave tip glued. Deep-unload then micro XY dither (right
                        # only; left frozen) before firm-grasp micro-lift.
                        stic_target = float(
                            a_cfg.get("surface_stiction_unload_n", 0.0)
                        )
                        stic_frames = int(
                            a_cfg.get("surface_stiction_unload_frames", 0)
                        )
                        stic_step = float(
                            a_cfg.get(
                                "surface_stiction_unload_step_m",
                                bleed_step,
                            )
                        )
                        dither_n = int(a_cfg.get("surface_stiction_dither_frames", 0))
                        dither_amp = float(
                            a_cfg.get("surface_stiction_dither_amp_m", 0.00035)
                        )
                        stic_used = 0
                        dither_used = 0
                        if stic_target > 0.0 and stic_frames > 0:
                            print(
                                f"pci: stiction deep-unload target≤{stic_target:.2f}N "
                                f"frames≤{stic_frames} step={stic_step*1e3:.2f}mm "
                                f"(left_share=0)",
                                flush=True,
                            )
                            ok_s = 0
                            for stic_used in range(1, stic_frames + 1):
                                contact, left_c, fz_r, fz_l = _log_force("stiction_unload")
                                resid_peak = max(resid_peak, contact)
                                if _grasp_lost():
                                    meta["grasp_abort_phase"] = "stiction_unload"
                                    print(
                                        f"pci: stiction-unload ABORT grasp_slip="
                                        f"{grasp_slip_peak*1e3:.1f}mm",
                                        flush=True,
                                    )
                                    break
                                if contact <= stic_target:
                                    ok_s += 1
                                    if ok_s >= 3:
                                        print(
                                            f"pci: stiction-unload OK j={stic_used} "
                                            f"|r|={contact:.2f}N",
                                            flush=True,
                                        )
                                        break
                                else:
                                    ok_s = 0
                                _apply_axial(-abs(stic_step), left_share=0.0)
                                if stic_used % 10 == 0 or stic_used == 1:
                                    print(
                                        f"pci: stiction-unload j={stic_used} "
                                        f"|r|={contact:.2f}N",
                                        flush=True,
                                    )
                            steps += int(stic_used)
                        meta["stiction_unload_frames"] = int(stic_used)
                        meta["stiction_unload_n"] = float(stic_target)
                        if dither_n > 0 and dither_amp > 0.0:
                            t1_d, t2_d, _ = hole_task_basis(hole_u)
                            print(
                                f"pci: stiction dither frames={dither_n} "
                                f"amp={dither_amp*1e3:.2f}mm (right only)",
                                flush=True,
                            )
                            for dither_used in range(1, dither_n + 1):
                                # square orbit in tip plane to break static friction
                                ph = (dither_used - 1) % 4
                                if ph == 0:
                                    d_xy = t1_d * dither_amp
                                elif ph == 1:
                                    d_xy = t2_d * dither_amp
                                elif ph == 2:
                                    d_xy = -t1_d * dither_amp
                                else:
                                    d_xy = -t2_d * dither_amp
                                hold_r = hold_r + d_xy
                                cmd = hold.copy()
                                cmd[0:3] = hold_r
                                cmd[6:22] = hold_r_hand
                                cmd[22:28] = hold_l
                                cmd[28:44] = hold_l_hand
                                step_action44(
                                    gym_env, cmd, ego_recorder=ego_recorder
                                )
                                contact, left_c, fz_r, fz_l = _log_force("stiction_dither")
                                resid_peak = max(resid_peak, contact)
                                if _grasp_lost():
                                    meta["grasp_abort_phase"] = "stiction_dither"
                                    print(
                                        f"pci: stiction-dither ABORT grasp_slip="
                                        f"{grasp_slip_peak*1e3:.1f}mm",
                                        flush=True,
                                    )
                                    break
                                if dither_used % 8 == 0 or dither_used == 1:
                                    print(
                                        f"pci: stiction-dither j={dither_used} "
                                        f"|r|={contact:.2f}N "
                                        f"slip={grasp_slip_peak*1e3:.1f}mm",
                                        flush=True,
                                    )
                            steps += int(dither_used)
                            # Center back: reverse last half-orbit average by 0 (hold).
                            site_d = actual_action44_from_sites(raw)
                            hold_r = site_d[0:3].copy()
                        meta["stiction_dither_frames"] = int(dither_used)
                        meta["stiction_dither_amp_m"] = float(dither_amp)

                        # Light re-touch to f_des after unload/dither (keep contact for lift).
                        n_pre_touch = int(
                            a_cfg.get("surface_stiction_retouch_frames", 0)
                        )
                        pre_touch_used = 0
                        for pre_touch_used in range(1, max(0, n_pre_touch) + 1):
                            contact, left_c, fz_r, fz_l = _log_force("stiction_retouch")
                            resid_peak = max(resid_peak, contact)
                            left_resid_peak = max(left_resid_peak, left_c)
                            if _grasp_lost():
                                meta["grasp_abort_phase"] = "stiction_retouch"
                                break
                            err = f_des - contact
                            step = float(np.clip(kp * err, -max_step, max_step))
                            if err > 0.0:
                                step = max(step, min(float(bleed_step) * 0.25, 0.00015))
                            _apply_axial(step, left_share=0.0)
                            if abs(contact - f_des) <= bleed_tol and pre_touch_used >= 4:
                                print(
                                    f"pci: stiction-retouch OK j={pre_touch_used} "
                                    f"|r|={contact:.2f}N",
                                    flush=True,
                                )
                                break
                            if pre_touch_used % 5 == 0 or pre_touch_used == 1:
                                print(
                                    f"pci: stiction-retouch j={pre_touch_used} "
                                    f"|r|={contact:.2f}N",
                                    flush=True,
                                )
                        steps += int(pre_touch_used)
                        meta["stiction_retouch_frames"] = int(pre_touch_used)

                        # --- Phase 1b: firm-grasp lift (skip if path-reverse already ran).
                        lift_m = float(a_cfg.get("surface_light_lift_m", 0.006))
                        tip_lift_need = float(a_cfg.get("surface_tip_lift_m", 0.0015))
                        n_lift_cap = int(a_cfg.get("surface_light_lift_frames", 50))
                        tip_servo_lift = bool(
                            a_cfg.get("surface_tip_servo_lift_enable", False)
                        )
                        per = float(
                            a_cfg.get(
                                "surface_light_lift_step_m",
                                max(lift_m / float(max(n_lift_cap, 1)), 0.00015)
                                if n_lift_cap > 0
                                else 0.00015,
                            )
                        )
                        per = float(np.clip(per, 1e-5, 0.001))
                        lift_r_scale = float(
                            a_cfg.get("surface_lift_right_grasp_scale", 1.0)
                        )
                        if not path_rev_done:
                            if lift_r_scale < 1.0 - 1e-9:
                                raise ValueError(
                                    "surface_lift_right_grasp_scale<1 forbidden "
                                    f"(got {lift_r_scale:.3f}); tip follow needs firm grasp"
                                )
                            if lift_r_scale > 1.0 + 1e-9:
                                # maximum(..., *scale): close fingers harder without opening.
                                hold_r_hand = np.clip(
                                    np.maximum(hold_r_hand, hold_r_hand * lift_r_scale),
                                    -1.5,
                                    1.5,
                                )
                                for _tg in range(3):
                                    cmd = hold.copy()
                                    cmd[0:3] = hold_r
                                    cmd[6:22] = hold_r_hand
                                    cmd[22:28] = hold_l
                                    cmd[28:44] = hold_l_hand
                                    step_action44(
                                        gym_env, cmd, ego_recorder=ego_recorder
                                    )
                                    _log_force("lift_tighten")
                                steps += 3
                                print(
                                    f"pci: lift tighten right grasp scale={lift_r_scale:.2f}",
                                    flush=True,
                                )
                            feat_lift0 = features_from_raw(raw)
                            along_lift0 = float(feat_lift0.along_m)
                            tip0 = np.asarray(feat_lift0.tip_pos, dtype=np.float64).reshape(3)
                            lift_used = 0
                            tip_d_xy = 0.0
                            along_rise = 0.0
                            tip_up = 0.0
                            if tip_servo_lift:
                                # Scheme D: lift tip OUT along hole opening (+hole);
                                # wrist := tip_target + frozen (wrist0-tip0).
                                tip_lift_tgt = float(
                                    a_cfg.get("surface_tip_servo_lift_m", 0.0025)
                                )
                                tip_lift_need = float(
                                    a_cfg.get(
                                        "surface_tip_servo_lift_need_m",
                                        a_cfg.get("surface_tip_lift_m", 0.0015),
                                    )
                                )
                                n_lift_cap = int(
                                    a_cfg.get("surface_tip_servo_lift_frames", 80)
                                )
                                max_w_step = float(
                                    a_cfg.get("surface_tip_servo_lift_wrist_step_m", 0.005)
                                )
                                # +hole_u = out of hole (= -press_ax). User "抬起沿 -hole"
                                # if hole:=insert; here hole_axis is opening, so +hole.
                                tip_target = tip0 + hole_u * tip_lift_tgt
                                site_lift0 = actual_action44_from_sites(raw)
                                offset_wt0 = site_lift0[0:3].copy() - tip0
                                print(
                                    f"pci: tip-servo-lift (scheme D) need≥{tip_lift_need*1e3:.1f}mm "
                                    f"tgt={tip_lift_tgt*1e3:.1f}mm wrist_step={max_w_step*1e3:.1f}mm "
                                    f"f_des={f_des:.2f}N (firm grasp, frozen wrist-tip offset)",
                                    flush=True,
                                )
                                for lift_used in range(1, max(0, n_lift_cap) + 1):
                                    feat_l = features_from_raw(raw)
                                    site_now = actual_action44_from_sites(raw)
                                    hold_r = _tip_servo_lift_wrist_cmd(
                                        tip_target,
                                        site_now[0:3],
                                        max_w_step,
                                        offset_wt0,
                                    )
                                    hold_l[0:3] = site_now[22:25].copy()
                                    cmd = hold.copy()
                                    cmd[0:3] = hold_r
                                    cmd[6:22] = hold_r_hand
                                    cmd[22:28] = hold_l
                                    cmd[28:44] = hold_l_hand
                                    step_action44(
                                        gym_env, cmd, ego_recorder=ego_recorder
                                    )
                                    contact, left_c, fz_r, fz_l = _log_force("lift")
                                    resid_peak = max(resid_peak, contact)
                                    left_resid_peak = max(left_resid_peak, left_c)
                                    feat_l = features_from_raw(raw)
                                    along_now = float(feat_l.along_m)
                                    tip_now = np.asarray(
                                        feat_l.tip_pos, dtype=np.float64
                                    ).reshape(3)
                                    d_tip = tip_now - tip0
                                    tip_d_xy = float(
                                        np.linalg.norm(
                                            d_tip - hole_u * float(np.dot(d_tip, hole_u))
                                        )
                                    )
                                    along_rise = along_now - along_lift0
                                    # tip_up = lift out along opening (+hole); gate wants ≥need.
                                    tip_up = float(np.dot(d_tip, hole_u))
                                    if _grasp_lost():
                                        print(
                                            f"pci: tip-servo-lift ABORT grasp_slip="
                                            f"{grasp_slip_peak*1e3:.1f}mm "
                                            f"> {grasp_slip_lim*1e3:.1f}mm j={lift_used}",
                                            flush=True,
                                        )
                                        meta["grasp_abort_phase"] = "tip_servo_lift"
                                        break
                                    if (
                                        along_rise >= tip_lift_need
                                        or tip_up >= tip_lift_need
                                    ):
                                        print(
                                            f"pci: tip-servo-lift done j={lift_used} "
                                            f"Δalong={along_rise*1e3:.1f}mm "
                                            f"tip_up={tip_up*1e3:.1f}mm "
                                            f"tip_xy={tip_d_xy*1e3:.1f}mm |r|={contact:.2f}N",
                                            flush=True,
                                        )
                                        break
                                    if lift_used % 15 == 0 or lift_used == 1:
                                        print(
                                            f"pci: tip-servo-lift j={lift_used} "
                                            f"tip_up={tip_up*1e3:.1f}mm "
                                            f"Δalong={along_rise*1e3:.1f}mm "
                                            f"|r|={contact:.2f}N",
                                            flush=True,
                                        )
                                # Refresh wrist hold from sites for subsequent axial P.
                                site_af = actual_action44_from_sites(raw)
                                hold_r = site_af[0:3].copy()
                                hold_l = site_af[22:28].copy()
                                steps += int(lift_used)
                                meta["surface_light_lift_m"] = float(tip_lift_tgt)
                                meta["surface_tip_servo_lift"] = True
                                meta["surface_tip_servo_lift_m"] = float(tip_lift_tgt)
                                meta["surface_tip_servo_lift_wrist_step_m"] = float(
                                    max_w_step
                                )
                                meta["surface_tip_along_rise_m"] = float(along_rise)
                                meta["surface_tip_up_m"] = float(tip_up)
                            else:
                                # Scheme C knobs (left yield / dual axial unload). Defaults
                                # keep prior firm-grasp behavior when unset.
                                lift_left_share = float(
                                    a_cfg.get(
                                        "surface_lift_left_share",
                                        left_share_cf,
                                    )
                                )
                                lift_left_admit_scale = float(
                                    a_cfg.get("surface_lift_left_admit_scale", 0.0)
                                )
                                lift_left_extra = float(
                                    a_cfg.get("surface_lift_left_extra_unload_m", 0.0)
                                )
                                lift_admit_c: LeftWristAdmitController | None = None
                                if lift_left_admit_scale > 1e-9:
                                    lift_admit_c = LeftWristAdmitController(
                                        _left_wrist_admit_config(cfg)
                                    )
                                    lift_admit_c.reset(read_wrist_wrench_world(raw)[1])
                                    lift_admit_c.config.enable = True
                                print(
                                    f"pci: tip-lift (scheme C left-yield) "
                                    f"need≥{tip_lift_need*1e3:.1f}mm "
                                    f"cmd_cap={lift_m*1e3:.1f}mm f_des={f_des:.2f}N "
                                    f"left_share={lift_left_share:.2f} "
                                    f"admit={lift_left_admit_scale:.2f} "
                                    f"extra_L={lift_left_extra*1e3:.2f}mm "
                                    f"r_scale={lift_r_scale:.2f}",
                                    flush=True,
                                )
                                for lift_used in range(1, max(0, n_lift_cap) + 1):
                                    _apply_axial(-per, left_share=lift_left_share)
                                    if lift_left_extra > 1e-9:
                                        hold_l[0:3] = hold_l[0:3] + (
                                            (-press_ax) * lift_left_extra
                                        )
                                        cmd = hold.copy()
                                        cmd[0:3] = hold_r
                                        cmd[6:22] = hold_r_hand
                                        cmd[22:28] = hold_l
                                        cmd[28:44] = hold_l_hand
                                        step_action44(
                                            gym_env, cmd, ego_recorder=ego_recorder
                                        )
                                    if lift_admit_c is not None:
                                        d_adm = (
                                            lift_admit_c.step(
                                                read_wrist_wrench_world(raw)[1],
                                                approach_axis=-hole_u,
                                            )
                                            * lift_left_admit_scale
                                        )
                                        d_ax = hole_u * float(np.dot(d_adm, hole_u))
                                        hold_l[0:3] = hold_l[0:3] + d_ax
                                        cmd = hold.copy()
                                        cmd[0:3] = hold_r
                                        cmd[6:22] = hold_r_hand
                                        cmd[22:28] = hold_l
                                        cmd[28:44] = hold_l_hand
                                        step_action44(
                                            gym_env, cmd, ego_recorder=ego_recorder
                                        )
                                    contact, left_c, fz_r, fz_l = _log_force("lift")
                                    resid_peak = max(resid_peak, contact)
                                    left_resid_peak = max(left_resid_peak, left_c)
                                    feat_l = features_from_raw(raw)
                                    along_now = float(feat_l.along_m)
                                    tip_now = np.asarray(
                                        feat_l.tip_pos, dtype=np.float64
                                    ).reshape(3)
                                    d_tip = tip_now - tip0
                                    tip_d_xy = float(
                                        np.linalg.norm(
                                            d_tip - hole_u * float(np.dot(d_tip, hole_u))
                                        )
                                    )
                                    along_rise = along_now - along_lift0
                                    # tip_up: rise away from surface (= +hole_u; press=-hole_u).
                                    tip_up = float(np.dot(d_tip, hole_u))
                                    if _grasp_lost():
                                        print(
                                            f"pci: tip-lift ABORT grasp_slip="
                                            f"{grasp_slip_peak*1e3:.1f}mm "
                                            f"> {grasp_slip_lim*1e3:.1f}mm j={lift_used}",
                                            flush=True,
                                        )
                                        meta["grasp_abort_phase"] = "tip_lift"
                                        break
                                    if (
                                        along_rise >= tip_lift_need
                                        or tip_up >= tip_lift_need
                                        or (lift_used * per) >= lift_m
                                    ):
                                        print(
                                            f"pci: tip-lift done j={lift_used} "
                                            f"cmd={lift_used*per*1e3:.1f}mm "
                                            f"Δalong={along_rise*1e3:.1f}mm "
                                            f"tip_up={tip_up*1e3:.1f}mm "
                                            f"tip_xy={tip_d_xy*1e3:.1f}mm |r|={contact:.2f}N",
                                            flush=True,
                                        )
                                        break
                                steps += int(lift_used)
                                meta["surface_light_lift_m"] = float(lift_used * per)
                                meta["surface_tip_along_rise_m"] = float(along_rise)
                                meta["surface_tip_up_m"] = float(tip_up)
                                meta["surface_lift_left_share"] = float(lift_left_share)
                                meta["surface_lift_left_admit_scale"] = float(
                                    lift_left_admit_scale
                                )
                                meta["surface_lift_left_extra_unload_m"] = float(
                                    lift_left_extra
                                )
                                meta["scheme_c_left_yield"] = True


                        # Optional soft re-touch (off by default — hard retouch jams tip).
                        n_retouch = int(a_cfg.get("surface_light_retouch_frames", 0))
                        j_rt = 0
                        for j_rt in range(1, max(0, n_retouch) + 1):
                            contact, left_c, fz_r, fz_l = _log_force("retouch")
                            resid_peak = max(resid_peak, contact)
                            left_resid_peak = max(left_resid_peak, left_c)
                            err = f_des - contact
                            step = float(np.clip(kp * err, -max_step, max_step))
                            if err > 0.0:
                                step = max(step, min(float(bleed_step) * 0.35, 0.0002))
                            elif err < 0.0:
                                step = float(
                                    np.clip(
                                        kp * unload_boost * err,
                                        -max_step * unload_boost,
                                        max_step,
                                    )
                                )
                            _apply_axial(step)
                            in_band = (
                                abs(contact - f_des) <= bleed_tol
                                and contact >= 0.5 * f_des
                            )
                            if in_band and j_rt >= 6:
                                print(
                                    f"pci: retouch OK j={j_rt} |r|={contact:.2f}N "
                                    f"left|r|={left_c:.2f}N",
                                    flush=True,
                                )
                                break
                        if n_retouch > 0:
                            steps += int(j_rt)

                        # --- Phase 2: light dual const-force hold (wrist P) ---
                        print(
                            f"pci: dual const-force hold f_des={f_des:.2f}N "
                            f"frames={n_cf} left_share={left_share_cf:.2f} "
                            f"bleed_ok={int(bleed_ok)}",
                            flush=True,
                        )
                        used = 0
                        for used in range(1, max(0, n_cf) + 1):
                            contact, left_c, fz_r, fz_l = _log_force("hold")
                            resid_peak = max(resid_peak, contact)
                            left_resid_peak = max(left_resid_peak, left_c)
                            err = f_des - contact
                            step = float(np.clip(kp * err, -max_step, max_step))
                            # Overshoot: unload faster than press.
                            if err < 0.0:
                                step = float(
                                    np.clip(
                                        kp * unload_boost * err,
                                        -max_step * unload_boost,
                                        max_step,
                                    )
                                )
                            feat_i = features_from_raw(raw)
                            sink = along0_cf - float(feat_i.along_m)
                            if (
                                float(feat_i.lateral_m) < min_lat_cf
                                or sink >= max_sink
                            ):
                                step = min(0.0, step)
                            _apply_axial(step)
                            tilt_i = _tray_tilt_entry_deg()
                            phase_a_tilt_peak = max(phase_a_tilt_peak, tilt_i)
                            phase_a_peg_tilt_peak = max(
                                phase_a_peg_tilt_peak, _peg_tilt_entry_deg()
                            )
                            out_i = env._labeler.compute(raw)
                            if (not out_i.tray_ok) or tilt_i > max_tilt_cf:
                                print(
                                    f"pci: const-force abort tilt={tilt_i:.1f}deg "
                                    f"tray_ok={int(bool(out_i.tray_ok))} j={used}",
                                    flush=True,
                                )
                                break
                            if used % 15 == 0 or used == 1:
                                print(
                                    f"pci: const-force j={used} |r|={contact:.2f}N "
                                    f"left|r|={left_c:.2f}N "
                                    f"FzR={fz_r:+.2f}N FzL={fz_l:+.2f}N "
                                    f"step={step*1e3:.2f}mm "
                                    f"lat={feat_i.lateral_m*1e3:.1f}mm "
                                    f"tilt={tilt_i:.1f}deg",
                                    flush=True,
                                )
                        steps += used
                        feat_s = features_from_raw(raw)
                        out_s = env._labeler.compute(raw)
                        meta["soft_latched"] = True
                        meta["latch_soft_contact"] = True
                        meta["stop_at_soft_contact"] = True
                        meta["const_force_hold"] = True
                        meta["const_force_des_n"] = float(f_des)
                        meta["const_force_frames"] = int(used)
                        meta["const_force_resid_peak_n"] = float(resid_peak)
                        meta["const_force_left_resid_peak_n"] = float(left_resid_peak)
                        meta["const_force_left_share"] = float(left_share_cf)
                        meta["fz_contact"] = float(fz)
                        meta["fz_delta"] = float(d_fz)
                        meta["fz_baseline_const"] = float(fz_base_c)
                        print(
                            f"pci: const-force done — |r|_peak={resid_peak:.2f}N "
                            f"left_peak={left_resid_peak:.2f}N "
                            f"lat={feat_s.lateral_m*1e3:.1f}mm "
                            f"along={feat_s.along_m*1e3:.1f}mm tip={feat_s.tip_socket_dist_m*1e3:.1f}mm "
                            f"tilt={phase_a_tilt_peak:.2f}deg → tip-spiral",
                            flush=True,
                        )
                        # Firm-grasp tip-plane Archimedean spiral + light force.
                        # Wrist mocap spiral alone does not move tip (finger stretch);
                        # servo wrist so TIP tracks the spiral (privileged_diagnostic).
                        _update_grasp_slip()
                        # Non-rigid grasp is expected: planar_C tip tracking exists for
                        # this. Never skip the whole spiral because peel stretch tripped
                        # the slip meter (ep07 never searched).
                        if (
                            grasp_slip_peak > grasp_slip_lim or meta.get("grasp_abort_phase")
                        ) and not (
                            bool(a_cfg.get("surface_planned_force_hole_detect", False))
                            and str(a_cfg.get("surface_tip_search_mode", ""))
                            .strip()
                            .lower()
                            in ("planned_spiral", "planned", "priv_spiral")
                        ):
                            feat_s = features_from_raw(raw)
                            out_s = env._labeler.compute(raw)
                            meta["align_lat_mm"] = feat_s.lateral_m * 1000
                            meta["align_along_mm"] = feat_s.along_m * 1000
                            meta["phase_a_tray_tilt_peak_deg"] = float(phase_a_tilt_peak)
                            meta["phase_a_peg_tilt_peak_deg"] = float(phase_a_peg_tilt_peak)
                            meta["phase_a_rel_rot_peak_rad"] = float(phase_a_rel_peak)
                            meta["spiral_frames"] = 0
                            meta["spiral_reason"] = "grasp_slip_pre_spiral"
                            meta["spiral_tip_xy_peak_mm"] = 0.0
                            meta["force_trace"] = force_trace
                            meta["force_trace_n"] = int(len(force_trace))
                            meta["grasp_slip_peak_m"] = float(grasp_slip_peak)
                            meta["grasp_latch_ok"] = True
                            gate = _priv_tip_spiral_gate(meta, cfg)
                            meta["priv_tip_spiral_gate"] = gate
                            meta.update(_final_geom_meta(feat_s, out_s, ctrl))
                            print(
                                f"pci: skip spiral — grasp_slip="
                                f"{grasp_slip_peak*1e3:.1f}mm "
                                f"GATE=FAIL grasp={gate.get('grasp_ok')}",
                                flush=True,
                            )
                            for rs in gate.get("reasons", []):
                                print(f"pci: GATE fail — {rs}", flush=True)
                            return steps, "priv_tip_spiral_gate_fail", meta
                        if grasp_slip_peak > grasp_slip_lim:
                            print(
                                f"pci: continue spiral despite pre-slip="
                                f"{grasp_slip_peak*1e3:.1f}mm (tip-track mode)",
                                flush=True,
                            )
                            meta.pop("grasp_abort_phase", None)
                        # Re-latch grasp baseline AFTER successful lift so spiral
                        # grasp_slip measures "drop during spiral", not peel stretch.
                        geom_sp0 = read_priv_grasp_geom(raw)
                        peg_in_r0 = _peg_in_right_wrist(geom_sp0)
                        grasp_slip_peak = 0.0
                        meta["grasp_slip_peak_m"] = 0.0
                        meta["grasp_relatch_after_lift"] = True
                        site_sp = actual_action44_from_sites(raw)
                        hold_r = site_sp[0:3].copy()
                        hold_l = site_sp[22:28].copy()
                        # Keep firm grasp command (don't drop back to stretched site q).
                        hold_r_hand = np.clip(
                            np.maximum(site_sp[6:22].copy(), hold_r_hand),
                            -1.5,
                            1.5,
                        )
                        hold_l_hand = site_sp[28:44].copy()
                        if getattr(ctrl, "_hold_l_hand", None) is not None:
                            hold_l_hand = np.asarray(
                                ctrl._hold_l_hand, dtype=np.float64
                            ).reshape(16)
                        # OIGS / GMHQP residuals; wrist laws read later.
                        tip_oigs_enable = bool(
                            a_cfg.get("surface_tip_oigs_enable", False)
                        ) or str(
                            a_cfg.get("surface_tip_baseline", "")
                        ).strip().lower() in (
                            "track_b_oigs",
                            "oigs",
                            "object_impedance_grasp_stiff",
                            "pfanne_grasp_stiff",
                        )
                        tip_gmhqp_enable = bool(
                            a_cfg.get("surface_tip_gmhqp_enable", False)
                        ) or str(
                            a_cfg.get("surface_tip_baseline", "")
                        ).strip().lower() in (
                            "track_b_gmhqp",
                            "gmhqp",
                            "grasp_map_hqp",
                            "grasp_map_operational_hqp",
                            "object_task_hqp",
                            "track_a_gmhqp_tip_obs",
                            "gmhqp_tip_obs",
                            "track_a_gmhqp",
                            "track_a_gmhqp_tip_fuse",
                            "gmhqp_tip_fuse",
                        )
                        tip_hybrid_gmhqp_enable = bool(
                            a_cfg.get("surface_tip_hybrid_gmhqp_enable", False)
                        ) or str(
                            a_cfg.get("surface_tip_baseline", "")
                        ).strip().lower() in (
                            "track_b_hybrid_gmhqp",
                            "hybrid_gmhqp",
                            "gmhqp_hybrid",
                            "tip_planar_c_hybrid_hqp",
                            "escande_hybrid_gmhqp",
                        )
                        oigs_state = OigsState()
                        oigs_cfg = OigsConfig()  # overwritten after baseline mode parse
                        gmhqp_state = GmhqpState()
                        gmhqp_cfg = GmhqpConfig()  # overwritten after baseline mode parse
                        hybrid_gmhqp_state = HybridGmhqpState()
                        hybrid_gmhqp_cfg = HybridGmhqpConfig()

                        # Arm peg-in-hand QP latch at tip-search start (not Phase-A latch).
                        if tip_search_pose_hold and use_priv_grasp_a:
                            if phase_a_qp is None:
                                phase_a_qp = PrivGraspOptController(
                                    _priv_grasp_config(cfg)
                                )
                            phase_a_qp.reset(
                                copy_priv_grasp_geom(geom_sp0),
                                hold_r_hand,
                                hold_l_hand,
                            )
                            phase_a_qp_ready = True
                            meta["tip_search_pose_hold"] = True
                            print(
                                "pci: tip-search pose-hold QP armed "
                                "(peg-in-hand / tip-wrist)",
                                flush=True,
                            )

                        def _fill_tip_search_hands(cmd44: np.ndarray) -> None:
                            """Locked fingers, or pose-hold QP Δq if enabled."""
                            nonlocal phase_a_rel_peak
                            cmd44[6:22] = hold_r_hand
                            cmd44[28:44] = hold_l_hand
                            if not tip_search_pose_hold:
                                return
                            if not (
                                use_priv_grasp_a
                                and phase_a_qp is not None
                                and phase_a_qp_ready
                                and force_labeler is not None
                            ):
                                return
                            geom_ph = read_priv_grasp_geom(raw)
                            f12_ph, lf12_ph = read_dual_finger_force24(
                                raw, force_labeler
                            )
                            po_ph = phase_a_qp.step(geom_ph, f12_ph, lf12_ph)
                            phase_a_rel_peak = max(
                                phase_a_rel_peak, float(po_ph.rel_rot_err_rad)
                            )
                            # OIGS: feed Pfanne object residual into wrist path alpha.
                            if tip_oigs_enable:
                                oigs_state.obj_rot_err_rad = max(
                                    float(po_ph.peg_inhand_rot_err_rad),
                                    float(po_ph.rel_rot_err_rad),
                                )
                                # Pos residual not returned by result; use 0 (rot-gated).
                                oigs_state.obj_pos_err_m = 0.0
                            cmd44[6:22] = np.clip(
                                hold_r_hand + po_ph.delta_right_hand16, -1.5, 1.5
                            )
                            if phase_a_left_qp_fingers:
                                cmd44[28:44] = np.clip(
                                    np.maximum(
                                        hold_l_hand + po_ph.delta_left_hand16,
                                        hold_l_hand * 0.98,
                                    ),
                                    -1.5,
                                    1.5,
                                )

                        feat_sp0 = features_from_raw(raw)
                        tip_sp0 = np.asarray(feat_sp0.tip_pos, dtype=np.float64).reshape(3)
                        # Spiral rotation axis = contact normal (peg↔hole surface),
                        # NOT peg shaft center. Flush face contact is the only case
                        # where peg/hole axis ≈ contact normal.
                        hole_u_sp = feat_sp0.hole_axis / (
                            np.linalg.norm(feat_sp0.hole_axis) + 1e-12
                        )
                        peg_u_sp = feat_sp0.peg_axis / (
                            np.linalg.norm(feat_sp0.peg_axis) + 1e-12
                        )
                        wr_sp0 = read_wrist_wrench_world(raw)
                        f_r_sp0 = np.asarray(wr_sp0[0][:3], dtype=np.float64).reshape(3)
                        f_n_sp0 = float(np.linalg.norm(f_r_sp0))
                        axis_mode = str(
                            a_cfg.get("surface_tip_spiral_axis", "contact_normal")
                        ).strip().lower()
                        if axis_mode in ("peg", "peg_axis"):
                            spiral_n = peg_u_sp.copy()
                            spiral_axis_src = "peg"
                        elif axis_mode in ("hole", "hole_axis"):
                            spiral_n = hole_u_sp.copy()
                            spiral_axis_src = "hole"
                        else:
                            # Force on wrist ≈ surface→peg push; align to hole opening.
                            if f_n_sp0 >= float(
                                a_cfg.get("surface_tip_spiral_contact_fmin_n", 0.03)
                            ):
                                spiral_n = f_r_sp0 / (f_n_sp0 + 1e-12)
                                if float(spiral_n @ hole_u_sp) < 0.0:
                                    spiral_n = -spiral_n
                                spiral_axis_src = "force"
                            else:
                                # Flat mouth fallback: socket face normal ≈ hole axis.
                                spiral_n = hole_u_sp.copy()
                                spiral_axis_src = "hole_fallback"
                        t1, t2, spiral_n = hole_task_basis(spiral_n)
                        # Press into the contact plane (ConnTact seeking along -normal).
                        press_ax_spiral = -spiral_n
                        ang_peg_n = float(
                            np.degrees(
                                np.arccos(
                                    float(
                                        np.clip(abs(float(peg_u_sp @ spiral_n)), 0.0, 1.0)
                                    )
                                )
                            )
                        )
                        ang_hole_n = float(
                            np.degrees(
                                np.arccos(
                                    float(
                                        np.clip(abs(float(hole_u_sp @ spiral_n)), 0.0, 1.0)
                                    )
                                )
                            )
                        )
                        print(
                            f"pci: spiral-axis src={spiral_axis_src} "
                            f"|F|={f_n_sp0:.3f}N "
                            f"ang(peg,n)={ang_peg_n:.1f}deg "
                            f"ang(hole,n)={ang_hole_n:.1f}deg "
                            f"axis_err={float(np.degrees(feat_sp0.axis_error_rad)):.1f}deg",
                            flush=True,
                        )
                        spiral_center_mode = str(
                            a_cfg.get("surface_tip_spiral_center", "tip")
                        ).strip().lower()
                        if spiral_center_mode == "socket":
                            spiral_center = _hole_axis_point_at_tip(
                                tip_sp0,
                                feat_sp0.socket_pos,
                                feat_sp0.hole_axis,
                            )
                        else:
                            spiral_center = tip_sp0.copy()
                        lat_shrink_only = bool(
                            a_cfg.get("surface_spiral_lat_shrink_only", False)
                        )
                        freeze_off = bool(
                            a_cfg.get("surface_tip_spiral_freeze_offset", True)
                        )
                        offset_wt0_sp = site_sp[0:3].copy() - tip_sp0
                        tip_baseline_mode = str(
                            a_cfg.get("surface_tip_baseline", "")
                        ).strip().lower()
                        if not tip_baseline_mode and bool(
                            a_cfg.get("surface_planned_spiral_wrist_ff", False)
                        ):
                            tip_baseline_mode = "tip_servo_ff"
                        # Track-A: observer → GMHQP tip task (not Oracle PD).
                        # TEC joint: same path; tip_ctrl gated by NIS/contact theory.
                        tip_tec_joint = bool(
                            is_tec_joint_mode(tip_baseline_mode)
                        ) or bool(a_cfg.get("surface_tip_tec_joint_enable", False))
                        tip_gmhqp_tip_obs = bool(
                            is_gmhqp_tip_obs_mode(tip_baseline_mode)
                        ) or tip_tec_joint or (
                            bool(a_cfg.get("surface_tip_gmhqp_use_tip_obs", False))
                            and (
                                tip_gmhqp_enable
                                or is_gmhqp_mode(tip_baseline_mode)
                            )
                        )
                        tec_joint_cfg = TecJointConfig(
                            nis_chi2_accept=float(
                                a_cfg.get(
                                    "surface_tip_tec_nis_chi2",
                                    TecJointConfig.nis_chi2_accept,
                                )
                                or TecJointConfig.nis_chi2_accept
                            ),
                            require_contact=bool(
                                a_cfg.get("surface_tip_tec_require_contact", True)
                            ),
                            reject_on_slip=bool(
                                a_cfg.get("surface_tip_tec_reject_on_slip", True)
                            ),
                            geom_disagree_sigma=float(
                                a_cfg.get("surface_tip_tec_geom_sigma", 3.0) or 3.0
                            ),
                            uncert_sigma=float(
                                a_cfg.get("surface_tip_tec_uncert_sigma", 3.0) or 3.0
                            ),
                        )
                        tec_joint_state = TecJointState()
                        tip_oracle_enable = tip_baseline_mode in (
                            "priv_oracle_tip_servo",
                            "oracle_tip",
                            "oracle",
                            "priv_oracle_tip_obs",
                            "track_a_est_oracle",
                            "est_oracle_tip_servo",
                            "est_oracle",
                            "track_a_theory_est",
                            "track_a_theory_ekf",
                            "theory_est_oracle",
                            "tec_slim_oracle",
                            "track_a_tip_axis",
                            "track_a_tip_axis_ekf",
                            "tip_axis_oracle",
                            "tec_tip_axis_oracle",
                        ) or bool(a_cfg.get("surface_tip_oracle_mouth", False))
                        # Hybrid must not steal Oracle mouth PD.
                        if tip_gmhqp_tip_obs:
                            tip_oracle_enable = False
                        tip_obs_sigma_mm = float(
                            a_cfg.get("surface_tip_obs_noise_mm", 0.0) or 0.0
                        )
                        tip_obs_lag_frames = int(
                            a_cfg.get("surface_tip_obs_lag_frames", 0) or 0
                        )
                        # Track-A: tip_hat (NOT tip_gt+noise). Theory / tip+axis EKF or v0.
                        _tip_est_backend = str(
                            a_cfg.get("surface_tip_est_backend", "") or ""
                        ).strip().lower()
                        if tip_gmhqp_tip_obs and not _tip_est_backend:
                            _tip_est_backend = "theory_ekf"
                        tip_axis_enable = (
                            (not tip_gmhqp_tip_obs)
                            and (
                                is_track_a_tip_axis_mode(tip_baseline_mode)
                                or (
                                    bool(a_cfg.get("surface_tip_est_enable", False))
                                    and _tip_est_backend
                                    in (
                                        "tip_axis_ekf",
                                        "tip_axis",
                                        "tec_tip_axis",
                                        "extrinsic_tip_axis",
                                    )
                                )
                            )
                        )
                        tip_theory_enable = (
                            tip_gmhqp_tip_obs
                            or is_track_a_theory_mode(tip_baseline_mode)
                            or (
                                bool(a_cfg.get("surface_tip_est_enable", False))
                                and _tip_est_backend
                                in (
                                    "theory_ekf",
                                    "tec_slim",
                                    "extrinsic_ekf",
                                    "theory",
                                    "tip_axis_ekf",
                                    "tip_axis",
                                    "tec_tip_axis",
                                    "extrinsic_tip_axis",
                                )
                            )
                        )
                        tip_est_enable = (
                            bool(a_cfg.get("surface_tip_est_enable", False))
                            or is_track_a_est_mode(tip_baseline_mode)
                            or tip_gmhqp_tip_obs
                        )
                        tip_obs_enable = (
                            (not tip_est_enable)
                            and (
                                tip_obs_sigma_mm > 0.0
                                or tip_baseline_mode in ("priv_oracle_tip_obs",)
                            )
                        )
                        tip_est: ContactTipEstimator | None = None
                        tip_theory_ekf: ExtrinsicTipStateEKF | ExtrinsicTipAxisEKF | None = (
                            None
                        )
                        tip_contact_sensor: ContactTipEstimator | None = None
                        tip_est_err_peak_m = 0.0
                        tip_est_err_sum_m = 0.0
                        tip_est_err_n = 0
                        tip_est_last_source = ""
                        tip_est_seated_frames = 0
                        tip_est_uncert_peak_m = 0.0
                        tip_axis_err_peak_rad = 0.0
                        tip_axis_hat: np.ndarray | None = None
                        tip_est_wrist_prev = (
                            np.asarray(site_sp[0:3], dtype=np.float64).reshape(3).copy()
                            if tip_est_enable
                            else None
                        )
                        if tip_theory_enable:
                            # TEC-slim tip or tip+axis EKF.
                            # Hybrid tip_obs→GMHQP: keep Escande law (no Oracle mouth PD).
                            if tip_gmhqp_tip_obs:
                                tip_oracle_enable = False
                                tip_gmhqp_enable = True
                                if not is_gmhqp_mode(tip_baseline_mode):
                                    if tip_tec_joint:
                                        tip_baseline_mode = "track_a_tec_joint_gmhqp"
                                    else:
                                        tip_baseline_mode = (
                                            "track_a_gmhqp_tip_fuse"
                                            if bool(
                                                a_cfg.get(
                                                    "surface_tip_theory_geom_prior",
                                                    False,
                                                )
                                            )
                                            else "track_a_gmhqp_tip_obs"
                                        )
                                tip_axis_enable = False
                            else:
                                tip_oracle_enable = True
                                if tip_axis_enable:
                                    if not is_track_a_tip_axis_mode(tip_baseline_mode):
                                        tip_baseline_mode = "track_a_tip_axis"
                                elif not is_track_a_theory_mode(tip_baseline_mode):
                                    tip_baseline_mode = "track_a_theory_est"
                            tip_contact_sensor = ContactTipEstimator(
                                raw,
                                ema=float(
                                    a_cfg.get("surface_tip_est_ema", 0.55) or 0.55
                                ),
                                min_force_n=float(
                                    a_cfg.get("surface_tip_est_min_force_n", 0.008)
                                    or 0.008
                                ),
                                seat_force_n=float(
                                    a_cfg.get("surface_tip_est_seat_force_n", 0.015)
                                    or 0.015
                                ),
                                cluster_radius_m=float(
                                    a_cfg.get(
                                        "surface_tip_est_cluster_radius_m", 0.008
                                    )
                                    or 0.008
                                ),
                                force_pow=float(
                                    a_cfg.get("surface_tip_est_force_pow", 1.5) or 1.5
                                ),
                                use_pca_locus=False,
                            )
                            _th_cfg = TheoryEstConfig(
                                process_tip=float(
                                    a_cfg.get(
                                        "surface_tip_theory_process_tip", 2.0e-6
                                    )
                                    or 2.0e-6
                                ),
                                process_c=float(
                                    a_cfg.get(
                                        "surface_tip_theory_process_c", 2.0e-5
                                    )
                                    or 2.0e-5
                                ),
                                process_axis=float(
                                    a_cfg.get(
                                        "surface_tip_theory_process_axis", 5.0e-5
                                    )
                                    or 5.0e-5
                                ),
                                meas_contact_var=float(
                                    a_cfg.get(
                                        "surface_tip_theory_meas_contact_var",
                                        2.0e-7,
                                    )
                                    or 2.0e-7
                                ),
                                meas_wrench_var=float(
                                    a_cfg.get(
                                        "surface_tip_theory_meas_wrench_var",
                                        4.0e-5,
                                    )
                                    or 4.0e-5
                                ),
                                meas_axis_finger_var=float(
                                    a_cfg.get(
                                        "surface_tip_theory_meas_axis_finger_var",
                                        8.0e-3,
                                    )
                                    or 8.0e-3
                                ),
                                meas_axis_wrist_var=float(
                                    a_cfg.get(
                                        "surface_tip_theory_meas_axis_wrist_var",
                                        4.0e-2,
                                    )
                                    or 4.0e-2
                                ),
                                meas_axis_contact_var=float(
                                    a_cfg.get(
                                        "surface_tip_theory_meas_axis_contact_var",
                                        1.0e-2,
                                    )
                                    or 1.0e-2
                                ),
                                meas_geom_tip_var=float(
                                    a_cfg.get(
                                        "surface_tip_theory_meas_geom_tip_var",
                                        1.5e-5,
                                    )
                                    or 1.5e-5
                                ),
                                geom_tip_prior=bool(
                                    a_cfg.get(
                                        "surface_tip_theory_geom_prior", False
                                    )
                                )
                                or bool(tip_tec_joint),
                            )
                            tip_theory_ekf = (
                                ExtrinsicTipAxisEKF(_th_cfg)
                                if tip_axis_enable
                                else ExtrinsicTipStateEKF(_th_cfg)
                            )
                            _w0 = (
                                np.asarray(site_sp[0:3], dtype=np.float64)
                                .reshape(3)
                                .copy()
                            )
                            _cen0, _f0, _n0 = tip_contact_sensor.measure_peg_tray_tip(
                                raw, spiral_n
                            )
                            if tip_axis_enable:
                                assert isinstance(tip_theory_ekf, ExtrinsicTipAxisEKF)
                                _wa0 = _wrist_approach_axis(raw)
                                tip_theory_ekf.reset(
                                    wrist_pos=_w0,
                                    plane_n=spiral_n,
                                    tip_seed=_cen0 if (_cen0 is not None and _n0 > 0) else None,
                                    axis_seed=_wa0,
                                    wrist_rot=_right_wrist_site_pose(raw)[1],
                                )
                                tip_axis_hat = tip_theory_ekf.axis_world(spiral_n)
                            else:
                                tip_theory_ekf.reset(
                                    wrist_pos=_w0,
                                    plane_n=spiral_n,
                                    tip_seed=_cen0 if (_cen0 is not None and _n0 > 0) else None,
                                )
                            tip_est_wrist_prev = _w0
                        elif tip_est_enable:
                            tip_oracle_enable = True
                            tip_baseline_mode = "track_a_est_oracle"
                            tip_est = ContactTipEstimator(
                                raw,
                                ema=float(
                                    a_cfg.get("surface_tip_est_ema", 0.55) or 0.55
                                ),
                                min_force_n=float(
                                    a_cfg.get("surface_tip_est_min_force_n", 0.008)
                                    or 0.008
                                ),
                                seat_force_n=float(
                                    a_cfg.get("surface_tip_est_seat_force_n", 0.015)
                                    or 0.015
                                ),
                                outlier_gate_m=float(
                                    a_cfg.get("surface_tip_est_outlier_gate_m", 0.012)
                                    or 0.012
                                ),
                                holdover_frames=int(
                                    a_cfg.get("surface_tip_est_holdover_frames", 12)
                                    or 12
                                ),
                                cluster_radius_m=float(
                                    a_cfg.get(
                                        "surface_tip_est_cluster_radius_m", 0.008
                                    )
                                    or 0.008
                                ),
                                force_pow=float(
                                    a_cfg.get("surface_tip_est_force_pow", 1.5) or 1.5
                                ),
                                use_pca_locus=bool(
                                    a_cfg.get("surface_tip_est_use_pca", False)
                                ),
                                pca_min_contacts=int(
                                    a_cfg.get("surface_tip_est_pca_min_n", 3) or 3
                                ),
                                holdover_mode=str(
                                    a_cfg.get(
                                        "surface_tip_est_holdover_mode",
                                        "freeze_planar",
                                    )
                                    or "freeze_planar"
                                ),
                                holdover_decay=float(
                                    a_cfg.get("surface_tip_est_holdover_decay", 0.7)
                                    or 0.7
                                ),
                                seat_only_offset=bool(
                                    a_cfg.get("surface_tip_est_seat_only_offset", True)
                                ),
                                freeze_require_seat=bool(
                                    a_cfg.get(
                                        "surface_tip_est_freeze_require_seat", True
                                    )
                                ),
                            )
                            # Prefer contact seed; tip_sp0 only if no peg↔tray contact.
                            _allow_seed = bool(
                                a_cfg.get("surface_tip_est_allow_gt_seed", False)
                            )
                            tip_est.reset(
                                raw,
                                wrist_pos=site_sp[0:3],
                                plane_n=spiral_n,
                                tip_seed=tip_sp0 if _allow_seed else None,
                            )
                            tip_est_wrist_prev = (
                                np.asarray(site_sp[0:3], dtype=np.float64)
                                .reshape(3)
                                .copy()
                            )
                        elif tip_obs_enable:
                            # INVALID path (tip_gt+noise); kept for tree compatibility.
                            tip_oracle_enable = True
                            if tip_baseline_mode not in (
                                "priv_oracle_tip_servo",
                                "oracle_tip",
                                "oracle",
                                "priv_oracle_tip_obs",
                            ):
                                tip_baseline_mode = "priv_oracle_tip_obs"
                            elif tip_baseline_mode != "priv_oracle_tip_obs":
                                tip_baseline_mode = "priv_oracle_tip_obs"
                        if tip_oracle_enable and tip_baseline_mode not in (
                            "priv_oracle_tip_servo",
                            "oracle_tip",
                            "oracle",
                            "priv_oracle_tip_obs",
                            "track_a_est_oracle",
                            "est_oracle_tip_servo",
                            "est_oracle",
                            "track_a_theory_est",
                            "track_a_theory_ekf",
                            "theory_est_oracle",
                            "tec_slim_oracle",
                            "track_a_tip_axis",
                            "track_a_tip_axis_ekf",
                            "tip_axis_oracle",
                            "tec_tip_axis_oracle",
                        ):
                            tip_baseline_mode = "priv_oracle_tip_servo"
                        # Episode-stable tip-obs RNG (socket_off already episode-specific).
                        tip_obs_lag_buf: list[np.ndarray] = []
                        tip_obs_rng: np.random.Generator | None = None
                        if tip_obs_enable:
                            _obs_seed = int(a_cfg.get("surface_tip_obs_seed", 0) or 0)
                            _off = np.asarray(socket_off, dtype=np.float64).reshape(-1)
                            _mix = (
                                _obs_seed
                                + abs(int(np.round(float(_off[0]) * 1e9)))
                                + abs(int(np.round(float(_off[1]) * 1e7)))
                            )
                            tip_obs_rng = np.random.default_rng(int(_mix) & 0x7FFFFFFF)
                        tip_coupling_est: TipCouplingEstimator | None = None
                        tip_planar_coupling_est: PlanarCouplingEstimator | None = None
                        tip_coupling_ekf: CouplingEKF | None = None
                        _phig_mode = tip_baseline_mode in (
                            "track_b_phig",
                            "phig",
                            "track_b",
                            "path_hybrid_grasp",
                        )
                        _hard_hqp_mode = tip_baseline_mode in (
                            "fail_driven_hard_hqp",
                            "hard_hqp",
                            "track_b_hard_hqp",
                            "escande_hard_hqp",
                        )
                        _clep_mode = tip_baseline_mode in (
                            "track_b_clep",
                            "clep",
                            "contact_line_extrinsic_pivot",
                            "active_extrinsic_clep",
                        )
                        _fasr_mode = tip_baseline_mode in (
                            "track_b_fasr",
                            "fasr",
                            "force_asymmetry_spiral_recenter",
                            "force_guided_radial_seek",
                        )
                        _oigs_mode = tip_baseline_mode in (
                            "track_b_oigs",
                            "oigs",
                            "object_impedance_grasp_stiff",
                            "pfanne_grasp_stiff",
                        )
                        _gmhqp_mode = tip_baseline_mode in (
                            "track_b_gmhqp",
                            "gmhqp",
                            "grasp_map_hqp",
                            "grasp_map_operational_hqp",
                            "object_task_hqp",
                            "track_a_gmhqp_tip_obs",
                            "gmhqp_tip_obs",
                            "track_a_gmhqp",
                            "track_a_gmhqp_tip_fuse",
                            "gmhqp_tip_fuse",
                            "track_a_tec_joint_gmhqp",
                            "tec_joint_gmhqp",
                            "tec_joint",
                            "track_a_tec_joint",
                            "gmhqp_tec_joint",
                        ) or bool(tip_gmhqp_tip_obs)
                        _gmhqp_ftip_mode = tip_baseline_mode in (
                            "track_b_gmhqp_ftip",
                            "gmhqp_ftip",
                            "gmhqp_ft_contact",
                            "gmhqp_ft_tip",
                            "grasp_map_hqp_ftip",
                        )
                        _gmhqp_compc_mode = tip_baseline_mode in (
                            "track_b_gmhqp_compc",
                            "gmhqp_compc",
                            "gmhqp_contact_opt",
                            "track_b_contact_opt_gmhqp",
                            "contact_opt_on_gmhqp",
                        )
                        _gmhqp_ac_mode = tip_baseline_mode in (
                            "track_b_gmhqp_ac",
                            "gmhqp_ac",
                            "adaptive_c_gmhqp",
                            "track_b_adaptive_c",
                            "gmhqp_adaptive_c",
                        )
                        _hybrid_gmhqp_mode = tip_baseline_mode in (
                            "track_b_hybrid_gmhqp",
                            "hybrid_gmhqp",
                            "gmhqp_hybrid",
                            "tip_planar_c_hybrid_hqp",
                            "escande_hybrid_gmhqp",
                        )
                        _planar_c_mode = (
                            tip_baseline_mode
                            in (
                                "planar_c",
                                "planar_c_matrix",
                                "s8",
                                "ekf_qp",
                                "s9",
                            )
                            or (
                                bool(a_cfg.get("surface_tip_coupling_planar_c", False))
                                and tip_baseline_mode
                                not in (
                                    "priv_coupling",
                                    "coupling",
                                    "ours",
                                    "c",
                                    "priv_oracle_tip_servo",
                                    "oracle_tip",
                                    "oracle",
                                    "priv_oracle_tip_obs",
                                    "track_a_est_oracle",
                                    "est_oracle_tip_servo",
                                    "est_oracle",
                                    "track_a_theory_est",
                                    "track_a_theory_ekf",
                                    "theory_est_oracle",
                                    "tec_slim_oracle",
                                    "track_a_tip_axis",
                                    "track_a_tip_axis_ekf",
                                    "tip_axis_oracle",
                                    "tec_tip_axis_oracle",
                                    "track_b_phig",
                                    "phig",
                                    "track_b",
                                    "path_hybrid_grasp",
                                    "fail_driven_hard_hqp",
                                    "hard_hqp",
                                    "track_b_hard_hqp",
                                    "escande_hard_hqp",
                                    "track_b_clep",
                                    "clep",
                                    "contact_line_extrinsic_pivot",
                                    "active_extrinsic_clep",
                                    "track_b_fasr",
                                    "fasr",
                                    "force_asymmetry_spiral_recenter",
                                    "force_guided_radial_seek",
                                    "track_b_oigs",
                                    "oigs",
                                    "object_impedance_grasp_stiff",
                                    "pfanne_grasp_stiff",
                                    "track_b_gmhqp",
                                    "gmhqp",
                                    "grasp_map_hqp",
                                    "grasp_map_operational_hqp",
                                    "object_task_hqp",
                                    "track_a_gmhqp_tip_obs",
                                    "gmhqp_tip_obs",
                                    "track_a_gmhqp",
                                    "track_a_gmhqp_tip_fuse",
                                    "gmhqp_tip_fuse",
                                    "track_a_tec_joint_gmhqp",
                                    "tec_joint_gmhqp",
                                    "tec_joint",
                                    "track_a_tec_joint",
                                    "gmhqp_tec_joint",
                                    "track_b_gmhqp_ftip",
                                    "gmhqp_ftip",
                                    "gmhqp_ft_contact",
                                    "gmhqp_ft_tip",
                                    "grasp_map_hqp_ftip",
                                    "track_b_gmhqp_compc",
                                    "gmhqp_compc",
                                    "gmhqp_contact_opt",
                                    "track_b_contact_opt_gmhqp",
                                    "contact_opt_on_gmhqp",
                                    "track_b_gmhqp_ac",
                                    "gmhqp_ac",
                                    "adaptive_c_gmhqp",
                                    "track_b_adaptive_c",
                                    "gmhqp_adaptive_c",
                                    "track_b_hybrid_gmhqp",
                                    "hybrid_gmhqp",
                                    "gmhqp_hybrid",
                                    "tip_planar_c_hybrid_hqp",
                                    "escande_hybrid_gmhqp",
                                )
                            )
                        ) and (not tip_oracle_enable) and (not _phig_mode) and (
                            not _hard_hqp_mode
                        ) and (not _clep_mode) and (not _fasr_mode) and (
                            not _oigs_mode
                        ) and (not _gmhqp_mode) and (not _gmhqp_ftip_mode) and (
                            not _gmhqp_compc_mode
                        ) and (not _gmhqp_ac_mode) and (
                            not _hybrid_gmhqp_mode
                        )
                        if tip_baseline_mode in ("ekf_qp", "s9"):
                            tip_coupling_ekf = CouplingEKF(
                                slip_ratio=float(
                                    a_cfg.get("surface_tip_ekf_slip_ratio", 0.25)
                                ),
                                slip_dw_m=float(
                                    a_cfg.get(
                                        "surface_tip_coupling_slip_reset_dw_m",
                                        0.0012,
                                    )
                                ),
                                alpha_floor=float(
                                    a_cfg.get(
                                        "surface_tip_coupling_alpha_floor", 0.18
                                    )
                                ),
                                alpha_ceil=float(
                                    a_cfg.get("surface_tip_coupling_alpha_ceil", 1.05)
                                ),
                            )
                            tip_coupling_ekf.reset(site_sp[0:3], tip_sp0, spiral_n)
                        elif _planar_c_mode:
                            tip_planar_coupling_est = PlanarCouplingEstimator(
                                window=int(
                                    a_cfg.get("surface_tip_coupling_window", 24)
                                ),
                                c_reg=float(
                                    a_cfg.get("surface_tip_coupling_c_reg", 0.15)
                                ),
                                alpha_floor=float(
                                    a_cfg.get(
                                        "surface_tip_coupling_alpha_floor", 0.18
                                    )
                                ),
                            )
                            tip_planar_coupling_est.reset(site_sp[0:3], tip_sp0)
                            if tip_baseline_mode not in (
                                "planar_c",
                                "planar_c_matrix",
                                "s8",
                            ):
                                tip_baseline_mode = "planar_c"
                        elif tip_baseline_mode in (
                            "priv_coupling",
                            "coupling",
                            "ours",
                            "c",
                            "planar_coupling",
                            "s7",
                        ):
                            tip_coupling_est = TipCouplingEstimator(
                                window=int(
                                    a_cfg.get("surface_tip_coupling_window", 24)
                                ),
                                ema_offset=float(
                                    a_cfg.get("surface_tip_coupling_ema", 0.88)
                                ),
                                alpha_floor=float(
                                    a_cfg.get(
                                        "surface_tip_coupling_alpha_floor", 0.18
                                    )
                                ),
                            )
                            tip_coupling_est.reset(site_sp[0:3], tip_sp0)
                        search_mode = str(
                            a_cfg.get("surface_tip_search_mode", "spiral")
                        ).strip().lower()
                        _priv_lat_modes = ("priv_lat", "priv_lat_lissajous")
                        if search_mode not in (
                            "spiral",
                            "planned_spiral",
                            "conntact_time_spiral",
                            "franka_spiral",
                            "psft_spiral",
                            "lissajous",
                            *_priv_lat_modes,
                        ):
                            search_mode = "spiral"
                        conntact_faithful = search_mode == "conntact_time_spiral"
                        franka_faithful = search_mode == "franka_spiral"
                        psft_faithful = search_mode == "psft_spiral"
                        conntact_params = ConnTactSpiralParams(
                            frequency_hz=float(
                                a_cfg.get("surface_conntact_frequency_hz", 0.15)
                            ),
                            min_amplitude_m=float(
                                a_cfg.get("surface_conntact_min_amplitude_m", 0.002)
                            ),
                            max_cycles=float(
                                a_cfg.get("surface_conntact_max_cycles", 62.83)
                            ),
                            safe_clearance_m=float(
                                a_cfg.get(
                                    "surface_conntact_safe_clearance_m",
                                    0.02 / 100.0,
                                )
                            ),
                            height_drop_m=float(
                                a_cfg.get("surface_conntact_height_drop_m", 0.0004)
                            ),
                        )
                        conntact_dt = float(
                            a_cfg.get("surface_conntact_dt_s", 1.0 / 30.0)
                        )
                        conntact_surface_h: float | None = None
                        conntact_height_enter = bool(
                            a_cfg.get("surface_conntact_height_enter", True)
                        )
                        # Soft-sim force scale: ref FORCE_THRESHOLD_N=2.0; default
                        # 0.05 matches planar jam/stuck band (disclose in doc).
                        franka_params = FrankaSpiralParams(
                            force_threshold_n=float(
                                a_cfg.get("surface_franka_force_threshold_n", 0.08)
                            ),
                            r_base_m=float(
                                a_cfg.get("surface_franka_r_base_m", 0.0002)
                            ),
                            r_growth_m=float(
                                a_cfg.get("surface_franka_r_growth_m", 0.0002)
                            ),
                            r_max_m=float(
                                a_cfg.get("surface_franka_r_max_m", 0.015)
                            ),
                            omega_rad=float(
                                a_cfg.get("surface_franka_omega_rad", 0.4)
                            ),
                            jam_lift_above_hole_m=float(
                                a_cfg.get("surface_franka_jam_lift_m", 0.040)
                            ),
                            insert_z_ramp_per_step_m=float(
                                a_cfg.get(
                                    "surface_franka_insert_z_ramp_m", 0.0005
                                )
                            ),
                            height_drop_m=float(
                                a_cfg.get("surface_franka_height_drop_m", 0.0004)
                            ),
                        )
                        franka_state = FrankaSpiralState()
                        franka_surface_h: float | None = None
                        franka_height_enter = bool(
                            a_cfg.get("surface_franka_height_enter", True)
                        )
                        franka_meta_last: dict = {}
                        if franka_faithful and not tip_baseline_mode:
                            tip_baseline_mode = "franka_tcp"
                        # Park RA-L 2020 PSFT (paper reimpl; no official code).
                        psft_params = PSFTParams(
                            h_m=float(a_cfg.get("surface_psft_h_m", 0.0005)),
                            d_m=float(a_cfg.get("surface_psft_d_m", 0.002)),
                            r_min_m=float(a_cfg.get("surface_psft_r_min_m", 0.001)),
                            r_max_m=float(a_cfg.get("surface_psft_r_max_m", 0.018)),
                            theta_max_rad=float(
                                a_cfg.get("surface_psft_theta_max_rad", 0.5 * np.pi)
                            ),
                            k_p=float(a_cfg.get("surface_psft_k_p", 300.0)),
                            k_omega=float(a_cfg.get("surface_psft_k_omega", 3.0)),
                            f_a_n=float(a_cfg.get("surface_psft_f_a_n", 7.0)),
                            tilt_rad=float(a_cfg.get("surface_psft_tilt_rad", 0.17)),
                            dt_s=float(a_cfg.get("surface_psft_dt_s", 0.003)),
                            height_drop_m=float(
                                a_cfg.get("surface_psft_height_drop_m", 0.0004)
                            ),
                        )
                        psft_state = PSFTState()
                        psft_state.reset(r_min_m=float(psft_params.r_min_m))
                        psft_surface_h: float | None = None
                        psft_height_enter = bool(
                            a_cfg.get("surface_psft_height_enter", True)
                        )
                        psft_meta_last: dict = {}
                        if psft_faithful and not tip_baseline_mode:
                            tip_baseline_mode = "psft_tcp"
                        n_sp = int(a_cfg.get("surface_spiral_frames", 180))
                        if search_mode in _priv_lat_modes:
                            n_search = int(
                                a_cfg.get("surface_priv_lat_frames", 480)
                            )
                        elif search_mode == "lissajous":
                            n_search = int(
                                a_cfg.get("surface_lissajous_frames", n_sp)
                            )
                        else:
                            n_search = n_sp
                        # Lissajous/priv_lat default to socket center; tip-centered
                        # ablation keeps pattern on current tip (no hole seek center).
                        want_tip_center = (
                            str(a_cfg.get("surface_tip_spiral_center", "tip"))
                            .strip()
                            .lower()
                            == "tip"
                        )
                        if (
                            search_mode in ("lissajous", *_priv_lat_modes)
                            or conntact_faithful
                            or franka_faithful
                            or psft_faithful
                        ):
                            if (
                                want_tip_center
                                and search_mode == "lissajous"
                                and not conntact_faithful
                                and not franka_faithful
                                and not psft_faithful
                            ):
                                spiral_center = tip_sp0.copy()
                                spiral_center_mode = "tip"
                            else:
                                # ConnTact/Franka/PSFT center spiral on known hole XY.
                                spiral_center = _hole_axis_point_at_tip(
                                    tip_sp0,
                                    feat_sp0.socket_pos,
                                    feat_sp0.hole_axis,
                                )
                                spiral_center_mode = "socket"
                        if conntact_faithful:
                            # ConnTact exit uses TCP world-Z vs surface_height.
                            conntact_surface_h = float(tip_sp0[2])
                            print(
                                f"pci: ConnTact faithful time-spiral "
                                f"f={conntact_params.frequency_hz}Hz "
                                f"amp0={conntact_params.min_amplitude_m*1e3:.2f}mm "
                                f"clear={conntact_params.safe_clearance_m*1e3:.3f}mm "
                                f"drop={conntact_params.height_drop_m*1e3:.2f}mm "
                                f"surface_z={conntact_surface_h:.5f}",
                                flush=True,
                            )
                        if franka_faithful:
                            franka_surface_h = float(tip_sp0[2])
                            franka_state.reset()
                            print(
                                f"pci: Franka faithful jam-spiral "
                                f"Fth={franka_params.force_threshold_n:.3f}N "
                                f"r0={franka_params.r_base_m*1e3:.2f}mm "
                                f"dr={franka_params.r_growth_m*1e3:.2f}mm "
                                f"rmax={franka_params.r_max_m*1e3:.1f}mm "
                                f"ω={franka_params.omega_rad:.2f} "
                                f"lift={franka_params.jam_lift_above_hole_m*1e3:.1f}mm "
                                f"surface_z={franka_surface_h:.5f}",
                                flush=True,
                            )
                        if psft_faithful:
                            psft_surface_h = float(tip_sp0[2])
                            psft_state.reset(r_min_m=float(psft_params.r_min_m))
                            print(
                                f"pci: PSFT paper-reimpl (no official code) "
                                f"θmax={psft_params.theta_max_rad:.3f} "
                                f"h={psft_params.h_m*1e3:.2f}mm "
                                f"d={psft_params.d_m*1e3:.2f}mm "
                                f"r=[{psft_params.r_min_m*1e3:.1f},"
                                f"{psft_params.r_max_m*1e3:.1f}]mm "
                                f"kp={psft_params.k_p:.0f} "
                                f"fa_paper={psft_params.f_a_n:.1f}N "
                                f"surface_z={psft_surface_h:.5f}",
                                flush=True,
                            )
                        mouth_lat = float(
                            a_cfg.get("surface_spiral_mouth_lat_m", 0.0045)
                        )
                        max_tilt_sp = float(
                            a_cfg.get(
                                "surface_spiral_max_tilt_deg",
                                a_cfg.get("surface_const_force_max_tilt_deg", 8.0),
                            )
                        )
                        pitch = float(
                            a_cfg.get(
                                "surface_tip_spiral_pitch_m",
                                cfg.get("compliant", {})
                                .get("search", {})
                                .get("spiral_pitch_m", 0.0025),
                            )
                        )
                        dtheta = float(a_cfg.get("surface_tip_spiral_step_rad", 0.16))
                        r0_cfg = float(a_cfg.get("surface_tip_spiral_r0_m", 0.002))
                        r0 = r0_cfg
                        r_max = float(a_cfg.get("surface_tip_spiral_rmax_m", 0.018))
                        r_min = float(
                            a_cfg.get(
                                "surface_tip_spiral_rmin_m",
                                a_cfg.get("surface_spiral_mouth_lat_m", 0.0045),
                            )
                        )
                        spiral_dir = str(
                            a_cfg.get("surface_tip_spiral_direction", "outward")
                        ).strip().lower()
                        if spiral_dir not in ("outward", "inward"):
                            spiral_dir = "outward"
                        track = float(a_cfg.get("surface_tip_spiral_track", 0.9))
                        max_xy = float(a_cfg.get("surface_tip_spiral_max_step_m", 0.0035))
                        left_share_sp = float(
                            a_cfg.get("surface_spiral_left_share", 0.05)
                        )
                        # ConnTact uses cartesian compliance while spiraling; we
                        # mirror with dual-arm yield: left shares planar motion +
                        # wrist admittance so the tray gives instead of stiction.
                        if search_mode == "planned_spiral":
                            # No open-loop share: left yields only via wrist FT admit.
                            left_share_sp = float(
                                a_cfg.get("surface_planned_spiral_left_share", 0.0)
                            )
                        spiral_left_admit_scale = float(
                            a_cfg.get(
                                "surface_spiral_left_admit_scale",
                                1.0 if search_mode == "planned_spiral" else 0.0,
                            )
                        )
                        spiral_admit: LeftWristAdmitController | None = None
                        if spiral_left_admit_scale > 1e-9:
                            spiral_admit = LeftWristAdmitController(
                                _left_wrist_admit_config(cfg)
                            )
                            wr_ad0 = read_wrist_wrench_world(raw)
                            spiral_admit.reset(wr_ad0[1])
                        spiral_lat_min = float(feat_sp0.lateral_m)
                        spiral_resid_peak = 0.0
                        spiral_reason = "spiral_timeout"
                        tip_xy_peak = 0.0
                        # Honest seat/orientation audit during tip spiral search.
                        spiral_tip_float_peak_m = 0.0
                        spiral_axis_err_peak_deg = 0.0
                        spiral_contact_ok_n = 0
                        spiral_contact_frames = 0
                        spiral_early_contact_ok_n = 0
                        spiral_early_frames_n = int(
                            a_cfg.get(
                                "surface_planned_priv_enter_early_frames", 60
                            )
                        )
                        seat_contact_n = float(
                            a_cfg.get("surface_planned_seat_contact_n", 0.08)
                        )
                        require_seat_search = bool(
                            a_cfg.get("surface_spiral_require_seat", True)
                        )
                        off0_xy = _tip_lat_offset_xy(
                            tip_sp0, spiral_center, spiral_n
                        )
                        r_start = float(np.linalg.norm(off0_xy))
                        theta_sp = (
                            float(np.arctan2(float(off0_xy @ t2), float(off0_xy @ t1)))
                            if r_start > 1e-9
                            else 0.0
                        )
                        if spiral_center_mode == "socket" and r_start > r0:
                            r0 = r_start
                        pre_lat_axis = _hole_axis_point_at_tip(
                            tip_sp0, feat_sp0.socket_pos, feat_sp0.hole_axis
                        )
                        pre_lat_n = int(a_cfg.get("surface_spiral_pre_lat_frames", 0))
                        pre_lat_step = float(
                            a_cfg.get("surface_spiral_pre_lat_step_m", 0.002)
                        )
                        pre_lat_used = 0
                        lis_wx = float(a_cfg.get("surface_lissajous_wx", 0.14))
                        lis_wy = float(a_cfg.get("surface_lissajous_wy", 0.19))
                        lis_phi = float(a_cfg.get("surface_lissajous_phase_rad", 1.5708))
                        lis_ay_ratio = float(a_cfg.get("surface_lissajous_ay_ratio", 0.75))
                        lis_shrink = bool(
                            a_cfg.get("surface_lissajous_shrink_enable", True)
                        )
                        lis_amin = float(
                            a_cfg.get(
                                "surface_lissajous_amin_m",
                                a_cfg.get("surface_spiral_mouth_lat_m", 0.0045),
                            )
                        )
                        lis_ax0 = float(a_cfg.get("surface_lissajous_ax_m", 0.0))
                        lis_ay0 = float(a_cfg.get("surface_lissajous_ay_m", 0.0))
                        priv_lat_step = float(
                            a_cfg.get("surface_priv_lat_step_m", 0.002)
                        )
                        priv_lat_axis_ff = float(
                            a_cfg.get("surface_priv_lat_axis_ff", 0.0)
                        )
                        priv_dith_ax = float(
                            a_cfg.get("surface_priv_lat_dither_ax_m", 0.004)
                        )
                        priv_dith_ay = float(
                            a_cfg.get(
                                "surface_priv_lat_dither_ay_m",
                                priv_dith_ax * lis_ay_ratio,
                            )
                        )
                        priv_dith_lat0 = float(
                            a_cfg.get("surface_priv_lat_dither_lat_scale_m", 0.012)
                        )
                        priv_lat_along_hold = bool(
                            a_cfg.get("surface_priv_lat_along_hold", False)
                        )
                        priv_lat_along_hold_max = float(
                            a_cfg.get(
                                "surface_priv_lat_along_hold_max_m",
                                a_cfg.get("mouth_max_along_m", 0.102),
                            )
                        )
                        # Tip/socket spiral: keep search on the contact plane (ConnTact).
                        spiral_along_hold = bool(
                            a_cfg.get("surface_spiral_along_hold", True)
                        )
                        along_hold_m: float | None = None
                        stick_enable = bool(
                            a_cfg.get("surface_priv_lat_stick_enable", True)
                        )
                        stick_win = int(
                            a_cfg.get("surface_priv_lat_stick_window", 20)
                        )
                        stick_tip_m = float(
                            a_cfg.get("surface_priv_lat_stick_tip_move_m", 0.0004)
                        )
                        stick_lift_gain = float(
                            a_cfg.get(
                                "surface_priv_lat_stick_lift_gain_m_per_rad", 0.004
                            )
                        )
                        stick_lat_lift = float(
                            a_cfg.get("surface_priv_lat_stick_lat_lift_gain", 0.02)
                        )
                        stick_max_lift = float(
                            a_cfg.get("surface_priv_lat_stick_max_lift_m", 0.0012)
                        )
                        stick_unload_gain = float(
                            a_cfg.get(
                                "surface_priv_lat_stick_unload_gain_n_per_rad", 0.8
                            )
                        )
                        stick_max_unload = float(
                            a_cfg.get("surface_priv_lat_stick_max_unload_n", 0.12)
                        )
                        stick_hop_every = int(
                            a_cfg.get("surface_priv_lat_stick_hop_every", 40)
                        )
                        stick_hop_lift = float(
                            a_cfg.get("surface_priv_lat_stick_hop_lift_m", 0.0025)
                        )
                        stick_lift_frames = int(
                            a_cfg.get("surface_priv_lat_stick_lift_frames", 18)
                        )
                        stick_slide_frames = int(
                            a_cfg.get("surface_priv_lat_stick_slide_frames", 30)
                        )
                        stick_slide_step = float(
                            a_cfg.get("surface_priv_lat_stick_slide_step_m", 0.004)
                        )
                        stick_press_frames = int(
                            a_cfg.get("surface_priv_lat_stick_press_frames", 16)
                        )
                        stick_hop_probe = bool(
                            a_cfg.get("surface_priv_lat_hop_probe", False)
                        )
                        stick_clear_lift = float(
                            a_cfg.get("surface_priv_lat_stick_clear_lift_m", 0.003)
                        )
                        stick_clear_n = float(
                            a_cfg.get("surface_priv_lat_stick_clear_contact_n", 0.06)
                        )
                        tip_hist_sp: list[np.ndarray] = []
                        # Privileged planned on-surface enter spiral (FF waypoints).
                        planned_wps = None
                        planned_rs = None
                        planned_ths = None
                        planned_i = 0
                        catch_miss_sp = 0
                        along0_spiral = float(feat_sp0.along_m)
                        priv_mouth_streak = 0
                        priv_enter_streak = 0
                        priv_hole_entered = False
                        priv_at_mouth_peak = False
                        force_hole_detect = bool(
                            a_cfg.get("surface_planned_force_hole_detect", False)
                        )
                        tip_hybrid_enable = bool(
                            a_cfg.get("surface_tip_hybrid_enable", True)
                        )
                        tip_hybrid_state = TipHybridState()
                        tip_hybrid_omega = np.zeros(3, dtype=np.float64)
                        tip_hybrid_soft_seat = False
                        tip_hybrid_mode_counts = {"seat": 0, "upright": 0, "spiral": 0}
                        tip_msar_enable = bool(a_cfg.get("surface_tip_msar_enable", False))
                        tip_msar_counts = {
                            "mouth_seek": 0,
                            "slip_reseat": 0,
                            "spiral_track": 0,
                        }
                        # Tip-STAR supersedes MSAR when enabled (no privileged mouth PD).
                        tip_star_enable = bool(
                            a_cfg.get("surface_tip_star_enable", False)
                        )
                        # Deployable Type-B slip restore: can run alone or under STAR.
                        tip_slip_enable = bool(
                            a_cfg.get("surface_tip_slip_enable", False)
                        ) or tip_star_enable
                        if tip_star_enable:
                            tip_msar_enable = False
                        # Track-B PHIG: wrist-path + deployable mouth/slip (no tip GT).
                        tip_phig_enable = bool(
                            a_cfg.get("surface_tip_phig_enable", False)
                        ) or bool(_phig_mode)
                        phig_state = PhigState()
                        phig_cfg = PhigConfig(
                            k_path=float(a_cfg.get("surface_tip_phig_k_path", 1.6)),
                            k_tang_admit=float(
                                a_cfg.get("surface_tip_phig_k_tang_admit", 0.00025)
                            ),
                            max_step_m=float(
                                a_cfg.get(
                                    "surface_tip_phig_max_step_m",
                                    a_cfg.get(
                                        "surface_planned_spiral_max_wrist_step_m",
                                        0.012,
                                    ),
                                )
                            ),
                            f_seat_n=float(
                                a_cfg.get(
                                    "surface_tip_hybrid_seat_contact_n", 0.05
                                )
                            ),
                            f_mouth_press_n=float(
                                a_cfg.get("surface_tip_phig_f_mouth_press_n", 0.08)
                            ),
                            mouth_r_m=float(
                                a_cfg.get("surface_tip_phig_mouth_r_m", 0.0045)
                            ),
                            mouth_press_m=float(
                                a_cfg.get("surface_tip_phig_mouth_press_m", 0.00045)
                            ),
                            mouth_hard_unload_n=float(
                                a_cfg.get(
                                    "surface_tip_phig_mouth_hard_unload_n", 0.90
                                )
                            ),
                            slip_tau_m=float(
                                a_cfg.get("surface_tip_phig_slip_tau_m", 0.12)
                            ),
                            reseat_frames=int(
                                a_cfg.get("surface_tip_phig_reseat_frames", 40)
                            ),
                            retrip_delta_m=float(
                                a_cfg.get("surface_tip_phig_retrip_delta_m", 0.05)
                            ),
                            grasp_scale=float(
                                a_cfg.get("surface_tip_phig_grasp_scale", 1.55)
                            ),
                            hold_enter_max_axis_deg=float(
                                a_cfg.get(
                                    "surface_tip_phig_hold_enter_max_axis_deg",
                                    a_cfg.get(
                                        "surface_planned_priv_enter_max_axis_err_deg",
                                        25.0,
                                    ),
                                )
                            ),
                        )
                        # Fail-driven Track-B HardHQP (seat ≻ upright ≻ path).
                        tip_hard_hqp_enable = bool(
                            a_cfg.get("surface_tip_hard_hqp_enable", False)
                        ) or bool(_hard_hqp_mode)
                        hard_hqp_cfg = HardHqpConfig(
                            f_seat_n=float(
                                a_cfg.get(
                                    "surface_tip_hard_hqp_f_seat_n",
                                    a_cfg.get(
                                        "surface_tip_hybrid_seat_contact_n", 0.05
                                    ),
                                )
                            ),
                            seat_press_m=float(
                                a_cfg.get(
                                    "surface_tip_hard_hqp_seat_press_m",
                                    a_cfg.get(
                                        "surface_tip_hybrid_seat_press_m", 0.00045
                                    ),
                                )
                            ),
                            upright_soft_deg=float(
                                a_cfg.get(
                                    "surface_tip_hard_hqp_upright_soft_deg",
                                    a_cfg.get(
                                        "surface_tip_hybrid_upright_soft_deg", 16.0
                                    ),
                                )
                            ),
                            upright_step_m=float(
                                a_cfg.get(
                                    "surface_tip_hard_hqp_upright_step_m", 0.004
                                )
                            ),
                            path_k=float(
                                a_cfg.get("surface_tip_hard_hqp_path_k", 1.4)
                            ),
                            max_step_m=float(
                                a_cfg.get(
                                    "surface_tip_hard_hqp_max_step_m",
                                    a_cfg.get(
                                        "surface_planned_spiral_max_wrist_step_m",
                                        0.012,
                                    ),
                                )
                            ),
                            mouth_r_m=float(
                                a_cfg.get(
                                    "surface_tip_hard_hqp_mouth_r_m", 0.0045
                                )
                            ),
                            mouth_press_m=float(
                                a_cfg.get(
                                    "surface_tip_hard_hqp_mouth_press_m", 0.00045
                                )
                            ),
                            path_when_hard=float(
                                a_cfg.get(
                                    "surface_tip_hard_hqp_path_when_hard", 0.0
                                )
                            ),
                        )
                        # Track-B CLEP: F/T contact-line extrinsic pivot (no tip GT).
                        tip_clep_enable = bool(
                            a_cfg.get("surface_tip_clep_enable", False)
                        ) or bool(_clep_mode)
                        clep_state = ClepState()
                        clep_cfg = ClepConfig(
                            k_contact=float(
                                a_cfg.get("surface_tip_clep_k_contact", 1.8)
                            ),
                            k_wrist_fallback=float(
                                a_cfg.get("surface_tip_clep_k_wrist_fallback", 0.6)
                            ),
                            max_step_m=float(
                                a_cfg.get(
                                    "surface_tip_clep_max_step_m",
                                    a_cfg.get(
                                        "surface_planned_spiral_max_wrist_step_m",
                                        0.012,
                                    ),
                                )
                            ),
                            ema=float(a_cfg.get("surface_tip_clep_ema", 0.55)),
                            f_min_n=float(a_cfg.get("surface_tip_clep_f_min_n", 0.02)),
                            f_seat_n=float(
                                a_cfg.get(
                                    "surface_tip_clep_f_seat_n",
                                    a_cfg.get(
                                        "surface_tip_hybrid_seat_contact_n", 0.05
                                    ),
                                )
                            ),
                            seat_press_m=float(
                                a_cfg.get(
                                    "surface_tip_clep_seat_press_m",
                                    a_cfg.get(
                                        "surface_tip_hybrid_seat_press_m", 0.00045
                                    ),
                                )
                            ),
                            seat_path_scale=float(
                                a_cfg.get("surface_tip_clep_seat_path_scale", 0.15)
                            ),
                            mouth_r_m=float(
                                a_cfg.get("surface_tip_clep_mouth_r_m", 0.0045)
                            ),
                            mouth_press_m=float(
                                a_cfg.get("surface_tip_clep_mouth_press_m", 0.00045)
                            ),
                            f_mouth_press_n=float(
                                a_cfg.get("surface_tip_clep_f_mouth_press_n", 0.08)
                            ),
                            max_contact_offset_m=float(
                                a_cfg.get(
                                    "surface_tip_clep_max_contact_offset_m", 0.08
                                )
                            ),
                            tang_gate_n=float(
                                a_cfg.get("surface_tip_clep_tang_gate_n", 0.55)
                            ),
                            tang_path_scale=float(
                                a_cfg.get("surface_tip_clep_tang_path_scale", 0.35)
                            ),
                        )
                        # Track-B FASR: F/T edge bias + spiral-center recenter (F1).
                        tip_fasr_enable = bool(
                            a_cfg.get("surface_tip_fasr_enable", False)
                        ) or bool(_fasr_mode)
                        fasr_state = FasrState()
                        fasr_cfg = FasrConfig(
                            k_path=float(a_cfg.get("surface_tip_fasr_k_path", 1.5)),
                            k_force=float(a_cfg.get("surface_tip_fasr_k_force", 0.004)),
                            f_edge_n=float(a_cfg.get("surface_tip_fasr_f_edge_n", 0.08)),
                            f_seat_n=float(
                                a_cfg.get(
                                    "surface_tip_fasr_f_seat_n",
                                    a_cfg.get(
                                        "surface_tip_hybrid_seat_contact_n", 0.05
                                    ),
                                )
                            ),
                            seat_press_m=float(
                                a_cfg.get(
                                    "surface_tip_fasr_seat_press_m",
                                    a_cfg.get(
                                        "surface_tip_hybrid_seat_press_m", 0.00045
                                    ),
                                )
                            ),
                            seat_path_scale=float(
                                a_cfg.get("surface_tip_fasr_seat_path_scale", 0.25)
                            ),
                            max_step_m=float(
                                a_cfg.get(
                                    "surface_tip_fasr_max_step_m",
                                    a_cfg.get(
                                        "surface_planned_spiral_max_wrist_step_m",
                                        0.012,
                                    ),
                                )
                            ),
                            recenter_ema=float(
                                a_cfg.get("surface_tip_fasr_recenter_ema", 0.08)
                            ),
                            recenter_step_m=float(
                                a_cfg.get("surface_tip_fasr_recenter_step_m", 0.00035)
                            ),
                            recenter_cap_m=float(
                                a_cfg.get("surface_tip_fasr_recenter_cap_m", 0.012)
                            ),
                            mouth_r_m=float(
                                a_cfg.get("surface_tip_fasr_mouth_r_m", 0.006)
                            ),
                            f_mouth_drop_n=float(
                                a_cfg.get("surface_tip_fasr_f_mouth_drop_n", 0.035)
                            ),
                            mouth_press_m=float(
                                a_cfg.get("surface_tip_fasr_mouth_press_m", 0.0005)
                            ),
                            mouth_path_scale=float(
                                a_cfg.get("surface_tip_fasr_mouth_path_scale", 0.4)
                            ),
                        )
                        # Track-B OIGS: refresh equation knobs after baseline parse.
                        tip_oigs_enable = bool(
                            a_cfg.get("surface_tip_oigs_enable", False)
                        ) or bool(_oigs_mode)
                        oigs_cfg = OigsConfig(
                            k_path=float(a_cfg.get("surface_tip_oigs_k_path", 1.4)),
                            max_step_m=float(
                                a_cfg.get(
                                    "surface_tip_oigs_max_step_m",
                                    a_cfg.get(
                                        "surface_planned_spiral_max_wrist_step_m",
                                        0.012,
                                    ),
                                )
                            ),
                            e_rigid_rad=float(
                                a_cfg.get("surface_tip_oigs_e_rigid_rad", 0.12)
                            ),
                            e_rigid_pos_m=float(
                                a_cfg.get("surface_tip_oigs_e_rigid_pos_m", 0.008)
                            ),
                            f_seat_n=float(
                                a_cfg.get(
                                    "surface_tip_oigs_f_seat_n",
                                    a_cfg.get(
                                        "surface_tip_hybrid_seat_contact_n", 0.05
                                    ),
                                )
                            ),
                            seat_press_m=float(
                                a_cfg.get(
                                    "surface_tip_oigs_seat_press_m",
                                    a_cfg.get(
                                        "surface_tip_hybrid_seat_press_m", 0.00045
                                    ),
                                )
                            ),
                            mouth_r_m=float(
                                a_cfg.get("surface_tip_oigs_mouth_r_m", 0.0045)
                            ),
                            mouth_press_m=float(
                                a_cfg.get("surface_tip_oigs_mouth_press_m", 0.0004)
                            ),
                            alpha_floor=float(
                                a_cfg.get("surface_tip_oigs_alpha_floor", 0.05)
                            ),
                        )
                        if tip_oigs_enable:
                            meta["tip_oigs_enable"] = True
                            meta["tip_oigs_theory"] = "pfanne_RA-L_2020+graspqp"
                        # Track-B GMHQP: tip-task through grasp-map C (Escande HQP).
                        tip_gmhqp_enable = bool(
                            a_cfg.get("surface_tip_gmhqp_enable", False)
                        ) or bool(_gmhqp_mode)
                        gmhqp_cfg = GmhqpConfig(
                            k_tip=float(a_cfg.get("surface_tip_gmhqp_k_tip", 1.4)),
                            max_step_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_max_step_m",
                                    a_cfg.get(
                                        "surface_planned_spiral_max_wrist_step_m",
                                        0.012,
                                    ),
                                )
                            ),
                            f_seat_n=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_f_seat_n",
                                    a_cfg.get(
                                        "surface_tip_hybrid_seat_contact_n", 0.05
                                    ),
                                )
                            ),
                            seat_press_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_seat_press_m",
                                    a_cfg.get(
                                        "surface_tip_hybrid_seat_press_m", 0.00045
                                    ),
                                )
                            ),
                            upright_soft_deg=float(
                                a_cfg.get("surface_tip_gmhqp_upright_soft_deg", 16.0)
                            ),
                            upright_step_m=float(
                                a_cfg.get("surface_tip_gmhqp_upright_step_m", 0.004)
                            ),
                            mouth_r_m=float(
                                a_cfg.get("surface_tip_gmhqp_mouth_r_m", 0.0045)
                            ),
                            mouth_press_m=float(
                                a_cfg.get("surface_tip_gmhqp_mouth_press_m", 0.0004)
                            ),
                            c_ridge=float(
                                a_cfg.get("surface_tip_gmhqp_c_ridge", 0.35)
                            ),
                            c_ema=float(a_cfg.get("surface_tip_gmhqp_c_ema", 0.85)),
                        )
                        if tip_gmhqp_enable:
                            meta["tip_gmhqp_enable"] = True
                            meta["tip_gmhqp_theory"] = (
                                "montana_grasp_map+pfanne+escande_hqp"
                            )
                            # Pose-hold GraspQP needed for finger side of G.
                            if tip_search_pose_hold and use_priv_grasp_a:
                                meta["tip_gmhqp_pose_hold"] = True
                        # Track-B GMHQP-FTIP: FT extrinsic tip → same GMHQP law (REUSE).
                        tip_gmhqp_ftip_enable = bool(
                            a_cfg.get("surface_tip_gmhqp_ftip_enable", False)
                        ) or bool(_gmhqp_ftip_mode)
                        gmhqp_ftip_state = GmhqpFtipState()
                        gmhqp_ftip_cfg = GmhqpFtipConfig(
                            k_tip=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_ftip_k_tip",
                                    a_cfg.get("surface_tip_gmhqp_k_tip", 1.4),
                                )
                            ),
                            max_step_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_ftip_max_step_m",
                                    a_cfg.get(
                                        "surface_tip_gmhqp_max_step_m",
                                        a_cfg.get(
                                            "surface_planned_spiral_max_wrist_step_m",
                                            0.012,
                                        ),
                                    ),
                                )
                            ),
                            f_seat_n=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_ftip_f_seat_n",
                                    a_cfg.get(
                                        "surface_tip_gmhqp_f_seat_n",
                                        a_cfg.get(
                                            "surface_tip_hybrid_seat_contact_n", 0.05
                                        ),
                                    ),
                                )
                            ),
                            seat_press_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_ftip_seat_press_m",
                                    a_cfg.get(
                                        "surface_tip_gmhqp_seat_press_m",
                                        a_cfg.get(
                                            "surface_tip_hybrid_seat_press_m", 0.00045
                                        ),
                                    ),
                                )
                            ),
                            upright_soft_deg=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_ftip_upright_soft_deg",
                                    a_cfg.get(
                                        "surface_tip_gmhqp_upright_soft_deg", 16.0
                                    ),
                                )
                            ),
                            upright_step_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_ftip_upright_step_m",
                                    a_cfg.get(
                                        "surface_tip_gmhqp_upright_step_m", 0.004
                                    ),
                                )
                            ),
                            mouth_r_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_ftip_mouth_r_m",
                                    a_cfg.get("surface_tip_gmhqp_mouth_r_m", 0.0045),
                                )
                            ),
                            mouth_press_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_ftip_mouth_press_m",
                                    a_cfg.get(
                                        "surface_tip_gmhqp_mouth_press_m", 0.0004
                                    ),
                                )
                            ),
                            c_ridge=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_ftip_c_ridge",
                                    a_cfg.get("surface_tip_gmhqp_c_ridge", 0.35),
                                )
                            ),
                            c_ema=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_ftip_c_ema",
                                    a_cfg.get("surface_tip_gmhqp_c_ema", 0.85),
                                )
                            ),
                            ft_ema=float(
                                a_cfg.get("surface_tip_gmhqp_ftip_ft_ema", 0.55)
                            ),
                            ft_f_min_n=float(
                                a_cfg.get("surface_tip_gmhqp_ftip_f_min_n", 0.02)
                            ),
                            ft_max_lever_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_ftip_max_lever_m", 0.08
                                )
                            ),
                        )
                        if tip_gmhqp_ftip_enable:
                            meta["tip_gmhqp_ftip_enable"] = True
                            meta["tip_gmhqp_ftip_theory"] = (
                                "doshi_ft_contact+montana_gmhqp_escande"
                            )
                            if tip_search_pose_hold and use_priv_grasp_a:
                                meta["tip_gmhqp_ftip_pose_hold"] = True
                        # Track-B CompC: contact short-horizon MPC on GMHQP (ĉ in cost only).
                        tip_gmhqp_compc_enable = bool(
                            a_cfg.get("surface_tip_gmhqp_compc_enable", False)
                        ) or bool(_gmhqp_compc_mode)
                        gmhqp_compc_state = GmhqpCompcState()
                        gmhqp_compc_cfg = GmhqpCompcConfig(
                            k_tip=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_compc_k_tip",
                                    a_cfg.get("surface_tip_gmhqp_k_tip", 1.4),
                                )
                            ),
                            max_step_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_compc_max_step_m",
                                    a_cfg.get(
                                        "surface_tip_gmhqp_max_step_m",
                                        a_cfg.get(
                                            "surface_planned_spiral_max_wrist_step_m",
                                            0.012,
                                        ),
                                    ),
                                )
                            ),
                            f_seat_n=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_compc_f_seat_n",
                                    a_cfg.get(
                                        "surface_tip_gmhqp_f_seat_n",
                                        a_cfg.get(
                                            "surface_tip_hybrid_seat_contact_n", 0.05
                                        ),
                                    ),
                                )
                            ),
                            seat_press_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_compc_seat_press_m",
                                    a_cfg.get(
                                        "surface_tip_gmhqp_seat_press_m",
                                        a_cfg.get(
                                            "surface_tip_hybrid_seat_press_m", 0.00045
                                        ),
                                    ),
                                )
                            ),
                            upright_soft_deg=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_compc_upright_soft_deg",
                                    a_cfg.get(
                                        "surface_tip_gmhqp_upright_soft_deg", 16.0
                                    ),
                                )
                            ),
                            upright_step_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_compc_upright_step_m",
                                    a_cfg.get(
                                        "surface_tip_gmhqp_upright_step_m", 0.004
                                    ),
                                )
                            ),
                            mouth_r_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_compc_mouth_r_m",
                                    a_cfg.get("surface_tip_gmhqp_mouth_r_m", 0.0045),
                                )
                            ),
                            mouth_press_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_compc_mouth_press_m",
                                    a_cfg.get(
                                        "surface_tip_gmhqp_mouth_press_m", 0.0004
                                    ),
                                )
                            ),
                            c_ridge=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_compc_c_ridge",
                                    a_cfg.get("surface_tip_gmhqp_c_ridge", 0.35),
                                )
                            ),
                            c_ema=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_compc_c_ema",
                                    a_cfg.get("surface_tip_gmhqp_c_ema", 0.85),
                                )
                            ),
                            horizon=int(
                                a_cfg.get("surface_tip_gmhqp_compc_horizon", 4)
                            ),
                            w_tip=float(
                                a_cfg.get("surface_tip_gmhqp_compc_w_tip", 1.0)
                            ),
                            w_contact=float(
                                a_cfg.get("surface_tip_gmhqp_compc_w_contact", 1.2)
                            ),
                            w_u=float(
                                a_cfg.get("surface_tip_gmhqp_compc_w_u", 0.08)
                            ),
                            sigma_false_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_compc_sigma_false_m", 0.005
                                )
                            ),
                            f_des_n=float(
                                a_cfg.get("surface_tip_gmhqp_compc_f_des_n", 0.08)
                            ),
                            k_force_m_per_n=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_compc_k_force_m_per_n", 0.0025
                                )
                            ),
                            force_press_cap_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_compc_force_press_cap_m",
                                    0.00035,
                                )
                            ),
                            ft_ema=float(
                                a_cfg.get("surface_tip_gmhqp_compc_ft_ema", 0.55)
                            ),
                            ft_f_min_n=float(
                                a_cfg.get("surface_tip_gmhqp_compc_f_min_n", 0.02)
                            ),
                            ft_max_lever_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_compc_max_lever_m", 0.08
                                )
                            ),
                            contact_step_cap_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_compc_contact_step_cap_m",
                                    0.008,
                                )
                            ),
                        )
                        if tip_gmhqp_compc_enable:
                            meta["tip_gmhqp_compc_enable"] = True
                            meta["tip_gmhqp_compc_theory"] = (
                                "active_extrinsic+letac_mpc_on_gmhqp"
                            )
                            if tip_search_pose_hold and use_priv_grasp_a:
                                meta["tip_gmhqp_compc_pose_hold"] = True
                        # Track-B Adaptive-C: CouplingEKF online C + uncertainty-weighted C⁺.
                        tip_gmhqp_ac_enable = bool(
                            a_cfg.get("surface_tip_gmhqp_ac_enable", False)
                        ) or bool(_gmhqp_ac_mode)
                        gmhqp_ac_state = GmhqpAcState()
                        gmhqp_ac_cfg = GmhqpAcConfig(
                            k_tip=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_ac_k_tip",
                                    a_cfg.get("surface_tip_gmhqp_k_tip", 1.4),
                                )
                            ),
                            max_step_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_ac_max_step_m",
                                    a_cfg.get(
                                        "surface_tip_gmhqp_max_step_m",
                                        a_cfg.get(
                                            "surface_planned_spiral_max_wrist_step_m",
                                            0.012,
                                        ),
                                    ),
                                )
                            ),
                            f_seat_n=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_ac_f_seat_n",
                                    a_cfg.get(
                                        "surface_tip_gmhqp_f_seat_n",
                                        a_cfg.get(
                                            "surface_tip_hybrid_seat_contact_n", 0.05
                                        ),
                                    ),
                                )
                            ),
                            seat_press_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_ac_seat_press_m",
                                    a_cfg.get(
                                        "surface_tip_gmhqp_seat_press_m",
                                        a_cfg.get(
                                            "surface_tip_hybrid_seat_press_m", 0.00045
                                        ),
                                    ),
                                )
                            ),
                            upright_soft_deg=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_ac_upright_soft_deg",
                                    a_cfg.get(
                                        "surface_tip_gmhqp_upright_soft_deg", 16.0
                                    ),
                                )
                            ),
                            upright_step_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_ac_upright_step_m",
                                    a_cfg.get(
                                        "surface_tip_gmhqp_upright_step_m", 0.004
                                    ),
                                )
                            ),
                            mouth_r_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_ac_mouth_r_m",
                                    a_cfg.get("surface_tip_gmhqp_mouth_r_m", 0.0045),
                                )
                            ),
                            mouth_press_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_ac_mouth_press_m",
                                    a_cfg.get(
                                        "surface_tip_gmhqp_mouth_press_m", 0.0004
                                    ),
                                )
                            ),
                            c_ridge0=float(
                                a_cfg.get("surface_tip_gmhqp_ac_c_ridge0", 0.25)
                            ),
                            c_ridge_p=float(
                                a_cfg.get("surface_tip_gmhqp_ac_c_ridge_p", 2.0)
                            ),
                            c_ridge_innov=float(
                                a_cfg.get("surface_tip_gmhqp_ac_c_ridge_innov", 8.0)
                            ),
                            c_cond_max=float(
                                a_cfg.get("surface_tip_gmhqp_ac_c_cond_max", 25.0)
                            ),
                            ekf_process_tip=float(
                                a_cfg.get("surface_tip_gmhqp_ac_ekf_process_tip", 1e-6)
                            ),
                            ekf_process_c=float(
                                a_cfg.get("surface_tip_gmhqp_ac_ekf_process_c", 1e-5)
                            ),
                            ekf_meas_var=float(
                                a_cfg.get("surface_tip_gmhqp_ac_ekf_meas_var", 4e-7)
                            ),
                            ekf_slip_ratio=float(
                                a_cfg.get("surface_tip_gmhqp_ac_ekf_slip_ratio", 0.25)
                            ),
                            ekf_slip_dw_m=float(
                                a_cfg.get(
                                    "surface_tip_gmhqp_ac_ekf_slip_dw_m", 0.0012
                                )
                            ),
                        )
                        if tip_gmhqp_ac_enable:
                            meta["tip_gmhqp_ac_enable"] = True
                            meta["tip_gmhqp_ac_theory"] = (
                                "coupling_ekf_adaptive_C+escande_gmhqp"
                            )
                            if tip_search_pose_hold and use_priv_grasp_a:
                                meta["tip_gmhqp_ac_pose_hold"] = True
                        # Track-B Hybrid GMHQP: tip_task ∨ planar_C ContactMotion fallback.
                        tip_hybrid_gmhqp_enable = bool(
                            a_cfg.get("surface_tip_hybrid_gmhqp_enable", False)
                        ) or bool(_hybrid_gmhqp_mode)
                        hybrid_gmhqp_cfg = HybridGmhqpConfig(
                            k_tip=float(
                                a_cfg.get("surface_tip_hybrid_gmhqp_k_tip", 1.4)
                            ),
                            max_step_m=float(
                                a_cfg.get(
                                    "surface_tip_hybrid_gmhqp_max_step_m",
                                    a_cfg.get(
                                        "surface_planned_spiral_max_wrist_step_m",
                                        0.012,
                                    ),
                                )
                            ),
                            f_seat_n=float(
                                a_cfg.get(
                                    "surface_tip_hybrid_gmhqp_f_seat_n",
                                    a_cfg.get(
                                        "surface_tip_hybrid_seat_contact_n", 0.05
                                    ),
                                )
                            ),
                            seat_press_m=float(
                                a_cfg.get(
                                    "surface_tip_hybrid_gmhqp_seat_press_m",
                                    a_cfg.get(
                                        "surface_tip_hybrid_seat_press_m", 0.00045
                                    ),
                                )
                            ),
                            upright_soft_deg=float(
                                a_cfg.get(
                                    "surface_tip_hybrid_gmhqp_upright_soft_deg", 16.0
                                )
                            ),
                            upright_step_m=float(
                                a_cfg.get(
                                    "surface_tip_hybrid_gmhqp_upright_step_m", 0.004
                                )
                            ),
                            mouth_r_m=float(
                                a_cfg.get(
                                    "surface_tip_hybrid_gmhqp_mouth_r_m", 0.0045
                                )
                            ),
                            mouth_press_m=float(
                                a_cfg.get(
                                    "surface_tip_hybrid_gmhqp_mouth_press_m", 0.0004
                                )
                            ),
                            c_ridge=float(
                                a_cfg.get("surface_tip_hybrid_gmhqp_c_ridge", 0.35)
                            ),
                            c_ema=float(
                                a_cfg.get("surface_tip_hybrid_gmhqp_c_ema", 0.85)
                            ),
                            k_path=float(
                                a_cfg.get("surface_tip_hybrid_gmhqp_k_path", 1.6)
                            ),
                            sat_err_m=float(
                                a_cfg.get("surface_tip_hybrid_gmhqp_sat_err_m", 0.008)
                            ),
                            sat_progress_m=float(
                                a_cfg.get(
                                    "surface_tip_hybrid_gmhqp_sat_progress_m", 0.0005
                                )
                            ),
                            sat_frames=int(
                                a_cfg.get("surface_tip_hybrid_gmhqp_sat_frames", 40)
                            ),
                            recover_err_m=float(
                                a_cfg.get(
                                    "surface_tip_hybrid_gmhqp_recover_err_m", 0.0055
                                )
                            ),
                        )
                        if tip_hybrid_gmhqp_enable:
                            meta["tip_hybrid_gmhqp_enable"] = True
                            meta["tip_hybrid_gmhqp_theory"] = (
                                "escande_hybrid_tip_task+planar_C_ContactMotion"
                            )
                            if tip_search_pose_hold and use_priv_grasp_a:
                                meta["tip_hybrid_gmhqp_pose_hold"] = True
                        # Deployable mouth force-enter after tip reaches mouth ring
                        # (planner r + F/T); yields spiral → SEAT + mouth_probe.
                        deploy_mouth_enter = bool(
                            a_cfg.get("surface_deploy_mouth_enter", False)
                        )
                        tip_star_counts = {
                            "spiral_track": 0,
                            "local_recover": 0,
                            "grasp_restore": 0,
                            "yield_mouth": 0,
                        }
                        along_seat_ref_m = float(feat_sp0.along_m)
                        force_hole_state = SpiralForceHoleDetectState()
                        search_cfg_sp = cfg.get("compliant", {}).get("search", {})
                        # Dual-track SEAT observation (docs/TRACK_AB_EXTRINSIC_SEAT.md).
                        seat_obs_mode = str(
                            a_cfg.get("surface_seat_obs_mode", "auto")
                        ).strip().lower()
                        tip_seat_sensor = tip_contact_sensor or tip_est
                        if tip_seat_sensor is None and seat_obs_mode in (
                            "tip_contact",
                            "tec",
                            "track_a",
                            "auto",
                        ):
                            _enable_tip_seat = bool(
                                a_cfg.get(
                                    "surface_seat_tip_contact_enable",
                                    seat_obs_mode
                                    in ("tip_contact", "tec", "track_a"),
                                )
                            )
                            if _enable_tip_seat and seat_obs_mode != "extrinsic":
                                tip_seat_sensor = ContactTipEstimator(
                                    raw,
                                    ema=0.55,
                                    min_force_n=float(
                                        a_cfg.get(
                                            "surface_tip_est_min_force_n", 0.008
                                        )
                                        or 0.008
                                    ),
                                    seat_force_n=float(
                                        a_cfg.get(
                                            "surface_tip_est_seat_force_n", 0.015
                                        )
                                        or 0.015
                                    ),
                                )
                                meta["seat_tip_contact_sensor"] = True
                        extrinsic_seat_cfg = ExtrinsicSeatConfig(
                            f_n_seat=float(
                                a_cfg.get(
                                    "surface_seat_extrinsic_fn_n",
                                    a_cfg.get(
                                        "surface_tip_hybrid_seat_contact_n", 0.05
                                    ),
                                )
                            ),
                            require_contact_est=bool(
                                a_cfg.get(
                                    "surface_seat_extrinsic_require_clep", True
                                )
                            ),
                        )
                        meta["seat_obs_mode"] = str(seat_obs_mode)

                        def _seat_verdict_now() -> object:
                            wr_s = read_wrist_wrench_world(raw)
                            # Residual along hole (same as |_wrist_contact|), NOT
                            # absolute wrench — abs |n·F|~6N false-seated AC ep07.
                            fz_r = float(
                                wrench_in_hole_frame(wr_s[0], hole_u)[2]
                            )
                            fn_resid = float(fz_r) - float(fz_base_c)
                            f_resid = (
                                np.asarray(hole_u, dtype=np.float64).reshape(3)
                                * float(fn_resid)
                            )
                            tau_s = np.asarray(wr_s[0][3:6], dtype=np.float64)
                            site_s = actual_action44_from_sites(raw)[0:3]
                            tip_n = tip_f = None
                            mode_s = seat_obs_mode
                            if tip_seat_sensor is not None and mode_s in (
                                "tip_contact",
                                "tec",
                                "track_a",
                                "auto",
                            ):
                                _c_s, tip_f, tip_n = tip_seat_sensor.measure_peg_tray_tip(
                                    raw, spiral_n
                                )
                                if mode_s == "auto":
                                    mode_s = "tip_contact"
                            elif mode_s == "auto":
                                mode_s = "extrinsic"
                            return resolve_seat(
                                mode=mode_s,
                                force_xyz=f_resid,
                                torque_xyz=tau_s,
                                normal=hole_u,
                                site_xyz=site_s,
                                tip_anchor=np.asarray(
                                    feat_sp0.tip_pos, dtype=np.float64
                                ).reshape(3),
                                tip_ncon=tip_n,
                                tip_force_n=tip_f,
                                tip_seat_force_n=float(
                                    a_cfg.get(
                                        "surface_tip_est_seat_force_n", 0.015
                                    )
                                    or 0.015
                                ),
                                extrinsic_cfg=extrinsic_seat_cfg,
                            )

                        def _seat_reseat_ax_m(seated: bool) -> float:
                            """ConnTact seeking into tray while unseated (Pfanne stiffen separate)."""
                            if seated:
                                return float(
                                    a_cfg.get(
                                        "surface_planned_probe_press_m", 0.00005
                                    )
                                )
                            return float(
                                a_cfg.get(
                                    "surface_seat_reseat_step_m",
                                    max(
                                        float(
                                            a_cfg.get(
                                                "surface_planned_surface_reseat_m",
                                                0.00012,
                                            )
                                        ),
                                        float(
                                            a_cfg.get(
                                                "surface_planned_seek_step_m",
                                                0.00012,
                                            )
                                        ),
                                    ),
                                )
                            )

                        def _seat_stiffen_right_hand(cmd44: np.ndarray) -> None:
                            """Pfanne-style grasp stiffen while reseating (tip follows wrist)."""
                            scale = float(
                                a_cfg.get("surface_seat_reseat_grasp_scale", 1.35)
                            )
                            if scale <= 1.0 + 1e-9:
                                return
                            h = np.asarray(cmd44[6:22], dtype=np.float64)
                            cmd44[6:22] = np.clip(
                                np.maximum(h, h * scale), -1.5, 1.5
                            )

                        priv_overforce_streak = 0
                        priv_overforce_streak_peak = 0
                        priv_along_drift_peak_m = 0.0
                        overload_cooldown_sp = 0
                        last_lat_improve_sp = 0
                        stuck_count_sp = 0
                        micro_lift_left = 0
                        tip_prev_stuck = tip_sp0.copy()
                        micro_lift_count = 0
                        lost_contact_count = 0
                        recontact_left = 0
                        recontact_count = 0
                        surface_depth0 = float(np.dot(tip_sp0, spiral_n))
                        if search_mode == "planned_spiral":
                            along_hold_m = surface_depth0
                        if search_mode == "planned_spiral":
                            # Surface approach: pull tip toward hole in the contact
                            # plane BEFORE spiraling. Bleed/const-force often leave
                            # tip 40–80mm out (ep02/10); a long spiral then loses tip.
                            approach_lat = float(
                                a_cfg.get("surface_planned_approach_lat_m", 0.022)
                            )
                            approach_n = int(
                                a_cfg.get("surface_planned_approach_frames", 360)
                            )
                            approach_step = float(
                                a_cfg.get("surface_planned_approach_step_m", 0.005)
                            )
                            approach_press = float(
                                a_cfg.get(
                                    "surface_planned_probe_press_m", 0.00008
                                )
                            )
                            ap_used = 0
                            feat_ap = features_from_raw(raw)
                            while (
                                float(feat_ap.lateral_m) > approach_lat
                                and ap_used < approach_n
                            ):
                                tip_ap = np.asarray(
                                    feat_ap.tip_pos, dtype=np.float64
                                ).reshape(3)
                                ctr_ap = _hole_axis_point_at_tip(
                                    tip_ap,
                                    feat_ap.socket_pos,
                                    feat_ap.hole_axis,
                                )
                                err_ap = ctr_ap - tip_ap
                                err_ap = err_ap - spiral_n * float(
                                    np.dot(err_ap, spiral_n)
                                )
                                en_ap = float(np.linalg.norm(err_ap))
                                if en_ap < 1e-5:
                                    break
                                d_ap = err_ap * min(1.0, approach_step / en_ap)
                                contact_ap = float(_wrist_contact()[0])
                                # SEAT ≻ planar (Raibert/ConnTact): no lat shrink while
                                # tip–tray seat observation is false (ban float approach).
                                seat_ap = _seat_verdict_now()
                                seated_ap = bool(getattr(seat_ap, "seated", False))
                                if not seated_ap:
                                    d_ap = d_ap * 0.0
                                # ConnTact seeking into tray + Pfanne grasp stiffen while
                                # unseated (tip_contact=0 / residual low). Use -hole_u.
                                ax_mag = _seat_reseat_ax_m(seated_ap)
                                if (
                                    (not seated_ap)
                                    or contact_ap < 0.05
                                    or float(feat_ap.tip_socket_dist_m) > 0.115
                                ):
                                    ax_mag = max(ax_mag, approach_press * 2.5, 0.0002)
                                else:
                                    ax_mag = max(ax_mag, approach_press)
                                # Pfanne: first frames grasp-stiffen only (no axial peel).
                                grasp_only_n = int(
                                    a_cfg.get("surface_seat_grasp_only_frames", 40)
                                )
                                if (not seated_ap) and ap_used < grasp_only_n:
                                    ax_mag = 0.0
                                ax_ap = (-hole_u) * float(ax_mag)
                                site_ap = actual_action44_from_sites(raw)
                                cmd_ap = np.zeros(44, dtype=np.float64)
                                cmd_ap[0:6] = site_ap[0:6].copy()
                                cmd_ap[0:3] = site_ap[0:3] + d_ap + ax_ap
                                _fill_tip_search_hands(cmd_ap)
                                if not seated_ap:
                                    _seat_stiffen_right_hand(cmd_ap)
                                cmd_ap[22:28] = site_ap[22:28].copy()
                                step_action44(
                                    gym_env, cmd_ap, ego_recorder=ego_recorder
                                )
                                ap_used += 1
                                steps += 1
                                feat_ap = features_from_raw(raw)
                                if ap_used == 1 or ap_used % 40 == 0:
                                    print(
                                        f"pci: surface-approach j={ap_used} "
                                        f"lat={feat_ap.lateral_m*1e3:.1f}mm "
                                        f"along={feat_ap.along_m*1e3:.1f}mm "
                                        f"|r|={_wrist_contact()[0]:.2f}N "
                                        f"seat={getattr(seat_ap, 'source', '?')}:"
                                        f"{int(seated_ap)} "
                                        f"fn={getattr(seat_ap, 'fn_n', 0):.3f}",
                                        flush=True,
                                    )
                            if ap_used > 0:
                                feat_sp0 = features_from_raw(raw)
                                tip_sp0 = np.asarray(
                                    feat_sp0.tip_pos, dtype=np.float64
                                ).reshape(3)
                                along0_spiral = float(feat_sp0.along_m)
                                surface_depth0 = float(np.dot(tip_sp0, spiral_n))
                                along_hold_m = surface_depth0
                                tip_prev_stuck = tip_sp0.copy()
                                site_sp = actual_action44_from_sites(raw)
                                print(
                                    f"pci: surface-approach done frames={ap_used} "
                                    f"lat={feat_sp0.lateral_m*1e3:.1f}mm "
                                    f"(then planned spiral)",
                                    flush=True,
                                )
                            # Enough steps to shrink from current tip lat to mouth.
                            lat_guess = max(
                                float(feat_sp0.lateral_m),
                                float(
                                    np.linalg.norm(
                                        _tip_lat_offset_xy(
                                            tip_sp0,
                                            _hole_axis_point_at_tip(
                                                tip_sp0,
                                                feat_sp0.socket_pos,
                                                feat_sp0.hole_axis,
                                            ),
                                            spiral_n,
                                        )
                                    )
                                ),
                            )
                            need = int(
                                max(240, (lat_guess - mouth_lat) / max(pitch, 1e-6)
                                    * (2.0 * np.pi) / max(dtheta, 1e-4) + 80)
                            )
                            planned_wps, planned_rs, planned_ths, planned_center, _basis = (
                                _plan_surface_enter_spiral(
                                    tip_sp0,
                                    feat_sp0.socket_pos,
                                    feat_sp0.hole_axis,
                                    spiral_n,
                                    pitch_m=pitch,
                                    dtheta=dtheta,
                                    r_min_m=max(r_min, mouth_lat),
                                    n_max=max(n_search, need),
                                )
                            )
                            spiral_center = planned_center
                            spiral_center_mode = "planned_hole"
                            n_wp = int(planned_wps.shape[0])
                            n_search = n_wp
                            # Tip-gated: 1 frame ≠ 1 waypoint; expand budget so tip
                            # can catch before radius shrinks away (ep01/03 lag).
                            tip_gate = int(
                                a_cfg.get(
                                    "surface_planned_spiral_soft_advance_miss",
                                    25,
                                )
                            ) <= 0 or (
                                float(
                                    a_cfg.get(
                                        "surface_planned_probe_press_m", 0.0
                                    )
                                )
                                > 1e-9
                                and int(
                                    a_cfg.get(
                                        "surface_planned_probe_soft_advance_miss",
                                        0,
                                    )
                                )
                                <= 0
                            )
                            if tip_gate:
                                fpp = int(
                                    a_cfg.get(
                                        "surface_planned_tip_gate_frames_per_wp",
                                        8,
                                    )
                                )
                                n_search = max(n_wp, n_wp * max(1, fpp))
                            t1 = _basis[0]
                            t2 = _basis[1]
                            print(
                                f"pci: planned-spiral N_wp={n_wp} "
                                f"n_frames={n_search} tip_gate={int(tip_gate)} "
                                f"r0={float(planned_rs[0])*1e3:.1f}mm "
                                f"rN={float(planned_rs[-1])*1e3:.1f}mm "
                                f"mouth={mouth_lat*1e3:.1f}mm "
                                f"(privileged surface→hole FF)",
                                flush=True,
                            )
                        stick_active = False
                        stick_count = 0
                        stick_streak = 0
                        stick_live_off = False
                        stick_phase = "search"
                        stick_phase_i = 0
                        stick_offset_wt: np.ndarray | None = None
                        stick_lat_enter = 0.0
                        stick_along0: float | None = None
                        stick_reached_mouth = False
                        stick_hop_count = 0
                        stick_lat_best = 1e9
                        stick_stall = 0
                        mouth_wiggle_enable = bool(
                            a_cfg.get("surface_mouth_wiggle_enable", False)
                        )
                        mouth_continue = bool(
                            a_cfg.get("surface_mouth_continue_on_near", True)
                        ) and mouth_wiggle_enable
                        n_mouth = int(a_cfg.get("surface_mouth_wiggle_frames", 180))
                        if stick_enable:
                            n_mouth = max(n_mouth, 420)
                        mouth_xy_m = float(
                            a_cfg.get("surface_mouth_wiggle_xy_m", 0.0015)
                        )
                        mouth_wx = float(
                            a_cfg.get(
                                "surface_mouth_wiggle_wx",
                                a_cfg.get("surface_lissajous_wx", 0.22),
                            )
                        )
                        mouth_wy = float(
                            a_cfg.get(
                                "surface_mouth_wiggle_wy",
                                a_cfg.get("surface_lissajous_wy", 0.31),
                            )
                        )
                        mouth_phi = float(
                            a_cfg.get("surface_mouth_wiggle_phase_rad", lis_phi)
                        )
                        mouth_rot_rad = float(
                            a_cfg.get("surface_mouth_wiggle_rot_rad", 0.008)
                        )
                        mouth_lat_step = float(
                            a_cfg.get("surface_mouth_wiggle_lat_step_m", 0.001)
                        )
                        mouth_axis_ff = float(
                            a_cfg.get("surface_mouth_wiggle_axis_ff", 0.3)
                        )
                        mouth_ax_step = float(
                            a_cfg.get("surface_mouth_wiggle_ax_step_m", 0.00008)
                        )
                        mouth_jam_fxy = float(
                            a_cfg.get("surface_mouth_jam_fxy_n", 0.35)
                        )
                        mouth_jam_frames = int(
                            a_cfg.get("surface_mouth_jam_frames", 8)
                        )
                        mouth_tray_admit = float(
                            a_cfg.get("surface_mouth_tray_admit_scale", 0.0)
                        )
                        mouth_tray_admit_max = float(
                            a_cfg.get("surface_mouth_tray_admit_max_m", 0.0005)
                        )
                        if search_mode == "priv_lat_lissajous":
                            print(
                                f"pci: priv-lat-liss+force f_des={f_des:.2f}N frames={n_search} "
                                f"lat_step={priv_lat_step*1e3:.2f}mm axis_ff={priv_lat_axis_ff:.2f} "
                                f"dith={priv_dith_ax*1e3:.1f}/{priv_dith_ay*1e3:.1f}mm "
                                f"lat0={feat_sp0.lateral_m*1e3:.1f}mm pre_lat={pre_lat_n}",
                                flush=True,
                            )
                        elif search_mode == "priv_lat":
                            print(
                                f"pci: priv-lat+force f_des={f_des:.2f}N frames={n_search} "
                                f"lat_step={priv_lat_step*1e3:.2f}mm axis_ff={priv_lat_axis_ff:.2f} "
                                f"lat0={feat_sp0.lateral_m*1e3:.1f}mm pre_lat={pre_lat_n}",
                                flush=True,
                            )
                        elif search_mode == "lissajous":
                            print(
                                f"pci: tip-lissajous+force f_des={f_des:.2f}N frames={n_search} "
                                f"wx={lis_wx:.2f} wy={lis_wy:.2f} shrink={int(lis_shrink)} "
                                f"lat0={feat_sp0.lateral_m*1e3:.1f}mm center=socket "
                                f"pre_lat={pre_lat_n} (firm grasp)",
                                flush=True,
                            )
                        elif search_mode == "planned_spiral":
                            print(
                                f"pci: planned-spiral+force f_des={f_des:.2f}N "
                                f"frames={n_search} track={track:.2f} "
                                f"lat0={feat_sp0.lateral_m*1e3:.1f}mm "
                                f"center={spiral_center_mode} "
                                f"(privileged FF tip track)",
                                flush=True,
                            )
                        else:
                            print(
                                f"pci: tip-spiral+force f_des={f_des:.2f}N frames={n_sp} "
                                f"pitch={pitch*1e3:.1f}mm track={track:.2f} "
                                f"r0={r0*1e3:.1f} rmax={r_max*1e3:.1f} "
                                f"lat0={feat_sp0.lateral_m*1e3:.1f}mm "
                                f"center={spiral_center_mode} dir={spiral_dir} "
                                f"shrink={int(lat_shrink_only)} pre_lat={pre_lat_n} "
                                f"(firm grasp)",
                                flush=True,
                            )
                        f_des_spiral = float(f_des)
                        if search_mode == "planned_spiral":
                            c0, _, _, _ = _wrist_contact()
                            if bool(
                                a_cfg.get("surface_planned_spiral_keep_f_des", False)
                            ):
                                f_des_spiral = float(
                                    a_cfg.get(
                                        "surface_planned_spiral_f_des_n", f_des
                                    )
                                )
                                print(
                                    f"pci: planned-spiral keep f_des="
                                    f"{f_des_spiral:.3f}N (contact0={c0:.3f}N)",
                                    flush=True,
                                )
                            else:
                                f_des_spiral = min(
                                    float(f_des),
                                    max(
                                        0.012,
                                        float(c0) * 1.1 + 0.004,
                                    ),
                                )
                                print(
                                    f"pci: planned-spiral f_des cap "
                                    f"{f_des_spiral:.3f}N (contact0={c0:.3f}N)",
                                    flush=True,
                                )
                        # Demo-only: push tip outward on the tray while holding
                        # contact, so spiral starts with a larger planar error
                        # (visible search) without breaking seat.
                        demo_out_lat = float(
                            a_cfg.get("surface_demo_outward_lat_m", 0.0)
                        )
                        demo_out_n = int(
                            a_cfg.get("surface_demo_outward_frames", 0)
                        )
                        demo_out_step = float(
                            a_cfg.get("surface_demo_outward_step_m", 0.0015)
                        )
                        if demo_out_lat > 1e-4 and demo_out_n > 0:
                            track_out = float(
                                a_cfg.get(
                                    "surface_demo_outward_track",
                                    a_cfg.get("surface_tip_oracle_track", 3.0),
                                )
                            )
                            max_cmd_out = float(
                                a_cfg.get(
                                    "surface_demo_outward_max_wrist_step_m",
                                    0.02,
                                )
                            )
                            f_des_out = float(
                                a_cfg.get(
                                    "surface_demo_outward_f_des_n",
                                    max(float(f_des), 0.08),
                                )
                            )
                            print(
                                f"pci: demo outward lat→{demo_out_lat*1e3:.1f}mm "
                                f"frames≤{demo_out_n} step={demo_out_step*1e3:.1f}mm "
                                f"track={track_out:.1f} f_des={f_des_out:.2f}N "
                                f"(tip-track keep seat)",
                                flush=True,
                            )
                            lat_out0 = float(features_from_raw(raw).lateral_m)
                            for _out_j in range(1, demo_out_n + 1):
                                feat_i = features_from_raw(raw)
                                site_now = actual_action44_from_sites(raw)
                                tip = np.asarray(
                                    feat_i.tip_pos, dtype=np.float64
                                ).reshape(3)
                                ax_i = feat_i.hole_axis / (
                                    np.linalg.norm(feat_i.hole_axis) + 1e-12
                                )
                                # Radial in contact plane: tip − hole-axis point.
                                ctr = _hole_axis_point_at_tip(
                                    tip, feat_i.socket_pos, feat_i.hole_axis
                                )
                                off_xy = tip - ctr
                                off_xy = off_xy - ax_i * float(np.dot(off_xy, ax_i))
                                lat_now = float(feat_i.lateral_m)
                                if lat_now >= demo_out_lat:
                                    print(
                                        f"pci: demo outward done j={_out_j} "
                                        f"lat={lat_now*1e3:.1f}mm "
                                        f"(from {lat_out0*1e3:.1f}mm)",
                                        flush=True,
                                    )
                                    break
                                n_xy = float(np.linalg.norm(off_xy))
                                if n_xy < 1e-6:
                                    rad = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                                    rad = rad - ax_i * float(np.dot(rad, ax_i))
                                    n_xy = float(np.linalg.norm(rad)) + 1e-12
                                    off_xy = rad * n_xy
                                step_v = off_xy * (
                                    min(demo_out_step, demo_out_lat - lat_now)
                                    / (n_xy + 1e-12)
                                )
                                target = tip + step_v
                                contact, left_c, fz_r, fz_l = _wrist_contact()
                                contact_sp = float(contact)
                                err_f = f_des_out - contact_sp
                                ax_step = float(
                                    np.clip(kp * err_f, -max_step, max_step)
                                )
                                site_xyz = site_now[0:3].copy()
                                # Soft grasp: C≠I — command wrist from tip error,
                                # not tip+offset (that barely moves tip).
                                tip_err = target - tip
                                tip_err = tip_err - ax_i * float(
                                    np.dot(tip_err, ax_i)
                                )
                                hold_r = (
                                    site_xyz
                                    + track_out * tip_err
                                    + press_ax * ax_step
                                )
                                d_cmd = hold_r - site_xyz
                                dn = float(np.linalg.norm(d_cmd))
                                if dn > max_cmd_out > 0.0:
                                    hold_r = site_xyz + d_cmd * (max_cmd_out / dn)
                                cmd = np.zeros(44, dtype=np.float64)
                                cmd[0:6] = site_now[0:6].copy()
                                cmd[0:3] = hold_r
                                _fill_tip_search_hands(cmd)
                                cmd[22:28] = hold_l
                                step_action44(
                                    gym_env, cmd, ego_recorder=ego_recorder
                                )
                                feat_after = features_from_raw(raw)
                                force_trace.append(
                                    {
                                        "t": float(t_force) * dt_f,
                                        "step": float(t_force),
                                        "phase": "demo_outward",
                                        "resid_r": float(contact_sp),
                                        "resid_l": float(left_c),
                                        "fz_r": float(fz_r),
                                        "fz_l": float(fz_l),
                                        "lat_mm": float(feat_after.lateral_m)
                                        * 1000.0,
                                        "along_mm": float(feat_after.along_m)
                                        * 1000.0,
                                        "f_des": float(f_des_out),
                                    }
                                )
                                t_force += 1
                                steps += 1
                            lat_out1 = float(features_from_raw(raw).lateral_m)
                            if lat_out1 + 1e-4 < demo_out_lat:
                                print(
                                    f"pci: demo outward SHORT lat={lat_out1*1e3:.1f}mm "
                                    f"< target {demo_out_lat*1e3:.1f}mm "
                                    f"(Δ={(lat_out1-lat_out0)*1e3:.1f}mm)",
                                    flush=True,
                                )
                            meta["demo_outward_lat_m"] = float(demo_out_lat)
                            meta["demo_outward_lat_achieved_m"] = float(lat_out1)
                            # Press reseat after outward so spiral does not start floating.
                            reseat_n = int(
                                a_cfg.get("surface_demo_outward_reseat_frames", 0)
                            )
                            if reseat_n > 0:
                                f_reseat = float(
                                    a_cfg.get(
                                        "surface_demo_outward_f_des_n",
                                        max(float(f_des), 0.1),
                                    )
                                )
                                print(
                                    f"pci: demo outward reseat frames={reseat_n} "
                                    f"f_des={f_reseat:.2f}N",
                                    flush=True,
                                )
                                for _rs in range(reseat_n):
                                    site_now = actual_action44_from_sites(raw)
                                    contact, left_c, fz_r, fz_l = _wrist_contact()
                                    err_f = f_reseat - float(contact)
                                    ax_step = float(
                                        np.clip(kp * err_f, -max_step, max_step)
                                    )
                                    cmd = np.zeros(44, dtype=np.float64)
                                    cmd[0:6] = site_now[0:6].copy()
                                    cmd[0:3] = site_now[0:3] + press_ax * ax_step
                                    _fill_tip_search_hands(cmd)
                                    cmd[22:28] = hold_l
                                    step_action44(
                                        gym_env, cmd, ego_recorder=ego_recorder
                                    )
                                    feat_after = features_from_raw(raw)
                                    force_trace.append(
                                        {
                                            "t": float(t_force) * dt_f,
                                            "step": float(t_force),
                                            "phase": "demo_outward_reseat",
                                            "resid_r": float(contact),
                                            "resid_l": float(left_c),
                                            "fz_r": float(fz_r),
                                            "fz_l": float(fz_l),
                                            "lat_mm": float(feat_after.lateral_m)
                                            * 1000.0,
                                            "along_mm": float(feat_after.along_m)
                                            * 1000.0,
                                            "f_des": float(f_reseat),
                                            "priv_peg_tray_ncon": float(
                                                getattr(
                                                    feat_after,
                                                    "priv_peg_tray_ncon",
                                                    0.0,
                                                )
                                                or 0.0
                                            ),
                                        }
                                    )
                                    t_force += 1
                                    steps += 1
                            # Re-plan spiral AFTER outward so waypoints start at the
                            # enlarged tip radius (old plan was built pre-outward).
                            if search_mode == "planned_spiral":
                                feat_sp0 = features_from_raw(raw)
                                tip_sp0 = np.asarray(
                                    feat_sp0.tip_pos, dtype=np.float64
                                ).reshape(3)
                                along0_spiral = float(feat_sp0.along_m)
                                surface_depth0 = float(np.dot(tip_sp0, spiral_n))
                                along_hold_m = surface_depth0
                                tip_prev_stuck = tip_sp0.copy()
                                lat_guess = max(
                                    float(feat_sp0.lateral_m),
                                    float(
                                        np.linalg.norm(
                                            _tip_lat_offset_xy(
                                                tip_sp0,
                                                _hole_axis_point_at_tip(
                                                    tip_sp0,
                                                    feat_sp0.socket_pos,
                                                    feat_sp0.hole_axis,
                                                ),
                                                spiral_n,
                                            )
                                        )
                                    ),
                                )
                                need = int(
                                    max(
                                        240,
                                        (lat_guess - mouth_lat)
                                        / max(pitch, 1e-6)
                                        * (2.0 * np.pi)
                                        / max(dtheta, 1e-4)
                                        + 80,
                                    )
                                )
                                (
                                    planned_wps,
                                    planned_rs,
                                    planned_ths,
                                    planned_center,
                                    _basis,
                                ) = _plan_surface_enter_spiral(
                                    tip_sp0,
                                    feat_sp0.socket_pos,
                                    feat_sp0.hole_axis,
                                    spiral_n,
                                    pitch_m=pitch,
                                    dtheta=dtheta,
                                    r_min_m=max(r_min, mouth_lat),
                                    n_max=max(n_search, need),
                                )
                                spiral_center = planned_center
                                spiral_center_mode = "planned_hole"
                                n_wp = int(planned_wps.shape[0])
                                n_search = n_wp
                                if tip_gate:
                                    fpp = int(
                                        a_cfg.get(
                                            "surface_planned_tip_gate_frames_per_wp",
                                            8,
                                        )
                                    )
                                    n_search = max(n_wp, n_wp * max(1, fpp))
                                t1 = _basis[0]
                                t2 = _basis[1]
                                print(
                                    f"pci: planned-spiral REPLAN after outward "
                                    f"N_wp={n_wp} n_frames={n_search} "
                                    f"r0={float(planned_rs[0])*1e3:.1f}mm "
                                    f"rN={float(planned_rs[-1])*1e3:.1f}mm "
                                    f"lat={lat_guess*1e3:.1f}mm",
                                    flush=True,
                                )
                        if pre_lat_n > 0:
                            for pre_lat_used in range(1, pre_lat_n + 1):
                                feat_i = features_from_raw(raw)
                                site_now = actual_action44_from_sites(raw)
                                tip = np.asarray(
                                    feat_i.tip_pos, dtype=np.float64
                                ).reshape(3)
                                ax_i = feat_i.hole_axis / (
                                    np.linalg.norm(feat_i.hole_axis) + 1e-12
                                )
                                off_xy = _tip_lat_offset_xy(
                                    tip, pre_lat_axis, feat_i.hole_axis
                                )
                                dist = float(np.linalg.norm(off_xy))
                                if float(feat_i.lateral_m) <= mouth_lat:
                                    print(
                                        f"pci: pre-lat done j={pre_lat_used} "
                                        f"lat={feat_i.lateral_m*1e3:.1f}mm",
                                        flush=True,
                                    )
                                    break
                                step_v = -off_xy * (
                                    min(pre_lat_step, dist) / (dist + 1e-12)
                                )
                                target = tip + step_v
                                err = target - tip
                                err = err - ax_i * float(np.dot(err, ax_i))
                                contact, left_c, fz_r, fz_l = _wrist_contact()
                                contact_sp = float(contact)
                                spiral_resid_peak = max(spiral_resid_peak, contact_sp)
                                err_f = f_des - contact_sp
                                ax_step = float(
                                    np.clip(kp * err_f, -max_step, max_step)
                                )
                                if err_f < 0.0:
                                    ax_step = float(
                                        np.clip(
                                            kp * unload_boost * err_f,
                                            -max_step * unload_boost,
                                            max_step,
                                        )
                                    )
                                site_xyz = site_now[0:3].copy()
                                offset_wt = (
                                    offset_wt0_sp
                                    if freeze_off
                                    else site_xyz - tip
                                )
                                hold_r = target + offset_wt + press_ax * ax_step
                                d_cmd = hold_r - site_xyz
                                dn = float(np.linalg.norm(d_cmd))
                                max_cmd = float(
                                    a_cfg.get(
                                        "surface_tip_spiral_max_wrist_step_m", 0.008
                                    )
                                )
                                if dn > max_cmd > 0.0:
                                    hold_r = site_xyz + d_cmd * (max_cmd / dn)
                                cmd = np.zeros(44, dtype=np.float64)
                                cmd[0:6] = site_now[0:6].copy()
                                cmd[0:3] = hold_r
                                _fill_tip_search_hands(cmd)
                                cmd[22:28] = hold_l
                                step_action44(
                                    gym_env, cmd, ego_recorder=ego_recorder
                                )
                                feat_after = features_from_raw(raw)
                                force_trace.append(
                                    {
                                        "t": float(t_force) * dt_f,
                                        "step": float(t_force),
                                        "phase": "pre_lat",
                                        "resid_r": float(contact_sp),
                                        "resid_l": float(left_c),
                                        "fz_r": float(fz_r),
                                        "fz_l": float(fz_l),
                                        "lat_mm": float(feat_after.lateral_m)
                                        * 1000.0,
                                        "along_mm": float(feat_after.along_m)
                                        * 1000.0,
                                        "f_des": float(f_des),
                                    }
                                )
                                t_force += 1
                                spiral_lat_min = min(
                                    spiral_lat_min, float(feat_after.lateral_m)
                                )
                                if _grasp_lost():
                                    spiral_reason = "grasp_slip"
                                    meta["grasp_abort_phase"] = "pre_lat"
                                    break
                            steps += int(pre_lat_used)
                            meta["spiral_pre_lat_frames"] = int(pre_lat_used)
                            if spiral_reason == "grasp_slip":
                                meta["spiral_frames"] = 0
                                meta["spiral_reason"] = spiral_reason
                                gate = _priv_tip_spiral_gate(meta, cfg)
                                meta["priv_tip_spiral_gate"] = gate
                                return steps, "priv_tip_spiral_gate_fail", meta
                            feat_sp0 = features_from_raw(raw)
                            tip_sp0 = np.asarray(
                                feat_sp0.tip_pos, dtype=np.float64
                            ).reshape(3)
                            off0_xy = _tip_lat_offset_xy(
                                tip_sp0, spiral_center, spiral_n
                            )
                            r_start = float(np.linalg.norm(off0_xy))
                            theta_sp = (
                                float(
                                    np.arctan2(
                                        float(off0_xy @ t2), float(off0_xy @ t1)
                                    )
                                )
                                if r_start > 1e-9
                                else 0.0
                            )
                            if spiral_center_mode == "socket" and r_start > r0:
                                r0 = r_start
                            if search_mode == "lissajous":
                                lat0_lis = float(feat_sp0.lateral_m)
                                if lis_ax0 <= 0.0:
                                    lis_ax0 = max(lat0_lis, r_start, lis_amin)
                                if lis_ay0 <= 0.0:
                                    lis_ay0 = lis_ax0 * lis_ay_ratio
                        # Pre-spiral SEAT gate (hard): no float search.
                        # Press-only until wrist residual seats, then start spiral.
                        if (
                            search_mode == "planned_spiral"
                            and bool(
                                a_cfg.get("surface_pre_spiral_seat_enable", True)
                            )
                        ):
                            pre_n = int(
                                a_cfg.get("surface_pre_spiral_seat_frames", 180)
                            )
                            pre_f = float(
                                a_cfg.get(
                                    "surface_pre_spiral_seat_f_des_n",
                                    max(float(f_des), 0.12),
                                )
                            )
                            pre_hold = int(
                                a_cfg.get("surface_pre_spiral_seat_hold_frames", 12)
                            )
                            seat_ok_streak = 0
                            print(
                                f"pci: pre-spiral SEAT frames≤{pre_n} "
                                f"f_des={pre_f:.2f}N mode={seat_obs_mode} "
                                f"(planar frozen)",
                                flush=True,
                            )
                            for _ps in range(max(0, pre_n)):
                                site_now = actual_action44_from_sites(raw)
                                contact, left_c, fz_r, fz_l = _wrist_contact()
                                seat_v = _seat_verdict_now()
                                seated_ps = bool(getattr(seat_v, "seated", False))
                                if seated_ps:
                                    seat_ok_streak += 1
                                    if seat_ok_streak >= pre_hold:
                                        print(
                                            f"pci: pre-spiral SEAT OK j={_ps} "
                                            f"|r|={contact:.2f}N "
                                            f"src={getattr(seat_v, 'source', '?')} "
                                            f"fn={getattr(seat_v, 'fn_n', 0):.3f} "
                                            f"ncon={getattr(seat_v, 'tip_contact_n', 0)}",
                                            flush=True,
                                        )
                                        break
                                else:
                                    seat_ok_streak = 0
                                # Unseated: Pfanne stiffen; axial only after grasp-only window.
                                # Wrist |r|→f_des chase is wrong (grasp resid ≠ tray).
                                grasp_only_n = int(
                                    a_cfg.get("surface_seat_grasp_only_frames", 40)
                                )
                                if (not seated_ps) and _ps < grasp_only_n:
                                    ax_step = 0.0
                                else:
                                    ax_step = _seat_reseat_ax_m(seated_ps)
                                    if not seated_ps:
                                        ax_step = max(ax_step, 1.2e-4)
                                    else:
                                        err_f = pre_f - float(contact)
                                        ax_step = float(
                                            np.clip(kp * err_f, -max_step, max_step)
                                        )
                                cmd = np.zeros(44, dtype=np.float64)
                                cmd[0:6] = site_now[0:6].copy()
                                cmd[0:3] = site_now[0:3] + (-hole_u) * float(ax_step)
                                _fill_tip_search_hands(cmd)
                                if not seated_ps:
                                    _seat_stiffen_right_hand(cmd)
                                cmd[22:28] = site_now[22:28].copy()
                                step_action44(
                                    gym_env, cmd, ego_recorder=ego_recorder
                                )
                                feat_ps = features_from_raw(raw)
                                force_trace.append(
                                    {
                                        "t": float(t_force) * dt_f,
                                        "step": float(t_force),
                                        "phase": "pre_spiral_seat",
                                        "resid_r": float(contact),
                                        "resid_l": float(left_c),
                                        "fz_r": float(fz_r),
                                        "fz_l": float(fz_l),
                                        "lat_mm": float(feat_ps.lateral_m) * 1000.0,
                                        "along_mm": float(feat_ps.along_m)
                                        * 1000.0,
                                        "f_des": float(pre_f),
                                        "seat_src": str(
                                            getattr(seat_v, "source", "")
                                        ),
                                        "seat_ok": int(
                                            bool(getattr(seat_v, "seated", False))
                                        ),
                                        "priv_peg_tray_ncon": float(
                                            getattr(seat_v, "tip_contact_n", 0) or 0
                                        ),
                                    }
                                )
                                t_force += 1
                                steps += 1
                            else:
                                seat_v = _seat_verdict_now()
                                print(
                                    f"pci: pre-spiral SEAT timeout |r|="
                                    f"{_wrist_contact()[0]:.2f}N "
                                    f"src={getattr(seat_v, 'source', '?')} "
                                    f"(spiral may float-reject)",
                                    flush=True,
                                )
                            feat_sp0 = features_from_raw(raw)
                            tip_sp0 = np.asarray(
                                feat_sp0.tip_pos, dtype=np.float64
                            ).reshape(3)
                            along0_spiral = float(feat_sp0.along_m)
                            surface_depth0 = float(np.dot(tip_sp0, spiral_n))
                            along_hold_m = surface_depth0
                            tip_prev_stuck = tip_sp0.copy()
                            meta["pre_spiral_seat"] = True
                            meta["pre_spiral_along0_mm"] = float(along0_spiral) * 1000.0
                        used_sp = 0
                        r_cmd = 0.0
                        _bl_meta: dict = {}
                        for used_sp in range(1, max(0, n_search) + 1):
                            feat_i = features_from_raw(raw)
                            site_now = actual_action44_from_sites(raw)
                            tip = np.asarray(feat_i.tip_pos, dtype=np.float64).reshape(3)
                            tip_gt = tip
                            tip_ctrl = tip
                            tip_ctrl_lat_m = float(feat_i.lateral_m)
                            if tip_theory_ekf is not None and tip_contact_sensor is not None:
                                _n_est = spiral_n
                                _hu_est = np.asarray(
                                    feat_i.hole_axis, dtype=np.float64
                                ).reshape(3)
                                _hn_est = float(np.linalg.norm(_hu_est))
                                if _hn_est > 1e-12:
                                    _n_est = _hu_est / _hn_est
                                _w_now = np.asarray(
                                    site_now[0:3], dtype=np.float64
                                ).reshape(3)
                                if tip_axis_enable and isinstance(
                                    tip_theory_ekf, ExtrinsicTipAxisEKF
                                ):
                                    _cen, _fsum, _ncon, _n_c = (
                                        tip_contact_sensor.measure_peg_tray_tip_ext(
                                            raw, _n_est
                                        )
                                    )
                                else:
                                    _cen, _fsum, _ncon = (
                                        tip_contact_sensor.measure_peg_tray_tip(
                                            raw, _n_est
                                        )
                                    )
                                    _n_c = None
                                # Intermittent contact only; absence ≠ negative seat.
                                _contact_tip = (
                                    _cen if (_cen is not None and _ncon > 0) else None
                                )
                                _wr = read_wrist_wrench_world(raw)
                                _force = np.asarray(
                                    _wr[0][:3], dtype=np.float64
                                ).reshape(3)
                                _torque = np.asarray(
                                    _wr[0][3:6], dtype=np.float64
                                ).reshape(3)
                                if tip_axis_enable and isinstance(
                                    tip_theory_ekf, ExtrinsicTipAxisEKF
                                ):
                                    _est = tip_theory_ekf.step(
                                        wrist_pos=_w_now,
                                        plane_n=_n_est,
                                        force_xyz=_force,
                                        torque_xyz=_torque,
                                        contact_tip=_contact_tip,
                                        contact_force_n=float(_fsum),
                                        contact_n=int(_ncon),
                                        contact_normal=_n_c,
                                        finger_grasp=tip_contact_sensor.finger_grasp_centroid(
                                            raw
                                        ),
                                        wrist_rot=_right_wrist_site_pose(raw)[1],
                                        wrist_approach=_wrist_approach_axis(raw),
                                        tip_gt=tip_gt,  # logging only
                                        axis_gt=feat_i.peg_axis,  # logging only
                                    )
                                    tip_axis_hat = np.asarray(
                                        _est.axis_hat, dtype=np.float64
                                    ).reshape(3)
                                    if np.isfinite(_est.axis_err_to_gt_rad):
                                        tip_axis_err_peak_rad = max(
                                            tip_axis_err_peak_rad,
                                            float(_est.axis_err_to_gt_rad),
                                        )
                                else:
                                    _est = tip_theory_ekf.step(
                                        wrist_pos=_w_now,
                                        plane_n=_n_est,
                                        force_xyz=_force,
                                        torque_xyz=_torque,
                                        contact_tip=_contact_tip,
                                        contact_force_n=float(_fsum),
                                        contact_n=int(_ncon),
                                        tip_gt=tip_gt,  # logging only
                                        # PoseDiff soft prior: same in-hand geom tip
                                        # GMHQP already uses (tactile/object-pose surrogate).
                                        geom_tip=tip_gt,
                                    )
                                tip_est_wrist_prev = _w_now.copy()
                                tip_ctrl = np.asarray(
                                    _est.tip_hat, dtype=np.float64
                                ).reshape(3)
                                # TEC joint: feed ˆt into GMHQP only when estimable.
                                if tip_tec_joint and isinstance(
                                    tip_theory_ekf, ExtrinsicTipStateEKF
                                ):
                                    tip_ctrl, tec_joint_state = select_tip_for_gmhqp(
                                        tip_hat=tip_ctrl,
                                        tip_geom=tip_gt,
                                        plane_n=_n_est,
                                        est=_est,
                                        ekf=tip_theory_ekf,
                                        joint_cfg=tec_joint_cfg,
                                        state=tec_joint_state,
                                    )
                                    tip_est_last_source = (
                                        f"{_est.source}|tec_{tec_joint_state.last_reason}"
                                    )
                                else:
                                    tip_est_last_source = str(_est.source)
                                if tip_tec_joint:
                                    pass  # tip_est_last_source set above
                                elif False:
                                    tip_est_last_source = str(_est.source)
                                if np.isfinite(_est.uncert_m):
                                    tip_est_uncert_peak_m = max(
                                        tip_est_uncert_peak_m, float(_est.uncert_m)
                                    )
                                if np.isfinite(_est.tip_err_to_gt_m):
                                    tip_est_err_peak_m = max(
                                        tip_est_err_peak_m, float(_est.tip_err_to_gt_m)
                                    )
                                    tip_est_err_sum_m += float(_est.tip_err_to_gt_m)
                                    tip_est_err_n += 1
                                    if bool(_est.seated):
                                        tip_est_seated_frames += 1
                                _mouth_est = _hole_axis_point_at_tip(
                                    tip_ctrl,
                                    feat_i.socket_pos,
                                    _n_est,
                                )
                                tip_ctrl_lat_m = float(
                                    np.linalg.norm(
                                        _tip_lat_offset_xy(
                                            tip_ctrl, _mouth_est, _n_est
                                        )
                                    )
                                )
                            elif tip_est_enable and tip_est is not None:
                                _n_est = spiral_n
                                _hu_est = np.asarray(
                                    feat_i.hole_axis, dtype=np.float64
                                ).reshape(3)
                                _hn_est = float(np.linalg.norm(_hu_est))
                                if _hn_est > 1e-12:
                                    _n_est = _hu_est / _hn_est
                                _w_now = np.asarray(
                                    site_now[0:3], dtype=np.float64
                                ).reshape(3)
                                _w_delta = None
                                if tip_est_wrist_prev is not None:
                                    _w_delta = _w_now - tip_est_wrist_prev
                                _est = tip_est.update(
                                    raw,
                                    wrist_pos=_w_now,
                                    plane_n=_n_est,
                                    tip_gt=tip_gt,  # logging only
                                    wrist_delta=_w_delta,
                                )
                                tip_est_wrist_prev = _w_now.copy()
                                tip_ctrl = np.asarray(
                                    _est.tip_hat, dtype=np.float64
                                ).reshape(3)
                                tip_est_last_source = str(_est.source)
                                if np.isfinite(_est.tip_err_to_gt_m):
                                    tip_est_err_peak_m = max(
                                        tip_est_err_peak_m, float(_est.tip_err_to_gt_m)
                                    )
                                    tip_est_err_sum_m += float(_est.tip_err_to_gt_m)
                                    tip_est_err_n += 1
                                    if bool(_est.seated):
                                        tip_est_seated_frames += 1
                                _mouth_est = _hole_axis_point_at_tip(
                                    tip_ctrl,
                                    feat_i.socket_pos,
                                    _n_est,
                                )
                                tip_ctrl_lat_m = float(
                                    np.linalg.norm(
                                        _tip_lat_offset_xy(
                                            tip_ctrl, _mouth_est, _n_est
                                        )
                                    )
                                )
                            elif tip_obs_enable and tip_obs_rng is not None:
                                _n_obs = spiral_n
                                _hu_obs = np.asarray(
                                    feat_i.hole_axis, dtype=np.float64
                                ).reshape(3)
                                _hn_obs = float(np.linalg.norm(_hu_obs))
                                if _hn_obs > 1e-12:
                                    _n_obs = _hu_obs / _hn_obs
                                tip_ctrl, _ = observe_tip_planar(
                                    tip_gt,
                                    _n_obs,
                                    sigma_m=float(tip_obs_sigma_mm) * 1e-3,
                                    rng=tip_obs_rng,
                                    lag_buf=tip_obs_lag_buf,
                                    lag_frames=int(tip_obs_lag_frames),
                                )
                                _mouth_obs = _hole_axis_point_at_tip(
                                    tip_ctrl,
                                    feat_i.socket_pos,
                                    _n_obs,
                                )
                                tip_ctrl_lat_m = float(
                                    np.linalg.norm(
                                        _tip_lat_offset_xy(
                                            tip_ctrl, _mouth_obs, _n_obs
                                        )
                                    )
                                )
                            contact_gate_sp = 0.0
                            unseated_sp = False
                            if (
                                search_mode == "planned_spiral" and stick_phase == "search"
                            ) or franka_faithful:
                                contact_gate_sp, _, _, _ = _wrist_contact()
                                # Hard seat: freeze planar while tip–tray seat obs false
                                # (wrist |r| alone is not tray seat under non-rigid grasp).
                                if require_seat_search:
                                    _sv = _seat_verdict_now()
                                    unseated_sp = not bool(
                                        getattr(_sv, "seated", False)
                                    )
                                else:
                                    unseated_sp = bool(
                                        float(contact_gate_sp) < seat_contact_n
                                    )
                                # Tilted tray/hole face: refresh contact normal from
                                # wrist force so planar track stays on the surface.
                                if search_mode == "planned_spiral" and bool(
                                    a_cfg.get(
                                        "surface_planned_update_contact_normal",
                                        True,
                                    )
                                ):
                                    wr_n = read_wrist_wrench_world(raw)
                                    f_n = np.asarray(
                                        wr_n[0][:3], dtype=np.float64
                                    ).reshape(3)
                                    fn = float(np.linalg.norm(f_n))
                                    fmin = float(
                                        a_cfg.get(
                                            "surface_tip_spiral_contact_fmin_n",
                                            0.03,
                                        )
                                    )
                                    if fn >= fmin:
                                        n_new = f_n / (fn + 1e-12)
                                        hole_u_now = feat_i.hole_axis / (
                                            np.linalg.norm(feat_i.hole_axis) + 1e-12
                                        )
                                        if float(n_new @ hole_u_now) < 0.0:
                                            n_new = -n_new
                                        blend = float(
                                            a_cfg.get(
                                                "surface_planned_normal_blend",
                                                0.15,
                                            )
                                        )
                                        spiral_n = spiral_n + blend * (
                                            n_new - spiral_n
                                        )
                                        spiral_n = spiral_n / (
                                            np.linalg.norm(spiral_n) + 1e-12
                                        )
                                        t1, t2, spiral_n = hole_task_basis(spiral_n)
                                        press_ax_spiral = -spiral_n
                            if (
                                stick_enable
                                and stick_hop_probe
                                and search_mode in _priv_lat_modes
                                and stick_phase == "search"
                                and used_sp == 1
                            ):
                                # User-requested probe: hop-contact-hop from first search step.
                                stick_phase = "lift"
                                stick_phase_i = 0
                                stick_active = True
                                stick_along0 = float(feat_i.along_m)
                                stick_lat_enter = float(feat_i.lateral_m)
                                stick_lat_best = float(feat_i.lateral_m)
                                stick_stall = 0
                                print(
                                    f"pci: hop-probe start lat={stick_lat_enter*1e3:.1f}mm "
                                    f"along={stick_along0*1e3:.1f}mm "
                                    f"hop={stick_slide_step*1e3:.1f}mm",
                                    flush=True,
                                )
                            ax_i = feat_i.hole_axis / (
                                np.linalg.norm(feat_i.hole_axis) + 1e-12
                            )
                            # Tip surface spiral: rotate/press about contact normal.
                            if (
                                spiral_along_hold
                                and search_mode in ("spiral", "planned_spiral")
                                and stick_phase == "search"
                            ):
                                ax_i = spiral_n
                            # Do not spin farther while tip has already drifted outside
                            # the commanded radius (that made lat only grow away).
                            tip_r_gate = float(
                                np.linalg.norm(
                                    _tip_lat_offset_xy(tip, spiral_center, ax_i)
                                )
                            )
                            spiral_track_gate = bool(
                                spiral_along_hold
                                and search_mode in ("spiral", "planned_spiral")
                                and stick_phase == "search"
                            )
                            if (not spiral_track_gate) or tip_r_gate <= max(
                                float(r_cmd) + 0.003, 0.004
                            ):
                                theta_sp = theta_sp + dtheta
                            turn = theta_sp / (2.0 * np.pi)
                            center_i = _hole_axis_point_at_tip(
                                tip, feat_i.socket_pos, feat_i.hole_axis
                            )
                            ax_j = 0.0
                            ay_j = 0.0
                            f_des_step = float(f_des_spiral)
                            stick_active = False
                            d_stick = np.zeros(3, dtype=np.float64)
                            if search_mode in _priv_lat_modes:
                                d_in = _priv_lat_inward_delta(
                                    tip,
                                    center_i,
                                    feat_i.hole_axis,
                                    feat_i.peg_axis,
                                    step_m=priv_lat_step,
                                    axis_ff_gain=priv_lat_axis_ff,
                                )
                                d_liss = np.zeros(3, dtype=np.float64)
                                if search_mode == "priv_lat_lissajous":
                                    lat_i = float(feat_i.lateral_m)
                                    dith_s = float(
                                        np.clip(
                                            priv_dith_lat0 / max(lat_i, 1e-9),
                                            0.15,
                                            1.0,
                                        )
                                    )
                                    ph = float(used_sp)
                                    d_liss = dith_s * (
                                        priv_dith_ax
                                        * np.sin(lis_wx * ph)
                                        * t1
                                        + priv_dith_ay
                                        * np.sin(lis_wy * ph + lis_phi)
                                        * t2
                                    )
                                tip_hist_sp.append(tip.copy())
                                if len(tip_hist_sp) > max(2, stick_win):
                                    tip_hist_sp[:] = tip_hist_sp[-stick_win:]
                                if stick_phase == "search":
                                    if (
                                        stick_enable
                                        and len(tip_hist_sp) >= stick_win
                                        and float(feat_i.lateral_m) > mouth_lat
                                    ):
                                        d_tip_w = tip_hist_sp[-1] - tip_hist_sp[0]
                                        tip_plan = float(
                                            np.linalg.norm(
                                                d_tip_w
                                                - ax_i * float(np.dot(d_tip_w, ax_i))
                                            )
                                        )
                                        if tip_plan < stick_tip_m:
                                            stick_active = True
                                            stick_count += 1
                                            stick_streak += 1
                                            stick_phase = "lift"
                                            stick_phase_i = 0
                                            stick_lat_enter = float(feat_i.lateral_m)
                                            stick_along0 = float(feat_i.along_m)
                                        else:
                                            stick_streak = 0
                                    else:
                                        stick_streak = 0
                                else:
                                    stick_active = True
                                    stick_count += 1
                                    stick_streak += 1
                                target = tip + d_in + d_liss
                                if priv_lat_along_hold and stick_phase == "search":
                                    if along_hold_m is None:
                                        along_hold_m = min(
                                            float(feat_i.along_m),
                                            priv_lat_along_hold_max,
                                        )
                                    d_move = target - tip
                                    d_ax = float(np.dot(d_move, press_ax))
                                    if d_ax > 0.0 or float(feat_i.along_m) > along_hold_m:
                                        target = tip + d_move - press_ax * max(
                                            0.0, d_ax
                                        )
                                r_cmd = float(feat_i.lateral_m)
                                ax_j = float(np.linalg.norm(d_in)) * 1000.0
                                ay_j = float(np.linalg.norm(d_liss)) * 1000.0
                            elif search_mode == "lissajous":
                                if lis_shrink and n_search > 1:
                                    frac = max(
                                        0.0, 1.0 - float(used_sp - 1) / float(n_search - 1)
                                    )
                                    lo = lis_amin / max(lis_ax0, 1e-9)
                                    scale = lo + (1.0 - lo) * frac
                                else:
                                    scale = 1.0
                                ax_j = lis_ax0 * scale
                                ay_j = lis_ay0 * scale
                                ph = float(used_sp)
                                target = spiral_center + ax_j * np.sin(
                                    lis_wx * ph
                                ) * t1 + ay_j * np.sin(lis_wy * ph + lis_phi) * t2
                                r_cmd = float(
                                    np.linalg.norm(
                                        _tip_lat_offset_xy(
                                            target, spiral_center, feat_i.hole_axis
                                        )
                                    )
                                )
                            elif search_mode == "planned_spiral":
                                assert planned_wps is not None and planned_rs is not None
                                # Strong tip FB to current waypoint; advance on planar catch.
                                catch = float(
                                    a_cfg.get("surface_planned_spiral_catch_m", 0.005)
                                )
                                if float(
                                    a_cfg.get("surface_planned_probe_press_m", 0.0)
                                ) > 1e-9:
                                    catch = max(
                                        catch,
                                        float(
                                            a_cfg.get(
                                                "surface_planned_probe_catch_m",
                                                0.0035,
                                            )
                                        ),
                                    )
                                max_adv = int(
                                    a_cfg.get("surface_planned_spiral_max_advance", 3)
                                )
                                soft_miss = int(
                                    a_cfg.get(
                                        "surface_planned_spiral_soft_advance_miss",
                                        25,
                                    )
                                )
                                # Tip-gated advance: wait until tip catches the waypoint.
                                # Open-loop soft_advance=1 made ep01/03 tip lag 20–50mm
                                # while radius shrank to mouth. Default probe miss=0.
                                if (
                                    soft_miss <= 0
                                    and float(
                                        a_cfg.get(
                                            "surface_planned_probe_press_m", 0.0
                                        )
                                    )
                                    > 1e-9
                                ):
                                    soft_miss = int(
                                        a_cfg.get(
                                            "surface_planned_probe_soft_advance_miss",
                                            0,
                                        )
                                    )
                                seat_frames = int(
                                    a_cfg.get(
                                        "surface_planned_force_seat_frames", 0
                                    )
                                )
                                planar_gate_n = float(
                                    a_cfg.get(
                                        "surface_planned_planar_force_gate_n",
                                        0.052,
                                    )
                                )
                                force_seat_sp = (
                                    seat_frames > 0 and used_sp <= seat_frames
                                )
                                freeze_path = force_seat_sp or (
                                    contact_gate_sp
                                    > planar_gate_n
                                )
                                # Probe spiral: never freeze radius advance on mild
                                # overload — that froze ep01 at r≈14.6mm while tip fled.
                                # Overload still slows planar track via force_gate.
                                if (
                                    float(
                                        a_cfg.get(
                                            "surface_planned_probe_press_m", 0.0
                                        )
                                    )
                                    > 1e-9
                                    and bool(
                                        a_cfg.get(
                                            "surface_planned_probe_no_freeze_path",
                                            True,
                                        )
                                    )
                                ):
                                    freeze_path = bool(force_seat_sp)
                                soft_miss_use = 0 if freeze_path else soft_miss
                                # Optional stiction scrub (default off): open-loop
                                # advance under high resid made ep01 tip lag 20mm+.
                                if (
                                    soft_miss_use <= 0
                                    and float(contact_gate_sp) > planar_gate_n
                                    and float(
                                        a_cfg.get(
                                            "surface_planned_probe_press_m", 0.0
                                        )
                                    )
                                    > 1e-9
                                ):
                                    soft_miss_use = int(
                                        a_cfg.get(
                                            "surface_planned_stiction_soft_advance_miss",
                                            0,
                                        )
                                    )
                                advanced = False
                                hyb_allow_adv = (not tip_hybrid_enable) or (
                                    tip_hybrid_state.mode == "spiral"
                                )
                                if unseated_sp:
                                    hyb_allow_adv = False
                                # Tip-MSAR Type B: freeze radius while reseating after slip.
                                if tip_msar_enable and int(
                                    tip_hybrid_state.slip_reseat_left
                                ) > 0:
                                    hyb_allow_adv = False
                                    tip_hybrid_state.slip_reseat_left = max(
                                        0, int(tip_hybrid_state.slip_reseat_left) - 1
                                    )
                                # Tip-STAR / Type-B: freeze while grasp-restore / local-recover.
                                if tip_slip_enable and int(
                                    tip_hybrid_state.star_reseat_left
                                ) > 0:
                                    hyb_allow_adv = False
                                    tip_hybrid_state.star_reseat_left = max(
                                        0, int(tip_hybrid_state.star_reseat_left) - 1
                                    )
                                    if tip_hybrid_state.star_reseat_left == 0 and (
                                        tip_hybrid_state.star_mode == "grasp_restore"
                                    ):
                                        tip_hybrid_state.star_mode = "spiral_track"
                                        print(
                                            "pci: Type-B GRASP_RESTORE done → resume spiral",
                                            flush=True,
                                        )
                                # Agent-D oracle v2: freeze planar while reseat+tighten.
                                if tip_oracle_enable and int(
                                    tip_hybrid_state.oracle_reseat_left
                                ) > 0:
                                    hyb_allow_adv = False
                                    tip_hybrid_state.oracle_reseat_left = max(
                                        0, int(tip_hybrid_state.oracle_reseat_left) - 1
                                    )
                                    if tip_hybrid_state.oracle_reseat_left == 0:
                                        print(
                                            "pci: ORACLE slip-reseat done → resume tip→mouth",
                                            flush=True,
                                        )
                                # Agent-R: EKF slip / large tip lag → freeze radius advance
                                # until tip catches (Type B).
                                if (
                                    tip_coupling_ekf is not None
                                    and tip_baseline_mode in ("ekf_qp", "s9")
                                    and bool(
                                        a_cfg.get(
                                            "surface_tip_ekf_freeze_adv_on_slip",
                                            True,
                                        )
                                    )
                                ):
                                    tip_err_mm = 0.0
                                    if planned_wps is not None:
                                        e_lag = planned_wps[planned_i] - tip
                                        e_lag = e_lag - spiral_n * float(
                                            np.dot(e_lag, spiral_n)
                                        )
                                        tip_err_mm = float(np.linalg.norm(e_lag)) * 1e3
                                    err_freeze = float(
                                        a_cfg.get(
                                            "surface_tip_ekf_freeze_tip_err_mm",
                                            8.0,
                                        )
                                    )
                                    if bool(tip_coupling_ekf.just_slipped) or (
                                        tip_err_mm > err_freeze
                                    ):
                                        hyb_allow_adv = False
                                if tip_msar_enable and planned_wps is not None:
                                    e_ms = planned_wps[planned_i] - tip
                                    e_ms = e_ms - spiral_n * float(
                                        np.dot(e_ms, spiral_n)
                                    )
                                    if tip_msar_slip_trip(
                                        grasp_slip_m=float(grasp_slip_peak),
                                        tip_err_m=float(np.linalg.norm(e_ms)),
                                        slip_tau_m=float(
                                            a_cfg.get(
                                                "surface_tip_msar_slip_tau_m", 0.08
                                            )
                                        ),
                                        tip_err_tau_m=float(
                                            a_cfg.get(
                                                "surface_tip_msar_tip_err_tau_m",
                                                0.012,
                                            )
                                        ),
                                    ):
                                        hyb_allow_adv = False
                                        tip_hybrid_state.msar_mode = "slip_reseat"
                                        tip_hybrid_state.slip_reseat_left = max(
                                            int(tip_hybrid_state.slip_reseat_left),
                                            int(
                                                a_cfg.get(
                                                    "surface_tip_msar_reseat_frames",
                                                    25,
                                                )
                                            ),
                                        )
                                        if tip_planar_coupling_est is not None:
                                            tip_planar_coupling_est._slip_reset()
                                        if tip_coupling_ekf is not None:
                                            tip_coupling_ekf.slip_reset()
                                if tip_slip_enable and planned_wps is not None:
                                    e_st = planned_wps[planned_i] - tip
                                    e_st = e_st - spiral_n * float(
                                        np.dot(e_st, spiral_n)
                                    )
                                    tip_off_st = _tip_lat_offset_xy(
                                        tip, spiral_center, spiral_n
                                    )
                                    tip_rho_st = float(np.linalg.norm(tip_off_st))
                                    r_cmd_st = float(planned_rs[planned_i])
                                    tip_err_st = float(np.linalg.norm(e_st))
                                    slip_tau = float(
                                        a_cfg.get(
                                            "surface_tip_slip_tau_m",
                                            a_cfg.get(
                                                "surface_tip_star_slip_tau_m", 0.12
                                            ),
                                        )
                                    )
                                    slip_tip_err = float(
                                        a_cfg.get(
                                            "surface_tip_slip_tip_err_m",
                                            a_cfg.get(
                                                "surface_tip_star_slip_tip_err_m",
                                                0.020,
                                            ),
                                        )
                                    )
                                    slip_lag = float(
                                        a_cfg.get(
                                            "surface_tip_slip_lag_margin_m",
                                            a_cfg.get(
                                                "surface_tip_star_lag_margin_m",
                                                0.003,
                                            ),
                                        )
                                    )
                                    trip_st = tip_star_slip_trip(
                                        grasp_slip_m=float(grasp_slip_peak),
                                        tip_err_m=tip_err_st,
                                        r_cmd_m=r_cmd_st,
                                        tip_rho_m=tip_rho_st,
                                        slip_tau_m=slip_tau,
                                        tip_err_tau_m=slip_tip_err,
                                        lag_margin_m=slip_lag,
                                        spiral_direction=str(spiral_dir),
                                        # Latch lives in tip_slip_can_enter_restore;
                                        # do not double-filter here.
                                        slip_armed_m=-1.0,
                                    )
                                    already = (
                                        tip_hybrid_state.star_mode == "grasp_restore"
                                        or int(tip_hybrid_state.star_reseat_left) > 0
                                    )
                                    if tip_slip_can_enter_restore(
                                        trip=trip_st,
                                        already_restoring=already,
                                        last_trip_slip_m=float(
                                            tip_hybrid_state.star_slip_armed_m
                                        ),
                                        slip_now_m=float(grasp_slip_peak),
                                        retrip_delta_m=float(
                                            a_cfg.get(
                                                "surface_tip_slip_retrip_delta_m",
                                                a_cfg.get(
                                                    "surface_tip_star_retrip_delta_m",
                                                    0.05,
                                                ),
                                            )
                                        ),
                                    ):
                                        hyb_allow_adv = False
                                        tip_hybrid_state.star_mode = "grasp_restore"
                                        tip_hybrid_state.star_slip_armed_m = float(
                                            grasp_slip_peak
                                        )
                                        tip_hybrid_state.star_reseat_left = max(
                                            int(tip_hybrid_state.star_reseat_left),
                                            int(
                                                a_cfg.get(
                                                    "surface_tip_slip_reseat_frames",
                                                    a_cfg.get(
                                                        "surface_tip_star_reseat_frames",
                                                        30,
                                                    ),
                                                )
                                            ),
                                        )
                                        # Reseat SEAT: freeze planar advance + hybrid SEAT.
                                        if bool(
                                            a_cfg.get(
                                                "surface_tip_slip_force_seat",
                                                a_cfg.get(
                                                    "surface_tip_star_force_seat",
                                                    True,
                                                ),
                                            )
                                        ):
                                            tip_hybrid_state.mode = "seat"
                                            tip_hybrid_state.seat_ok_streak = 0
                                            tip_hybrid_state.upright_ok_streak = 0
                                        if tip_planar_coupling_est is not None:
                                            tip_planar_coupling_est._slip_reset()
                                        if tip_coupling_ekf is not None:
                                            tip_coupling_ekf.slip_reset()
                                        # Real grasp tighten — must bypass handoff freeze.
                                        boost = float(
                                            a_cfg.get(
                                                "surface_tip_slip_grasp_scale",
                                                a_cfg.get(
                                                    "surface_tip_star_grasp_scale",
                                                    1.45,
                                                ),
                                            )
                                        )
                                        bypass = bool(
                                            a_cfg.get(
                                                "surface_tip_slip_grasp_bypass_freeze",
                                                a_cfg.get(
                                                    "surface_tip_star_grasp_bypass_freeze",
                                                    True,
                                                ),
                                            )
                                        )
                                        tight = _tighten_left_grasp(
                                            max(float(grasp_ramp_scale), boost),
                                            bypass_freeze=bypass,
                                        )
                                        tip_hybrid_state.star_grasp_boosted = bool(
                                            tip_hybrid_state.star_grasp_boosted
                                            or tight.get("ok")
                                        )
                                        if bool(tight.get("ok")):
                                            tip_hybrid_state.star_grasp_mean_abs_before = (
                                                float(tight.get("mean_abs_before", 0.0))
                                            )
                                            tip_hybrid_state.star_grasp_mean_abs_after = (
                                                float(tight.get("mean_abs_after", 0.0))
                                            )
                                            meta["tip_slip_grasp_tighten_ok"] = True
                                            meta["tip_slip_hand_mean_abs_before"] = (
                                                float(tight.get("mean_abs_before", 0.0))
                                            )
                                            meta["tip_slip_hand_mean_abs_after"] = float(
                                                tight.get("mean_abs_after", 0.0)
                                            )
                                        meta["tip_slip_grasp_bypass_freeze"] = bool(
                                            tight.get("bypassed_freeze")
                                        )
                                        meta["tip_slip_grasp_scale"] = float(
                                            tight.get("scale", boost)
                                        )
                                        meta["tip_slip_trips"] = int(
                                            meta.get("tip_slip_trips", 0)
                                        ) + 1
                                        print(
                                            f"pci: Type-B GRASP_RESTORE trip "
                                            f"slip={grasp_slip_peak*1e3:.1f}mm "
                                            f"tighten_ok={bool(tight.get('ok'))} "
                                            f"mean|q|="
                                            f"{float(tight.get('mean_abs_before', 0)):.3f}"
                                            f"→{float(tight.get('mean_abs_after', 0)):.3f}",
                                            flush=True,
                                        )
                                # Agent-D oracle v2: large slip → freeze planar, reseat+tighten,
                                # then resume tip→mouth (NOT deployable).
                                if tip_oracle_enable and bool(
                                    a_cfg.get("surface_tip_oracle_slip_reseat", False)
                                ):
                                    trip_or = tip_oracle_slip_trip(
                                        grasp_slip_m=float(grasp_slip_peak),
                                        tip_lat_m=float(
                                            tip_ctrl_lat_m
                                            if (tip_obs_enable or tip_est_enable)
                                            else feat_i.lateral_m
                                        ),
                                        slip_tau_m=float(
                                            a_cfg.get(
                                                "surface_tip_oracle_slip_tau_m", 0.035
                                            )
                                        ),
                                        tip_lag_m=float(
                                            a_cfg.get(
                                                "surface_tip_oracle_tip_lag_m", 0.008
                                            )
                                        ),
                                        mouth_m=float(
                                            a_cfg.get(
                                                "surface_spiral_mouth_lat_m", 0.0045
                                            )
                                        ),
                                        slip_armed_m=float(
                                            tip_hybrid_state.oracle_slip_armed_m
                                        ),
                                    )
                                    already_or = (
                                        int(tip_hybrid_state.oracle_reseat_left) > 0
                                    )
                                    if tip_slip_can_enter_restore(
                                        trip=trip_or,
                                        already_restoring=already_or,
                                        last_trip_slip_m=float(
                                            tip_hybrid_state.oracle_slip_armed_m
                                        ),
                                        slip_now_m=float(grasp_slip_peak),
                                        retrip_delta_m=float(
                                            a_cfg.get(
                                                "surface_tip_oracle_retrip_delta_m",
                                                0.04,
                                            )
                                        ),
                                    ):
                                        hyb_allow_adv = False
                                        tip_hybrid_state.oracle_slip_armed_m = float(
                                            grasp_slip_peak
                                        )
                                        tip_hybrid_state.oracle_reseat_left = max(
                                            int(tip_hybrid_state.oracle_reseat_left),
                                            int(
                                                a_cfg.get(
                                                    "surface_tip_oracle_reseat_frames",
                                                    40,
                                                )
                                            ),
                                        )
                                        # Soft SEAT cue only if floated; keep mouth PD.
                                        float_or = max(
                                            0.0,
                                            _priv_tip_surface_depth_m(
                                                tip, tip_sp0, spiral_n
                                            ),
                                        )
                                        if float_or > float(
                                            a_cfg.get(
                                                "surface_tip_hybrid_float_tol_m",
                                                0.0015,
                                            )
                                        ):
                                            tip_hybrid_state.mode = "seat"
                                            tip_hybrid_state.seat_ok_streak = 0
                                        tip_hybrid_state.upright_ok_streak = 0
                                        boost_or = float(
                                            a_cfg.get(
                                                "surface_tip_oracle_grasp_scale", 1.55
                                            )
                                        )
                                        tight_or = _tighten_left_grasp(
                                            max(float(grasp_ramp_scale), boost_or),
                                            bypass_freeze=bool(
                                                a_cfg.get(
                                                    "surface_tip_oracle_grasp_bypass_freeze",
                                                    True,
                                                )
                                            ),
                                        )
                                        tip_hybrid_state.oracle_grasp_boosted = bool(
                                            tip_hybrid_state.oracle_grasp_boosted
                                            or tight_or.get("ok")
                                        )
                                        meta["tip_oracle_slip_trips"] = int(
                                            meta.get("tip_oracle_slip_trips", 0)
                                        ) + 1
                                        print(
                                            f"pci: ORACLE slip-reseat trip "
                                            f"slip={grasp_slip_peak*1e3:.1f}mm "
                                            f"lat={feat_i.lateral_m*1e3:.1f}mm "
                                            f"tighten_ok={bool(tight_or.get('ok'))}",
                                            flush=True,
                                        )
                                # Track-B PHIG: proprio slip + r_cmd/F/T mouth (no tip GT).
                                if tip_phig_enable and planned_wps is not None:
                                    r_cmd_ph = float(planned_rs[planned_i])
                                    axis_deg_ph = float(
                                        np.degrees(
                                            float(
                                                getattr(
                                                    feat_i, "axis_error_rad", 0.0
                                                )
                                            )
                                        )
                                    )
                                    # contact_sp is assigned later in the frame; read now.
                                    _c_ph, _, _, _ = _wrist_contact()
                                    prev_phig_mode = str(phig_state.mode)
                                    prev_phig_reseat = int(phig_state.reseat_left)
                                    phig_state = phig_select_mode(
                                        phig_state,
                                        hybrid_mode=str(tip_hybrid_state.mode),
                                        r_cmd_m=r_cmd_ph,
                                        contact_resid_n=float(_c_ph),
                                        force_hole_cue=bool(force_hole_state.entered),
                                        grasp_slip_m=float(grasp_slip_peak),
                                        axis_err_deg=axis_deg_ph,
                                        cfg=phig_cfg,
                                    )
                                    if phig_state.mode in (
                                        "slip_reseat",
                                        "mouth_press",
                                        "enter_hold",
                                    ):
                                        hyb_allow_adv = False
                                    if phig_state.mode == "enter_hold":
                                        tip_hybrid_state.mode = "upright"
                                        tip_hybrid_state.upright_ok_streak = 0
                                    new_slip = (
                                        phig_state.mode == "slip_reseat"
                                        and (
                                            prev_phig_reseat <= 0
                                            or prev_phig_mode != "slip_reseat"
                                        )
                                    )
                                    if new_slip:
                                        float_ph = max(
                                            0.0,
                                            _priv_tip_surface_depth_m(
                                                tip, tip_sp0, spiral_n
                                            ),
                                        )
                                        if float_ph > float(
                                            a_cfg.get(
                                                "surface_tip_hybrid_float_tol_m",
                                                0.0015,
                                            )
                                        ):
                                            tip_hybrid_state.mode = "seat"
                                            tip_hybrid_state.seat_ok_streak = 0
                                        tip_hybrid_state.upright_ok_streak = 0
                                        if tip_planar_coupling_est is not None:
                                            tip_planar_coupling_est._slip_reset()
                                        if tip_coupling_ekf is not None:
                                            tip_coupling_ekf.slip_reset()
                                        boost_ph = float(phig_cfg.grasp_scale)
                                        tight_ph = _tighten_left_grasp(
                                            max(float(grasp_ramp_scale), boost_ph),
                                            bypass_freeze=bool(
                                                a_cfg.get(
                                                    "surface_tip_phig_grasp_bypass_freeze",
                                                    True,
                                                )
                                            ),
                                        )
                                        phig_state.grasp_boosted = bool(
                                            phig_state.grasp_boosted
                                            or tight_ph.get("ok")
                                        )
                                        meta["tip_phig_slip_trips"] = int(
                                            meta.get("tip_phig_slip_trips", 0)
                                        ) + 1
                                        meta["tip_phig_grasp_bypass_freeze"] = bool(
                                            tight_ph.get("bypassed_freeze")
                                        )
                                        meta["tip_phig_grasp_scale"] = float(
                                            tight_ph.get("scale", boost_ph)
                                        )
                                        if bool(tight_ph.get("ok")):
                                            meta["tip_phig_grasp_tighten_ok"] = True
                                            meta["tip_phig_hand_mean_abs_before"] = (
                                                float(
                                                    tight_ph.get(
                                                        "mean_abs_before", 0.0
                                                    )
                                                )
                                            )
                                            meta["tip_phig_hand_mean_abs_after"] = (
                                                float(
                                                    tight_ph.get(
                                                        "mean_abs_after", 0.0
                                                    )
                                                )
                                            )
                                        print(
                                            f"pci: PHIG slip-reseat trip "
                                            f"slip={grasp_slip_peak*1e3:.1f}mm "
                                            f"r_cmd={r_cmd_ph*1e3:.1f}mm "
                                            f"tighten_ok={bool(tight_ph.get('ok'))}",
                                            flush=True,
                                        )
                                if not force_seat_sp and hyb_allow_adv:
                                    for _adv in range(max(1, max_adv)):
                                        target = planned_wps[planned_i].copy()
                                        e_pl = target - tip
                                        e_pl = e_pl - spiral_n * float(
                                            np.dot(e_pl, spiral_n)
                                        )
                                        if float(np.linalg.norm(e_pl)) > catch:
                                            break
                                        if planned_i + 1 >= planned_wps.shape[0]:
                                            break
                                        planned_i += 1
                                        advanced = True
                                else:
                                    target = planned_wps[planned_i].copy()
                                if advanced:
                                    catch_miss_sp = 0
                                else:
                                    catch_miss_sp += 1
                                    # Soft-advance only when enabled (>0); else follow path.
                                    if (
                                        hyb_allow_adv
                                        and soft_miss_use > 0
                                        and catch_miss_sp >= soft_miss_use
                                        and planned_i + 1 < planned_wps.shape[0]
                                    ):
                                        planned_i += 1
                                        catch_miss_sp = 0
                                target = planned_wps[planned_i].copy()
                                r_cmd = float(planned_rs[planned_i])
                                theta_sp = float(planned_ths[planned_i])
                                # Tip-MSAR Type A: blend spiral waypoint → mouth attractor.
                                if tip_msar_enable and tip_hybrid_state.mode == "spiral":
                                    target, msar_alpha, msar_m = tip_msar_blend_target(
                                        tip=tip,
                                        spiral_target=target,
                                        hole_center=spiral_center,
                                        normal=spiral_n,
                                        mouth_m=float(
                                            a_cfg.get(
                                                "surface_tip_msar_mouth_m",
                                                a_cfg.get(
                                                    "surface_spiral_mouth_lat_m",
                                                    0.0045,
                                                ),
                                            )
                                        ),
                                        blend_m=float(
                                            a_cfg.get(
                                                "surface_tip_msar_blend_m", 0.012
                                            )
                                        ),
                                    )
                                    if tip_hybrid_state.msar_mode != "slip_reseat":
                                        tip_hybrid_state.msar_mode = str(msar_m)
                                    tip_msar_counts[tip_hybrid_state.msar_mode] = (
                                        tip_msar_counts.get(
                                            tip_hybrid_state.msar_mode, 0
                                        )
                                        + 1
                                    )
                                    # Far from mouth: do not expand spiral further.
                                    if float(msar_alpha) > 0.35:
                                        hyb_allow_adv = False
                                # Tip-STAR: track-saturate local recover (no hole PD).
                                # Type-B-only (`surface_tip_slip_enable`) counts restore
                                # frames without enabling Type-A local_recover / yield.
                                if (
                                    tip_slip_enable
                                    and tip_hybrid_state.star_mode == "grasp_restore"
                                ):
                                    tip_star_counts["grasp_restore"] = (
                                        tip_star_counts.get("grasp_restore", 0) + 1
                                    )
                                    hyb_allow_adv = False
                                elif tip_star_enable and tip_hybrid_state.mode == "spiral":
                                    tip_off_sr = _tip_lat_offset_xy(
                                        tip, spiral_center, spiral_n
                                    )
                                    tip_rho_sr = float(np.linalg.norm(tip_off_sr))
                                    e_sr = target - tip
                                    e_sr = e_sr - spiral_n * float(
                                        np.dot(e_sr, spiral_n)
                                    )
                                    tip_err_sr = float(np.linalg.norm(e_sr))
                                    mouth_sr = float(
                                        a_cfg.get(
                                            "surface_tip_star_mouth_m",
                                            a_cfg.get(
                                                "surface_spiral_mouth_lat_m",
                                                0.0045,
                                            ),
                                        )
                                    )
                                    force_cue = bool(
                                        force_hole_detect
                                        and force_hole_state.entered
                                    )
                                    if tip_star_yield_mouth(
                                        r_cmd_m=float(r_cmd),
                                        tip_rho_m=tip_rho_sr,
                                        tip_err_m=tip_err_sr,
                                        mouth_m=mouth_sr,
                                        tip_err_near_m=float(
                                            a_cfg.get(
                                                "surface_tip_star_near_err_m",
                                                0.008,
                                            )
                                        ),
                                        force_mouth_cue=force_cue,
                                    ):
                                        tip_hybrid_state.star_mode = "yield_mouth"
                                        tip_hybrid_state.star_sat_streak = 0
                                        tip_star_counts["yield_mouth"] = (
                                            tip_star_counts.get("yield_mouth", 0) + 1
                                        )
                                        hyb_allow_adv = False
                                        # Keep spiral target; SEAT/mouth-probe own enter.
                                    else:
                                        sat = tip_star_track_saturate(
                                            tip_err_m=tip_err_sr,
                                            r_cmd_m=float(r_cmd),
                                            tip_rho_m=tip_rho_sr,
                                            tip_err_tau_m=float(
                                                a_cfg.get(
                                                    "surface_tip_star_sat_err_m",
                                                    0.010,
                                                )
                                            ),
                                            lag_margin_m=float(
                                                a_cfg.get(
                                                    "surface_tip_star_lag_margin_m",
                                                    0.003,
                                                )
                                            ),
                                            spiral_direction=str(spiral_dir),
                                        )
                                        if sat:
                                            tip_hybrid_state.star_sat_streak = (
                                                int(tip_hybrid_state.star_sat_streak)
                                                + 1
                                            )
                                        else:
                                            tip_hybrid_state.star_sat_streak = 0
                                        n_sat = int(
                                            a_cfg.get(
                                                "surface_tip_star_sat_frames", 8
                                            )
                                        )
                                        # STAR ep02 lesson: never planar-recover while
                                        # axis is soft-bad — Tip-Hybrid UPRIGHT first.
                                        _, axis_err_sat, _ = _tip_hole_pose_err(
                                            tip,
                                            spiral_center,
                                            feat_i.hole_axis,
                                            feat_i.peg_axis,
                                        )
                                        upright_soft_sat = float(
                                            np.radians(
                                                a_cfg.get(
                                                    "surface_tip_hybrid_upright_soft_deg",
                                                    18.0,
                                                )
                                            )
                                        )
                                        axis_blocks_recover = bool(
                                            a_cfg.get(
                                                "surface_tip_star_upright_before_recover",
                                                True,
                                            )
                                        ) and float(axis_err_sat) > upright_soft_sat
                                        if (
                                            int(tip_hybrid_state.star_sat_streak)
                                            >= n_sat
                                            and not axis_blocks_recover
                                        ):
                                            tip_hybrid_state.star_mode = "local_recover"
                                            target = tip_star_local_target(
                                                tip=tip,
                                                spiral_target=target,
                                                spiral_center=spiral_center,
                                                normal=spiral_n,
                                                r_cmd_m=float(r_cmd),
                                                spiral_direction=str(spiral_dir),
                                                mouth_m=float(mouth_sr),
                                                shrink_step_m=float(
                                                    a_cfg.get(
                                                        "surface_tip_star_recover_shrink_m",
                                                        0.001,
                                                    )
                                                ),
                                            )
                                            hyb_allow_adv = False
                                            tip_star_counts["local_recover"] = (
                                                tip_star_counts.get(
                                                    "local_recover", 0
                                                )
                                                + 1
                                            )
                                        elif axis_blocks_recover and (
                                            sat
                                            or int(tip_hybrid_state.star_sat_streak)
                                            >= n_sat
                                        ):
                                            # Force hybrid UPRIGHT — do not keep
                                            # planar spiral target that tips peg ~50°.
                                            tip_hybrid_state.mode = "upright"
                                            tip_hybrid_state.upright_ok_streak = 0
                                            tip_hybrid_state.star_mode = "yield_upright"
                                            tip_star_counts["yield_upright"] = (
                                                tip_star_counts.get(
                                                    "yield_upright", 0
                                                )
                                                + 1
                                            )
                                            hyb_allow_adv = False
                                        else:
                                            tip_hybrid_state.star_mode = "spiral_track"
                                            tip_star_counts["spiral_track"] = (
                                                tip_star_counts.get(
                                                    "spiral_track", 0
                                                )
                                                + 1
                                            )
                                # Agent-D oracle: privileged tip→mouth (not spiral chase).
                                # Mouth = live hole-axis point at tip height (same geom as
                                # feat.lateral_m / priv_planar). Frozen spiral_center can
                                # diverge when spiral_n ≠ hole_axis (ep02: tip_err≈0 while
                                # priv_planar≈15mm).
                                # v2: keep mouth PD during SEAT/reseat (ep08 tip lag).
                                if tip_oracle_enable and (
                                    (not tip_hybrid_enable)
                                    or tip_hybrid_state.mode
                                    in ("spiral", "upright", "seat")
                                ):
                                    hole_u_or = np.asarray(
                                        feat_i.hole_axis, dtype=np.float64
                                    ).reshape(3)
                                    hn = float(np.linalg.norm(hole_u_or))
                                    if hn > 1e-12:
                                        hole_u_or = hole_u_or / hn
                                    # Control tip: tip_hat under obs_noise, else GT.
                                    tip_servo = tip_ctrl
                                    mouth_or = _hole_axis_point_at_tip(
                                        tip_servo,
                                        feat_i.socket_pos,
                                        hole_u_or,
                                    )
                                    target = priv_oracle_mouth_target(
                                        tip=tip_servo,
                                        hole_center=mouth_or,
                                        normal=hole_u_or,
                                    )
                                    hyb_allow_adv = False
                                    tip_lat_or = float(
                                        np.linalg.norm(
                                            _tip_lat_offset_xy(
                                                tip_servo, mouth_or, hole_u_or
                                            )
                                        )
                                    )
                                    if tip_obs_enable or tip_est_enable:
                                        tip_lat_or = max(
                                            tip_lat_or, float(tip_ctrl_lat_m)
                                        )
                                    else:
                                        # Prefer feature lat (audit metric) when GT.
                                        tip_lat_or = max(
                                            tip_lat_or, float(feat_i.lateral_m)
                                        )
                                    r_cmd = max(tip_lat_or, 1e-6)
                                # Tip drifted outside commanded ring (ep08/09): blend
                                # target toward hole-centered radius so tip is pulled
                                # back instead of chasing a lost spiral phase.
                                tip_off = _tip_lat_offset_xy(
                                    tip, spiral_center, spiral_n
                                )
                                tip_lat_n = float(np.linalg.norm(tip_off))
                                if (
                                    (not tip_oracle_enable)
                                    and tip_lat_n > float(r_cmd) + 0.008
                                    and tip_lat_n > 1e-9
                                ):
                                    ring = spiral_center + tip_off * (
                                        float(r_cmd) / tip_lat_n
                                    )
                                    target = 0.5 * target + 0.5 * ring
                            elif search_mode == "conntact_time_spiral":
                                # Faithful ConnTact SpiralToFindHole time spiral.
                                t_sec = float(used_sp) * float(conntact_dt)
                                target, amp_ct = spiral_target_on_plane(
                                    t_sec,
                                    center=spiral_center,
                                    t1=t1,
                                    t2=t2,
                                    params=conntact_params,
                                )
                                r_cmd = float(amp_ct)
                            elif search_mode == "franka_spiral":
                                # Faithful Franka jam spiral (ScriptedExpertPegV1).
                                tip_h_now = float(tip[2])
                                surf_h = (
                                    float(franka_surface_h)
                                    if franka_surface_h is not None
                                    else tip_h_now
                                )
                                above_hole = tip_h_now > (
                                    surf_h - float(franka_params.height_drop_m)
                                )
                                target, franka_meta_last = (
                                    franka_spiral_target_on_plane(
                                        franka_state,
                                        center=spiral_center,
                                        t1=t1,
                                        t2=t2,
                                        normal=spiral_n,
                                        upward_force_n=float(contact_gate_sp),
                                        above_hole=bool(above_hole),
                                        params=franka_params,
                                    )
                                )
                                r_cmd = float(franka_meta_last.get("r_m", 0.0))
                            elif search_mode == "psft_spiral":
                                # Park RA-L 2020 PSFT eq (10) → p_t on plane.
                                target, psft_meta_last = psft_spiral_target_on_plane(
                                    psft_state,
                                    center=spiral_center,
                                    t1=t1,
                                    t2=t2,
                                    params=psft_params,
                                )
                                r_cmd = float(psft_meta_last.get("r_m", 0.0))
                            elif spiral_dir == "inward":
                                r_cmd = max(r_min, r0 - pitch * turn)
                                target = spiral_center + r_cmd * (
                                    np.cos(theta_sp) * t1 + np.sin(theta_sp) * t2
                                )
                            else:
                                r_cmd = min(r_max, r0 + pitch * turn)
                                target = spiral_center + r_cmd * (
                                    np.cos(theta_sp) * t1 + np.sin(theta_sp) * t2
                                )
                            err = target - tip
                            # ConnTact SpiralToFindHole: command XY only; z=current
                            # (force/wrench owns seeking). Dual-arm yield breaks stiction.
                            conntact_surface_spiral = bool(
                                spiral_along_hold
                                and stick_phase == "search"
                                and search_mode
                                in (
                                    "spiral",
                                    "planned_spiral",
                                    "conntact_time_spiral",
                                    "franka_spiral",
                                    "psft_spiral",
                                )
                            )
                            if stick_phase == "search":
                                # Always planar pose error on contact/hole axis.
                                err = err - ax_i * float(np.dot(err, ax_i))
                            if conntact_surface_spiral:
                                if along_hold_m is None:
                                    along_hold_m = float(np.dot(tip_sp0, spiral_n))
                                surface_lock_err = bool(
                                    search_mode == "planned_spiral"
                                    and a_cfg.get(
                                        "surface_planned_spiral_surface_lock",
                                        True,
                                    )
                                )
                                if not surface_lock_err:
                                    err = err + press_ax_spiral * 0.0002
                                    depth_now = float(np.dot(tip, spiral_n))
                                    if depth_now > along_hold_m + 0.0010:
                                        err = err + press_ax_spiral * min(
                                            0.0015,
                                            0.8 * (depth_now - along_hold_m),
                                        )
                            if (
                                stick_phase == "search"
                                and (
                                    lat_shrink_only
                                    or search_mode
                                    in (
                                        "lissajous",
                                        *_priv_lat_modes,
                                    )
                                )
                            ):
                                off_xy = _tip_lat_offset_xy(
                                    tip, spiral_center, feat_i.hole_axis
                                )
                                dist = float(np.linalg.norm(off_xy))
                                if dist > 1e-9:
                                    radial_u = off_xy / dist
                                    outward = float(np.dot(err, radial_u))
                                    if outward > 0.0:
                                        err = err - radial_u * outward
                            en = float(np.linalg.norm(err))
                            # Allow larger chase when tip lags spiral (sticky surface).
                            step_lim = max_xy
                            track_use = float(track)
                            planned_stuck = False
                            if search_mode == "planned_spiral" and stick_phase == "search":
                                step_lim = max(
                                    step_lim,
                                    float(
                                        a_cfg.get(
                                            "surface_planned_spiral_max_step_m", 0.006
                                        )
                                    ),
                                )
                                track_use = max(
                                    track_use,
                                    float(
                                        a_cfg.get(
                                            "surface_planned_spiral_track", 1.2
                                        )
                                    ),
                                )
                                tip_move = float(
                                    np.linalg.norm(tip - tip_prev_stuck)
                                )
                                tip_prev_stuck = tip.copy()
                                # Soften if already lifting; force-gated stuck starts after FT.
                                if micro_lift_left > 0:
                                    planned_stuck = True
                                    track_use = float(
                                        a_cfg.get(
                                            "surface_planned_stuck_track", 0.30
                                        )
                                    )
                                    step_lim = float(
                                        a_cfg.get(
                                            "surface_planned_stuck_max_step_m",
                                            0.0012,
                                        )
                                    )
                                elif recontact_left > 0:
                                    # Hold XY nearly still; press into surface.
                                    track_use = float(
                                        a_cfg.get(
                                            "surface_planned_recontact_track", 0.25
                                        )
                                    )
                                    step_lim = float(
                                        a_cfg.get(
                                            "surface_planned_recontact_max_step_m",
                                            0.0010,
                                        )
                                    )
                                else:
                                    slide_err_m = float(
                                        a_cfg.get(
                                            "surface_planned_slide_err_m", 0.003
                                        )
                                    )
                                    surface_lock = bool(
                                        a_cfg.get(
                                            "surface_planned_spiral_surface_lock",
                                            True,
                                        )
                                    )
                                    if (
                                        en > slide_err_m
                                        and not surface_lock
                                    ):
                                        track_use = max(
                                            track_use,
                                            float(
                                                a_cfg.get(
                                                    "surface_planned_slide_track",
                                                    1.6,
                                                )
                                            ),
                                        )
                                        step_lim = max(
                                            step_lim,
                                            float(
                                                a_cfg.get(
                                                    "surface_planned_slide_max_step_m",
                                                    0.008,
                                                )
                                            ),
                                        )
                            if en > 0.004 or stick_phase != "search":
                                step_lim = max(step_lim, 0.006)
                            if stick_phase == "slide":
                                step_lim = max(step_lim, stick_slide_step)
                            if en > step_lim > 0.0:
                                err = err * (step_lim / en)
                            d_xy = err * track_use
                            contact, left_c, fz_r, fz_l = _wrist_contact()
                            contact_sp = float(contact)
                            spiral_resid_peak = max(spiral_resid_peak, contact_sp)
                            # Tip-hybrid FSM (SEAT→UPRIGHT→SPIRAL). Uses tip-tray
                            # contact when available; wrist resid as deployable fallback.
                            tip_hybrid_omega = np.zeros(3, dtype=np.float64)
                            tip_hybrid_soft_seat = False
                            if (
                                tip_hybrid_enable
                                and search_mode == "planned_spiral"
                                and stick_phase == "search"
                            ):
                                float_m_h = max(
                                    0.0,
                                    _priv_tip_surface_depth_m(tip, tip_sp0, spiral_n),
                                )
                                # Floating over mouth: lat small but along still high.
                                mouth_lat_h = float(
                                    a_cfg.get("surface_spiral_mouth_lat_m", 0.0045)
                                )
                                along_cap_h = float(
                                    a_cfg.get(
                                        "surface_tip_hybrid_max_along_m", 0.100
                                    )
                                )
                                near_mouth_h = float(feat_i.lateral_m) <= mouth_lat_h * 3.0
                                along_high_h = float(feat_i.along_m) > along_cap_h
                                if near_mouth_h and along_high_h:
                                    float_m_h = max(
                                        float_m_h,
                                        float(feat_i.along_m) - along_cap_h,
                                    )
                                priv_fn_h, priv_n_h = _priv_peg_tray_contact_force_n(
                                    raw
                                )
                                seat_n_h = (
                                    float(priv_fn_h)
                                    if int(priv_n_h) > 0
                                    else float(contact_sp)
                                )
                                _, axis_err_h, _ = _tip_hole_pose_err(
                                    tip,
                                    spiral_center,
                                    feat_i.hole_axis,
                                    feat_i.peg_axis,
                                )
                                # Hard SEAT only on geometric float / mouth-high-along.
                                # Low contact alone → soft press boost, still spiral.
                                hard_seat_h = float_m_h > float(
                                    a_cfg.get(
                                        "surface_tip_hybrid_float_tol_m", 0.0015
                                    )
                                ) or (near_mouth_h and along_high_h)
                                hyb_mode = tip_hybrid_select(
                                    tip_hybrid_state,
                                    float_m=float_m_h,
                                    contact_n=seat_n_h,
                                    axis_err_rad=float(axis_err_h),
                                    seat_float_tol_m=float(
                                        a_cfg.get(
                                            "surface_tip_hybrid_float_tol_m", 0.0015
                                        )
                                    ),
                                    seat_contact_n=float(
                                        a_cfg.get(
                                            "surface_tip_hybrid_seat_contact_n", 0.05
                                        )
                                    ),
                                    upright_soft_rad=float(
                                        np.radians(
                                            a_cfg.get(
                                                "surface_tip_hybrid_upright_soft_deg",
                                                18.0,
                                            )
                                        )
                                    ),
                                    upright_confirm=int(
                                        a_cfg.get(
                                            "surface_tip_hybrid_upright_confirm", 3
                                        )
                                    ),
                                    hard_seat=bool(hard_seat_h),
                                )
                                # Type-B GRASP_RESTORE: keep Tip-Hybrid on SEAT while
                                # spiral advance is frozen (reseat face contact).
                                if tip_slip_enable and (
                                    tip_hybrid_state.star_mode == "grasp_restore"
                                ):
                                    tip_hybrid_state.mode = "seat"
                                    tip_hybrid_state.seat_ok_streak = 0
                                    hyb_mode = "seat"
                                tip_hybrid_mode_counts[hyb_mode] = (
                                    tip_hybrid_mode_counts.get(hyb_mode, 0) + 1
                                )
                                # Soft reseat: boost press without freezing spiral.
                                if (
                                    hyb_mode == "spiral"
                                    and seat_n_h
                                    < float(
                                        a_cfg.get(
                                            "surface_tip_hybrid_seat_contact_n", 0.05
                                        )
                                    )
                                ):
                                    tip_hybrid_soft_seat = True
                                    tip_hybrid_mode_counts["soft_seat"] = (
                                        tip_hybrid_mode_counts.get("soft_seat", 0) + 1
                                    )
                            if search_mode == "planned_spiral" and force_hole_detect:
                                hole_ax_h = feat_i.hole_axis / (
                                    np.linalg.norm(feat_i.hole_axis) + 1e-12
                                )
                                wr_h = read_wrist_wrench_world(raw)
                                wh_h = wrench_in_hole_frame(wr_h[0], hole_ax_h)
                                r_sp_now = (
                                    float(planned_rs[min(planned_i, planned_rs.shape[0] - 1)])
                                    if planned_rs is not None
                                    else float(r_cmd)
                                )
                                # Oracle: radius + tip gate must use tip→hole planar,
                                # not frozen spiral waypoint index / peg-body lat.
                                tip_lat_hole = (
                                    float(tip_ctrl_lat_m)
                                    if (tip_obs_enable or tip_est_enable)
                                    else float(feat_i.lateral_m)
                                )
                                if tip_oracle_enable:
                                    r_sp_now = max(tip_lat_hole, 1e-6)
                                if update_spiral_force_hole_detect(
                                    force_hole_state,
                                    fz_axial=float(wh_h[2]),
                                    f_lat=float(np.linalg.norm(wh_h[:2])),
                                    spiral_radius_m=r_sp_now,
                                    step_idx=int(used_sp),
                                    search_cfg=search_cfg_sp,
                                    contact_resid_n=float(contact_sp),
                                    tip_lat_m=tip_lat_hole,
                                ):
                                    print(
                                        f"pci: FORCE hole-detect unload="
                                        f"{force_hole_state.peak_abs_fz - abs(float(wh_h[2])):.2f}N "
                                        f"r={r_sp_now*1e3:.1f}mm lat="
                                        f"{tip_lat_hole*1e3:.1f}mm i={planned_i}",
                                        flush=True,
                                    )
                            if search_mode == "planned_spiral" and stick_phase == "search":
                                stuck_move_m = float(
                                    a_cfg.get(
                                        "surface_planned_stuck_move_m", 0.0004
                                    )
                                )
                                stuck_err_m = float(
                                    a_cfg.get(
                                        "surface_planned_stuck_err_m", 0.004
                                    )
                                )
                                stuck_f_n = float(
                                    a_cfg.get(
                                        "surface_planned_stuck_force_n", 0.08
                                    )
                                )
                                stuck_need = int(
                                    a_cfg.get(
                                        "surface_planned_stuck_frames", 10
                                    )
                                )
                                # Real stiction: pressed into surface AND tip not moving.
                                near_hole_lat = float(
                                    a_cfg.get(
                                        "surface_planned_near_hole_lat_m", 0.016
                                    )
                                )
                                if (
                                    micro_lift_left <= 0
                                    and float(feat_i.lateral_m) > near_hole_lat
                                    and contact_sp >= stuck_f_n
                                    and en >= stuck_err_m
                                    and tip_move < stuck_move_m
                                ):
                                    stuck_count_sp += 1
                                elif micro_lift_left <= 0:
                                    stuck_count_sp = 0
                                if (
                                    bool(a_cfg.get("surface_planned_micro_lift_enable", False))
                                    and micro_lift_left <= 0
                                    and stuck_count_sp >= stuck_need
                                ):
                                    micro_lift_left = int(
                                        a_cfg.get(
                                            "surface_planned_micro_lift_frames", 8
                                        )
                                    )
                                    stuck_count_sp = 0
                                    micro_lift_count += 1
                                    planned_stuck = True
                                    print(
                                        f"pci: stiction micro-lift "
                                        f"#{micro_lift_count} "
                                        f"|r|={contact_sp:.2f}N "
                                        f"err={en*1e3:.1f}mm "
                                        f"move={tip_move*1e3:.2f}mm "
                                        f"i={planned_i}",
                                        flush=True,
                                    )
                                # Lost contact while searching → re-establish surface press.
                                lost_n = float(
                                    a_cfg.get(
                                        "surface_planned_lost_contact_n", 0.035
                                    )
                                )
                                lost_need = int(
                                    a_cfg.get(
                                        "surface_planned_lost_contact_frames", 6
                                    )
                                )
                                slide_err_m = float(
                                    a_cfg.get(
                                        "surface_planned_slide_err_m", 0.003
                                    )
                                )
                                if (
                                    recontact_left <= 0
                                    and micro_lift_left <= 0
                                    and contact_sp < lost_n
                                    and en
                                    <= float(
                                        a_cfg.get(
                                            "surface_planned_recontact_max_planar_m",
                                            0.004,
                                        )
                                    )
                                ):
                                    lost_contact_count += 1
                                else:
                                    lost_contact_count = 0
                                if (
                                    recontact_left <= 0
                                    and micro_lift_left <= 0
                                    and lost_contact_count >= lost_need
                                ):
                                    recontact_left = int(
                                        a_cfg.get(
                                            "surface_planned_recontact_frames", 20
                                        )
                                    )
                                    lost_contact_count = 0
                                    recontact_count += 1
                                    # Cancel pending micro-lift; need press not raise.
                                    micro_lift_left = 0
                                    print(
                                        f"pci: recontact #{recontact_count} "
                                        f"|r|={contact_sp:.2f}N "
                                        f"err={en*1e3:.1f}mm i={planned_i}",
                                        flush=True,
                                    )
                            if search_mode in _priv_lat_modes and stick_phase != "search":
                                d_stick, f_des_step, next_ph, want_live = (
                                    _tip_stick_fsm_delta(
                                        phase=stick_phase,
                                        tip_pos=tip,
                                        center=center_i,
                                        hole_axis=feat_i.hole_axis,
                                        peg_axis=feat_i.peg_axis,
                                        press_ax=press_ax,
                                        contact_n=contact_sp,
                                        f_des=f_des,
                                        phase_i=stick_phase_i,
                                        lift_frames=stick_lift_frames,
                                        slide_frames=stick_slide_frames,
                                        slide_step_m=stick_slide_step,
                                        clear_lift_m=stick_clear_lift,
                                        clear_contact_n=stick_clear_n,
                                        mouth_lat_m=mouth_lat,
                                        lat_enter_m=stick_lat_enter,
                                        press_frames=stick_press_frames,
                                    )
                                )
                                target = tip + d_stick
                                # Only during press: keep from floating too high after hop.
                                if (
                                    stick_along0 is not None
                                    and stick_phase in ("press", "retouch")
                                ):
                                    along_now = float(feat_i.along_m)
                                    if along_now > stick_along0 + 0.010:
                                        target = target + press_ax * min(
                                            0.0015, along_now - stick_along0 - 0.006
                                        )
                                # Track best lat; stall → re-lift with refreshed tip-servo offset.
                                lat_now = float(feat_i.lateral_m)
                                if stick_phase == "slide":
                                    if lat_now < stick_lat_best - 0.0005:
                                        stick_lat_best = lat_now
                                        stick_stall = 0
                                        # Refresh frozen offset as tip actually moves.
                                        stick_offset_wt = site_now[0:3].copy() - tip
                                    else:
                                        stick_stall += 1
                                    if stick_stall >= 35 and lat_now > mouth_lat * 1.25:
                                        next_ph = "lift"
                                        stick_stall = 0
                                stick_phase_i += 1
                                if next_ph != stick_phase:
                                    if stick_phase == "lift" and next_ph == "slide":
                                        stick_offset_wt = (
                                            site_now[0:3].copy() - tip
                                        )
                                        stick_lat_enter = min(
                                            stick_lat_enter, lat_now
                                        )
                                        stick_lat_best = lat_now
                                        stick_stall = 0
                                    if stick_phase == "slide" and next_ph == "press":
                                        stick_hop_count += 1
                                    if next_ph == "mouth":
                                        stick_reached_mouth = True
                                        stick_phase = "mouth"
                                        stick_phase_i = 0
                                    elif next_ph == "search":
                                        stick_streak = 0
                                        tip_hist_sp.clear()
                                        stick_offset_wt = None
                                        stick_along0 = None
                                        stick_phase = next_ph
                                        stick_phase_i = 0
                                    else:
                                        stick_phase = next_ph
                                        stick_phase_i = 0
                                stick_live_off = bool(want_live)
                                stick_active = True
                                if stick_reached_mouth or (
                                    lat_now <= mouth_lat * 1.25
                                    and stick_phase
                                    in ("press", "mouth", "retouch")
                                ):
                                    stick_reached_mouth = True
                                    spiral_reason = "near_mouth"
                                    print(
                                        f"pci: stick FSM reached mouth "
                                        f"lat={lat_now*1e3:.1f}mm "
                                        f"along={feat_i.along_m*1e3:.1f}mm "
                                        f"tip_xy_peak={tip_xy_peak*1e3:.1f}mm",
                                        flush=True,
                                    )
                                    break
                            elif search_mode in _priv_lat_modes and stick_active:
                                stick_live_off = True
                                f_des_step = min(float(f_des), 0.03)
                            else:
                                stick_live_off = False
                            if search_mode == "planned_spiral" and stick_phase == "search":
                                # Constant light seeking force + probe press at every
                                # waypoint (ConnTact-style). Do NOT raise f_des from
                                # priv lat — hole enter is F/T drop only.
                                f_des_step = min(
                                    float(f_des_step),
                                    float(
                                        a_cfg.get(
                                            "surface_planned_spiral_f_des_n", 0.06
                                        )
                                    ),
                                )
                                if (
                                    not bool(
                                        a_cfg.get(
                                            "surface_planned_force_hole_only", False
                                        )
                                    )
                                    and float(feat_i.lateral_m)
                                    <= float(
                                        a_cfg.get(
                                            "surface_planned_near_hole_lat_m", 0.016
                                        )
                                    )
                                    and not bool(
                                        a_cfg.get(
                                            "surface_planned_spiral_surface_lock",
                                            True,
                                        )
                                    )
                                ):
                                    f_des_step = max(
                                        float(f_des_step),
                                        float(
                                            a_cfg.get(
                                                "surface_planned_near_hole_f_des_n",
                                                0.18,
                                            )
                                        ),
                                    )
                                    micro_lift_left = 0
                                    stuck_count_sp = 0
                            err_f = f_des_step - contact_sp
                            ax_step = float(np.clip(kp * err_f, -max_step, max_step))
                            if err_f < 0.0:
                                ax_step = float(
                                    np.clip(
                                        kp * unload_boost * err_f,
                                        -max_step * unload_boost,
                                        max_step,
                                    )
                                )
                            # Planned spiral: displacement along desired-force direction
                            # (into surface) to HOLD contact while XY tracks the path.
                            if (
                                search_mode == "planned_spiral"
                                and stick_phase == "search"
                                and micro_lift_left <= 0
                                and bool(
                                    a_cfg.get(
                                        "surface_planned_spiral_surface_lock", True
                                    )
                                )
                            ):
                                depth_err_ax = _priv_tip_surface_depth_m(
                                    tip, tip_sp0, spiral_n
                                )
                                ax_step = _priv_planned_surface_ax_step(
                                    a_cfg,
                                    contact_sp=contact_sp,
                                    f_des_step=f_des_step,
                                    depth_err=depth_err_ax,
                                    planar_err_m=en,
                                    kp=kp,
                                    max_step=max_step,
                                )
                                # Hard clamp: probe spiral never large lift-off,
                                # but allow tiny unload when residual is heavy.
                                if (
                                    float(
                                        a_cfg.get(
                                            "surface_planned_probe_press_m", 0.0
                                        )
                                    )
                                    > 1e-9
                                    and float(ax_step) < 0.0
                                ):
                                    heavy_n = float(
                                        a_cfg.get(
                                            "surface_planned_heavy_resid_n",
                                            0.45,
                                        )
                                    )
                                    if float(contact_sp) < heavy_n:
                                        ax_step = 0.0
                                    else:
                                        ax_step = float(
                                            max(
                                                ax_step,
                                                -float(
                                                    a_cfg.get(
                                                        "surface_planned_seek_unload_cap_m",
                                                        0.00012,
                                                    )
                                                ),
                                            )
                                        )
                            elif (
                                search_mode == "planned_spiral"
                                and stick_phase == "search"
                                and micro_lift_left <= 0
                                and not bool(
                                    a_cfg.get(
                                        "surface_planned_spiral_surface_lock", True
                                    )
                                )
                            ):
                                # Non-lock seek: press along hole normal + probe floor.
                                hole_u_seek = feat_i.hole_axis / (
                                    np.linalg.norm(feat_i.hole_axis) + 1e-12
                                )
                                press_ax_spiral = -hole_u_seek
                                seek_max = float(
                                    a_cfg.get(
                                        "surface_planned_seek_step_m", 0.00055
                                    )
                                )
                                probe_m = float(
                                    a_cfg.get("surface_planned_probe_press_m", 0.00006)
                                )
                                if err_f > 0.0:
                                    frac = float(
                                        np.clip(
                                            err_f / max(float(f_des_step), 1e-6),
                                            0.0,
                                            1.0,
                                        )
                                    )
                                    ax_step = max(ax_step, seek_max * frac)
                                elif err_f < 0.0:
                                    over = float(
                                        np.clip(
                                            (-err_f)
                                            / max(float(f_des_step), 1e-6),
                                            0.0,
                                            2.0,
                                        )
                                    )
                                    ax_step = min(
                                        ax_step,
                                        -seek_max * max(0.35, over),
                                    )
                                else:
                                    ax_step = max(float(ax_step), probe_m)
                                if contact_sp <= float(
                                    a_cfg.get(
                                        "surface_planned_force_overload_n", 0.055
                                    )
                                ):
                                    ax_step = max(float(ax_step), probe_m)
                            if (
                                search_mode == "planned_spiral"
                                and stick_phase == "search"
                                and micro_lift_left > 0
                            ):
                                lift_m = float(
                                    a_cfg.get(
                                        "surface_planned_micro_lift_m", 0.0008
                                    )
                                )
                                # Raise along contact normal (opposite seeking).
                                ax_step = -lift_m
                                micro_lift_left -= 1
                                # Already unloaded → stop lifting early, resume track.
                                if contact_sp < float(
                                    a_cfg.get(
                                        "surface_planned_micro_lift_end_n", 0.04
                                    )
                                ):
                                    micro_lift_left = 0
                                if used_sp % 5 == 0 or micro_lift_left == 0:
                                    print(
                                        f"pci: micro-lift step left={micro_lift_left} "
                                        f"|r|={contact_sp:.2f}N err={en*1e3:.1f}mm",
                                        flush=True,
                                    )
                            elif (
                                search_mode == "planned_spiral"
                                and stick_phase == "search"
                                and recontact_left > 0
                            ):
                                # Press back onto surface (ConnTact seeking).
                                press_m = float(
                                    a_cfg.get(
                                        "surface_planned_recontact_step_m", 0.0006
                                    )
                                )
                                # Prefer hole-axis into-surface if contact normal drifted.
                                hole_u_now = feat_i.hole_axis / (
                                    np.linalg.norm(feat_i.hole_axis) + 1e-12
                                )
                                press_re = -hole_u_now
                                ax_step = max(ax_step, press_m)
                                # Also reseat tip toward start contact depth along hole.
                                along0 = float(feat_sp0.along_m)
                                along_now = float(feat_i.along_m)
                                if along_now > along0 + 0.001:
                                    # tip rose out of hole → push back along insert.
                                    hold_boost = min(
                                        0.0020, 0.5 * (along_now - along0)
                                    )
                                    # Inject via larger ax_step (applied with press_use below).
                                    ax_step = max(ax_step, press_m + hold_boost)
                                    # Switch press axis to hole for this frame.
                                    press_ax_spiral = press_re
                                f_des_step = max(
                                    float(f_des_step),
                                    float(
                                        a_cfg.get(
                                            "surface_planned_recontact_f_des_n", 0.18
                                        )
                                    ),
                                )
                                recontact_left -= 1
                                ok_n = float(
                                    a_cfg.get(
                                        "surface_planned_recontact_ok_n", 0.08
                                    )
                                )
                                if contact_sp >= ok_n and along_now <= along0 + 0.008:
                                    recontact_left = 0
                                    print(
                                        f"pci: recontact OK |r|={contact_sp:.2f}N "
                                        f"along={along_now*1e3:.1f}mm i={planned_i}",
                                        flush=True,
                                    )
                                elif used_sp % 5 == 0 or recontact_left == 0:
                                    print(
                                        f"pci: recontact step left={recontact_left} "
                                        f"|r|={contact_sp:.2f}N "
                                        f"along={along_now*1e3:.1f}mm",
                                        flush=True,
                                    )
                            # Extra unload when stuck with pose error.
                            if stick_phase != "search" and contact_sp > f_des_step:
                                ax_step = min(
                                    ax_step,
                                    -max_step * unload_boost,
                                )
                            # Lift: unload only. Slide: keep ConnTact-style seeking force.
                            if stick_phase == "lift":
                                ax_step = min(0.0, ax_step)
                            if (
                                search_mode == "planned_spiral"
                                and stick_phase == "search"
                                and micro_lift_left <= 0
                                and float(feat_i.lateral_m)
                                <= float(
                                    a_cfg.get(
                                        "surface_planned_insert_lat_m", 0.010
                                    )
                                )
                                and (
                                    force_hole_state.entered
                                    if force_hole_detect
                                    else priv_hole_entered
                                )
                            ):
                                # Hole found — force-control into mouth.
                                ax_step = max(
                                    ax_step,
                                    float(
                                        a_cfg.get(
                                            "surface_planned_insert_step_m", 0.00045
                                        )
                                    ),
                                )
                                f_des_step = max(
                                    float(f_des_step),
                                    float(
                                        a_cfg.get(
                                            "surface_planned_insert_f_des_n", 0.22
                                        )
                                    ),
                                )
                                press_ax_spiral = -(
                                    feat_i.hole_axis
                                    / (np.linalg.norm(feat_i.hole_axis) + 1e-12)
                                )
                            # Wrist := tip_target + offset.
                            site_xyz = site_now[0:3].copy()
                            if stick_phase == "slide" and stick_offset_wt is not None:
                                offset_wt = stick_offset_wt
                            elif search_mode == "planned_spiral" and stick_phase == "search":
                                if bool(
                                    a_cfg.get(
                                        "surface_planned_spiral_freeze_offset", True
                                    )
                                ):
                                    offset_wt = offset_wt0_sp
                                else:
                                    offset_wt = site_xyz - tip
                            elif (
                                freeze_off
                                and (not stick_live_off)
                                and stick_phase == "search"
                            ):
                                offset_wt = offset_wt0_sp
                            else:
                                offset_wt = site_xyz - tip
                            press_use = (
                                press_ax_spiral
                                if (
                                    spiral_along_hold
                                    and search_mode
                                    in (
                                        "spiral",
                                        "planned_spiral",
                                        "franka_spiral",
                                        "psft_spiral",
                                        "conntact_time_spiral",
                                    )
                                    and stick_phase == "search"
                                )
                                else press_ax
                            )
                            if (
                                franka_faithful
                                and tip_baseline_mode
                                and stick_phase == "search"
                            ):
                                # Franka TCP track: wrist → spiral target (rigid-EE assumption).
                                # Jam lift: ref sets absolute z=hole_top+0.040; soft sim
                                # applies repeated unload along contact normal (disclose).
                                _jammed_now = bool(
                                    franka_meta_last.get("jammed", False)
                                )
                                if _jammed_now:
                                    ax_step = -float(
                                        a_cfg.get(
                                            "surface_franka_jam_lift_step_m",
                                            0.003,
                                        )
                                    )
                                hold_r, _bl_meta, ax_step = baseline_hold_r(
                                    tip_baseline_mode,
                                    target=target,
                                    tip=tip,
                                    site_xyz=site_xyz,
                                    offset_frozen=offset_wt0_sp,
                                    coupling=tip_coupling_est,
                                    planar_coupling=tip_planar_coupling_est,
                                    coupling_ekf=tip_coupling_ekf,
                                    qp_lambda_reg=float(
                                        a_cfg.get("surface_tip_qp_lambda_reg", 0.12)
                                    ),
                                    spiral_n=spiral_n,
                                    press_ax=press_use,
                                    ax_step=ax_step,
                                    track=float(track_use),
                                    max_step_m=float(
                                        a_cfg.get(
                                            "surface_planned_spiral_max_wrist_step_m",
                                            0.008,
                                        )
                                    ),
                                    along_now_m=float(feat_i.along_m),
                                    along0_m=along0_spiral,
                                    hole_axis=feat_i.hole_axis,
                                    along_slack_m=float(
                                        a_cfg.get(
                                            "surface_planned_along_hold_slack_m",
                                            0.0025,
                                        )
                                    ),
                                    tip_anchor=None,
                                    contact_resid_n=float(contact_sp),
                                    f_des_n=float(f_des_step),
                                    planar_gate_n=float(
                                        a_cfg.get(
                                            "surface_planned_planar_force_gate_n",
                                            0.052,
                                        )
                                    ),
                                    overload_n=float(
                                        a_cfg.get(
                                            "surface_planned_force_overload_n",
                                            0.070,
                                        )
                                    ),
                                    overload_planar_scale=float(
                                        a_cfg.get(
                                            "surface_planned_overload_planar_scale",
                                            0.12,
                                        )
                                    ),
                                    planar_ramp=1.0,
                                    heavy_unload_m=float(
                                        a_cfg.get(
                                            "surface_planned_heavy_unload_m",
                                            0.0015,
                                        )
                                    ),
                                    grasp_slip_m=float(_update_grasp_slip()),
                                    force_gate_enable=False,
                                    force_gate_axial=False,
                                )
                            elif (
                                psft_faithful
                                and tip_baseline_mode
                                and stick_phase == "search"
                            ):
                                # PSFT paper eqs (3)/(10)/(13): TCP toward p_t (rigid EE).
                                hold_r, _bl_meta, ax_step = baseline_hold_r(
                                    tip_baseline_mode,
                                    target=target,
                                    tip=tip,
                                    site_xyz=site_xyz,
                                    offset_frozen=offset_wt0_sp,
                                    coupling=tip_coupling_est,
                                    planar_coupling=tip_planar_coupling_est,
                                    coupling_ekf=tip_coupling_ekf,
                                    qp_lambda_reg=float(
                                        a_cfg.get("surface_tip_qp_lambda_reg", 0.12)
                                    ),
                                    spiral_n=spiral_n,
                                    press_ax=press_use,
                                    ax_step=ax_step,
                                    track=float(track_use),
                                    max_step_m=float(
                                        a_cfg.get(
                                            "surface_psft_max_wrist_step_m",
                                            a_cfg.get(
                                                "surface_planned_spiral_max_wrist_step_m",
                                                0.008,
                                            ),
                                        )
                                    ),
                                    along_now_m=float(feat_i.along_m),
                                    along0_m=along0_spiral,
                                    hole_axis=feat_i.hole_axis,
                                    along_slack_m=float(
                                        a_cfg.get(
                                            "surface_planned_along_hold_slack_m",
                                            0.0025,
                                        )
                                    ),
                                    tip_anchor=None,
                                    contact_resid_n=float(contact_sp),
                                    f_des_n=float(f_des_step),
                                    planar_gate_n=float(
                                        a_cfg.get(
                                            "surface_planned_planar_force_gate_n",
                                            0.052,
                                        )
                                    ),
                                    overload_n=float(
                                        a_cfg.get(
                                            "surface_planned_force_overload_n",
                                            0.070,
                                        )
                                    ),
                                    overload_planar_scale=float(
                                        a_cfg.get(
                                            "surface_planned_overload_planar_scale",
                                            0.12,
                                        )
                                    ),
                                    planar_ramp=1.0,
                                    heavy_unload_m=float(
                                        a_cfg.get(
                                            "surface_planned_heavy_unload_m",
                                            0.0015,
                                        )
                                    ),
                                    grasp_slip_m=float(_update_grasp_slip()),
                                    force_gate_enable=False,
                                    force_gate_axial=False,
                                )
                            elif search_mode == "planned_spiral" and stick_phase == "search":
                                # Privileged on-surface spiral: planar FF + light press along contact normal.
                                press_use = press_ax_spiral
                                if tip_baseline_mode:
                                    _paper_planar_c = tip_baseline_mode in (
                                        "planar_c",
                                        "planar_c_matrix",
                                        "s8",
                                        "ekf_qp",
                                        "s9",
                                        "priv_oracle_tip_servo",
                                        "oracle_tip",
                                        "oracle",
                                        "priv_oracle_tip_obs",
                                    ) or bool(tip_oracle_enable)
                                    _bl_plane = (
                                        float(
                                            a_cfg.get(
                                                "surface_baseline_plane_lift_m",
                                                0.0,
                                            )
                                        )
                                        if tip_baseline_mode
                                        in ("planar_coupling", "s7")
                                        else 0.0
                                    )
                                    _bl_press = (
                                        float(
                                            a_cfg.get(
                                                "surface_baseline_plane_press_m",
                                                0.0,
                                            )
                                        )
                                        if tip_baseline_mode
                                        in ("planar_coupling", "s7")
                                        else 0.0
                                    )
                                    _bl_hard_along = (
                                        float(
                                            a_cfg.get(
                                                "surface_baseline_hard_along_m",
                                                0.006,
                                            )
                                        )
                                        if tip_baseline_mode
                                        in ("planar_coupling", "s7")
                                        else None
                                    )
                                    _grasp_slip_now = float(_update_grasp_slip())
                                    _planar_ramp = float(
                                        np.clip(
                                            (
                                                float(used_sp)
                                                - max(
                                                    int(
                                                        a_cfg.get(
                                                            "surface_planned_force_seat_frames",
                                                            0,
                                                        )
                                                    ),
                                                    1,
                                                )
                                            )
                                            / max(
                                                float(
                                                    a_cfg.get(
                                                        "surface_planned_planar_ramp_frames",
                                                        120,
                                                    )
                                                ),
                                                1.0,
                                            ),
                                            0.0,
                                            1.0,
                                        )
                                    )
                                    _or_track = float(track_use)
                                    _or_max = float(
                                        a_cfg.get(
                                            "surface_planned_spiral_max_wrist_step_m",
                                            0.008,
                                        )
                                    )
                                    _or_gate = True
                                    if tip_oracle_enable:
                                        # Upper-bound tip PD: do not crush XY chase.
                                        _or_track = max(
                                            _or_track,
                                            float(
                                                a_cfg.get(
                                                    "surface_tip_oracle_track", 2.5
                                                )
                                            ),
                                        )
                                        _or_max = max(
                                            _or_max,
                                            float(
                                                a_cfg.get(
                                                    "surface_tip_oracle_max_step_m",
                                                    0.018,
                                                )
                                            ),
                                        )
                                        _or_gate = bool(
                                            a_cfg.get(
                                                "surface_tip_oracle_force_gate",
                                                False,
                                            )
                                        )
                                        _planar_ramp = 1.0
                                    if tip_phig_enable:
                                        # Known-path wrist track: keep planar alive.
                                        _or_gate = bool(
                                            a_cfg.get(
                                                "surface_tip_phig_force_gate", False
                                            )
                                        )
                                        _planar_ramp = 1.0
                                        _or_max = max(
                                            _or_max,
                                            float(phig_cfg.max_step_m),
                                        )
                                    _or_n = spiral_n
                                    if tip_oracle_enable:
                                        _hu = np.asarray(
                                            feat_i.hole_axis, dtype=np.float64
                                        ).reshape(3)
                                        _hn = float(np.linalg.norm(_hu))
                                        if _hn > 1e-12:
                                            _or_n = _hu / _hn
                                    hold_r, _bl_meta, ax_step = baseline_hold_r(
                                        tip_baseline_mode,
                                        target=target,
                                        tip=(
                                            tip_ctrl
                                            if (
                                                tip_oracle_enable
                                                or (
                                                    tip_gmhqp_tip_obs
                                                    and tip_theory_ekf is not None
                                                )
                                            )
                                            else tip
                                        ),
                                        site_xyz=site_xyz,
                                        offset_frozen=offset_wt0_sp,
                                        coupling=tip_coupling_est,
                                        planar_coupling=tip_planar_coupling_est,
                                        coupling_ekf=tip_coupling_ekf,
                                        qp_lambda_reg=float(
                                            a_cfg.get(
                                                "surface_tip_qp_lambda_reg", 0.12
                                            )
                                        ),
                                        spiral_n=_or_n,
                                        press_ax=press_use,
                                        ax_step=ax_step,
                                        track=_or_track,
                                        max_step_m=_or_max,
                                        along_now_m=float(feat_i.along_m),
                                        along0_m=along0_spiral,
                                        hole_axis=feat_i.hole_axis,
                                        along_slack_m=float(
                                            a_cfg.get(
                                                "surface_planned_along_hold_slack_m",
                                                0.0025,
                                            )
                                        ),
                                        tip_anchor=tip_sp0
                                        if _paper_planar_c
                                        or _bl_plane > 0
                                        or _bl_press > 0
                                        or tip_gmhqp_ftip_enable
                                        or tip_gmhqp_compc_enable
                                        or tip_gmhqp_ac_enable
                                        else None,
                                        surface_lock_lift_m=_bl_plane,
                                        surface_lock_press_m=_bl_press,
                                        hard_along_slack_m=_bl_hard_along,
                                        contact_resid_n=float(contact_sp),
                                        f_des_n=float(f_des_step),
                                        planar_gate_n=float(
                                            a_cfg.get(
                                                "surface_planned_planar_force_gate_n",
                                                0.052,
                                            )
                                        ),
                                        overload_n=float(
                                            a_cfg.get(
                                                "surface_planned_force_overload_n",
                                                0.070,
                                            )
                                        ),
                                        overload_planar_scale=float(
                                            a_cfg.get(
                                                "surface_planned_overload_planar_scale",
                                                0.12,
                                            )
                                        ),
                                        planar_ramp=_planar_ramp,
                                        heavy_unload_m=float(
                                            a_cfg.get(
                                                "surface_planned_heavy_unload_m",
                                                0.0015,
                                            )
                                        ),
                                        grasp_slip_m=_grasp_slip_now,
                                        force_gate_enable=_or_gate,
                                        # Keep axial probe during spiral; planar gate
                                        # still scales XY chase on overload/slip.
                                        force_gate_axial=False
                                        if float(
                                            a_cfg.get(
                                                "surface_planned_probe_press_m", 0.0
                                            )
                                        )
                                        > 1e-9
                                        else not (
                                            force_hole_detect
                                            and force_hole_state.entered
                                            and bool(
                                                a_cfg.get(
                                                    "surface_planned_insert_on_force_only",
                                                    True,
                                                )
                                            )
                                        ),
                                        phig_cfg=phig_cfg if tip_phig_enable else None,
                                        wrench_xyz=(
                                            np.asarray(
                                                read_wrist_wrench_world(raw)[0][:3],
                                                dtype=np.float64,
                                            ).reshape(3)
                                            if (
                                                tip_phig_enable
                                                or tip_clep_enable
                                                or tip_fasr_enable
                                                or tip_gmhqp_ftip_enable
                                                or tip_gmhqp_compc_enable
                                            )
                                            else None
                                        ),
                                        wrench_tau_xyz=(
                                            np.asarray(
                                                read_wrist_wrench_world(raw)[0][3:6],
                                                dtype=np.float64,
                                            ).reshape(3)
                                            if (
                                                tip_clep_enable
                                                or tip_gmhqp_ftip_enable
                                                or tip_gmhqp_compc_enable
                                            )
                                            else None
                                        ),
                                        clep_cfg=(
                                            clep_cfg if tip_clep_enable else None
                                        ),
                                        clep_state=(
                                            clep_state if tip_clep_enable else None
                                        ),
                                        fasr_cfg=(
                                            fasr_cfg if tip_fasr_enable else None
                                        ),
                                        fasr_state=(
                                            fasr_state if tip_fasr_enable else None
                                        ),
                                        oigs_cfg=(
                                            oigs_cfg if tip_oigs_enable else None
                                        ),
                                        oigs_state=(
                                            oigs_state if tip_oigs_enable else None
                                        ),
                                        gmhqp_cfg=(
                                            gmhqp_cfg if tip_gmhqp_enable else None
                                        ),
                                        gmhqp_state=(
                                            gmhqp_state if tip_gmhqp_enable else None
                                        ),
                                        gmhqp_ftip_cfg=(
                                            gmhqp_ftip_cfg
                                            if tip_gmhqp_ftip_enable
                                            else None
                                        ),
                                        gmhqp_ftip_state=(
                                            gmhqp_ftip_state
                                            if tip_gmhqp_ftip_enable
                                            else None
                                        ),
                                        gmhqp_compc_cfg=(
                                            gmhqp_compc_cfg
                                            if tip_gmhqp_compc_enable
                                            else None
                                        ),
                                        gmhqp_compc_state=(
                                            gmhqp_compc_state
                                            if tip_gmhqp_compc_enable
                                            else None
                                        ),
                                        gmhqp_ac_cfg=(
                                            gmhqp_ac_cfg
                                            if tip_gmhqp_ac_enable
                                            else None
                                        ),
                                        gmhqp_ac_state=(
                                            gmhqp_ac_state
                                            if tip_gmhqp_ac_enable
                                            else None
                                        ),
                                        hybrid_gmhqp_cfg=(
                                            hybrid_gmhqp_cfg
                                            if tip_hybrid_gmhqp_enable
                                            else None
                                        ),
                                        hybrid_gmhqp_state=(
                                            hybrid_gmhqp_state
                                            if tip_hybrid_gmhqp_enable
                                            else None
                                        ),
                                        hard_hqp_cfg=(
                                            hard_hqp_cfg
                                            if tip_hard_hqp_enable
                                            else None
                                        ),
                                        axis_err_deg=float(
                                            np.degrees(
                                                float(
                                                    getattr(
                                                        feat_i, "axis_error_rad", 0.0
                                                    )
                                                    or 0.0
                                                )
                                            )
                                        ),
                                        peg_axis=getattr(feat_i, "peg_axis", None),
                                        r_cmd_m=float(r_cmd),
                                        mouth_xyz=(
                                            np.asarray(
                                                spiral_center, dtype=np.float64
                                            ).reshape(3)
                                            if tip_gmhqp_compc_enable
                                            else None
                                        ),
                                    )
                                    # Enforce probe floor after baseline (never cancel).
                                    _probe_floor = float(
                                        a_cfg.get(
                                            "surface_planned_probe_press_m", 0.0
                                        )
                                    )
                                    _hard_ax = float(
                                        a_cfg.get(
                                            "surface_planned_spiral_hard_unload_n",
                                            0.35,
                                        )
                                    )
                                    if (
                                        _probe_floor > 1e-9
                                        and float(contact_sp) <= _hard_ax
                                        and stick_phase == "search"
                                        and micro_lift_left <= 0
                                        and float(ax_step) >= -1e-12
                                    ):
                                        # Never cancel an unload with probe floor.
                                        ax_step = max(float(ax_step), _probe_floor)
                                        _pu = press_use / (
                                            np.linalg.norm(press_use) + 1e-12
                                        )
                                        _ax_now = float(
                                            np.dot(hold_r - site_xyz, _pu)
                                        )
                                        if _ax_now + 1e-9 < float(ax_step):
                                            hold_r = hold_r + _pu * (
                                                float(ax_step) - _ax_now
                                            )
                                    # Agent-D oracle v2: stronger axial press near mouth
                                    # (ep03 force-enter) and reseat press on slip freeze.
                                    if tip_oracle_enable:
                                        _pu_or = press_use / (
                                            np.linalg.norm(press_use) + 1e-12
                                        )
                                        mouth_lat_or = float(
                                            a_cfg.get(
                                                "surface_tip_oracle_mouth_press_lat_m",
                                                a_cfg.get(
                                                    "surface_spiral_mouth_lat_m",
                                                    0.0045,
                                                ),
                                            )
                                        )
                                        tip_lat_or_now = (
                                            float(tip_ctrl_lat_m)
                                            if (tip_obs_enable or tip_est_enable)
                                            else float(feat_i.lateral_m)
                                        )
                                        axis_deg_or = float(
                                            np.degrees(
                                                float(
                                                    getattr(
                                                        feat_i, "axis_error_rad", 0.0
                                                    )
                                                )
                                            )
                                        )
                                        max_axis_or = float(
                                            a_cfg.get(
                                                "surface_planned_priv_enter_max_axis_err_deg",
                                                25.0,
                                            )
                                        )
                                        reseat_left_or = int(
                                            tip_hybrid_state.oracle_reseat_left
                                        )
                                        if reseat_left_or > 0 and bool(
                                            a_cfg.get(
                                                "surface_tip_oracle_slip_reseat",
                                                False,
                                            )
                                        ):
                                            # Axial reseat + grasp already tightened;
                                            # KEEP tip→mouth planar (scaled), do not
                                            # zero XY — that starved ep08 mouth reach.
                                            reseat_press = float(
                                                a_cfg.get(
                                                    "surface_tip_oracle_reseat_press_m",
                                                    0.00045,
                                                )
                                            )
                                            ax_step = max(float(ax_step), reseat_press)
                                            planar_scale_or = float(
                                                a_cfg.get(
                                                    "surface_tip_oracle_reseat_planar_scale",
                                                    0.35,
                                                )
                                            )
                                            d_h = hold_r - site_xyz
                                            ax_comp = float(np.dot(d_h, _pu_or))
                                            planar_h = d_h - _pu_or * ax_comp
                                            hold_r = (
                                                site_xyz
                                                + planar_h * planar_scale_or
                                                + _pu_or * float(ax_step)
                                            )
                                        elif (
                                            bool(
                                                a_cfg.get(
                                                    "surface_tip_oracle_mouth_press",
                                                    False,
                                                )
                                            )
                                            and tip_lat_or_now <= mouth_lat_or
                                            and stick_phase == "search"
                                            and micro_lift_left <= 0
                                            and reseat_left_or <= 0
                                        ):
                                            # Tip-Hybrid: if axis bad, UPRIGHT first —
                                            # do not jam sideways into mouth (ep03/07).
                                            if axis_deg_or > max_axis_or:
                                                tip_hybrid_state.mode = "upright"
                                                tip_hybrid_state.upright_ok_streak = 0
                                            else:
                                                press_or = float(
                                                    a_cfg.get(
                                                        "surface_tip_oracle_mouth_press_m",
                                                        0.00035,
                                                    )
                                                )
                                                hard_or = float(
                                                    a_cfg.get(
                                                        "surface_tip_oracle_mouth_hard_unload_n",
                                                        0.85,
                                                    )
                                                )
                                                if float(contact_sp) <= hard_or:
                                                    ax_step = max(
                                                        float(ax_step), press_or
                                                    )
                                                    hole_u_mp = np.asarray(
                                                        feat_i.hole_axis,
                                                        dtype=np.float64,
                                                    ).reshape(3)
                                                    hn_mp = float(
                                                        np.linalg.norm(hole_u_mp)
                                                    )
                                                    if hn_mp > 1e-12:
                                                        press_use = -(
                                                            hole_u_mp / hn_mp
                                                        )
                                                        _pu_or = press_use
                                                    _ax_now2 = float(
                                                        np.dot(
                                                            hold_r - site_xyz, _pu_or
                                                        )
                                                    )
                                                    if _ax_now2 + 1e-9 < float(
                                                        ax_step
                                                    ):
                                                        hold_r = hold_r + _pu_or * (
                                                            float(ax_step) - _ax_now2
                                                        )
                                                    tip_hybrid_state.oracle_mouth_press_frames = (
                                                        int(
                                                            tip_hybrid_state.oracle_mouth_press_frames
                                                        )
                                                        + 1
                                                    )
                                    # Track-B PHIG: mouth press / slip reseat axial
                                    # (r_cmd + F/T only — no tip_lat GT gate).
                                    if tip_phig_enable:
                                        _pu_ph = press_use / (
                                            np.linalg.norm(press_use) + 1e-12
                                        )
                                        if phig_state.mode == "slip_reseat":
                                            ax_step = phig_axial_override(
                                                mode="slip_reseat",
                                                ax_step=float(ax_step),
                                                cfg=phig_cfg,
                                            )
                                            planar_scale_ph = float(
                                                a_cfg.get(
                                                    "surface_tip_phig_reseat_planar_scale",
                                                    0.45,
                                                )
                                            )
                                            d_h = hold_r - site_xyz
                                            ax_comp = float(np.dot(d_h, _pu_ph))
                                            planar_h = d_h - _pu_ph * ax_comp
                                            hold_r = (
                                                site_xyz
                                                + planar_h * planar_scale_ph
                                                + _pu_ph * float(ax_step)
                                            )
                                        elif (
                                            phig_state.mode == "mouth_press"
                                            and stick_phase == "search"
                                            and micro_lift_left <= 0
                                        ):
                                            # Re-check deployable near-mouth (no tip GT).
                                            r_cmd_mp = (
                                                float(planned_rs[planned_i])
                                                if planned_rs is not None
                                                else 1.0
                                            )
                                            near_mp = phig_near_mouth(
                                                r_cmd_m=r_cmd_mp,
                                                contact_resid_n=float(contact_sp),
                                                force_hole_cue=bool(
                                                    force_hole_state.entered
                                                ),
                                                mouth_r_m=float(phig_cfg.mouth_r_m),
                                                f_mouth_press_n=float(
                                                    phig_cfg.f_mouth_press_n
                                                ),
                                            )
                                            if near_mp and float(contact_sp) <= float(
                                                phig_cfg.mouth_hard_unload_n
                                            ):
                                                ax_step = phig_axial_override(
                                                    mode="mouth_press",
                                                    ax_step=float(ax_step),
                                                    cfg=phig_cfg,
                                                )
                                                hole_u_mp = np.asarray(
                                                    feat_i.hole_axis,
                                                    dtype=np.float64,
                                                ).reshape(3)
                                                hn_mp = float(
                                                    np.linalg.norm(hole_u_mp)
                                                )
                                                if hn_mp > 1e-12:
                                                    press_use = -(hole_u_mp / hn_mp)
                                                    _pu_ph = press_use
                                                _ax_now_ph = float(
                                                    np.dot(hold_r - site_xyz, _pu_ph)
                                                )
                                                if _ax_now_ph + 1e-9 < float(ax_step):
                                                    hold_r = hold_r + _pu_ph * (
                                                        float(ax_step) - _ax_now_ph
                                                    )
                                                phig_state.mouth_press_frames = int(
                                                    phig_state.mouth_press_frames
                                                ) + 1
                                        elif phig_state.mode == "enter_hold":
                                            tip_hybrid_state.mode = "upright"
                                            tip_hybrid_state.upright_ok_streak = 0
                                else:
                                    surface_lock = bool(
                                        a_cfg.get(
                                            "surface_planned_spiral_surface_lock",
                                            True,
                                        )
                                    )
                                if not tip_baseline_mode and surface_lock:
                                    planar = target - tip
                                    planar = planar - spiral_n * float(
                                        np.dot(planar, spiral_n)
                                    )
                                    planar_scale = 1.0
                                    overload_n = float(
                                        a_cfg.get(
                                            "surface_planned_force_overload_n",
                                            0.070,
                                        )
                                    )
                                    planar_gate_n = float(
                                        a_cfg.get(
                                            "surface_planned_planar_force_gate_n",
                                            0.052,
                                        )
                                    )
                                    seat_frames = int(
                                        a_cfg.get(
                                            "surface_planned_force_seat_frames", 0
                                        )
                                    )
                                    force_seat_sp = (
                                        seat_frames > 0 and used_sp <= seat_frames
                                    )
                                    lift_tol = float(
                                        a_cfg.get(
                                            "surface_planned_surface_lift_tol_m",
                                            0.0006,
                                        )
                                    )
                                    heavy_unload = float(
                                        a_cfg.get(
                                            "surface_planned_heavy_unload_m",
                                            0.0015,
                                        )
                                    )
                                    overloaded = (
                                        float(contact_sp) > planar_gate_n
                                        or float(contact_sp)
                                        > float(f_des_step) * 1.15
                                    )
                                    if float(contact_sp) > overload_n:
                                        overload_cooldown_sp = max(
                                            overload_cooldown_sp,
                                            int(
                                                a_cfg.get(
                                                    "surface_planned_overload_cooldown_frames",
                                                    18,
                                                )
                                            ),
                                        )
                                    if overload_cooldown_sp > 0:
                                        overload_cooldown_sp -= 1
                                        overloaded = True
                                    if force_seat_sp or overloaded:
                                        planar_scale = 0.0
                                    if unseated_sp:
                                        # Float ban: zero planar, ConnTact press + grasp stiffen.
                                        planar_scale = 0.0
                                        ax_step = max(
                                            float(ax_step),
                                            float(_seat_reseat_ax_m(False)),
                                        )
                                    if force_seat_sp and not overloaded:
                                        f_tol = float(
                                            a_cfg.get(
                                                "surface_planned_force_ok_tol_n",
                                                0.008,
                                            )
                                        )
                                        if (
                                            float(contact_sp) >= 0.006
                                            and float(contact_sp)
                                            <= float(f_des_step) + f_tol
                                        ):
                                            ax_step = 0.0
                                    elif overloaded:
                                        over = max(
                                            float(contact_sp) - float(f_des_step),
                                            0.0,
                                        )
                                        ax_step = -max(
                                            heavy_unload,
                                            min(
                                                0.004,
                                                heavy_unload
                                                * max(
                                                    1.0,
                                                    over
                                                    / max(
                                                        float(f_des_step), 1e-6
                                                    ),
                                                ),
                                            ),
                                        )
                                        lift_tol = float(
                                            a_cfg.get(
                                                "surface_planned_recovery_lift_m",
                                                0.0035,
                                            )
                                        )
                                    elif not force_seat_sp:
                                        ramp = float(
                                            np.clip(
                                                (
                                                    float(used_sp)
                                                    - max(seat_frames, 1)
                                                )
                                                / max(
                                                    float(
                                                        a_cfg.get(
                                                            "surface_planned_planar_ramp_frames",
                                                            120,
                                                        )
                                                    ),
                                                    1.0,
                                                ),
                                                0.0,
                                                1.0,
                                            )
                                        )
                                        if float(contact_sp) > float(f_des_step):
                                            planar_scale = float(
                                                a_cfg.get(
                                                    "surface_planned_overload_planar_scale",
                                                    0.12,
                                                )
                                            )
                                        planar_scale *= max(ramp, 0.08)
                                    hold_r = (
                                        tip
                                        + offset_wt
                                        + planar * track_use * planar_scale
                                        + press_use * ax_step
                                    )
                                    depth_tip = _priv_tip_surface_depth_m(
                                        tip, tip_sp0, spiral_n
                                    )
                                    press_deep = float(
                                        a_cfg.get(
                                            "surface_planned_surface_press_deep_m",
                                            0.0008,
                                        )
                                    )
                                    if depth_tip < -0.5 * press_deep:
                                        hold_r = hold_r - spiral_n * min(
                                            0.0025,
                                            abs(depth_tip) * 0.55,
                                        )
                                    hold_r = _priv_clamp_hold_on_surface(
                                        hold_r,
                                        anchor=tip_sp0,
                                        normal=spiral_n,
                                        max_lift_m=lift_tol,
                                    )
                                    along_slack = float(
                                        a_cfg.get(
                                            "surface_planned_along_hold_slack_m",
                                            0.0025,
                                        )
                                    )
                                    along_now_h = float(feat_i.along_m)
                                    if along_now_h > along0_spiral + along_slack:
                                        hole_u_hold = feat_i.hole_axis / (
                                            np.linalg.norm(feat_i.hole_axis)
                                            + 1e-12
                                        )
                                        excess_a = along_now_h - (
                                            along0_spiral + along_slack
                                        )
                                        hold_r = hold_r - hole_u_hold * min(
                                            0.0015, 0.35 * excess_a
                                        )
                                elif not tip_baseline_mode and bool(
                                    a_cfg.get(
                                        "surface_planned_spiral_wrist_ff", False
                                    )
                                ):
                                    hold_r = target + offset_wt + press_use * ax_step
                                elif not tip_baseline_mode:
                                    hole_u_cmd = feat_i.hole_axis / (
                                        np.linalg.norm(feat_i.hole_axis) + 1e-12
                                    )
                                    d_xy_only = target - tip
                                    d_xy_only = d_xy_only - hole_u_cmd * float(
                                        np.dot(d_xy_only, hole_u_cmd)
                                    )
                                    hold_r = (
                                        tip
                                        + offset_wt
                                        + d_xy_only * track_use
                                        + press_use * ax_step
                                    )
                            else:
                                hold_r = target + offset_wt + press_use * ax_step
                            d_cmd = hold_r - site_xyz
                            dn = float(np.linalg.norm(d_cmd))
                            max_cmd = float(
                                a_cfg.get("surface_tip_spiral_max_wrist_step_m", 0.008)
                            )
                            if search_mode == "planned_spiral" and stick_phase == "search":
                                if micro_lift_left > 0 or planned_stuck or recontact_left > 0:
                                    max_cmd = float(
                                        a_cfg.get(
                                            "surface_planned_stuck_max_wrist_step_m",
                                            0.003,
                                        )
                                    )
                                else:
                                    max_cmd = max(
                                        max_cmd,
                                        float(
                                            a_cfg.get(
                                                "surface_planned_spiral_max_wrist_step_m",
                                                0.008,
                                            )
                                        ),
                                    )
                                    slide_err_m = float(
                                        a_cfg.get(
                                            "surface_planned_slide_err_m", 0.003
                                        )
                                    )
                                    if (
                                        en
                                        > float(
                                            a_cfg.get(
                                                "surface_planned_slide_err_m", 0.003
                                            )
                                        )
                                        and not bool(
                                            a_cfg.get(
                                                "surface_planned_spiral_surface_lock",
                                                True,
                                            )
                                        )
                                    ):
                                        max_cmd = max(
                                            max_cmd,
                                            float(
                                                a_cfg.get(
                                                    "surface_planned_slide_max_wrist_step_m",
                                                    0.010,
                                                )
                                            ),
                                        )
                            if stick_phase == "slide":
                                max_cmd = max(
                                    max_cmd,
                                    float(
                                        a_cfg.get(
                                            "surface_priv_lat_stick_wrist_step_m",
                                            0.014,
                                        )
                                    ),
                                )
                            elif stick_phase != "search":
                                max_cmd = max(
                                    max_cmd,
                                    float(
                                        a_cfg.get(
                                            "surface_priv_lat_stick_wrist_step_m",
                                            0.012,
                                        )
                                    ),
                                )
                            # If tip lags commanded spiral radius, push harder.
                            # Measure in the spiral contact plane (not peg/hole shaft).
                            plane_ax = (
                                spiral_n
                                if (
                                    spiral_along_hold
                                    and search_mode in ("spiral", "planned_spiral")
                                    and stick_phase == "search"
                                )
                                else feat_i.hole_axis
                            )
                            tip_off_xy = _tip_lat_offset_xy(
                                tip, spiral_center, plane_ax
                            )
                            tip_r_now = float(np.linalg.norm(tip_off_xy))
                            cmd_off = _tip_lat_offset_xy(
                                target, spiral_center, plane_ax
                            )
                            cmd_r = float(np.linalg.norm(cmd_off))
                            # Pull back if tip fled outside spiral; push out if lagging.
                            # Never boost-yank during stiction micro-lift.
                            skip_spiral_boost = False
                            if search_mode == "planned_spiral" and stick_phase == "search":
                                seat_b = int(
                                    a_cfg.get(
                                        "surface_planned_force_seat_frames", 0
                                    )
                                )
                                gate_b = float(
                                    a_cfg.get(
                                        "surface_planned_planar_force_gate_n",
                                        0.052,
                                    )
                                )
                                skip_spiral_boost = bool(
                                    a_cfg.get(
                                        "surface_planned_spiral_surface_lock",
                                        True,
                                    )
                                ) and (
                                    (seat_b > 0 and used_sp <= seat_b)
                                    or float(contact_sp) > gate_b
                                )
                            if micro_lift_left > 0 and search_mode == "planned_spiral":
                                tip_r_now = cmd_r  # skip boost branches
                            elif skip_spiral_boost:
                                pass
                            elif tip_r_now - cmd_r > 0.002:
                                max_cmd = max(
                                    max_cmd,
                                    float(
                                        a_cfg.get(
                                            "surface_tip_spiral_boost_wrist_step_m",
                                            0.012,
                                        )
                                    ),
                                )
                                # Extra planar pull toward commanded spiral point.
                                if spiral_track_gate:
                                    pull = target - tip
                                    pull = pull - ax_i * float(np.dot(pull, ax_i))
                                    pn = float(np.linalg.norm(pull))
                                    if pn > 1e-9:
                                        hold_r = hold_r + pull * min(1.0, 0.006 / pn)
                                        d_cmd = hold_r - site_xyz
                                        dn = float(np.linalg.norm(d_cmd))
                            elif skip_spiral_boost:
                                pass
                            elif r_cmd - tip_r_now > 0.003:
                                max_cmd = max(
                                    max_cmd,
                                    float(
                                        a_cfg.get(
                                            "surface_tip_spiral_boost_wrist_step_m",
                                            0.012,
                                        )
                                    ),
                                )
                            # Type-A LOCAL_RECOVER: stronger wrist step so radial
                            # ratchet is not stuck behind soft max_cmd (ep08).
                            if (
                                tip_star_enable
                                and tip_hybrid_state.star_mode == "local_recover"
                                and stick_phase == "search"
                            ):
                                max_cmd = max(
                                    max_cmd,
                                    float(
                                        a_cfg.get(
                                            "surface_tip_star_recover_wrist_step_m",
                                            a_cfg.get(
                                                "surface_tip_spiral_boost_wrist_step_m",
                                                0.012,
                                            ),
                                        )
                                    ),
                                )
                            if dn > max_cmd > 0.0:
                                # Separate planar vs axial limits so spiral XY chase
                                # does not dilute the light probe press (ep05 bug).
                                probe_on = (
                                    search_mode == "planned_spiral"
                                    and stick_phase == "search"
                                    and float(
                                        a_cfg.get(
                                            "surface_planned_probe_press_m", 0.0
                                        )
                                    )
                                    > 1e-9
                                )
                                if probe_on:
                                    pu = press_use / (
                                        np.linalg.norm(press_use) + 1e-12
                                    )
                                    ax_c = float(np.dot(d_cmd, pu))
                                    planar_c = d_cmd - pu * ax_c
                                    pn = float(np.linalg.norm(planar_c))
                                    if pn > max_cmd > 0.0:
                                        planar_c = planar_c * (max_cmd / pn)
                                    ax_cap = float(
                                        a_cfg.get(
                                            "surface_planned_probe_press_m", 0.00015
                                        )
                                    )
                                    hard_n = float(
                                        a_cfg.get(
                                            "surface_planned_spiral_hard_unload_n",
                                            0.35,
                                        )
                                    )
                                    if float(ax_step) < -1e-12:
                                        # Probe spiral: never command lift-off.
                                        ax_c = 0.0
                                    elif float(contact_sp) > hard_n:
                                        ax_c = min(ax_c, 0.0)
                                    else:
                                        # Reinject probe floor only while seeking.
                                        ax_want = max(float(ax_step), ax_cap)
                                        ax_c = float(max(ax_c, ax_want))
                                        ax_c = float(
                                            np.clip(ax_c, 0.0, max(ax_want, ax_cap))
                                        )
                                    hold_r = site_xyz + planar_c + pu * ax_c
                                else:
                                    hold_r = site_xyz + d_cmd * (max_cmd / dn)
                            elif (
                                search_mode == "planned_spiral"
                                and stick_phase == "search"
                                and float(
                                    a_cfg.get("surface_planned_probe_press_m", 0.0)
                                )
                                > 1e-9
                                and float(ax_step) >= -1e-12
                                and float(contact_sp)
                                <= float(
                                    a_cfg.get(
                                        "surface_planned_spiral_hard_unload_n",
                                        0.35,
                                    )
                                )
                            ):
                                # dn ≤ max_cmd path: still reinject axial probe.
                                pu = press_use / (np.linalg.norm(press_use) + 1e-12)
                                ax_c = float(np.dot(hold_r - site_xyz, pu))
                                ax_want = max(
                                    float(ax_step),
                                    float(
                                        a_cfg.get(
                                            "surface_planned_probe_press_m", 0.00015
                                        )
                                    ),
                                )
                                if ax_c + 1e-9 < ax_want:
                                    hold_r = hold_r + pu * (ax_want - ax_c)
                            # Tip-hybrid: reseat press + tip-pivot upright about tip.
                            if (
                                tip_hybrid_enable
                                and search_mode == "planned_spiral"
                                and stick_phase == "search"
                            ):
                                hyb_mode = str(tip_hybrid_state.mode)
                                pu_h = press_use / (
                                    np.linalg.norm(press_use) + 1e-12
                                )
                                if hyb_mode == "seat":
                                    seat_press = float(
                                        a_cfg.get(
                                            "surface_tip_hybrid_seat_press_m",
                                            0.00045,
                                        )
                                    )
                                    planar_h = hold_r - site_xyz
                                    ax_h = float(np.dot(planar_h, pu_h))
                                    planar_h = planar_h - pu_h * ax_h
                                    planar_h *= float(
                                        a_cfg.get(
                                            "surface_tip_hybrid_seat_track", 0.15
                                        )
                                    )
                                    hold_r = (
                                        site_xyz
                                        + planar_h
                                        + pu_h * max(ax_h, seat_press)
                                    )
                                elif tip_hybrid_soft_seat:
                                    soft_press = float(
                                        a_cfg.get(
                                            "surface_tip_hybrid_soft_press_m",
                                            0.00028,
                                        )
                                    )
                                    ax_h = float(np.dot(hold_r - site_xyz, pu_h))
                                    if ax_h < soft_press:
                                        hold_r = hold_r + pu_h * (soft_press - ax_h)
                                if hyb_mode in ("upright", "spiral"):
                                    max_ang = float(
                                        np.radians(
                                            a_cfg.get(
                                                "surface_tip_hybrid_upright_step_deg",
                                                1.2,
                                            )
                                        )
                                    )
                                    # Full upright while STAR planar-recover / hold-enter
                                    # (ep02 axis-50° lesson): do not use spiral scale-down.
                                    star_full_up = bool(
                                        (
                                            tip_star_enable
                                            or tip_slip_enable
                                        )
                                        and tip_hybrid_state.star_mode
                                        in (
                                            "local_recover",
                                            "yield_upright",
                                            "grasp_restore",
                                        )
                                    )
                                    if hyb_mode == "spiral" and not star_full_up:
                                        max_ang *= float(
                                            a_cfg.get(
                                                "surface_tip_hybrid_spiral_orient_scale",
                                                0.35,
                                            )
                                        )
                                    soft_deg = float(
                                        a_cfg.get(
                                            "surface_tip_hybrid_upright_soft_deg",
                                            18.0,
                                        )
                                    )
                                    _, axis_err_now, _ = _tip_hole_pose_err(
                                        tip_ctrl
                                        if (
                                            tip_axis_enable
                                            and tip_theory_ekf is not None
                                        )
                                        else tip,
                                        spiral_center,
                                        feat_i.hole_axis,
                                        tip_axis_hat
                                        if (
                                            tip_axis_enable
                                            and tip_axis_hat is not None
                                        )
                                        else feat_i.peg_axis,
                                    )
                                    if (
                                        hyb_mode == "upright"
                                        or star_full_up
                                        or float(axis_err_now)
                                        > np.radians(soft_deg) * 0.5
                                    ):
                                        d_pos_h, omega_h, _ = tip_pivot_upright(
                                            site_xyz=site_xyz,
                                            tip=(
                                                tip_ctrl
                                                if (
                                                    tip_axis_enable
                                                    and tip_theory_ekf is not None
                                                )
                                                else tip
                                            ),
                                            peg_axis=(
                                                tip_axis_hat
                                                if (
                                                    tip_axis_enable
                                                    and tip_axis_hat is not None
                                                )
                                                else feat_i.peg_axis
                                            ),
                                            hole_axis=feat_i.hole_axis,
                                            max_ang_rad=max_ang,
                                        )
                                        hold_r = hold_r + d_pos_h
                                        tip_hybrid_omega = omega_h
                                        if hyb_mode == "upright" or star_full_up:
                                            # Keep light press while uprighting.
                                            ax_h = float(
                                                np.dot(hold_r - site_xyz, pu_h)
                                            )
                                            hold_r = hold_r + pu_h * max(
                                                0.0,
                                                float(
                                                    a_cfg.get(
                                                        "surface_planned_probe_press_m",
                                                        0.0002,
                                                    )
                                                )
                                                - ax_h,
                                            )
                            d_left = d_xy * left_share_sp
                            if spiral_admit is not None and spiral_left_admit_scale > 0.0:
                                wr_l_now = read_wrist_wrench_world(raw)
                                d_adm = spiral_admit.step(
                                    wr_l_now[1], approach_axis=press_use
                                ) * spiral_left_admit_scale
                                # Force-feedback yield only in contact plane (not into/out).
                                d_adm = d_adm - press_use * float(np.dot(d_adm, press_use))
                                d_left = d_left + d_adm
                            # Cap left yield so tray does not run away in one step.
                            left_cap = float(
                                a_cfg.get("surface_spiral_left_max_step_m", 0.004)
                            )
                            ln = float(np.linalg.norm(d_left))
                            if ln > left_cap > 0.0:
                                d_left = d_left * (left_cap / ln)
                            hold_l[0:3] = site_now[22:25].copy() + d_left
                            cmd = np.zeros(44, dtype=np.float64)
                            cmd[0:6] = site_now[0:6].copy()
                            cmd[0:3] = hold_r
                            if float(np.linalg.norm(tip_hybrid_omega)) > 1e-9:
                                from scipy.spatial.transform import Rotation as R

                                R_cur = R.from_rotvec(cmd[3:6])
                                cmd[3:6] = (
                                    R.from_rotvec(tip_hybrid_omega) * R_cur
                                ).as_rotvec()
                            _fill_tip_search_hands(cmd)
                            if unseated_sp:
                                _seat_stiffen_right_hand(cmd)
                            cmd[22:28] = hold_l
                            step_action44(gym_env, cmd, ego_recorder=ego_recorder)
                            feat_after_sp = features_from_raw(raw)
                            tip_a = np.asarray(
                                feat_after_sp.tip_pos, dtype=np.float64
                            ).reshape(3)
                            d_tip = tip_a - tip_sp0
                            tip_xy = float(
                                np.linalg.norm(d_tip - ax_i * float(np.dot(d_tip, ax_i)))
                            )
                            tip_xy_peak = max(tip_xy_peak, tip_xy)
                            if search_mode in _priv_lat_modes:
                                phase_name = str(search_mode)
                            elif search_mode == "lissajous":
                                phase_name = "lissajous"
                            elif search_mode == "planned_spiral":
                                phase_name = "planned_spiral"
                            else:
                                phase_name = "spiral"
                            priv_fn, priv_ncon = _priv_peg_tray_contact_force_n(raw)
                            priv_mouth_now = False
                            priv_enter_now = False
                            priv_along_enter_mm = 0.0
                            if search_mode == "planned_spiral":
                                priv_mouth_lat = float(
                                    a_cfg.get(
                                        "surface_planned_priv_mouth_lat_m",
                                        mouth_lat * 2.0,
                                    )
                                )
                                priv_enter_lat = float(
                                    a_cfg.get(
                                        "surface_planned_priv_enter_lat_m",
                                        mouth_lat,
                                    )
                                )
                                priv_enter_along = float(
                                    a_cfg.get(
                                        "surface_planned_priv_enter_along_m",
                                        0.003,
                                    )
                                )
                                priv_mouth_now, priv_enter_now, priv_along_enter_mm = (
                                    _priv_planned_hole_diag(
                                        feat_after_sp,
                                        along0_m=along0_spiral,
                                        mouth_lat_m=priv_mouth_lat,
                                        enter_lat_m=priv_enter_lat,
                                        enter_along_min_m=priv_enter_along,
                                    )
                                )
                                if priv_mouth_now:
                                    priv_mouth_streak += 1
                                    priv_at_mouth_peak = True
                                else:
                                    priv_mouth_streak = 0
                                if priv_enter_now:
                                    priv_enter_streak += 1
                                else:
                                    priv_enter_streak = 0
                                priv_confirm = int(
                                    a_cfg.get(
                                        "surface_planned_priv_enter_confirm", 4
                                    )
                                )
                                if (
                                    bool(
                                        a_cfg.get(
                                            "surface_planned_priv_detect_enable",
                                            True,
                                        )
                                    )
                                    and priv_enter_streak >= max(1, priv_confirm)
                                ):
                                    priv_hole_entered = True
                                overload_n = float(
                                    a_cfg.get(
                                        "surface_planned_force_overload_n", 0.055
                                    )
                                )
                                if float(feat_after_sp.lateral_m) < spiral_lat_min - 0.00025:
                                    last_lat_improve_sp = used_sp
                                if float(contact_sp) > overload_n:
                                    priv_overforce_streak += 1
                                else:
                                    priv_overforce_streak = 0
                                priv_overforce_streak_peak = max(
                                    priv_overforce_streak_peak, priv_overforce_streak
                                )
                                priv_along_drift_peak_m = max(
                                    priv_along_drift_peak_m,
                                    max(
                                        0.0,
                                        float(feat_after_sp.along_m) - along0_spiral,
                                    ),
                                )
                                over_need = int(
                                    a_cfg.get(
                                        "surface_planned_priv_overforce_frames", 12
                                    )
                                )
                                lat_stall_need = int(
                                    a_cfg.get(
                                        "surface_planned_priv_lat_stall_frames",
                                        180,
                                    )
                                )
                                drift_lim = float(
                                    a_cfg.get(
                                        "surface_planned_priv_along_drift_abort_m",
                                        0.006,
                                    )
                                )
                                lat_stall = used_sp - last_lat_improve_sp
                                if (
                                    bool(
                                        a_cfg.get(
                                            "surface_planned_priv_guard_abort", True
                                        )
                                    )
                                    and used_sp
                                    > int(
                                        a_cfg.get(
                                            "surface_planned_spiral_grace_frames", 45
                                        )
                                    )
                                    and priv_overforce_streak >= over_need
                                    and lat_stall >= lat_stall_need
                                    and spiral_lat_min
                                    > float(
                                        a_cfg.get(
                                            "surface_planned_priv_mouth_lat_m",
                                            0.008,
                                        )
                                    )
                                    # Probe spiral: ConnTact stays in contact; do not
                                    # abort on mild residual (ep01 stopped at 0.16N).
                                    and not (
                                        float(
                                            a_cfg.get(
                                                "surface_planned_probe_press_m", 0.0
                                            )
                                        )
                                        > 1e-9
                                        and bool(
                                            a_cfg.get(
                                                "surface_planned_probe_ignore_overforce_abort",
                                                True,
                                            )
                                        )
                                    )
                                ):
                                    spiral_reason = "priv_overforce"
                                    print(
                                        f"pci: PRIV abort overforce |r|="
                                        f"{contact_sp:.2f}N>{overload_n:.2f}N "
                                        f"streak={priv_overforce_streak} "
                                        f"lat_stall={lat_stall} "
                                        f"lat={feat_after_sp.lateral_m*1e3:.1f}mm",
                                        flush=True,
                                    )
                                    break
                                if (
                                    bool(
                                        a_cfg.get(
                                            "surface_planned_priv_guard_abort", True
                                        )
                                    )
                                    and priv_along_drift_peak_m > drift_lim
                                    # Probe spiral: along wobble under stiction is not
                                    # tip flyaway — ep01 aborted at +8mm while |r|~2N.
                                    and not (
                                        float(
                                            a_cfg.get(
                                                "surface_planned_probe_press_m", 0.0
                                            )
                                        )
                                        > 1e-9
                                        and bool(
                                            a_cfg.get(
                                                "surface_planned_probe_ignore_along_drift_abort",
                                                True,
                                            )
                                        )
                                    )
                                ):
                                    spiral_reason = "priv_along_drift"
                                    print(
                                        f"pci: PRIV abort along drift "
                                        f"{priv_along_drift_peak_m*1e3:.1f}mm "
                                        f"lat={feat_after_sp.lateral_m*1e3:.1f}mm",
                                        flush=True,
                                    )
                                    break
                            wr_abs = read_wrist_wrench_world(raw)
                            f_abs = float(
                                __import__("numpy").linalg.norm(
                                    __import__("numpy").asarray(
                                        wr_abs[0][:3], dtype=__import__("numpy").float64
                                    )
                                )
                            )
                            _slip_mm_log = float(_update_grasp_slip()) * 1000.0
                            force_trace.append(
                                {
                                    "t": float(t_force) * dt_f,
                                    "step": float(t_force),
                                    "phase": phase_name,
                                    "resid_r": float(contact_sp),
                                    "resid_l": float(left_c),
                                    "fz_r": float(fz_r),
                                    "fz_l": float(fz_l),
                                    "f_abs_r": float(f_abs),
                                    "priv_peg_tray_fn": float(priv_fn),
                                    "priv_peg_tray_ncon": float(priv_ncon),
                                    "priv_at_mouth": int(priv_mouth_now),
                                    "priv_hole_entered": int(priv_enter_now),
                                    "priv_along_enter_mm": float(
                                        priv_along_enter_mm
                                    )
                                    * 1000.0,
                                    "priv_depth_mm": float(
                                        _priv_tip_surface_depth_m(
                                            tip_a, tip_sp0, spiral_n
                                        )
                                    )
                                    * 1000.0
                                    if search_mode == "planned_spiral"
                                    else 0.0,
                                    "priv_planar_mm": float(
                                        feat_after_sp.lateral_m
                                    )
                                    * 1000.0
                                    if search_mode == "planned_spiral"
                                    else 0.0,
                                    "tip_track_err_mm": float(en) * 1000.0
                                    if search_mode == "planned_spiral"
                                    else 0.0,
                                    "lat_mm": float(feat_after_sp.lateral_m) * 1000.0,
                                    "along_mm": float(feat_after_sp.along_m) * 1000.0,
                                    "f_des": float(f_des_step),
                                    "spiral_resid": float(contact_sp),
                                    "tip_xy_mm": float(tip_xy) * 1000.0,
                                    "spiral_r_mm": float(r_cmd) * 1000.0,
                                    "spiral_theta": float(theta_sp),
                                    "stick": int(stick_active),
                                    "stick_phase": str(stick_phase),
                                    "stick_hop": int(stick_hop_count),
                                    "stick_lift_mm": float(
                                        abs(float(np.dot(d_stick, press_ax)))
                                    )
                                    * 1000.0
                                    if stick_active
                                    else 0.0,
                                    "liss_ax_mm": float(ax_j) * 1000.0
                                    if search_mode == "lissajous"
                                    else 0.0,
                                    "liss_ay_mm": float(ay_j) * 1000.0
                                    if search_mode == "lissajous"
                                    else 0.0,
                                    "bl_path": str(_bl_meta.get("path", "")),
                                    "bl_tip_err_mm": float(
                                        _bl_meta.get("tip_err_mm", 0.0)
                                    ),
                                    "bl_alpha": float(_bl_meta.get("alpha", 0.0)),
                                    "bl_C_trace": float(
                                        _bl_meta.get("C_trace", 0.0)
                                    ),
                                    "bl_ax_step_mm": float(ax_step) * 1000.0,
                                    "bl_force_scale": float(
                                        _bl_meta.get("force_scale", 1.0)
                                    ),
                                    "grasp_slip_mm": _slip_mm_log,
                                    "tip_spiral_err_mm": float(
                                        abs(float(r_cmd) - float(tip_xy))
                                    )
                                    * 1000.0,
                                    "force_hole_entered": int(
                                        force_hole_state.entered
                                        if force_hole_detect
                                        else 0
                                    ),
                                }
                            )
                            t_force += 1
                            tilt_i = _tray_tilt_entry_deg()
                            phase_a_tilt_peak = max(phase_a_tilt_peak, tilt_i)
                            phase_a_peg_tilt_peak = max(
                                phase_a_peg_tilt_peak, _peg_tilt_entry_deg()
                            )
                            spiral_lat_min = min(
                                spiral_lat_min, float(feat_after_sp.lateral_m)
                            )
                            # Tip seat / axis telemetry (privileged audit only).
                            tip_now_sp = np.asarray(
                                feat_after_sp.tip_pos, dtype=np.float64
                            ).reshape(3)
                            float_m = max(
                                0.0,
                                _priv_tip_surface_depth_m(
                                    tip_now_sp, tip_sp0, spiral_n
                                ),
                            )
                            spiral_tip_float_peak_m = max(
                                spiral_tip_float_peak_m, float(float_m)
                            )
                            spiral_axis_err_peak_deg = max(
                                spiral_axis_err_peak_deg,
                                float(np.degrees(feat_after_sp.axis_error_rad)),
                            )
                            spiral_contact_frames += 1
                            if float(contact_sp) >= seat_contact_n:
                                spiral_contact_ok_n += 1
                                if spiral_contact_frames <= spiral_early_frames_n:
                                    spiral_early_contact_ok_n += 1
                            out_i = env._labeler.compute(raw)
                            if _grasp_lost():
                                # Planned spiral: tip-tracking mode expects peel;
                                # never abort mid-search on slip meter (ep09).
                                hard_slip = float(
                                    a_cfg.get(
                                        "surface_planned_grasp_slip_hard_m",
                                        max(grasp_slip_lim * 1.5, 0.12),
                                    )
                                )
                                if search_mode == "planned_spiral":
                                    if used_sp % 30 == 0:
                                        print(
                                            f"pci: planned continue despite slip="
                                            f"{grasp_slip_peak*1e3:.1f}mm "
                                            f"j={used_sp}",
                                            flush=True,
                                        )
                                elif grasp_slip_peak <= hard_slip:
                                    if used_sp % 30 == 0:
                                        print(
                                            f"pci: continue despite slip="
                                            f"{grasp_slip_peak*1e3:.1f}mm "
                                            f"j={used_sp}",
                                            flush=True,
                                        )
                                else:
                                    spiral_reason = "grasp_slip"
                                    print(
                                        f"pci: tip-spiral ABORT grasp_slip="
                                        f"{grasp_slip_peak*1e3:.1f}mm "
                                        f"> {grasp_slip_lim*1e3:.1f}mm j={used_sp}",
                                        flush=True,
                                    )
                                    meta["grasp_abort_phase"] = "spiral"
                                    break
                            tilt_hard = float(
                                a_cfg.get(
                                    "surface_planned_spiral_max_tilt_hard_deg",
                                    max(max_tilt_sp * 2.0, 18.0),
                                )
                            )
                            if not out_i.tray_ok:
                                # Tray contact label flickers under slip/tilt; do not
                                # kill planned tip spiral (ep02/08/10 tray_ok=0 abort).
                                if search_mode in ("planned_spiral", "franka_spiral"):
                                    if used_sp % 30 == 0:
                                        print(
                                            f"pci: planned continue despite "
                                            f"tray_ok=0 j={used_sp}",
                                            flush=True,
                                        )
                                else:
                                    spiral_reason = "spiral_tilt_or_tray"
                                    print(
                                        f"pci: tip-spiral abort tray_ok=0 j={used_sp}",
                                        flush=True,
                                    )
                                    break
                            if tilt_i > max_tilt_sp:
                                if (
                                    search_mode
                                    in ("planned_spiral", "franka_spiral")
                                    and tilt_i <= tilt_hard
                                ):
                                    if used_sp % 30 == 0:
                                        print(
                                            f"pci: planned continue despite tilt="
                                            f"{tilt_i:.1f}deg "
                                            f"(soft={max_tilt_sp:.1f} "
                                            f"hard={tilt_hard:.1f}) j={used_sp}",
                                            flush=True,
                                        )
                                else:
                                    spiral_reason = "spiral_tilt_or_tray"
                                    print(
                                        f"pci: tip-spiral abort tilt={tilt_i:.1f}deg "
                                        f"tray_ok={int(bool(out_i.tray_ok))} j={used_sp}",
                                        flush=True,
                                    )
                                    break
                            if (
                                search_mode == "planned_spiral"
                                and force_hole_detect
                                and force_hole_state.entered
                                and bool(
                                    a_cfg.get(
                                        "surface_planned_force_stop_on_hole", True
                                    )
                                )
                            ):
                                # Tip-STAR / ours v11: hold stop until axis upright
                                # (STAR smoke ep02: mouth reach + axis~50° reject).
                                hold_until_up = bool(
                                    (
                                        (
                                            tip_star_enable
                                            or deploy_mouth_enter
                                        )
                                        and a_cfg.get(
                                            "surface_tip_star_hold_enter_until_upright",
                                            True,
                                        )
                                    )
                                    or (
                                        tip_phig_enable
                                        and bool(
                                            a_cfg.get(
                                                "surface_tip_phig_hold_enter", True
                                            )
                                        )
                                    )
                                )
                                axis_now_deg = float(
                                    np.degrees(
                                        float(
                                            getattr(
                                                feat_after_sp,
                                                "axis_error_rad",
                                                0.0,
                                            )
                                        )
                                    )
                                )
                                max_axis_hold = float(
                                    a_cfg.get(
                                        "surface_planned_priv_enter_max_axis_err_deg",
                                        25.0,
                                    )
                                )
                                # Type-A / STAR: hold to soft upright (stricter than
                                # audit 25°) so ep02 cannot stop near mouth then tip
                                # to ~40–50° before GATE.
                                if tip_star_enable and bool(
                                    a_cfg.get(
                                        "surface_tip_star_hold_enter_until_upright",
                                        True,
                                    )
                                ):
                                    soft_hold = float(
                                        a_cfg.get(
                                            "surface_tip_star_hold_enter_max_axis_deg",
                                            a_cfg.get(
                                                "surface_tip_hybrid_upright_soft_deg",
                                                18.0,
                                            ),
                                        )
                                    )
                                    max_axis_hold = min(max_axis_hold, soft_hold)
                                # Agent-D oracle v2: also hold until along seat
                                # (ep07 force_hole_along_reject ~234mm float).
                                hold_oracle_seat = bool(
                                    tip_oracle_enable
                                    and a_cfg.get(
                                        "surface_tip_oracle_hold_enter_seat", True
                                    )
                                )
                                # Track-B PHIG: same seat/axis hold, deployable cues only.
                                hold_phig_seat = bool(
                                    tip_phig_enable
                                    and a_cfg.get("surface_tip_phig_hold_enter", True)
                                )
                                along_now_h = float(feat_after_sp.along_m)
                                tip_now_h = float(feat_after_sp.tip_socket_dist_m)
                                max_along_h = float(
                                    a_cfg.get(
                                        "surface_planned_priv_enter_max_along_m",
                                        0.100,
                                    )
                                )
                                tip_in_bore_h = tip_now_h <= float(
                                    a_cfg.get(
                                        "surface_planned_priv_enter_tip_m", 0.095
                                    )
                                ) or along_now_h <= float(
                                    a_cfg.get(
                                        "surface_planned_priv_enter_along_m", 0.095
                                    )
                                )
                                along_ok_h = tip_in_bore_h or along_now_h <= max_along_h
                                axis_ok_h = axis_now_deg <= max_axis_hold
                                hold_confirm_n = int(
                                    a_cfg.get(
                                        "surface_tip_star_hold_enter_confirm", 4
                                    )
                                )
                                need_hold_up = bool(hold_until_up) and (
                                    axis_now_deg > max_axis_hold
                                    or int(tip_hybrid_state.upright_ok_streak)
                                    < hold_confirm_n
                                )
                                if need_hold_up:
                                    tip_hybrid_state.mode = "upright"
                                    if axis_now_deg > max_axis_hold:
                                        tip_hybrid_state.upright_ok_streak = 0
                                    tip_hybrid_state.star_mode = "yield_upright"
                                    tip_star_counts["hold_enter_upright"] = (
                                        tip_star_counts.get("hold_enter_upright", 0)
                                        + 1
                                    )
                                    if used_sp % 30 == 0:
                                        print(
                                            f"pci: FORCE hole cue held for upright "
                                            f"axis={axis_now_deg:.1f}>{max_axis_hold:.1f}deg "
                                            f"streak={int(tip_hybrid_state.upright_ok_streak)}/"
                                            f"{hold_confirm_n} "
                                            f"lat={feat_after_sp.lateral_m*1e3:.1f}mm "
                                            f"j={used_sp}",
                                            flush=True,
                                        )
                                elif hold_oracle_seat and (
                                    (not along_ok_h) or (not axis_ok_h)
                                ):
                                    if not along_ok_h:
                                        tip_hybrid_state.mode = "seat"
                                        tip_hybrid_state.seat_ok_streak = 0
                                        tip_hybrid_state.oracle_reseat_left = max(
                                            int(tip_hybrid_state.oracle_reseat_left),
                                            int(
                                                a_cfg.get(
                                                    "surface_tip_oracle_reseat_frames",
                                                    40,
                                                )
                                            )
                                            // 2,
                                        )
                                    else:
                                        tip_hybrid_state.mode = "upright"
                                        tip_hybrid_state.upright_ok_streak = 0
                                    if used_sp % 30 == 0:
                                        print(
                                            f"pci: ORACLE hole cue held for seat "
                                            f"along={along_now_h*1e3:.1f}mm "
                                            f"axis={axis_now_deg:.1f}deg "
                                            f"lat={feat_after_sp.lateral_m*1e3:.1f}mm "
                                            f"j={used_sp}",
                                            flush=True,
                                        )
                                elif hold_phig_seat and (
                                    (not along_ok_h) or (not axis_ok_h)
                                ):
                                    if not along_ok_h:
                                        tip_hybrid_state.mode = "seat"
                                        tip_hybrid_state.seat_ok_streak = 0
                                        phig_state.mode = "slip_reseat"
                                        phig_state.reseat_left = max(
                                            int(phig_state.reseat_left),
                                            int(phig_cfg.reseat_frames) // 2,
                                        )
                                    else:
                                        tip_hybrid_state.mode = "upright"
                                        tip_hybrid_state.upright_ok_streak = 0
                                        phig_state.mode = "enter_hold"
                                    if used_sp % 30 == 0:
                                        print(
                                            f"pci: PHIG hole cue held for seat "
                                            f"along={along_now_h*1e3:.1f}mm "
                                            f"axis={axis_now_deg:.1f}deg "
                                            f"r_cmd="
                                            f"{(float(planned_rs[planned_i]) if planned_rs is not None else -1)*1e3:.1f}mm "
                                            f"j={used_sp}",
                                            flush=True,
                                        )
                                else:
                                    spiral_reason = "hole_detected"
                                    print(
                                        f"pci: FORCE hole-enter stop lat="
                                        f"{feat_after_sp.lateral_m*1e3:.1f}mm "
                                        f"along={feat_after_sp.along_m*1e3:.1f}mm "
                                        f"|r|={contact_sp:.2f}N i={planned_i}",
                                        flush=True,
                                    )
                                    break
                            if (
                                conntact_faithful
                                and conntact_height_enter
                                and conntact_surface_h is not None
                            ):
                                tip_h = float(
                                    np.asarray(
                                        feat_after_sp.tip_pos, dtype=np.float64
                                    ).reshape(3)[2]
                                )
                                if height_drop_enter(
                                    tip_h,
                                    float(conntact_surface_h),
                                    drop_m=float(conntact_params.height_drop_m),
                                ):
                                    force_hole_state.entered = True
                                    spiral_reason = "hole_detected"
                                    print(
                                        f"pci: ConnTact height-drop enter "
                                        f"z={tip_h:.5f} surf={conntact_surface_h:.5f} "
                                        f"drop≥{conntact_params.height_drop_m*1e3:.2f}mm "
                                        f"lat={feat_after_sp.lateral_m*1e3:.1f}mm "
                                        f"along={feat_after_sp.along_m*1e3:.1f}mm "
                                        f"j={used_sp}",
                                        flush=True,
                                    )
                                    break
                            if (
                                franka_faithful
                                and franka_height_enter
                                and franka_surface_h is not None
                            ):
                                tip_h = float(
                                    np.asarray(
                                        feat_after_sp.tip_pos, dtype=np.float64
                                    ).reshape(3)[2]
                                )
                                if franka_height_drop_enter(
                                    tip_h,
                                    float(franka_surface_h),
                                    drop_m=float(franka_params.height_drop_m),
                                ):
                                    force_hole_state.entered = True
                                    spiral_reason = "hole_detected"
                                    print(
                                        f"pci: Franka height-drop enter "
                                        f"z={tip_h:.5f} surf={franka_surface_h:.5f} "
                                        f"drop≥{franka_params.height_drop_m*1e3:.2f}mm "
                                        f"jam_step={franka_state.spiral_step} "
                                        f"lat={feat_after_sp.lateral_m*1e3:.1f}mm "
                                        f"along={feat_after_sp.along_m*1e3:.1f}mm "
                                        f"j={used_sp}",
                                        flush=True,
                                    )
                                    break
                            if (
                                psft_faithful
                                and psft_height_enter
                                and psft_surface_h is not None
                            ):
                                tip_h = float(
                                    np.asarray(
                                        feat_after_sp.tip_pos, dtype=np.float64
                                    ).reshape(3)[2]
                                )
                                if psft_height_drop_enter(
                                    tip_h,
                                    float(psft_surface_h),
                                    drop_m=float(psft_params.height_drop_m),
                                ):
                                    force_hole_state.entered = True
                                    spiral_reason = "hole_detected"
                                    print(
                                        f"pci: PSFT height-drop enter "
                                        f"z={tip_h:.5f} surf={psft_surface_h:.5f} "
                                        f"drop≥{psft_params.height_drop_m*1e3:.2f}mm "
                                        f"r={psft_state.r_m*1e3:.1f}mm "
                                        f"θ={psft_state.theta_rad:.2f} "
                                        f"lat={feat_after_sp.lateral_m*1e3:.1f}mm "
                                        f"along={feat_after_sp.along_m*1e3:.1f}mm "
                                        f"j={used_sp}",
                                        flush=True,
                                    )
                                    break
                            if (
                                search_mode == "planned_spiral"
                                and priv_hole_entered
                                and bool(
                                    a_cfg.get(
                                        "surface_planned_priv_stop_on_enter", True
                                    )
                                )
                                and not (
                                    force_hole_detect
                                    and bool(
                                        a_cfg.get(
                                            "surface_planned_force_hole_only", True
                                        )
                                    )
                                )
                            ):
                                spiral_reason = "priv_hole_entered"
                                print(
                                    f"pci: PRIV hole-enter lat="
                                    f"{feat_after_sp.lateral_m*1e3:.1f}mm "
                                    f"along={feat_after_sp.along_m*1e3:.1f}mm "
                                    f"enter={priv_along_enter_mm*1e3:.1f}mm "
                                    f"|r|={contact_sp:.2f}N priv_fn={priv_fn:.3f} "
                                    f"i={planned_i}",
                                    flush=True,
                                )
                                break
                            if (
                                search_mode == "planned_spiral"
                                and planned_wps is not None
                                and planned_i >= planned_wps.shape[0] - 1
                                and float(np.linalg.norm(planned_wps[-1] - tip_a))
                                <= float(a_cfg.get("surface_planned_spiral_catch_m", 0.003))
                            ):
                                spiral_reason = "planned_spiral_done"
                                print(
                                    f"pci: planned-spiral done "
                                    f"lat={feat_after_sp.lateral_m*1e3:.1f}mm "
                                    f"along={feat_after_sp.along_m*1e3:.1f}mm "
                                    f"i={planned_i}/{planned_wps.shape[0]}",
                                    flush=True,
                                )
                                break
                            if float(feat_after_sp.lateral_m) <= mouth_lat:
                                # Tip at mouth radius: stop and mouth-probe.
                                # Keep threshold = mouth (not 1.5x) so we do not probe
                                # beside the hole (v2 false force / audit rejects).
                                # Agent-D oracle: if floated (no contact), keep tip→mouth
                                # + reseat instead of entering mouth-probe (ep08 flyaway).
                                oracle_need_seat = bool(
                                    tip_oracle_enable
                                    and float(contact_sp)
                                    < float(
                                        a_cfg.get(
                                            "surface_tip_hybrid_seat_contact_n", 0.05
                                        )
                                    )
                                    * 0.5
                                )
                                # Oracle: already tip-in-bore under SUCCESS_STANDARD →
                                # accept without mouth-probe (ep08 along~91 then flew).
                                if tip_oracle_enable:
                                    along_nm = float(feat_after_sp.along_m)
                                    tip_nm = float(feat_after_sp.tip_socket_dist_m)
                                    axis_nm = float(
                                        np.degrees(
                                            float(feat_after_sp.axis_error_rad)
                                        )
                                    )
                                    tip_bore_nm = tip_nm <= float(
                                        a_cfg.get(
                                            "surface_planned_priv_enter_tip_m",
                                            0.095,
                                        )
                                    ) or along_nm <= float(
                                        a_cfg.get(
                                            "surface_planned_priv_enter_along_m",
                                            0.095,
                                        )
                                    )
                                    axis_ok_nm = axis_nm <= float(
                                        a_cfg.get(
                                            "surface_planned_priv_enter_max_axis_err_deg",
                                            25.0,
                                        )
                                    )
                                    if tip_bore_nm and axis_ok_nm:
                                        force_hole_state.entered = True
                                        spiral_reason = "hole_detected"
                                        print(
                                            f"pci: ORACLE tip-in-bore enter "
                                            f"lat={feat_after_sp.lateral_m*1e3:.1f}mm "
                                            f"along={along_nm*1e3:.1f}mm "
                                            f"axis={axis_nm:.1f}deg "
                                            f"|r|={contact_sp:.3f}N j={used_sp}",
                                            flush=True,
                                        )
                                        break
                                    if tip_bore_nm and (not axis_ok_nm):
                                        tip_hybrid_state.mode = "upright"
                                        tip_hybrid_state.upright_ok_streak = 0
                                        if used_sp % 30 == 0:
                                            print(
                                                f"pci: ORACLE tip-in-bore hold upright "
                                                f"axis={axis_nm:.1f}deg "
                                                f"along={along_nm*1e3:.1f}mm "
                                                f"j={used_sp}",
                                                flush=True,
                                            )
                                        # Do not mouth-probe while tilted.
                                        continue
                                if oracle_need_seat:
                                    tip_hybrid_state.oracle_reseat_left = max(
                                        int(tip_hybrid_state.oracle_reseat_left),
                                        int(
                                            a_cfg.get(
                                                "surface_tip_oracle_reseat_frames",
                                                40,
                                            )
                                        )
                                        // 2,
                                    )
                                    if used_sp % 30 == 0:
                                        print(
                                            f"pci: ORACLE near mouth but float "
                                            f"|r|={contact_sp:.3f}N — reseat "
                                            f"lat={feat_after_sp.lateral_m*1e3:.1f}mm "
                                            f"j={used_sp}",
                                            flush=True,
                                        )
                                elif mouth_continue and search_mode != "planned_spiral":
                                    if used_sp == 1 or used_sp % 30 == 0:
                                        print(
                                            f"pci: near mouth continue search "
                                            f"lat={feat_after_sp.lateral_m*1e3:.1f}mm "
                                            f"along={feat_after_sp.along_m*1e3:.1f}mm "
                                            f"j={used_sp}",
                                            flush=True,
                                        )
                                else:
                                    spiral_reason = "near_mouth"
                                    print(
                                        f"pci: tip-spiral near mouth "
                                        f"lat={feat_after_sp.lateral_m*1e3:.1f}mm "
                                        f"along={feat_after_sp.along_m*1e3:.1f}mm "
                                        f"|r|={contact_sp:.2f}N tip_xy={tip_xy*1e3:.1f}mm",
                                        flush=True,
                                    )
                                    break
                            # Deployable mouth yield (optional). Default OFF — premature
                            # ρ-to-planner-center yield aborted healthy Tip-Hybrid
                            # spirals and false-fired mouth-probe (ep01/03).
                            if (
                                deploy_mouth_enter
                                and bool(
                                    a_cfg.get(
                                        "surface_deploy_mouth_yield_spiral", False
                                    )
                                )
                                and search_mode == "planned_spiral"
                                and (not force_hole_state.entered)
                            ):
                                tip_a_dm = np.asarray(
                                    feat_after_sp.tip_pos, dtype=np.float64
                                ).reshape(3)
                                tip_off_dm = _tip_lat_offset_xy(
                                    tip_a_dm, spiral_center, spiral_n
                                )
                                tip_rho_dm = float(np.linalg.norm(tip_off_dm))
                                e_dm = target - tip_a_dm
                                e_dm = e_dm - spiral_n * float(
                                    np.dot(e_dm, spiral_n)
                                )
                                tip_err_dm = float(np.linalg.norm(e_dm))
                                seat_n_dm = float(
                                    a_cfg.get(
                                        "surface_tip_hybrid_seat_contact_n", 0.05
                                    )
                                )
                                if (
                                    float(contact_sp) >= seat_n_dm * 0.5
                                    and deploy_mouth_yield_cue(
                                        r_cmd_m=float(r_cmd),
                                        tip_rho_m=tip_rho_dm,
                                        tip_err_m=tip_err_dm,
                                        mouth_m=float(mouth_lat),
                                        tip_err_near_m=float(
                                            a_cfg.get(
                                                "surface_deploy_mouth_yield_err_m",
                                                0.004,
                                            )
                                        ),
                                        contact_resid_n=float(contact_sp),
                                        peak_contact_resid_n=float(
                                            force_hole_state.peak_contact_resid
                                        ),
                                        resid_dip_n=float(
                                            a_cfg.get(
                                                "surface_deploy_mouth_resid_dip_n",
                                                0.05,
                                            )
                                        ),
                                        seat_contact_n=seat_n_dm,
                                        allow_resid_dip=bool(
                                            a_cfg.get(
                                                "surface_deploy_mouth_allow_resid_dip",
                                                False,
                                            )
                                        ),
                                    )
                                ):
                                    spiral_reason = "near_mouth"
                                    tip_star_counts["yield_mouth"] = (
                                        tip_star_counts.get("yield_mouth", 0) + 1
                                    )
                                    print(
                                        f"pci: deploy-mouth yield "
                                        f"r={float(r_cmd)*1e3:.1f}mm "
                                        f"ρ={tip_rho_dm*1e3:.1f}mm "
                                        f"err={tip_err_dm*1e3:.1f}mm "
                                        f"|r|={contact_sp:.2f}N "
                                        f"→ SEAT+mouth-probe",
                                        flush=True,
                                    )
                                    break
                            if used_sp % 30 == 0 or used_sp == 1:
                                if search_mode in _priv_lat_modes:
                                    print(
                                        f"pci: {search_mode} j={used_sp} |r|={contact_sp:.2f}N "
                                        f"lat={feat_after_sp.lateral_m*1e3:.1f}mm "
                                        f"along={feat_after_sp.along_m*1e3:.1f}mm "
                                        f"axis={feat_after_sp.axis_error_rad*180/np.pi:.1f}deg "
                                        f"din={ax_j:.2f}mm dith={ay_j:.2f}mm "
                                        f"tip_xy={tip_xy*1e3:.1f}mm tilt={tilt_i:.1f}deg "
                                        f"stick={stick_phase} "
                                        f"f_des={f_des_step:.2f}N",
                                        flush=True,
                                    )
                                elif search_mode == "lissajous":
                                    print(
                                        f"pci: tip-lissajous j={used_sp} |r|={contact_sp:.2f}N "
                                        f"lat={feat_after_sp.lateral_m*1e3:.1f}mm "
                                        f"along={feat_after_sp.along_m*1e3:.1f}mm "
                                        f"ax={ax_j*1e3:.1f} ay={ay_j*1e3:.1f}mm "
                                        f"tip_xy={tip_xy*1e3:.1f}mm tilt={tilt_i:.1f}deg",
                                        flush=True,
                                    )
                                else:
                                    print(
                                        f"pci: tip-spiral j={used_sp} |r|={contact_sp:.2f}N "
                                        f"lat={feat_after_sp.lateral_m*1e3:.1f}mm "
                                        f"along={feat_after_sp.along_m*1e3:.1f}mm "
                                        f"θ={theta_sp:.1f} r={r_cmd*1e3:.1f}mm "
                                        f"tip_xy={tip_xy*1e3:.1f}mm tilt={tilt_i:.1f}deg",
                                        flush=True,
                                    )
                        steps += used_sp
                        mouth_wiggle_used = 0
                        mouth_wiggle_ok = False
                        mouth_tip_xy_peak = 0.0
                        mouth_wiggle_reason = "skip"
                        if mouth_wiggle_enable and spiral_reason != "grasp_slip":
                            feat_m0 = features_from_raw(raw)
                            run_mouth = (
                                spiral_reason == "near_mouth"
                                or bool(stick_reached_mouth)
                                or float(feat_m0.lateral_m) <= mouth_lat * 2.5
                                or (
                                    search_mode == "planned_spiral"
                                    and spiral_reason
                                    in (
                                        "priv_overforce",
                                        "spiral_timeout",
                                        "planned_spiral_done",
                                    )
                                    and float(feat_m0.lateral_m)
                                    <= float(
                                        a_cfg.get(
                                            "surface_mouth_fallback_lat_m", 0.028
                                        )
                                    )
                                )
                            )
                            if run_mouth and n_mouth > 0:
                                mouth_wiggle_reason = "run"
                                tip_m0 = np.asarray(
                                    feat_m0.tip_pos, dtype=np.float64
                                ).reshape(3)
                                along_m0 = float(feat_m0.along_m)
                                tip_dist_m0 = float(feat_m0.tip_socket_dist_m)
                                jam_streak = 0
                                jam_mode = False
                                # Freeze wrist-tip offset at mouth entry (after stick
                                # slide). Never reuse spiral-start offset_wt0_sp —
                                # tip may have moved tens of mm.
                                site_m0 = actual_action44_from_sites(raw)
                                if stick_offset_wt is not None:
                                    mouth_offset_wt = stick_offset_wt.copy()
                                else:
                                    mouth_offset_wt = (
                                        site_m0[0:3].copy() - tip_m0
                                    )
                                mouth_lat_best = float(feat_m0.lateral_m)
                                mouth_along_best = float(feat_m0.along_m)
                                mouth_along_stall = 0
                                mouth_wrist_step = float(
                                    a_cfg.get(
                                        "surface_priv_lat_stick_wrist_step_m",
                                        0.014,
                                    )
                                )
                                print(
                                    f"pci: mouth-wiggle start frames={n_mouth} "
                                    f"lat={feat_m0.lateral_m*1e3:.1f}mm "
                                    f"along={feat_m0.along_m*1e3:.1f}mm "
                                    f"tray_admit={mouth_tray_admit:.2f}",
                                    flush=True,
                                )
                                for mouth_wiggle_used in range(1, n_mouth + 1):
                                    feat_i = features_from_raw(raw)
                                    site_now = actual_action44_from_sites(raw)
                                    tip = np.asarray(
                                        feat_i.tip_pos, dtype=np.float64
                                    ).reshape(3)
                                    ax_i = feat_i.hole_axis / (
                                        np.linalg.norm(feat_i.hole_axis) + 1e-12
                                    )
                                    center_i = _hole_axis_point_at_tip(
                                        tip,
                                        feat_i.socket_pos,
                                        feat_i.hole_axis,
                                    )
                                    lat_now_i = float(feat_i.lateral_m)
                                    along_now_i = float(feat_i.along_m)
                                    # Absolute tip → hole axis; only push depth once
                                    # lat is tight (else rim-jam at ~98mm like ep1).
                                    tip_goal = np.asarray(
                                        center_i, dtype=np.float64
                                    ).reshape(3)
                                    lat_seat = max(float(mouth_lat) * 0.85, 0.0035)
                                    # Already near bore: keep seating even if lat wobbles.
                                    if along_now_i <= 0.095:
                                        lat_seat = max(lat_seat, float(mouth_lat) * 1.2)
                                    if lat_now_i <= lat_seat or along_now_i <= 0.094:
                                        seat = float(
                                            np.clip(
                                                (along_now_i - 0.085) * 0.65,
                                                0.002,
                                                0.016,
                                            )
                                        )
                                        if along_now_i <= 0.094:
                                            seat = max(seat, 0.004)
                                        tip_goal = tip_goal + press_ax * seat
                                    d_abs = tip_goal - tip
                                    if lat_now_i > lat_seat and along_now_i > 0.094:
                                        # Far / mid: planar-only chase first.
                                        d_abs = d_abs - ax_i * float(
                                            np.dot(d_abs, ax_i)
                                        )
                                    dn_abs = float(np.linalg.norm(d_abs))
                                    slide_cap = max(
                                        float(
                                            a_cfg.get(
                                                "surface_priv_lat_stick_slide_step_m",
                                                0.008,
                                            )
                                        ),
                                        0.008,
                                    )
                                    if lat_now_i <= lat_seat:
                                        slide_cap = max(slide_cap, 0.012)
                                    if dn_abs > slide_cap > 0.0:
                                        d_abs = d_abs * (slide_cap / dn_abs)
                                    # Tiny dither only once lat is already near mouth.
                                    ph = float(mouth_wiggle_used)
                                    d_wig = mouth_xy_m * (
                                        np.sin(mouth_wx * ph) * t1
                                        + np.sin(mouth_wy * ph + mouth_phi) * t2
                                    )
                                    if lat_now_i > mouth_lat * 1.5:
                                        d_wig = d_wig * 0.15
                                    elif lat_now_i <= lat_seat:
                                        d_wig = d_wig * 0.35
                                    d_in_n = dn_abs
                                    d_wig_n = float(np.linalg.norm(d_wig))
                                    # Light press for force contact / chamfer seat.
                                    press_m = max(mouth_ax_step, 0.00008)
                                    if lat_now_i <= lat_seat or along_now_i <= 0.094:
                                        press_m = max(press_m, 0.0004)
                                    elif lat_now_i <= mouth_lat * 1.25:
                                        # Still recentering: tiny contact only.
                                        press_m = min(press_m, 0.00005)
                                    if along_now_i <= 0.092:
                                        # Lock depth: kill dither that knocks tip out.
                                        d_wig = d_wig * 0.05
                                        press_m = max(press_m, 0.00055)
                                        # Prefer axial over further lateral chase.
                                        d_plan = d_abs - ax_i * float(
                                            np.dot(d_abs, ax_i)
                                        )
                                        d_ax_only = press_ax * float(
                                            np.dot(d_abs, press_ax)
                                        )
                                        if float(np.dot(d_abs, press_ax)) < 0.0:
                                            d_ax_only = press_ax * 0.004
                                        d_abs = 0.25 * d_plan + d_ax_only
                                    if jam_mode:
                                        press_m = -mouth_ax_step * 2.0
                                    target = tip + d_abs + 0.25 * d_wig + press_ax * press_m
                                    # Cap along float-away (don't climb out).
                                    if along_now_i > along_m0 + 0.004:
                                        target = target + press_ax * min(
                                            0.002, along_now_i - along_m0
                                        )
                                    off_xy = _tip_lat_offset_xy(
                                        tip, center_i, feat_i.hole_axis
                                    )
                                    dist = float(np.linalg.norm(off_xy))
                                    d_xy = d_abs * track
                                    contact, left_c, fz_r, fz_l = _wrist_contact()
                                    contact_sp = float(contact)
                                    spiral_resid_peak = max(
                                        spiral_resid_peak, contact_sp
                                    )
                                    jam_thresh = float(mouth_jam_fxy)
                                    if lat_now_i <= mouth_lat * 1.5:
                                        # Chamfer seating contact is expected; don't jam-lock.
                                        jam_thresh = max(jam_thresh, 1.6)
                                    if contact_sp >= jam_thresh:
                                        jam_streak += 1
                                    else:
                                        jam_streak = 0
                                    jam_mode = jam_streak >= mouth_jam_frames
                                    # Near-mouth jam hop: brief unload → planar recenter → press.
                                    if (
                                        jam_mode
                                        and lat_now_i <= mouth_lat * 1.75
                                        and along_now_i > 0.092
                                    ):
                                        hop = int(mouth_wiggle_used) % 30
                                        if hop < 8:
                                            target = tip + (-press_ax) * 0.00055 + 0.2 * d_abs
                                        elif hop < 18:
                                            target = tip + d_abs + 0.5 * d_wig
                                        else:
                                            target = tip + d_abs + press_ax * 0.00045
                                        jam_mode = hop < 8  # only unload on lift half
                                    # Along stall on rim: forced hop even without force-jam.
                                    along_stall_hop = False
                                    if along_now_i < mouth_along_best - 0.0004:
                                        mouth_along_best = along_now_i
                                        mouth_along_stall = 0
                                    else:
                                        mouth_along_stall += 1
                                    if (
                                        mouth_along_stall >= 35
                                        and lat_now_i <= mouth_lat * 1.75
                                        and along_now_i > 0.096
                                    ):
                                        along_stall_hop = True
                                        hop = int(mouth_along_stall) % 36
                                        live_off = site_now[0:3].copy() - tip
                                        if hop < 10:
                                            target = tip + (-press_ax) * 0.0008
                                            mouth_offset_wt = live_off
                                        elif hop < 22:
                                            target = tip + d_abs * 1.2
                                            mouth_offset_wt = live_off
                                        else:
                                            target = tip + d_abs + press_ax * 0.0006
                                        if hop == 35:
                                            mouth_along_stall = 0

                                    # Prefer light contact; unload hard jam.
                                    f_mouth = min(float(f_des), 0.08)
                                    if lat_now_i <= mouth_lat * 1.25 and not jam_mode:
                                        f_mouth = max(f_mouth, 0.35)
                                    err_f = f_mouth - contact_sp
                                    ax_step = float(
                                        np.clip(kp * err_f, -max_step, max_step)
                                    )
                                    if err_f < 0.0:
                                        ax_step = float(
                                            np.clip(
                                                kp * unload_boost * err_f,
                                                -max_step * unload_boost,
                                                max_step,
                                            )
                                        )
                                    # Seating into mouth: never unload tip back out
                                    # (was stuck ~98mm while force loop retracted).
                                    if (
                                        lat_now_i <= mouth_lat * 1.5
                                        and along_now_i > 0.090
                                        and contact_sp < 2.2
                                        and not jam_mode
                                    ):
                                        ax_step = max(0.0, ax_step)
                                    if along_stall_hop:
                                        ax_step = 0.0
                                    site_xyz = site_now[0:3].copy()
                                    # Refresh frozen offset when tip actually moves in.
                                    if lat_now_i < mouth_lat_best - 0.0005:
                                        mouth_lat_best = lat_now_i
                                        mouth_offset_wt = site_xyz - tip
                                    # If lat blows out, force live offset so tip follows.
                                    if lat_now_i > mouth_lat * 2.0:
                                        mouth_offset_wt = site_xyz - tip
                                    # Seating: always live offset so axial tip chase works
                                    # (frozen offset after stick slide often kills Z push).
                                    if (
                                        lat_now_i <= mouth_lat * 1.75
                                        and along_now_i > 0.090
                                    ):
                                        offset_wt = site_xyz - tip
                                    else:
                                        offset_wt = mouth_offset_wt
                                    hold_r = target + offset_wt + press_ax * ax_step
                                    d_cmd = hold_r - site_xyz
                                    dn = float(np.linalg.norm(d_cmd))
                                    max_cmd = max(mouth_wrist_step, 0.012)
                                    if (
                                        lat_now_i <= mouth_lat * 1.5
                                        and along_now_i > 0.090
                                    ):
                                        max_cmd = max(max_cmd, 0.018)
                                    if dn > max_cmd > 0.0:
                                        hold_r = site_xyz + d_cmd * (max_cmd / dn)
                                    hold_l[0:3] = site_now[22:25].copy() + d_xy * left_share_sp
                                    if mouth_tray_admit > 0.0 and dist > 1e-9:
                                        admit = min(
                                            mouth_tray_admit * mouth_lat_step,
                                            mouth_tray_admit_max,
                                        )
                                        hold_l[0:3] = site_now[22:25].copy() - off_xy / dist * admit
                                    cmd = np.zeros(44, dtype=np.float64)
                                    cmd[0:6] = site_now[0:6].copy()
                                    cmd[0:3] = hold_r
                                    _fill_tip_search_hands(cmd)
                                    cmd[22:28] = hold_l
                                    step_action44(
                                        gym_env, cmd, ego_recorder=ego_recorder
                                    )
                                    feat_after_m = features_from_raw(raw)
                                    tip_a = np.asarray(
                                        feat_after_m.tip_pos, dtype=np.float64
                                    ).reshape(3)
                                    d_tip = tip_a - tip_m0
                                    tip_xy_m = float(
                                        np.linalg.norm(
                                            d_tip - ax_i * float(np.dot(d_tip, ax_i))
                                        )
                                    )
                                    mouth_tip_xy_peak = max(
                                        mouth_tip_xy_peak, tip_xy_m
                                    )
                                    tip_xy_peak = max(tip_xy_peak, tip_xy_m)
                                    force_trace.append(
                                        {
                                            "t": float(t_force) * dt_f,
                                            "step": float(t_force),
                                            "phase": "mouth_wiggle",
                                            "resid_r": float(contact_sp),
                                            "resid_l": float(left_c),
                                            "fz_r": float(fz_r),
                                            "fz_l": float(fz_l),
                                            "lat_mm": float(feat_after_m.lateral_m)
                                            * 1000.0,
                                            "along_mm": float(feat_after_m.along_m)
                                            * 1000.0,
                                            "f_des": float(f_des),
                                            "tip_xy_mm": float(tip_xy_m) * 1000.0,
                                            "mouth_din_mm": float(d_in_n) * 1000.0,
                                            "mouth_wig_mm": float(d_wig_n) * 1000.0,
                                            "jam_mode": int(jam_mode),
                                        }
                                    )
                                    t_force += 1
                                    tilt_i = _tray_tilt_entry_deg()
                                    phase_a_tilt_peak = max(phase_a_tilt_peak, tilt_i)
                                    phase_a_peg_tilt_peak = max(
                                        phase_a_peg_tilt_peak, _peg_tilt_entry_deg()
                                    )
                                    spiral_lat_min = min(
                                        spiral_lat_min,
                                        float(feat_after_m.lateral_m),
                                    )
                                    if _grasp_lost():
                                        # Privileged mouth seat: do not abort on slip;
                                        # gate still records peak. Only bail if tip
                                        # fled far from the hole.
                                        if (
                                            float(feat_after_m.lateral_m)
                                            > mouth_lat * 4.0
                                            and float(feat_after_m.along_m) > 0.110
                                        ):
                                            mouth_wiggle_reason = "grasp_slip"
                                            meta["grasp_abort_phase"] = "mouth_wiggle"
                                            break
                                        # else keep seating


                                    out_i = env._labeler.compute(raw)
                                    mouth_tilt_lim = max(float(max_tilt_sp), 16.0)
                                    if (not out_i.tray_ok) or tilt_i > mouth_tilt_lim:
                                        mouth_wiggle_reason = "mouth_tilt_or_tray"
                                        break
                                    along_gain = float(feat_after_m.along_m) - along_m0
                                    tip_gain = tip_dist_m0 - float(
                                        feat_after_m.tip_socket_dist_m
                                    )
                                    lat_now_m = float(feat_after_m.lateral_m)
                                    along_now_m = float(feat_after_m.along_m)
                                    tip_now_m = float(
                                        feat_after_m.tip_socket_dist_m
                                    )
                                    # Real enter: tip/along into bore (≤91mm).
                                    entered = (
                                        along_now_m <= 0.091 or tip_now_m <= 0.091
                                    )
                                    if entered and lat_now_m <= mouth_lat * 2.0:
                                        mouth_wiggle_reason = "mouth_insert_progress"
                                        spiral_reason = "mouth_insert"
                                        mouth_wiggle_ok = True
                                        print(
                                            f"pci: mouth-wiggle ENTER "
                                            f"along={along_now_m*1e3:.1f}mm "
                                            f"tip={tip_now_m*1e3:.1f}mm "
                                            f"lat={lat_now_m*1e3:.1f}mm "
                                            f"j={mouth_wiggle_used}",
                                            flush=True,
                                        )
                                        break
                                    # Keep pressing once lat is seated on axis.
                                    if (
                                        lat_now_m <= mouth_lat * 1.25
                                        and mouth_wiggle_used > 40
                                        and not jam_mode
                                    ):
                                        # Extra axial push baked into next target via
                                        # larger press — bump mouth_ax_step locally.
                                        mouth_ax_step = max(
                                            mouth_ax_step, 0.00018
                                        )
                                    if mouth_wiggle_used % 30 == 0 or mouth_wiggle_used == 1:
                                        print(
                                            f"pci: mouth-wiggle j={mouth_wiggle_used} "
                                            f"|r|={contact_sp:.2f}N jam={int(jam_mode)} "
                                            f"lat={feat_after_m.lateral_m*1e3:.1f}mm "
                                            f"along={feat_after_m.along_m*1e3:.1f}mm "
                                            f"tip_xy={tip_xy_m*1e3:.1f}mm",
                                            flush=True,
                                        )
                                steps += int(mouth_wiggle_used)
                                if (
                                    mouth_wiggle_reason == "run"
                                    and mouth_wiggle_used >= 24
                                    and mouth_tip_xy_peak >= 0.003
                                ):
                                    mouth_wiggle_ok = True
                                if (
                                    spiral_reason in ("spiral_timeout", "near_mouth")
                                    and mouth_wiggle_used > 0
                                ):
                                    spiral_reason = (
                                        mouth_wiggle_reason
                                        if mouth_wiggle_reason != "run"
                                        else "mouth_wiggle_done"
                                    )
                        meta["mouth_wiggle_enable"] = bool(mouth_wiggle_enable)
                        meta["mouth_wiggle_frames"] = int(mouth_wiggle_used)
                        meta["mouth_wiggle_ok"] = bool(mouth_wiggle_ok)
                        meta["mouth_wiggle_reason"] = str(mouth_wiggle_reason)
                        meta["mouth_tip_xy_peak_mm"] = float(mouth_tip_xy_peak) * 1000.0
                        meta["priv_lat_along_hold"] = bool(priv_lat_along_hold)
                        meta["spiral_along_hold"] = bool(spiral_along_hold)
                        meta["priv_lat_stick_enable"] = bool(stick_enable)
                        meta["priv_lat_stick_frames"] = int(stick_count)
                        meta["priv_lat_stick_phase"] = str(stick_phase)
                        meta["stick_reached_mouth"] = bool(stick_reached_mouth)
                        meta["stick_hop_count"] = int(stick_hop_count)
                        if along_hold_m is not None:
                            meta["priv_lat_along_hold_m"] = float(along_hold_m)
                        if mouth_wiggle_reason == "grasp_slip":
                            feat_s = features_from_raw(raw)
                            out_s = env._labeler.compute(raw)
                            meta["spiral_reason"] = "grasp_slip"
                            meta["force_trace"] = force_trace
                            meta["grasp_latch_ok"] = True
                            gate = _priv_tip_spiral_gate(meta, cfg)
                            meta["priv_tip_spiral_gate"] = gate
                            meta.update(_final_geom_meta(feat_s, out_s, ctrl))
                            return steps, "priv_tip_spiral_gate_fail", meta
                        feat_s = features_from_raw(raw)
                        out_s = env._labeler.compute(raw)
                        meta["align_lat_mm"] = feat_s.lateral_m * 1000
                        meta["align_along_mm"] = feat_s.along_m * 1000
                        meta["phase_a_tray_tilt_peak_deg"] = float(phase_a_tilt_peak)
                        meta["phase_a_peg_tilt_peak_deg"] = float(phase_a_peg_tilt_peak)
                        meta["phase_a_rel_rot_peak_rad"] = float(phase_a_rel_peak)
                        meta["spiral_frames"] = int(used_sp)
                        meta["spiral_reason"] = str(spiral_reason)
                        meta["priv_planned_hole_entered"] = bool(priv_hole_entered)
                        meta["force_planned_hole_entered"] = bool(
                            (
                                force_hole_state.entered
                                if (
                                    force_hole_detect
                                    or conntact_faithful
                                    or franka_faithful
                                    or psft_faithful
                                )
                                else False
                            )
                        )
                        if conntact_faithful:
                            meta["conntact_faithful"] = True
                            meta["conntact_height_enter"] = bool(
                                spiral_reason == "hole_detected"
                                and force_hole_state.entered
                            )
                            meta["conntact_surface_z"] = (
                                float(conntact_surface_h)
                                if conntact_surface_h is not None
                                else None
                            )
                        if franka_faithful:
                            meta["franka_faithful"] = True
                            meta["franka_height_enter"] = bool(
                                spiral_reason == "hole_detected"
                                and force_hole_state.entered
                            )
                            meta["franka_spiral_step"] = int(franka_state.spiral_step)
                            meta["franka_surface_z"] = (
                                float(franka_surface_h)
                                if franka_surface_h is not None
                                else None
                            )
                        if psft_faithful:
                            meta["psft_faithful"] = True
                            meta["psft_paper_reimpl"] = True
                            meta["psft_no_official_code"] = True
                            meta["psft_height_enter"] = bool(
                                spiral_reason == "hole_detected"
                                and force_hole_state.entered
                            )
                            meta["psft_r_m"] = float(psft_state.r_m)
                            meta["psft_theta_rad"] = float(psft_state.theta_rad)
                            meta["psft_step"] = int(psft_state.step)
                            meta["psft_surface_z"] = (
                                float(psft_surface_h)
                                if psft_surface_h is not None
                                else None
                            )
                            meta["franka_last"] = dict(franka_meta_last)
                        meta["priv_planned_at_mouth"] = bool(priv_at_mouth_peak)
                        meta["priv_planned_overforce_streak_peak"] = int(
                            priv_overforce_streak_peak
                        )
                        meta["priv_planned_along_drift_peak_mm"] = float(
                            priv_along_drift_peak_m
                        ) * 1000.0
                        meta["spiral_lat_min_mm"] = float(spiral_lat_min) * 1000
                        meta["spiral_resid_peak_n"] = float(spiral_resid_peak)
                        meta["spiral_tip_float_peak_m"] = float(spiral_tip_float_peak_m)
                        meta["spiral_axis_err_peak_deg"] = float(
                            spiral_axis_err_peak_deg
                        )
                        meta["spiral_contact_ok_n"] = int(spiral_contact_ok_n)
                        meta["spiral_contact_frames"] = int(spiral_contact_frames)
                        meta["spiral_contact_frac"] = float(
                            spiral_contact_ok_n / max(spiral_contact_frames, 1)
                        )
                        _early_den = max(
                            1, min(spiral_contact_frames, spiral_early_frames_n)
                        )
                        meta["spiral_early_contact_ok_n"] = int(
                            spiral_early_contact_ok_n
                        )
                        meta["spiral_early_contact_frames"] = int(_early_den)
                        meta["spiral_early_contact_frac"] = float(
                            spiral_early_contact_ok_n / float(_early_den)
                        )
                        meta["tip_hybrid_enable"] = bool(tip_hybrid_enable)
                        meta["tip_hybrid_mode_final"] = str(tip_hybrid_state.mode)
                        meta["tip_hybrid_seat_frames"] = int(
                            tip_hybrid_mode_counts.get("seat", 0)
                        )
                        meta["tip_hybrid_upright_frames"] = int(
                            tip_hybrid_mode_counts.get("upright", 0)
                        )
                        meta["tip_hybrid_spiral_frames"] = int(
                            tip_hybrid_mode_counts.get("spiral", 0)
                        )
                        meta["tip_msar_enable"] = bool(tip_msar_enable)
                        meta["tip_msar_mode_final"] = str(
                            tip_hybrid_state.msar_mode
                        )
                        meta["tip_msar_mouth_seek_frames"] = int(
                            tip_msar_counts.get("mouth_seek", 0)
                        )
                        meta["tip_msar_slip_reseat_frames"] = int(
                            tip_msar_counts.get("slip_reseat", 0)
                        )
                        meta["tip_msar_spiral_track_frames"] = int(
                            tip_msar_counts.get("spiral_track", 0)
                        )
                        meta["tip_star_enable"] = bool(tip_star_enable)
                        meta["tip_slip_enable"] = bool(tip_slip_enable)
                        meta["tip_star_mode_final"] = str(tip_hybrid_state.star_mode)
                        meta["tip_star_local_recover_frames"] = int(
                            tip_star_counts.get("local_recover", 0)
                        )
                        meta["tip_star_grasp_restore_frames"] = int(
                            tip_star_counts.get("grasp_restore", 0)
                        )
                        meta["tip_slip_grasp_restore_frames"] = int(
                            tip_star_counts.get("grasp_restore", 0)
                        )
                        meta["tip_star_yield_mouth_frames"] = int(
                            tip_star_counts.get("yield_mouth", 0)
                        )
                        meta["tip_star_yield_upright_frames"] = int(
                            tip_star_counts.get("yield_upright", 0)
                        )
                        meta["tip_star_hold_enter_upright_frames"] = int(
                            tip_star_counts.get("hold_enter_upright", 0)
                        )
                        meta["tip_star_spiral_track_frames"] = int(
                            tip_star_counts.get("spiral_track", 0)
                        )
                        meta["tip_star_grasp_boosted"] = bool(
                            tip_hybrid_state.star_grasp_boosted
                        )
                        meta["tip_slip_grasp_boosted"] = bool(
                            tip_hybrid_state.star_grasp_boosted
                        )
                        if "tip_slip_hand_mean_abs_before" not in meta:
                            meta["tip_slip_hand_mean_abs_before"] = float(
                                tip_hybrid_state.star_grasp_mean_abs_before
                            )
                            meta["tip_slip_hand_mean_abs_after"] = float(
                                tip_hybrid_state.star_grasp_mean_abs_after
                            )
                            meta["tip_slip_grasp_tighten_ok"] = bool(
                                tip_hybrid_state.star_grasp_boosted
                                and float(
                                    tip_hybrid_state.star_grasp_mean_abs_after
                                )
                                > float(
                                    tip_hybrid_state.star_grasp_mean_abs_before
                                )
                                + 1e-6
                            )
                        meta["tip_oracle_enable"] = bool(tip_oracle_enable)
                        meta["tip_baseline_mode"] = str(tip_baseline_mode)
                        meta["privileged_oracle_tip_servo"] = bool(
                            tip_oracle_enable
                            and (not tip_obs_enable)
                            and (not tip_est_enable)
                        )
                        meta["track_a_tip_est"] = bool(tip_est_enable)
                        meta["tip_est_backend"] = (
                            "tip_axis_ekf"
                            if tip_axis_enable and tip_theory_ekf is not None
                            else (
                                "theory_ekf"
                                if tip_theory_ekf is not None
                                else ("contact_v0" if tip_est is not None else "")
                            )
                        )
                        meta["tip_est_err_peak_mm"] = float(tip_est_err_peak_m) * 1000.0
                        meta["tip_est_err_mean_mm"] = (
                            float(tip_est_err_sum_m) / float(tip_est_err_n) * 1000.0
                            if tip_est_err_n > 0
                            else -1.0
                        )
                        meta["tip_est_err_mean_seated_mm"] = float(
                            meta["tip_est_err_mean_mm"]
                        )
                        meta["tip_est_seated_frames"] = int(tip_est_seated_frames)
                        meta["tip_est_err_frames"] = int(tip_est_err_n)
                        meta["tip_est_last_source"] = str(tip_est_last_source)
                        meta["tip_est_uncert_peak_mm"] = (
                            float(tip_est_uncert_peak_m) * 1000.0
                            if tip_theory_ekf is not None
                            else -1.0
                        )
                        meta["tip_est_axis_err_peak_deg"] = (
                            float(np.degrees(tip_axis_err_peak_rad))
                            if tip_axis_enable and tip_theory_ekf is not None
                            else -1.0
                        )
                        meta["tip_est_uses_axis_hat"] = bool(
                            tip_axis_enable and tip_theory_ekf is not None
                        )
                        meta["tip_est_residual_privilege"] = (
                            "mujoco_peg_tray_contact_pos_intermittent"
                            "+wrench_eligible"
                            "+finger_proprio_axis"
                            "+wrist_approach_prior"
                            "+priv_grasp_slip"
                            if tip_axis_enable and tip_theory_ekf is not None
                            else (
                                "mujoco_peg_tray_contact_pos_intermittent"
                                "+wrench_eligible"
                                "+priv_grasp_slip"
                                "+geom_tip_posediff"
                                "+gmhqp_tip_task_tec_gated_hat"
                                if tip_tec_joint and tip_theory_ekf is not None
                                else (
                                    "mujoco_peg_tray_contact_pos_intermittent"
                                    "+wrench_eligible"
                                    "+priv_grasp_slip"
                                    "+gmhqp_tip_task_uses_tip_hat"
                                    if tip_gmhqp_tip_obs and tip_theory_ekf is not None
                                    else (
                                        "mujoco_peg_tray_contact_pos_intermittent+wrench_eligible+priv_grasp_slip"
                                        if tip_theory_ekf is not None
                                        else (
                                            "mujoco_peg_tray_contact_pos+priv_grasp_slip"
                                            if tip_est_enable
                                            else ""
                                        )
                                    )
                                )
                            )
                        )
                        meta["tec_joint_enable"] = bool(tip_tec_joint)
                        meta["tec_joint_gated_on_frac"] = float(
                            tec_joint_state.meta.get("tec_joint_gated_on_frac", 0.0)
                        ) if tip_tec_joint else -1.0
                        meta["tec_joint_last_reason"] = (
                            str(tec_joint_state.last_reason) if tip_tec_joint else ""
                        )
                        meta["privileged_observation_noise"] = bool(tip_obs_enable)
                        meta["tip_obs_noise_mm"] = float(tip_obs_sigma_mm)
                        meta["tip_obs_lag_frames"] = int(tip_obs_lag_frames)
                        meta["obs_noise"] = bool(tip_obs_enable)
                        meta["tip_oracle_slip_trips"] = int(
                            meta.get("tip_oracle_slip_trips", 0)
                        )
                        meta["tip_oracle_mouth_press_frames"] = int(
                            tip_hybrid_state.oracle_mouth_press_frames
                        )
                        meta["tip_oracle_grasp_boosted"] = bool(
                            tip_hybrid_state.oracle_grasp_boosted
                        )
                        meta["tip_phig_enable"] = bool(tip_phig_enable)
                        meta["tip_phig_mode"] = str(phig_state.mode)
                        meta["tip_phig_mouth_press_frames"] = int(
                            phig_state.mouth_press_frames
                        )
                        meta["tip_phig_grasp_boosted"] = bool(phig_state.grasp_boosted)
                        meta["tip_phig_slip_trips"] = int(
                            meta.get("tip_phig_slip_trips", 0)
                        )
                        meta["spiral_theta_rad"] = float(theta_sp)
                        meta["spiral_mode"] = (
                            str(search_mode)
                            if search_mode in _priv_lat_modes
                            else (
                                "lissajous_socket"
                                if search_mode == "lissajous"
                                else "tip_track_firm_grasp"
                            )
                        )
                        meta["search_mode"] = str(search_mode)
                        meta["spiral_center"] = str(spiral_center_mode)
                        meta["spiral_direction"] = str(spiral_dir)
                        meta["spiral_lat_shrink_only"] = bool(lat_shrink_only)
                        meta["spiral_pre_lat_frames"] = int(
                            meta.get("spiral_pre_lat_frames", pre_lat_used)
                        )
                        meta["spiral_tip_xy_peak_mm"] = float(tip_xy_peak) * 1000.0
                        meta["spiral_axis_src"] = str(spiral_axis_src)
                        meta["spiral_axis_ang_peg_deg"] = float(ang_peg_n)
                        meta["spiral_axis_ang_hole_deg"] = float(ang_hole_n)
                        meta["force_trace"] = force_trace
                        meta["force_trace_n"] = int(len(force_trace))
                        meta["grasp_slip_peak_m"] = float(grasp_slip_peak)
                        meta["grasp_latch_ok"] = True
                        # ep06-style: already at mouth with N≈1 spiral. Replace priv
                        # lat→insert with XY-hold + light probe until F/T hole drop.
                        mouth_probe_n = int(
                            a_cfg.get("surface_planned_mouth_probe_frames", 120)
                        )
                        mouth_probe_used = 0
                        r_cmd_end = (
                            float(planned_rs[min(planned_i, planned_rs.shape[0] - 1)])
                            if planned_rs is not None
                            else 1e9
                        )
                        feat_probe0 = features_from_raw(raw)
                        mouth_probe_r_max = float(
                            a_cfg.get(
                                "surface_planned_mouth_probe_r_max_m",
                                mouth_lat * 2.0,
                            )
                        )
                        tip_probe0 = np.asarray(
                            feat_probe0.tip_pos, dtype=np.float64
                        ).reshape(3)
                        tip_rho_probe0 = float(
                            np.linalg.norm(
                                _tip_lat_offset_xy(
                                    tip_probe0, spiral_center, spiral_n
                                )
                            )
                        )
                        # Tip must currently be at/inside mouth radius.
                        # Keep base lat / near_mouth gate (Tip-Hybrid already uses
                        # tip–hole geometry for this stop). Do NOT gate on planner
                        # tip_rho alone — frozen spiral_center can be 10–40 mm off
                        # (false yield / skipped probe on ep01/04).
                        near_for_mouth_probe = float(feat_probe0.lateral_m) <= (
                            mouth_lat * 1.25
                        ) or (
                            str(spiral_reason) == "near_mouth"
                            and float(feat_probe0.lateral_m) <= mouth_lat * 1.5
                        )
                        if (
                            force_hole_detect
                            and (not force_hole_state.entered)
                            and mouth_probe_n > 0
                            and search_mode == "planned_spiral"
                            and near_for_mouth_probe
                        ):
                            hold_p = current_action44(raw)
                            if getattr(ctrl, "_hold_l_hand", None) is not None:
                                hold_p = hold_p.copy()
                                hold_p[28:44] = np.asarray(
                                    ctrl._hold_l_hand, dtype=np.float64
                                ).reshape(16)
                            probe_m = float(
                                a_cfg.get(
                                    "surface_planned_mouth_probe_press_m",
                                    max(
                                        float(
                                            a_cfg.get(
                                                "surface_planned_probe_press_m",
                                                0.00008,
                                            )
                                        ),
                                        0.00012,
                                    ),
                                )
                            )
                            hard_unload_n = float(
                                a_cfg.get(
                                    "surface_planned_mouth_probe_hard_unload_n", 0.35
                                )
                            )
                            # Agent-D oracle v2: stronger mouth press / higher unload bar.
                            if tip_oracle_enable and bool(
                                a_cfg.get("surface_tip_oracle_mouth_press", False)
                            ):
                                probe_m = max(
                                    probe_m,
                                    float(
                                        a_cfg.get(
                                            "surface_tip_oracle_mouth_press_m",
                                            0.00035,
                                        )
                                    ),
                                )
                                hard_unload_n = max(
                                    hard_unload_n,
                                    float(
                                        a_cfg.get(
                                            "surface_tip_oracle_mouth_hard_unload_n",
                                            0.85,
                                        )
                                    ),
                                )
                                mouth_probe_n = max(
                                    mouth_probe_n,
                                    int(
                                        a_cfg.get(
                                            "surface_tip_oracle_mouth_probe_frames",
                                            mouth_probe_n,
                                        )
                                    ),
                                )
                            # Track-B PHIG: same deployable mouth press boost (no tip GT).
                            if tip_phig_enable:
                                probe_m = max(
                                    probe_m, float(phig_cfg.mouth_press_m)
                                )
                                hard_unload_n = max(
                                    hard_unload_n,
                                    float(phig_cfg.mouth_hard_unload_n),
                                )
                            along_mp0 = float(feat_probe0.along_m)
                            tip_mp0 = float(feat_probe0.tip_socket_dist_m)
                            # Use spiral press axis (into surface). Do NOT unload on
                            # mild residual — that was lifting the tip away from mouth.
                            press_p = np.asarray(
                                press_ax_spiral, dtype=np.float64
                            ).reshape(3)
                            pn = float(np.linalg.norm(press_p))
                            if pn < 1e-9:
                                press_p = -spiral_n / (np.linalg.norm(spiral_n) + 1e-12)
                            else:
                                press_p = press_p / pn
                            # Agent-D oracle: prefer hole-axis into-surface (ep08 float).
                            if tip_oracle_enable:
                                hole_u_mp0 = np.asarray(
                                    feat_probe0.hole_axis, dtype=np.float64
                                ).reshape(3)
                                hn0 = float(np.linalg.norm(hole_u_mp0))
                                if hn0 > 1e-12:
                                    press_p = -(hole_u_mp0 / hn0)
                            # Deployable: soft seat resid (~0.7N) must press into bore,
                            # not retract (oracle/MSAR ep03: hard_unload=0.35 → float).
                            jam_unload_n = float(
                                a_cfg.get(
                                    "surface_planned_mouth_probe_jam_n",
                                    max(hard_unload_n, 1.2)
                                    if deploy_mouth_enter
                                    else hard_unload_n,
                                )
                            )
                            seat_press_scale = float(
                                a_cfg.get(
                                    "surface_planned_mouth_probe_seat_press_scale",
                                    2.5
                                    if (
                                        deploy_mouth_enter
                                        or tip_phig_enable
                                        or (
                                            tip_oracle_enable
                                            and bool(
                                                a_cfg.get(
                                                    "surface_tip_oracle_mouth_press",
                                                    False,
                                                )
                                            )
                                        )
                                    )
                                    else 1.0,
                                )
                            )
                            seat_n_mp = float(
                                a_cfg.get(
                                    "surface_tip_hybrid_seat_contact_n", 0.05
                                )
                            )
                            height_require_seat = bool(
                                a_cfg.get(
                                    "surface_planned_mouth_height_require_seat",
                                    bool(deploy_mouth_enter),
                                )
                            )
                            max_along_mp = float(
                                a_cfg.get(
                                    "surface_planned_priv_enter_max_along_m", 0.100
                                )
                            )
                            # Fresh peak for mouth press: don't treat spiral soft
                            # residual as a hole drop.
                            force_hole_state.peak_contact_resid = 0.0
                            force_hole_state.peak_abs_fz = 0.0
                            force_hole_state.hole_confirm = 0
                            force_hole_state.abs_fz_ema = 0.0
                            xy_mode = (
                                "hold_xy"
                                if deploy_mouth_enter
                                else "hole_axis"
                            )
                            print(
                                f"pci: mouth-probe hold frames≤{mouth_probe_n} "
                                f"probe={probe_m*1e3:.2f}mm/step "
                                f"jam_unload>{jam_unload_n:.2f}N "
                                f"xy={xy_mode} seat_scale={seat_press_scale:.1f}",
                                flush=True,
                            )
                            mouth_track = float(
                                a_cfg.get("surface_planned_mouth_probe_track", 1.2)
                            )
                            mouth_xy = float(
                                a_cfg.get(
                                    "surface_planned_mouth_probe_max_step_m",
                                    0.006,
                                )
                            )
                            for _mp in range(mouth_probe_n):
                                contact_mp, left_mp, fz_r_mp, fz_l_mp = _wrist_contact()
                                contact_mp = float(contact_mp)
                                feat_pre = features_from_raw(raw)
                                tip_pre = np.asarray(
                                    feat_pre.tip_pos, dtype=np.float64
                                ).reshape(3)
                                if deploy_mouth_enter:
                                    # Planner spiral center only — no privileged hole.
                                    ctr_pre = np.asarray(
                                        spiral_center, dtype=np.float64
                                    ).reshape(3)
                                else:
                                    ctr_pre = _hole_axis_point_at_tip(
                                        tip_pre,
                                        feat_pre.socket_pos,
                                        feat_pre.hole_axis,
                                    )
                                err_pre = ctr_pre - tip_pre
                                err_pre = err_pre - spiral_n * float(
                                    np.dot(err_pre, spiral_n)
                                )
                                en_pre = float(np.linalg.norm(err_pre))
                                tip_rho_pre = float(
                                    np.linalg.norm(
                                        _tip_lat_offset_xy(
                                            tip_pre, spiral_center, spiral_n
                                        )
                                    )
                                )
                                at_mouth_est = tip_rho_pre <= mouth_lat * 1.25 or (
                                    float(feat_pre.lateral_m) <= mouth_lat * 1.25
                                    and not deploy_mouth_enter
                                )
                                if deploy_mouth_enter:
                                    # Hold tip XY (SEAT ownership). Do not chase
                                    # planner center — frozen center can be 10–30 mm
                                    # from true mouth and pulls tip off-hole (ep03).
                                    d_xy = np.zeros(3, dtype=np.float64)
                                else:
                                    if en_pre > 1e-9:
                                        d_xy = err_pre * min(
                                            mouth_track, mouth_xy / en_pre
                                        )
                                    else:
                                        d_xy = np.zeros(3, dtype=np.float64)
                                site_pre = actual_action44_from_sites(raw)
                                hold_p = current_action44(raw).copy()
                                tip_high = float(feat_pre.tip_socket_dist_m) > 0.110
                                # Oracle: no contact / tip high → reseat press only,
                                # freeze XY so slip cannot fly tip off mouth (ep08).
                                if tip_oracle_enable and (
                                    tip_high or contact_mp < float(seat_n_mp) * 0.5
                                ):
                                    d_xy = np.zeros(3, dtype=np.float64)
                                    hole_u_live = np.asarray(
                                        feat_pre.hole_axis, dtype=np.float64
                                    ).reshape(3)
                                    hn_live = float(np.linalg.norm(hole_u_live))
                                    if hn_live > 1e-12:
                                        press_p = -(hole_u_live / hn_live)
                                if contact_mp > jam_unload_n and (not tip_high):
                                    # True jam only — soft seat resid must not retract.
                                    ax_p = -press_p * probe_m
                                elif tip_high or (
                                    tip_oracle_enable
                                    and contact_mp < float(seat_n_mp) * 0.5
                                ):
                                    # Tip high / float: dive to the face along hole.
                                    ax_p = press_p * max(probe_m * 4.0, 0.00045)
                                elif at_mouth_est and contact_mp >= seat_n_mp * 0.5:
                                    # SEAT ownership at mouth: press into bore.
                                    ax_p = press_p * max(
                                        probe_m * seat_press_scale, 0.00025
                                    )
                                elif contact_mp < 0.03:
                                    ax_p = press_p * max(probe_m * 2.0, 0.0002)
                                else:
                                    ax_p = press_p * probe_m
                                hold_p[0:3] = site_pre[0:3] + d_xy + ax_p
                                if getattr(ctrl, "_hold_l_hand", None) is not None:
                                    hold_p[28:44] = np.asarray(
                                        ctrl._hold_l_hand, dtype=np.float64
                                    ).reshape(16)
                                step_action44(
                                    gym_env, hold_p, ego_recorder=ego_recorder
                                )
                                mouth_probe_used += 1
                                steps += 1
                                used_sp += 1
                                feat_mp = features_from_raw(raw)
                                wr_mp = read_wrist_wrench_world(raw)
                                # Deployable: wrist-tool Fz, not hole-frame priv.
                                _p_mp, rot_mp = _right_wrist_site_pose(raw)
                                tf_mp = TaskFrame.from_wrist_site(hold_p[0:3], rot_mp)
                                wh_mp = tf_mp.wrench_tool(wr_mp[0])
                                tip_mp = np.asarray(
                                    feat_mp.tip_pos, dtype=np.float64
                                ).reshape(3)
                                tip_rho_mp = float(
                                    np.linalg.norm(
                                        _tip_lat_offset_xy(
                                            tip_mp, spiral_center, spiral_n
                                        )
                                    )
                                )
                                tip_lat_gate = float(feat_mp.lateral_m)
                                r_mp = min(float(mouth_lat), 0.0045)
                                axis_mp_deg = float(
                                    np.degrees(float(feat_mp.axis_error_rad))
                                )
                                max_axis_mp = float(
                                    a_cfg.get(
                                        "surface_planned_priv_enter_max_axis_err_deg",
                                        25.0,
                                    )
                                )
                                # Deployable mouth-enter and oracle v2 both require
                                # axis≤25 before accepting force-hole (SUCCESS_STANDARD).
                                hold_axis_mp = bool(deploy_mouth_enter) or (
                                    tip_oracle_enable
                                    and bool(
                                        a_cfg.get(
                                            "surface_tip_oracle_hold_enter_seat",
                                            True,
                                        )
                                    )
                                )
                                upright_ok_mp = (not hold_axis_mp) or (
                                    axis_mp_deg <= max_axis_mp
                                )
                                if (not upright_ok_mp) and tip_oracle_enable:
                                    tip_hybrid_state.mode = "upright"
                                    tip_hybrid_state.upright_ok_streak = 0
                                along_mp_now = float(feat_mp.along_m)
                                tip_mp_now = float(feat_mp.tip_socket_dist_m)
                                tip_bore_mp = tip_mp_now <= float(
                                    a_cfg.get(
                                        "surface_planned_priv_enter_tip_m", 0.095
                                    )
                                ) or along_mp_now <= float(
                                    a_cfg.get(
                                        "surface_planned_priv_enter_along_m", 0.095
                                    )
                                )
                                along_ok_mp = tip_bore_mp or along_mp_now <= float(
                                    a_cfg.get(
                                        "surface_planned_priv_enter_max_along_m",
                                        0.100,
                                    )
                                )
                                seat_gate_mp = upright_ok_mp and (
                                    (not tip_oracle_enable)
                                    or (not bool(
                                        a_cfg.get(
                                            "surface_tip_oracle_hold_enter_seat",
                                            True,
                                        )
                                    ))
                                    or along_ok_mp
                                )
                                if seat_gate_mp and update_spiral_force_hole_detect(
                                    force_hole_state,
                                    fz_axial=float(wh_mp[2]),
                                    f_lat=float(np.linalg.norm(wh_mp[:2])),
                                    spiral_radius_m=r_mp,
                                    step_idx=int(mouth_probe_used),
                                    search_cfg=search_cfg_sp,
                                    contact_resid_n=float(contact_mp),
                                    tip_lat_m=float(tip_lat_gate),
                                ):
                                    spiral_reason = "hole_detected"
                                    print(
                                        f"pci: FORCE hole-enter via mouth-probe "
                                        f"j={mouth_probe_used} |r|={contact_mp:.3f}N "
                                        f"along={feat_mp.along_m*1e3:.1f}mm "
                                        f"ρ={tip_rho_mp*1e3:.1f}mm "
                                        f"lat={feat_mp.lateral_m*1e3:.1f}mm "
                                        f"axis={axis_mp_deg:.1f}deg",
                                        flush=True,
                                    )
                                    break
                                # ConnTact-style: tip already at mouth + height drop
                                # into bore (ep01: along 99→90 without F resid drop).
                                drop_need_m = float(
                                    a_cfg.get(
                                        "surface_planned_mouth_along_drop_m",
                                        0.004,
                                    )
                                )
                                along_drop = along_mp0 - float(feat_mp.along_m)
                                tip_drop = tip_mp0 - float(
                                    feat_mp.tip_socket_dist_m
                                )
                                lat_ok_h = tip_lat_gate <= mouth_lat * (
                                    1.05 if deploy_mouth_enter else 1.25
                                )
                                seat_ok_h = (not height_require_seat) or (
                                    contact_mp >= seat_n_mp * 0.5
                                    and float(feat_mp.along_m) <= max_along_mp
                                )
                                if (
                                    upright_ok_mp
                                    and seat_gate_mp
                                    and lat_ok_h
                                    and seat_ok_h
                                    and max(along_drop, tip_drop) >= drop_need_m
                                ):
                                    force_hole_state.entered = True
                                    spiral_reason = "hole_detected"
                                    print(
                                        f"pci: FORCE hole-enter via mouth-height "
                                        f"j={mouth_probe_used} "
                                        f"along_drop={along_drop*1e3:.1f}mm "
                                        f"tip_drop={tip_drop*1e3:.1f}mm "
                                        f"ρ={tip_rho_mp*1e3:.1f}mm "
                                        f"lat={feat_mp.lateral_m*1e3:.1f}mm "
                                        f"axis={axis_mp_deg:.1f}deg "
                                        f"|r|={contact_mp:.3f}N",
                                        flush=True,
                                    )
                                    break
                                # While axis soft-bad, keep probing with SEAT press
                                # but do not latch force-hole (audit axis gate).
                                if (
                                    deploy_mouth_enter
                                    and tip_hybrid_enable
                                    and axis_mp_deg > max_axis_mp
                                    and mouth_probe_used % 30 == 0
                                ):
                                    print(
                                        f"pci: mouth-probe hold for upright "
                                        f"axis={axis_mp_deg:.1f}>{max_axis_mp:.1f}deg "
                                        f"ρ={tip_rho_mp*1e3:.1f}mm "
                                        f"|r|={contact_mp:.2f}N",
                                        flush=True,
                                    )
                                contact_mp_log, left_mp_log, fz_r_log, fz_l_log = (
                                    _wrist_contact()
                                )
                                f_des_mp = float(
                                    a_cfg.get(
                                        "surface_planned_near_hole_f_des_n",
                                        a_cfg.get(
                                            "surface_planned_spiral_f_des_n",
                                            a_cfg.get(
                                                "surface_const_force_des_n", 0.05
                                            ),
                                        ),
                                    )
                                )
                                force_trace.append(
                                    {
                                        "t": float(t_force) * dt_f,
                                        "step": float(t_force),
                                        "phase": "mouth_probe",
                                        "resid_r": float(contact_mp_log),
                                        "resid_l": float(left_mp_log),
                                        "fz_r": float(fz_r_log),
                                        "fz_l": float(fz_l_log),
                                        "f_des": float(f_des_mp),
                                        "lat_mm": float(feat_mp.lateral_m) * 1000.0,
                                        "along_mm": float(feat_mp.along_m) * 1000.0,
                                        "force_hole_entered": int(
                                            force_hole_state.entered
                                        ),
                                    }
                                )
                                t_force += 1
                                if mouth_probe_used % 30 == 0:
                                    print(
                                        f"pci: mouth-probe j={mouth_probe_used} "
                                        f"|r|={contact_mp:.2f}N "
                                        f"ρ={tip_rho_mp*1e3:.1f}mm "
                                        f"lat={feat_mp.lateral_m*1e3:.1f}mm "
                                        f"along={feat_mp.along_m*1e3:.1f}mm",
                                        flush=True,
                                    )
                                out_mp = env._labeler.compute(raw)
                                if not out_mp.tray_ok:
                                    break
                            meta["mouth_probe_frames"] = int(mouth_probe_used)
                            meta["force_planned_hole_entered"] = bool(
                                force_hole_state.entered
                            )
                            meta["deploy_mouth_enter"] = bool(deploy_mouth_enter)
                            feat_s = features_from_raw(raw)
                            out_s = env._labeler.compute(raw)
                            spiral_lat_min = min(
                                spiral_lat_min, float(feat_s.lateral_m)
                            )
                            meta["spiral_lat_min_mm"] = float(spiral_lat_min) * 1000
                            meta["spiral_reason"] = str(spiral_reason)
                        ins_steps, t_force, ins_meta = _run_planned_spiral_axial_insert(
                            env,
                            gym_env,
                            cfg=cfg,
                            ctrl=ctrl,
                            ego_recorder=ego_recorder,
                            force_trace=force_trace,
                            t_force=t_force,
                            dt_f=dt_f,
                            spiral_reason=str(spiral_reason),
                            force_hole_entered=bool(
                                force_hole_state.entered if force_hole_detect else False
                            ),
                            spiral_lat_min_m=float(spiral_lat_min),
                        )
                        if ins_meta:
                            meta.update(ins_meta)
                            meta["force_trace"] = force_trace
                            meta["force_trace_n"] = int(len(force_trace))
                            steps += int(ins_steps)
                            feat_s = features_from_raw(raw)
                            out_s = env._labeler.compute(raw)
                        gate = _priv_tip_spiral_gate(meta, cfg)
                        meta["priv_tip_spiral_gate"] = gate
                        meta.update(_final_geom_meta(feat_s, out_s, ctrl))
                        print(
                            f"pci: tip-spiral done reason={spiral_reason} "
                            f"lat={feat_s.lateral_m*1e3:.1f}mm "
                            f"along={feat_s.along_m*1e3:.1f}mm tip={feat_s.tip_socket_dist_m*1e3:.1f}mm "
                            f"|r|_peak={spiral_resid_peak:.2f}N "
                            f"tip_xy_peak={tip_xy_peak*1e3:.1f}mm "
                            f"grasp_slip={grasp_slip_peak*1e3:.1f}mm "
                            f"tilt={phase_a_tilt_peak:.2f}deg "
                            f"GATE={'PASS' if gate['ok'] else 'FAIL'} "
                            f"lift={gate['lift_ok']} spiral={gate['spiral_ok']} "
                            f"force={gate['force_ok']} grasp={gate.get('grasp_ok')}",
                            flush=True,
                        )
                        if not gate["ok"]:
                            for rs in gate.get("reasons", []):
                                print(f"pci: GATE fail — {rs}", flush=True)
                        why = "surface_press" if gate["ok"] else "priv_tip_spiral_gate_fail"
                        return steps, why, meta
                    # r24: first latch + arm Phase-A QP even if soft_contact_relatch=false.
                    if use_pose_qp:
                        latch_existing = meta.get("latch_priv_geom")
                        if not isinstance(latch_existing, PrivGraspGeom):
                            snap0 = _maybe_latch_priv_geom(tag="soft_contact")
                            if isinstance(snap0, PrivGraspGeom):
                                _ensure_phase_a_qp(snap0)
                        else:
                            _ensure_phase_a_qp(latch_existing)
                    if use_pose_qp and soft_contact_relatch:
                        geom_soft = read_priv_grasp_geom(raw)
                        tilt_soft = _tray_tilt_entry_deg()
                        rel_thresh = float(a_cfg.get("early_latch_rel_rot_rad", 0.12))
                        tilt_thresh = float(a_cfg.get("early_latch_max_tilt_deg", 8.0))
                        latch_existing = meta.get("latch_priv_geom")
                        rel_soft = (
                            _rel_rot_err_rad(latch_existing, geom_soft)
                            if isinstance(latch_existing, PrivGraspGeom)
                            else 0.0
                        )
                        do_relatch = isinstance(latch_existing, PrivGraspGeom) and (
                            rel_soft < rel_thresh and tilt_soft < tilt_thresh
                        )
                        if not isinstance(latch_existing, PrivGraspGeom):
                            do_relatch = tilt_soft <= soft_latch_max_tilt_deg
                        if do_relatch:
                            snap = copy_priv_grasp_geom(geom_soft)
                            meta["latch_priv_geom"] = snap
                            meta["latch_priv_geom_ok"] = True
                            meta["relatch_soft_contact"] = True
                            meta["relatch_rel_rot_rad"] = float(rel_soft)
                            meta["relatch_tilt_deg"] = float(tilt_soft)
                            if _phase_a_qp_enable(cfg):
                                _reset_phase_a_qp(snap)
                            print(
                                f"pci: soft-contact re-latch rel={rel_soft:.3f}rad "
                                f"tilt={tilt_soft:.1f}deg",
                                flush=True,
                            )

            # Tip for deliver geom gate; optional mid-ALIGN tip abort (scheme B).
            tilt_now = _tray_tilt_entry_deg()
            peg_now = _peg_tilt_entry_deg()
            abort_tray = float(a_cfg.get("align_tip_abort_deg", 0.0))
            abort_peg = float(a_cfg.get("align_peg_abort_deg", 0.0))
            if soft_latched and (
                (abort_tray > 1e-6 and tilt_now > abort_tray)
                or (abort_peg > 1e-6 and peg_now > abort_peg)
            ):
                # Unload then fail — do not keep soft-pressing into tipped pose.
                n_u = int(a_cfg.get("align_abort_unload_frames", 15))
                u_step = float(a_cfg.get("surface_resid_unload_step_m", 0.0004))
                hole_u_ab = feat.hole_axis / (np.linalg.norm(feat.hole_axis) + 1e-12)
                hold_ab = actual_action44_from_sites(raw)
                for _u in range(max(0, n_u)):
                    hold_ab = hold_ab.copy()
                    hold_ab[0:3] = hold_ab[0:3] + hole_u_ab * u_step
                    if getattr(ctrl, "_hold_l_hand", None) is not None:
                        hold_ab[28:44] = np.asarray(ctrl._hold_l_hand, dtype=np.float64)
                    step_action44(gym_env, hold_ab, ego_recorder=ego_recorder)
                    steps += 1
                if ctrl.active:
                    ctrl._deactivate()  # noqa: SLF001
                meta["align_tip_abort"] = True
                meta["align_tip_abort_tray_deg"] = float(tilt_now)
                meta["align_tip_abort_peg_deg"] = float(peg_now)
                meta["phase_a_tray_tilt_peak_deg"] = float(
                    max(phase_a_tilt_peak, tilt_now)
                )
                meta["phase_a_peg_tilt_peak_deg"] = float(
                    max(phase_a_peg_tilt_peak, peg_now)
                )
                print(
                    f"pci: ALIGN tip-abort tray={tilt_now:.1f}deg peg={peg_now:.1f}deg "
                    f"— unload {n_u} then fail",
                    flush=True,
                )
                return steps, "align_tip_abort", meta

            if contact_mag >= force_thresh:
                # Geom gate: refuse far false-contact deliveries (r25 ep3/4 latched at along>150mm).
                max_along = float(a_cfg.get("surface_soft_latch_max_along_m", 0.120))
                max_tip = float(a_cfg.get("surface_deliver_max_tip_m", 0.130))
                tip_now = float(feat.tip_socket_dist_m)
                along_abs = abs(float(feat.along_m))
                lat_abs = abs(float(feat.lateral_m))
                max_lat = float(a_cfg.get("surface_deliver_max_lat_m", 0.028))
                if (
                    along_abs > max_along
                    or tip_now > max_tip
                    or lat_abs > max_lat
                    or tilt_now > soft_latch_max_tilt_deg
                ):
                    contact_streak = 0
                    ctrl.config.pbvs_lambda_z = min(
                        float(ctrl.config.pbvs_lambda_z),
                        float(a_cfg.get("soft_contact_lambda_z", 0.015)),
                    )
                    if steps % 20 == 0:
                        print(
                            f"pci: refuse deliver geom along={along_abs*1e3:.0f} "
                            f"lat={lat_abs*1e3:.0f} tip={tip_now*1e3:.0f} "
                            f"tilt={tilt_now:.1f} — damp Z",
                            flush=True,
                        )
                else:
                    contact_streak += 1
            else:
                contact_streak = 0
            if contact_streak >= force_confirm:
                if _deliver_geom_ok(feat):
                    if bool(a_cfg.get("stop_after_surface", False)):
                        contact_streak = 0
                    else:
                        return _deliver_surface(feat, outcome, fz=fz, d_fz=d_fz)
                contact_streak = 0
                ctrl.config.pbvs_lambda_z = min(
                    float(ctrl.config.pbvs_lambda_z),
                    float(a_cfg.get("soft_contact_lambda_z", 0.015)),
                )

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
    episode_index: int = 0,
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
    bias_cfg = _socket_bias_cfg(cfg)
    base_seed = 0 if bias_cfg.seed is None else int(bias_cfg.seed)
    ep_rng = np.random.default_rng(base_seed + int(episode_index) * 1009 + 17)
    priv_ctrl = _priv_in_control(cfg)
    exp_tag = _experiment_tag(cfg)
    compliance_tag = _compliance_label(cfg)

    # Phase A: force-gated biased surface press (not open-loop INSERT).
    align_steps, align_reason, surface_meta = run_pbvs_biased_surface_press(
        env, hybrid, gym_env, cfg=cfg, ego_recorder=ego_recorder, rng=ep_rng,
        force_labeler=force_labeler,
    )
    surface_ok = align_reason in (
        "surface_press",
        "surface_press_but_bad_geom",
    )
    handoff_insert = bool(
        cfg.get("approach", {}).get("surface_handoff_insert_after_gate", False)
    )
    # Soft-land demo: skip settle unload (along-hole) and SEARCH.
    if bool(cfg.get("approach", {}).get("stop_after_surface", False)) and not handoff_insert:
        feat = features_from_raw(raw)
        out = env._labeler.compute(raw)
        clean = {
            k: v
            for k, v in dict(surface_meta).items()
            if not str(k).startswith("_")
            and not hasattr(v, "dtype")
            and not isinstance(v, PrivGraspGeom)
        }
        clean["stop_after_surface"] = True
        clean["stop_at_soft_contact"] = bool(surface_meta.get("stop_at_soft_contact"))
        tilt_peak = float(surface_meta.get("phase_a_tray_tilt_peak_deg", 0.0))
        max_tilt_ok = float(
            cfg.get("approach", {}).get("surface_settle_max_tray_tilt_deg", 8.0)
        )
        gate = surface_meta.get("priv_tip_spiral_gate")
        if not isinstance(gate, dict):
            gate = _priv_tip_spiral_gate(surface_meta, cfg)
            clean["priv_tip_spiral_gate"] = gate
        tray_ok_demo = bool(out.tray_ok) and tilt_peak <= max_tilt_ok
        surface_ok_demo = bool(tray_ok_demo and gate.get("ok") and surface_ok)
        insert_ok_demo = bool(clean.get("insert_ok", False))
        mouth_enter_demo = bool(
            clean.get("force_planned_hole_entered", False)
            or clean.get("force_hole_entered", False)
            or clean.get("mouth_enter_ok", False)
        )
        # Privileged audit (reporting only): tip seat + upright enter.
        # Standard: tip-on-surface spiral AND axis not too tilted at enter.
        # Reject: float search (ep03), sideways "enter" (ep02).
        a_cfg = cfg.get("approach", {})
        mouth_audit = float(a_cfg.get("surface_spiral_mouth_lat_m", 0.0045))
        lat_min_m = float(clean.get("spiral_lat_min_mm", 1e9)) / 1000.0
        along_drift_mm = float(clean.get("priv_planned_along_drift_peak_mm", 0.0))
        tip_rise_m = float(clean.get("surface_tip_along_rise_m", 0.0))
        tip_now_m = float(feat.tip_socket_dist_m)
        along_now_m = float(feat.along_m)
        axis_err_deg = float(np.degrees(feat.axis_error_rad))
        clean["final_axis_err_deg"] = float(axis_err_deg)
        float_peak_m = float(clean.get("spiral_tip_float_peak_m") or 0.0)
        _cf = clean.get("spiral_contact_frac", None)
        contact_frac = 1.0 if _cf is None else float(_cf)
        _ecf = clean.get("spiral_early_contact_frac", None)
        early_contact_frac = contact_frac if _ecf is None else float(_ecf)
        tip_in_bore = tip_now_m <= float(
            a_cfg.get("surface_planned_priv_enter_tip_m", 0.095)
        ) or along_now_m <= float(
            a_cfg.get("surface_planned_priv_enter_along_m", 0.095)
        )
        near_mouth_audit = lat_min_m <= mouth_audit * 1.25
        max_axis = float(
            a_cfg.get("surface_planned_priv_enter_max_axis_err_deg", 25.0)
        )
        max_float = float(
            a_cfg.get("surface_planned_priv_enter_max_float_m", 0.004)
        )
        # Hard seat (2026-09-04): float search is never success.
        min_contact_frac = float(
            a_cfg.get("surface_planned_priv_enter_min_contact_frac", 0.80)
        )
        min_early_contact_frac = float(
            a_cfg.get(
                "surface_planned_priv_enter_min_early_contact_frac", 0.85
            )
        )
        min_spiral_frames = int(
            a_cfg.get("surface_planned_priv_enter_min_spiral_frames", 30)
        )
        max_along = float(
            a_cfg.get("surface_planned_priv_enter_max_along_m", 0.100)
        )
        max_tip_rise = float(
            a_cfg.get("surface_planned_priv_enter_max_tip_rise_m", 0.006)
        )
        axis_ok = axis_err_deg <= max_axis
        # Floating over mouth (lat_min ok but along still high) is not enter.
        along_ok = tip_in_bore or along_now_m <= max_along
        # Geometry seat + contact seat (hard): reject START_FLOAT / DEGENERATE.
        seat_ok = float_peak_m <= max_float and tip_rise_m <= max_tip_rise
        n_c = int(clean.get("spiral_contact_frames") or 0)
        degenerate = n_c < min_spiral_frames
        if degenerate:
            seat_ok = False
        if min_contact_frac > 1e-9 and n_c >= min_spiral_frames:
            seat_ok = seat_ok and contact_frac >= min_contact_frac
        if min_early_contact_frac > 1e-9 and n_c >= min(min_spiral_frames, 20):
            seat_ok = seat_ok and early_contact_frac >= min_early_contact_frac
        lat_ok = near_mouth_audit or tip_in_bore
        priv_enter_ok = bool(lat_ok and along_ok and axis_ok and seat_ok)
        # Success: force hole + seated tip search + upright enter.
        enter_ok = bool(mouth_enter_demo and priv_enter_ok)
        mouth_ok_demo = enter_ok
        print(
            f"pci: stop_after_surface — tray_ok={int(bool(out.tray_ok))} "
            f"tilt_peak={tilt_peak:.2f}deg GATE={'PASS' if gate.get('ok') else 'FAIL'} "
            f"lat={feat.lateral_m*1e3:.1f}mm along={feat.along_m*1e3:.1f}mm "
            f"tip={feat.tip_socket_dist_m*1e3:.1f}mm "
            f"axis={axis_err_deg:.1f}deg float={float_peak_m*1e3:.1f}mm "
            f"cfrac={contact_frac:.2f} early={early_contact_frac:.2f} "
            f"force_hole={int(mouth_enter_demo)} priv_audit={int(priv_enter_ok)} "
            f"enter_ok={int(enter_ok)} insert_ok={int(insert_ok_demo)}",
            flush=True,
        )
        fail_reason = ""
        if not enter_ok:
            if mouth_enter_demo and not priv_enter_ok:
                if not axis_ok:
                    fail_reason = "force_hole_axis_reject"
                elif degenerate:
                    fail_reason = "force_hole_degenerate_reject"
                elif not seat_ok:
                    fail_reason = "force_hole_float_reject"
                elif not along_ok:
                    fail_reason = "force_hole_along_reject"
                else:
                    fail_reason = "force_hole_priv_reject"
            elif align_reason == "priv_tip_spiral_gate_fail" or not gate.get("ok"):
                fail_reason = "priv_tip_spiral_gate_fail"
            elif not tray_ok_demo:
                fail_reason = "surface_tilt_or_tray"
            else:
                fail_reason = align_reason or "no_force_hole"
        return {
            # Success = force hole + tip seat spiral + upright enter audit.
            "success": bool(enter_ok),
            "mouth_ok": bool(mouth_ok_demo),
            "mouth_enter_ok": bool(enter_ok),
            "force_hole_raw": bool(mouth_enter_demo),
            "priv_enter_audit": bool(priv_enter_ok),
            "priv_axis_ok": bool(axis_ok),
            "priv_seat_ok": bool(seat_ok),
            "insert_ok": bool(insert_ok_demo),
            "surface_gate_ok": bool(surface_ok_demo),
            "tray_ok": bool(out.tray_ok),
            "peg_ok": bool(out.peg_ok),
            "fail_reason": fail_reason,
            "align_phase": "SURFACE",
            "align_steps": align_steps,
            "control_steps": 0,
            "final_phase": "SURFACE_DONE",
            "final_tip_dist_m": feat.tip_socket_dist_m,
            "final_lat_m": feat.lateral_m,
            "final_along_m": feat.along_m,
            "eval_only": True,
            "hybrid_summary": hybrid.episode_summary(),
            "traj_tail": [],
            "traj": [],
            "phase_a_mode": "pbvs_biased_surface_press",
            "approach_noise": clean,
            "surface_meta": clean,
            "surface_reason": "stop_at_soft_contact",
            "compliance": compliance_tag,
            "priv_gate": True,
            "priv_tip_spiral_gate": gate,
        }

    if not surface_ok:
        feat = features_from_raw(raw)
        return {
            "success": False,
            "insert_ok": False,
            "fail_reason": align_reason,
            "align_phase": "SURFACE",
            "align_steps": align_steps,
            "control_steps": 0,
            "final_phase": "APPROACH",
            "final_tip_dist_m": feat.tip_socket_dist_m,
            "eval_only": True,
            "hybrid_summary": hybrid.episode_summary(),
            "traj_tail": [],
            "phase_a_mode": "pbvs_biased_surface_press",
            "approach_noise": surface_meta,
            "surface_meta": surface_meta,
        }

    # Post-surface settle: FT unload + left wrist yield (not rigid dual freeze forever).
    settle_s = float(cfg.get("approach", {}).get("surface_settle_seconds", 1.0))
    settle_frames = max(1, int(round(settle_s / _sim_dt(cfg))))
    max_tilt_deg = float(cfg.get("approach", {}).get("surface_settle_max_tray_tilt_deg", 8.0))
    settle44 = actual_action44_from_sites(raw)
    hold_l_hand = settle44[28:44].copy()
    hold_l_wrist = settle44[22:28].copy()
    # Prefer handoff-locked left command if surface froze it.
    ctrl_s = hybrid.controller
    if ctrl_s is not None and getattr(ctrl_s, "_hold_l_hand", None) is not None:
        hold_l_hand = np.asarray(ctrl_s._hold_l_hand, dtype=np.float64).reshape(16).copy()
    if ctrl_s is not None and getattr(ctrl_s, "_hold_l_arm", None) is not None:
        arm_h = np.asarray(ctrl_s._hold_l_arm, dtype=np.float64).reshape(22)
        hold_l_wrist = arm_h[0:6].copy()
        hold_l_hand = arm_h[6:22].copy()
    settle44[22:28] = hold_l_wrist
    settle44[28:44] = hold_l_hand
    from scipy.spatial.transform import Rotation as R

    # Settle tilt vs handoff (not re-baselined at contact) — privileged_diagnostic.
    if isinstance(surface_meta.get("tray_R_handoff"), list):
        tray_R0 = np.asarray(surface_meta["tray_R_handoff"], dtype=np.float64).reshape(3, 3)
    else:
        tray_R0 = read_priv_grasp_geom(raw).tray_rot.copy()
    surface_meta["settle_tilt_ref"] = "handoff"
    out0 = env._labeler.compute(raw)
    if not out0.tray_ok:
        print("pci: PRIV GATE fail — tray_ok already false entering settle", flush=True)
        feat = features_from_raw(raw)
        return {
            "success": False,
            "insert_ok": False,
            "fail_reason": "contact_tray_lost",
            "align_phase": "SURFACE",
            "align_steps": align_steps,
            "control_steps": 0,
            "final_phase": "SETTLE",
            "final_tip_dist_m": feat.tip_socket_dist_m,
            "eval_only": True,
            "hybrid_summary": hybrid.episode_summary(),
            "traj_tail": [],
            "phase_a_mode": "pbvs_biased_surface_press",
            "approach_noise": surface_meta,
            "surface_meta": dict(surface_meta),
            "compliance": "force_tactile",
            "priv_gate": True,
        }

    def _tray_tilt_deg() -> float:
        R_now = read_priv_grasp_geom(raw).tray_rot
        return float(np.linalg.norm(R.from_matrix(tray_R0.T @ R_now).as_rotvec()) * 180.0 / np.pi)

    unload_total = float(cfg.get("approach", {}).get("surface_settle_unload_m", 0.0015))
    # Control unload axis: wrist approach when sensor mode; hole_axis only for priv diag.
    if priv_ctrl:
        feat_settle0 = features_from_raw(raw)
        hole_u = np.asarray(feat_settle0.hole_axis, dtype=np.float64).reshape(3)
        hn = float(np.linalg.norm(hole_u))
        if hn > 1e-9:
            hole_u = hole_u / hn
        else:
            hole_u = np.array([0.0, 0.0, 1.0])
    else:
        # Wrist +Z is insert approach; retreat = -approach.
        hole_u = -_wrist_approach_axis(raw)
    # Insert = -hole_u when hole_u is hole axis; retreat = +hole_u.
    if unload_total > 0.0:
        settle44[0:3] = settle44[0:3] + hole_u * unload_total
    print(
        f"pci: surface settle {settle_s:.1f}s ({settle_frames} frames) "
        f"— unload {unload_total*1e3:.1f}mm + left hold + priv tray gate "
        f"(priv_ctrl={priv_ctrl})",
        flush=True,
    )
    settle_tilt_peak = 0.0
    settle_need_stabilize = False
    settle_need_straighten = False
    for i in range(settle_frames):
        cmd = settle44.copy()
        cmd[22:28] = hold_l_wrist
        cmd[28:44] = hold_l_hand
        step_action44(gym_env, cmd, ego_recorder=ego_recorder)
        outcome_s = env._labeler.compute(raw)
        tilt = _tray_tilt_deg()
        settle_tilt_peak = max(settle_tilt_peak, tilt)
        if not outcome_s.tray_ok:
            print(f"pci: PRIV GATE fail — tray_ok lost at settle frame {i}", flush=True)
            feat = features_from_raw(raw)
            return {
                "success": False,
                "insert_ok": False,
                "fail_reason": "settle_tray_lost",
                "align_phase": "SURFACE",
                "align_steps": align_steps,
                "control_steps": i + 1,
                "final_phase": "SETTLE",
                "final_tip_dist_m": feat.tip_socket_dist_m,
                "eval_only": True,
                "hybrid_summary": hybrid.episode_summary(),
                "traj_tail": [],
                "phase_a_mode": "pbvs_biased_surface_press",
                "approach_noise": surface_meta,
                "surface_meta": {**dict(surface_meta), "settle_tray_lost_frame": i},
                "compliance": compliance_tag,
                "experiment_tag": exp_tag,
                "priv_gate": True,
            }
        if tilt > max_tilt_deg:
            hard_settle = float(
                cfg.get("approach", {}).get("surface_settle_hard_tilt_deg", 32.0)
            )
            if tilt <= hard_settle:
                if not priv_ctrl:
                    # Sensor mode: FT admit yield, then require peg∧tray stable — not soft-continue.
                    print(
                        f"pci: settle tip soft {tilt:.1f}deg > {max_tilt_deg:.1f} "
                        f"— left admit stabilize (sensor)",
                        flush=True,
                    )
                    settle_need_stabilize = True
                    surface_meta["settle_tip_soft_admit"] = True
                    surface_meta["settle_tilt_peak_deg"] = settle_tilt_peak
                    break
                # Privileged: straighten before SEARCH (no 20° tip open-search).
                print(
                    f"pci: settle tip soft {tilt:.1f}deg > {max_tilt_deg:.1f} "
                    f"— priv straighten before SEARCH",
                    flush=True,
                )
                settle_need_straighten = True
                surface_meta["settle_tip_soft_straighten"] = True
                surface_meta["settle_tilt_peak_deg"] = settle_tilt_peak
                break
            print(
                f"pci: PRIV GATE fail — tray tilt {tilt:.1f}deg > {hard_settle:.1f} "
                f"at settle frame {i}",
                flush=True,
            )
            feat = features_from_raw(raw)
            return {
                "success": False,
                "insert_ok": False,
                "mouth_ok": False,
                "fail_reason": "settle_tray_tilt",
                "align_phase": "SURFACE",
                "align_steps": align_steps,
                "control_steps": i + 1,
                "final_phase": "SETTLE",
                "final_tip_dist_m": feat.tip_socket_dist_m,
                "eval_only": True,
                "hybrid_summary": hybrid.episode_summary(),
                "traj_tail": [],
                "phase_a_mode": "pbvs_biased_surface_press",
                "approach_noise": surface_meta,
                "surface_meta": {
                    **dict(surface_meta),
                    "settle_tilt_peak_deg": settle_tilt_peak,
                },
                "compliance": compliance_tag,
                "experiment_tag": exp_tag,
                "priv_gate": True,
            }

    # r15: privileged settle Phase-A QP + left admit to hold rel_rot before SEARCH.
    search_cfg_pre = cfg.get("compliant", {}).get("search", {})
    settle_qp_n = int(search_cfg_pre.get("settle_qp_frames", 0))
    use_priv_grasp_settle = bool(
        cfg.get("compliant", {}).get("priv_grasp_opt", {}).get("enable", False)
    )
    latch_settle = surface_meta.get("latch_priv_geom")
    settle_soft_admit = float(cfg.get("approach", {}).get("surface_soft_left_admit_scale", 0.0))
    if (
        priv_ctrl
        and _phase_a_qp_enable(cfg)
        and use_priv_grasp_settle
        and settle_qp_n > 0
        and isinstance(latch_settle, PrivGraspGeom)
        and float(surface_meta.get("rel_rot_at_contact_rad", 999.0)) > float(
            search_cfg_pre.get("settle_max_rel_rot_rad", 0.12)
        )
    ):
        settle_qp_ctrl = PrivGraspOptController(_priv_grasp_config(cfg))
        settle_qp_ctrl.reset(latch_settle, settle44[6:22].copy(), hold_l_hand.copy())
        settle_qp_admit: LeftWristAdmitController | None = None
        if settle_soft_admit > 1e-9:
            settle_qp_admit = LeftWristAdmitController(_left_wrist_admit_config(cfg))
            settle_qp_admit.reset(read_wrist_wrench_world(raw)[1])
            settle_qp_admit.config.enable = True
        settle_rel_peak = float(surface_meta.get("phase_a_rel_rot_peak_rad", 0.0))
        print(f"pci: settle Phase-A QP {settle_qp_n} frames + left_admit", flush=True)
        for _sq in range(settle_qp_n):
            geom_sq = read_priv_grasp_geom(raw)
            f12_sq, lf12_sq = read_dual_finger_force24(raw, force_labeler)
            po_sq = settle_qp_ctrl.step(geom_sq, f12_sq, lf12_sq)
            settle_rel_peak = max(settle_rel_peak, float(po_sq.rel_rot_err_rad))
            cmd = settle44.copy()
            cmd[6:22] = settle44[6:22] + po_sq.delta_right_hand16
            if bool(surface_meta.get("handoff_left_locked", False)):
                cmd[28:44] = hold_l_hand
            else:
                cmd[28:44] = np.maximum(
                    hold_l_hand + po_sq.delta_left_hand16, hold_l_hand * 0.98
                )
            if settle_qp_admit is not None and not bool(
                surface_meta.get("handoff_left_locked", False)
            ):
                d_left = (
                    settle_qp_admit.step(read_wrist_wrench_world(raw)[1], approach_axis=-hole_u)
                    * settle_soft_admit
                )
                hold_l_wrist[0:3] = hold_l_wrist[0:3] + d_left
            cmd[22:28] = hold_l_wrist
            step_action44(gym_env, cmd, ego_recorder=ego_recorder)
            tilt_sq = _tray_tilt_deg()
            settle_tilt_peak = max(settle_tilt_peak, tilt_sq)
        surface_meta["settle_qp_frames"] = settle_qp_n
        surface_meta["phase_a_rel_rot_peak_rad"] = settle_rel_peak
        print(f"pci: settle QP done rel_peak={settle_rel_peak:.3f}rad", flush=True)

    # Sensor settle stabilize: left FT admit yield, then peg_ok∧tray_ok for N frames.
    if settle_need_stabilize and not priv_ctrl:
        admit_frames = int(
            cfg.get("compliant", {}).get("search", {}).get("settle_admit_frames", 24)
        )
        stable_need = int(
            cfg.get("compliant", {}).get("search", {}).get("settle_stable_frames", 8)
        )
        left_scale = float(
            cfg.get("compliant", {}).get("search", {}).get("settle_admit_left_scale", 0.0)
        )
        unload_step = float(
            cfg.get("compliant", {})
            .get("search", {})
            .get(
                "settle_unload_step_m",
                cfg.get("compliant", {}).get("search", {}).get("search_unload_step_m", 0.0003),
            )
        )
        soft_continue = bool(
            cfg.get("compliant", {})
            .get("search", {})
            .get("settle_soft_continue_search", False)
        )
        # No motion requested: skip admit loop (avoids tip growth from residual push).
        if left_scale <= 1e-9 and unload_step <= 1e-12 and soft_continue:
            out_last = env._labeler.compute(raw)
            tilt_last = _tray_tilt_deg()
            hard_settle = float(
                cfg.get("approach", {}).get("surface_settle_hard_tilt_deg", 32.0)
            )
            if bool(out_last.peg_ok) and bool(out_last.tray_ok) and tilt_last <= hard_settle:
                surface_meta["settle_soft_continue_search"] = True
                surface_meta["settle_tilt_peak_deg"] = settle_tilt_peak
                print(
                    f"pci: settle soft-continue SEARCH (hold-only, tilt={tilt_last:.1f}deg)",
                    flush=True,
                )
            else:
                feat = features_from_raw(raw)
                return {
                    "success": False,
                    "insert_ok": False,
                    "mouth_ok": False,
                    "fail_reason": "settle_tray_tilt"
                    if tilt_last > hard_settle
                    else "settle_unstable",
                    "align_phase": "SURFACE",
                    "align_steps": align_steps,
                    "control_steps": settle_frames,
                    "final_phase": "SETTLE_ADMIT",
                    "final_tip_dist_m": feat.tip_socket_dist_m,
                    "eval_only": True,
                    "hybrid_summary": hybrid.episode_summary(),
                    "traj_tail": [],
                    "phase_a_mode": "pbvs_biased_surface_press",
                    "approach_noise": surface_meta,
                    "surface_meta": {
                        **dict(surface_meta),
                        "settle_tilt_peak_deg": settle_tilt_peak,
                    },
                    "compliance": compliance_tag,
                    "experiment_tag": exp_tag,
                    "priv_gate": True,
                }
        else:
            settle_admit = LeftWristAdmitController(_left_wrist_admit_config(cfg))
            settle_admit.reset(read_wrist_wrench_world(raw)[1])
            settle_admit.config.enable = bool(left_scale > 1e-9)
            wrist_ax = _wrist_approach_axis(raw)
            for j in range(admit_frames):
                wr = read_wrist_wrench_world(raw)
                if settle_admit.config.enable:
                    d_left = settle_admit.step(wr[1], approach_axis=wrist_ax) * left_scale
                    hold_l_wrist[0:3] = hold_l_wrist[0:3] + d_left
                # Soft axial unload on right (sensor FT path), not hole seek.
                if unload_step > 0.0:
                    settle44[0:3] = settle44[0:3] - wrist_ax * unload_step
                cmd = settle44.copy()
                cmd[22:28] = hold_l_wrist
                cmd[28:44] = hold_l_hand
                step_action44(gym_env, cmd, ego_recorder=ego_recorder)
                tilt = _tray_tilt_deg()
                settle_tilt_peak = max(settle_tilt_peak, tilt)
                hard_settle = float(
                    cfg.get("approach", {}).get("surface_settle_hard_tilt_deg", 32.0)
                )
                if tilt > hard_settle:
                    feat = features_from_raw(raw)
                    return {
                        "success": False,
                        "insert_ok": False,
                        "mouth_ok": False,
                        "fail_reason": "settle_tray_tilt",
                        "align_phase": "SURFACE",
                        "align_steps": align_steps,
                        "control_steps": settle_frames + j + 1,
                        "final_phase": "SETTLE_ADMIT",
                        "final_tip_dist_m": feat.tip_socket_dist_m,
                        "eval_only": True,
                        "hybrid_summary": hybrid.episode_summary(),
                        "traj_tail": [],
                        "phase_a_mode": "pbvs_biased_surface_press",
                        "approach_noise": surface_meta,
                        "surface_meta": {
                            **dict(surface_meta),
                            "settle_tilt_peak_deg": settle_tilt_peak,
                        },
                        "compliance": compliance_tag,
                        "experiment_tag": exp_tag,
                        "priv_gate": True,
                    }
            stable_streak = 0
            for k in range(max(stable_need * 3, stable_need)):
                cmd = settle44.copy()
                cmd[22:28] = hold_l_wrist
                cmd[28:44] = hold_l_hand
                step_action44(gym_env, cmd, ego_recorder=ego_recorder)
                out_k = env._labeler.compute(raw)
                tilt = _tray_tilt_deg()
                settle_tilt_peak = max(settle_tilt_peak, tilt)
                if bool(out_k.peg_ok) and bool(out_k.tray_ok) and tilt <= max_tilt_deg:
                    stable_streak += 1
                else:
                    stable_streak = 0
                if stable_streak >= stable_need:
                    surface_meta["settle_stable_frames"] = stable_streak
                    print(
                        f"pci: settle stabilize OK peg∧tray streak={stable_streak}",
                        flush=True,
                    )
                    break
            else:
                soft_continue = bool(
                    cfg.get("compliant", {})
                    .get("search", {})
                    .get("settle_soft_continue_search", False)
                )
                out_last = env._labeler.compute(raw)
                tilt_last = _tray_tilt_deg()
                hard_settle = float(
                    cfg.get("approach", {}).get("surface_settle_hard_tilt_deg", 32.0)
                )
                if (
                    soft_continue
                    and bool(out_last.peg_ok)
                    and bool(out_last.tray_ok)
                    and tilt_last <= hard_settle
                ):
                    surface_meta["settle_soft_continue_search"] = True
                    surface_meta["settle_stable_streak"] = stable_streak
                    print(
                        f"pci: settle soft-continue SEARCH "
                        f"(tilt={tilt_last:.1f}deg streak={stable_streak}/{stable_need})",
                        flush=True,
                    )
                else:
                    print(
                        f"pci: PRIV GATE fail — settle_unstable "
                        f"(peg∧tray streak={stable_streak}/{stable_need})",
                        flush=True,
                    )
                    feat = features_from_raw(raw)
                    return {
                        "success": False,
                        "insert_ok": False,
                        "mouth_ok": False,
                        "fail_reason": "settle_unstable",
                        "align_phase": "SURFACE",
                        "align_steps": align_steps,
                        "control_steps": settle_frames + admit_frames + k + 1,
                        "final_phase": "SETTLE_UNSTABLE",
                        "final_tip_dist_m": feat.tip_socket_dist_m,
                        "eval_only": True,
                        "hybrid_summary": hybrid.episode_summary(),
                        "traj_tail": [],
                        "phase_a_mode": "pbvs_biased_surface_press",
                        "approach_noise": surface_meta,
                        "surface_meta": {
                            **dict(surface_meta),
                            "settle_tilt_peak_deg": settle_tilt_peak,
                            "settle_stable_streak": stable_streak,
                        },
                        "compliance": compliance_tag,
                        "experiment_tag": exp_tag,
                        "priv_gate": True,
                    }

    # Privileged pre-SEARCH straighten: left_tray_follow + light right unload.
    # Gate: tilt < settle_straighten_max_tilt_deg AND peg∧tray streak ≥ N.
    if priv_ctrl:
        scfg_s = cfg.get("compliant", {}).get("search", {})
        straighten_tilt = float(scfg_s.get("settle_straighten_max_tilt_deg", 12.0))
        straighten_need = max(1, int(scfg_s.get("settle_straighten_stable_frames", 10)))
        straighten_max = max(
            straighten_need,
            int(scfg_s.get("settle_straighten_max_frames", 150)),
        )
        unload_step_s = float(scfg_s.get("settle_unload_step_m", 0.00025))
        soft_cont = bool(scfg_s.get("settle_soft_continue_search", False))
        tilt_gate = _tray_tilt_deg()
        out_gate = env._labeler.compute(raw)
        need_straighten = bool(settle_need_straighten) or (
            float(tilt_gate) > straighten_tilt
            or (not bool(out_gate.peg_ok))
            or (not bool(out_gate.tray_ok))
        )
        if need_straighten:
            open_search_tilt = float(
                scfg_s.get("settle_open_search_max_tilt_deg", 20.0)
            )
            # r11/r12: tip 已在 soft 开搜带内时勿 hold。
            # r12: 不要求 peg∧tray（ep3 入口闪烁会误进 hold 把 tip 顶翻）。
            if soft_cont and float(tilt_gate) <= open_search_tilt:
                surface_meta["settle_straighten_soft"] = True
                surface_meta["settle_straighten_skip_hold"] = True
                surface_meta["settle_straighten_tilt_deg"] = float(tilt_gate)
                surface_meta["settle_skip_peg"] = bool(out_gate.peg_ok)
                surface_meta["settle_skip_tray"] = bool(out_gate.tray_ok)
                print(
                    f"pci: priv straighten skip-hold soft-continue "
                    f"tilt={tilt_gate:.1f}deg≤{open_search_tilt:.1f} "
                    f"peg={int(bool(out_gate.peg_ok))} tray={int(bool(out_gate.tray_ok))}",
                    flush=True,
                )
            else:
                print(
                    f"pci: priv straighten before SEARCH "
                    f"tilt={tilt_gate:.1f}deg target<{straighten_tilt:.1f} "
                    f"streak={straighten_need} max={straighten_max}",
                    flush=True,
                )
                hard_settle = float(
                    cfg.get("approach", {}).get("surface_settle_hard_tilt_deg", 32.0)
                )
                # Blend left wrist toward handoff pose while unloading right (recover tip).
                handoff_arm = surface_meta.get("left_arm_handoff")
                handoff_l6 = None
                if isinstance(handoff_arm, list) and len(handoff_arm) >= 6:
                    handoff_l6 = np.asarray(handoff_arm, dtype=np.float64).reshape(-1)[0:6]
                blend = float(scfg_s.get("settle_handoff_left_blend", 0.15))
                wrist_ax_s = hole_u.copy()
                unload_cap_s = float(scfg_s.get("settle_unload_cap_m", 0.002))
                unload_acc_s = 0.0
                stable_s = 0
                ok_s = False
                for j in range(straighten_max):
                    # Force-gated unload: only retreat while |Fz| still high.
                    fz_now = abs(float(_right_fz_hole(raw))) if priv_ctrl else 0.0
                    if (
                        unload_step_s > 0.0
                        and unload_acc_s < unload_cap_s - 1e-12
                        and fz_now > 1.5
                    ):
                        step_u = min(unload_step_s, unload_cap_s - unload_acc_s)
                        settle44[0:3] = settle44[0:3] + wrist_ax_s * step_u
                        unload_acc_s += step_u
                    if handoff_l6 is not None and blend > 0.0:
                        hold_l_wrist = (1.0 - blend) * hold_l_wrist + blend * handoff_l6
                    cmd = settle44.copy()
                    cmd[22:28] = hold_l_wrist
                    cmd[28:44] = hold_l_hand
                    step_action44(gym_env, cmd, ego_recorder=ego_recorder)
                    out_j = env._labeler.compute(raw)
                    tilt_j = _tray_tilt_deg()
                    settle_tilt_peak = max(settle_tilt_peak, tilt_j)
                    if tilt_j > hard_settle:
                        # Soft: if peg∧tray still ok, stop unload and hold (don't abort yet).
                        if bool(out_j.peg_ok) and bool(out_j.tray_ok):
                            unload_acc_s = unload_cap_s  # freeze further unload
                            stable_s = 0
                            continue
                        feat = features_from_raw(raw)
                        return {
                            "success": False,
                            "insert_ok": False,
                            "mouth_ok": False,
                            "fail_reason": "settle_tray_tilt",
                            "align_phase": "SURFACE",
                            "align_steps": align_steps,
                            "control_steps": settle_frames + j + 1,
                            "final_phase": "SETTLE_STRAIGHTEN",
                            "final_tip_dist_m": feat.tip_socket_dist_m,
                            "eval_only": True,
                            "hybrid_summary": hybrid.episode_summary(),
                            "traj_tail": [],
                            "phase_a_mode": "pbvs_biased_surface_press",
                            "approach_noise": surface_meta,
                            "surface_meta": {
                                **dict(surface_meta),
                                "settle_tilt_peak_deg": settle_tilt_peak,
                                "settle_straighten_frames": j + 1,
                            },
                            "compliance": compliance_tag,
                            "experiment_tag": exp_tag,
                            "priv_gate": True,
                        }
                    if (
                        bool(out_j.peg_ok)
                        and bool(out_j.tray_ok)
                        and float(tilt_j) <= straighten_tilt
                    ):
                        stable_s += 1
                    else:
                        stable_s = 0
                    if stable_s >= straighten_need:
                        ok_s = True
                        surface_meta["settle_straighten_ok"] = True
                        surface_meta["settle_straighten_frames"] = j + 1
                        surface_meta["settle_straighten_tilt_deg"] = float(tilt_j)
                        print(
                            f"pci: priv straighten OK tilt={tilt_j:.1f}deg "
                            f"streak={stable_s}/{straighten_need} frames={j + 1}",
                            flush=True,
                        )
                        break
                if not ok_s and not surface_meta.get("settle_straighten_skip_hold"):
                    tilt_last = _tray_tilt_deg()
                    out_last = env._labeler.compute(raw)
                    # Soft continue: peg∧tray and tip ≤ open_search cap (r10: ≤20°).
                    if (
                        soft_cont
                        and bool(out_last.peg_ok)
                        and bool(out_last.tray_ok)
                        and float(tilt_last) <= open_search_tilt
                    ):
                        surface_meta["settle_straighten_soft"] = True
                        surface_meta["settle_straighten_tilt_deg"] = float(tilt_last)
                        surface_meta["settle_straighten_streak"] = stable_s
                        print(
                            f"pci: priv straighten soft-continue "
                            f"tilt={tilt_last:.1f}deg≤{open_search_tilt:.1f} "
                            f"streak={stable_s}/{straighten_need}",
                            flush=True,
                        )
                    else:
                        print(
                            f"pci: PRIV GATE fail — settle_straighten "
                            f"tilt={tilt_last:.1f}deg streak={stable_s}/{straighten_need}",
                            flush=True,
                        )
                        feat = features_from_raw(raw)
                        return {
                            "success": False,
                            "insert_ok": False,
                            "mouth_ok": False,
                            "fail_reason": "settle_straighten_fail",
                            "align_phase": "SURFACE",
                            "align_steps": align_steps,
                            "control_steps": settle_frames + straighten_max,
                            "final_phase": "SETTLE_STRAIGHTEN",
                            "final_tip_dist_m": feat.tip_socket_dist_m,
                            "eval_only": True,
                            "hybrid_summary": hybrid.episode_summary(),
                            "traj_tail": [],
                            "phase_a_mode": "pbvs_biased_surface_press",
                            "approach_noise": surface_meta,
                            "surface_meta": {
                                **dict(surface_meta),
                                "settle_tilt_peak_deg": settle_tilt_peak,
                                "settle_straighten_streak": stable_s,
                                "settle_straighten_tilt_deg": float(tilt_last),
                            },
                            "compliance": compliance_tag,
                            "experiment_tag": exp_tag,
                            "priv_gate": True,
                        }

    # Residual-gated micro-unload (FT): until |Fz-baseline| small, left fully pinned.
    fz_base = surface_meta.get("fz_baseline")
    resid_ok = float(cfg.get("approach", {}).get("surface_resid_ok_n", 0.5))
    micro_step = float(cfg.get("approach", {}).get("surface_resid_unload_step_m", 0.00035))
    micro_max = int(cfg.get("approach", {}).get("surface_resid_unload_max_steps", 40))
    micro_acc = 0.0
    micro_cap = float(cfg.get("approach", {}).get("surface_resid_unload_cap_m", 0.008))
    task_ax = -hole_u  # insert direction
    if fz_base is not None and micro_step > 0.0 and micro_max > 0:
        print(
            f"pci: residual unload target |r|<{resid_ok:.2f}N "
            f"step={micro_step*1e3:.2f}mm cap={micro_cap*1e3:.1f}mm",
            flush=True,
        )
        for j in range(micro_max):
            resid = float(_right_fz_hole(raw)) - float(fz_base)
            if abs(resid) <= resid_ok or micro_acc >= micro_cap - 1e-9:
                print(
                    f"pci: residual unload done j={j} r={resid:+.2f}N "
                    f"acc={micro_acc*1e3:.1f}mm",
                    flush=True,
                )
                surface_meta["resid_after_unload"] = resid
                surface_meta["resid_unload_acc_m"] = micro_acc
                break
            settle44[0:3] = settle44[0:3] - task_ax * micro_step
            micro_acc += micro_step
            cmd = settle44.copy()
            cmd[22:28] = hold_l_wrist
            cmd[28:44] = hold_l_hand
            step_action44(gym_env, cmd, ego_recorder=ego_recorder)
            outcome_s = env._labeler.compute(raw)
            tilt = _tray_tilt_deg()
            settle_tilt_peak = max(settle_tilt_peak, tilt)
            if not outcome_s.tray_ok:
                print(f"pci: PRIV GATE fail — tray_ok lost in resid unload j={j}", flush=True)
                feat = features_from_raw(raw)
                return {
                    "success": False,
                    "insert_ok": False,
                    "fail_reason": "settle_tray_lost",
                    "align_phase": "SURFACE",
                    "align_steps": align_steps,
                    "control_steps": settle_frames + j + 1,
                    "final_phase": "RESID_UNLOAD",
                    "final_tip_dist_m": feat.tip_socket_dist_m,
                    "eval_only": True,
                    "hybrid_summary": hybrid.episode_summary(),
                    "traj_tail": [],
                    "phase_a_mode": "pbvs_biased_surface_press",
                    "approach_noise": surface_meta,
                    "surface_meta": {
                        **dict(surface_meta),
                        "settle_tilt_peak_deg": settle_tilt_peak,
                        "resid_unload_acc_m": micro_acc,
                    },
                    "compliance": "force_tactile",
                    "priv_gate": True,
                }
            if tilt > max_tilt_deg:
                print(
                    f"pci: PRIV GATE fail — tray tilt {tilt:.1f}deg in resid unload",
                    flush=True,
                )
                feat = features_from_raw(raw)
                return {
                    "success": False,
                    "insert_ok": False,
                    "fail_reason": "settle_tray_tilt",
                    "align_phase": "SURFACE",
                    "align_steps": align_steps,
                    "control_steps": settle_frames + j + 1,
                    "final_phase": "RESID_UNLOAD",
                    "final_tip_dist_m": feat.tip_socket_dist_m,
                    "eval_only": True,
                    "hybrid_summary": hybrid.episode_summary(),
                    "traj_tail": [],
                    "phase_a_mode": "pbvs_biased_surface_press",
                    "approach_noise": surface_meta,
                    "surface_meta": {
                        **dict(surface_meta),
                        "settle_tilt_peak_deg": settle_tilt_peak,
                        "resid_unload_acc_m": micro_acc,
                    },
                    "compliance": "force_tactile",
                    "priv_gate": True,
                }
        else:
            resid = float(_right_fz_hole(raw)) - float(fz_base)
            surface_meta["resid_after_unload"] = resid
            surface_meta["resid_unload_acc_m"] = micro_acc
            print(
                f"pci: residual unload maxed r={resid:+.2f}N acc={micro_acc*1e3:.1f}mm",
                flush=True,
            )

    # Post-settle: right from sites (after unload); left keep settle freeze pose.
    action44 = actual_action44_from_sites(raw)
    action44[22:28] = hold_l_wrist
    action44[28:44] = hold_l_hand
    if priv_ctrl:
        feat_b = features_from_raw(raw)
        task_frame = TaskFrame.from_hole_axis(
            action44[0:3],
            feat_b.hole_axis,
            peg_axis_world=feat_b.peg_axis,
        )
    else:
        task_frame = _task_frame_from_wrist(raw, origin_world=action44[0:3])
    wrench = read_wrist_wrench_world(raw)
    finger12, left_finger12 = read_dual_finger_force24(raw, force_labeler)
    hold_fingers = action44[6:22].copy()
    r_scale = float(cfg.get("approach", {}).get("surface_right_grasp_scale", 1.0))
    if r_scale > 1.0:
        hold_fingers = np.clip(
            np.maximum(hold_fingers, hold_fingers * r_scale), -1.5, 1.5
        )
        print(f"pci: right grasp tighten scale={r_scale:.2f}", flush=True)
    hold_left_fingers = hold_l_hand.copy()
    hold_left_wrist28 = hold_l_wrist.copy()
    hold_right_wrist6 = action44[0:6].copy()
    # Phase-B re-tighten left fingers (disabled when handoff-locked scale≤1).
    l_scale_b = float(
        cfg.get("compliant", {}).get("release", {}).get("left_grasp_scale", 1.0)
    )
    l_scale_b = max(
        l_scale_b,
        float(cfg.get("approach", {}).get("phase_b_left_grasp_scale", 1.55)),
    )
    if bool(surface_meta.get("handoff_left_locked", False)):
        l_scale_b = 1.0
    if l_scale_b > 1.0:
        hold_left_fingers = np.clip(
            np.maximum(hold_left_fingers, hold_left_fingers * l_scale_b), -1.5, 1.5
        )
        print(f"pci: left grasp tighten scale={l_scale_b:.2f}", flush=True)
    use_priv_grasp = bool(
        cfg.get("compliant", {}).get("priv_grasp_opt", {}).get("enable", False)
    )
    left_admit = LeftWristAdmitController(_left_wrist_admit_config(cfg))
    left_admit.reset(wrench[1])
    left_follow = LeftTrayFollowController(_left_tray_follow_config(cfg))
    # Prefer surface-deliver latch (pose-hold theory); else current geom.
    latch_snap = surface_meta.get("latch_priv_geom")
    if isinstance(latch_snap, PrivGraspGeom):
        priv_geom_latch = copy_priv_grasp_geom(latch_snap)
    else:
        priv_geom_latch = read_priv_grasp_geom(raw)
        surface_meta = dict(surface_meta)
        surface_meta["latch_priv_geom"] = copy_priv_grasp_geom(priv_geom_latch)
        surface_meta["latch_priv_geom_ok"] = True
    left_follow.reset(priv_geom_latch)
    # Privileged geom only if research QP enabled (off in force-tactile mode).
    priv_geom0 = priv_geom_latch if use_priv_grasp else None
    theory_path = _theory_pose_qp(cfg)
    print(
        f"pci: settle OK tilt_peak={settle_tilt_peak:.2f}deg → "
        f"{compliance_tag} search (priv_ctrl={priv_ctrl}, priv_grasp={use_priv_grasp}, "
        f"left_admit={left_admit.config.enable}, "
        f"left_tray_follow={left_follow.config.enable}, "
        f"theory_pose_qp={theory_path})",
        flush=True,
    )
    # Demo / audit: stop at soft-land (no SEARCH).
    if bool(cfg.get("approach", {}).get("stop_after_surface", False)) and not handoff_insert:
        feat_s = features_from_raw(raw)
        out_s = env._labeler.compute(raw)
        max_tilt_ok = float(
            cfg.get("approach", {}).get("surface_settle_max_tray_tilt_deg", 8.0)
        )
        surface_ok = bool(out_s.tray_ok) and float(settle_tilt_peak) <= max_tilt_ok
        clean = {
            k: v
            for k, v in dict(surface_meta).items()
            if not str(k).startswith("_")
            and not hasattr(v, "dtype")
            and not isinstance(v, PrivGraspGeom)
        }
        clean["stop_after_surface"] = True
        clean["settle_tilt_peak_deg"] = float(settle_tilt_peak)
        print(
            f"pci: stop_after_surface — tray_ok={int(bool(out_s.tray_ok))} "
            f"settle_tilt={settle_tilt_peak:.2f}deg "
            f"ok={int(surface_ok)}",
            flush=True,
        )
        return {
            "success": bool(surface_ok),
            "mouth_ok": False,
            "insert_ok": False,
            "tray_ok": bool(out_s.tray_ok),
            "peg_ok": bool(out_s.peg_ok),
            "fail_reason": "" if surface_ok else "surface_tilt_or_tray",
            "align_phase": "SURFACE",
            "align_steps": align_steps,
            "control_steps": 0,
            "final_phase": "SURFACE_DONE",
            "final_tip_dist_m": feat_s.tip_socket_dist_m,
            "eval_only": True,
            "hybrid_summary": hybrid.episode_summary(),
            "traj_tail": [],
            "traj": [],
            "phase_a_mode": "pbvs_biased_surface_press",
            "approach_noise": clean,
            "surface_meta": clean,
            "surface_reason": "stop_after_surface",
            "compliance": compliance_tag,
            "priv_gate": True,
        }
    pipeline.begin_compliant(
        task_frame,
        wrench[0],
        finger12,
        hold_fingers,
        action44[0:3],
        already_on_surface=True,
        hold_left_hand16=hold_left_fingers,
        priv_geom=priv_geom0,
        left_finger_force12=left_finger12,
    )
    surface_meta = dict(surface_meta)
    # Always re-arm Fz baseline in wrist/tool frame at SEARCH entry.
    # Phase-A hole-axis baseline is incomparable → fake |resid|~12N → permanent unload/slam.
    surface_meta["fz_baseline"] = float(_right_fz_sensor(raw, task_frame))
    surface_meta["fz_baseline_frame"] = "wrist_tool"
    print(
        f"pci: Fz baseline re-armed (wrist_tool)={surface_meta['fz_baseline']:+.2f}N "
        f"priv_ctrl={bool(priv_ctrl)}",
        flush=True,
    )
    surface_meta["settle_seconds"] = settle_s
    surface_meta["settle_frames"] = settle_frames
    surface_meta["settle_tilt_peak_deg"] = settle_tilt_peak
    if isinstance(priv_geom_latch, PrivGraspGeom):
        surface_meta["rel_rot_at_search_start_rad"] = float(
            _rel_rot_err_rad(priv_geom_latch, read_priv_grasp_geom(raw))
        )
    surface_meta["control_mode"] = (
        "sensor_control_priv_monitor"
        if not priv_ctrl
        else (
            "force_tactile_left_tray_follow"
            if left_follow.config.enable
            else (
                "force_tactile_left_admit"
                if left_admit.config.enable
                else "force_tactile_left_pin"
            )
        )
    )
    surface_meta["priv_gate"] = True
    surface_meta["priv_in_control"] = bool(priv_ctrl)
    surface_meta["priv_grasp_in_loop"] = bool(use_priv_grasp)
    surface_meta["left_wrist_admit"] = bool(left_admit.config.enable)
    surface_meta["left_tray_follow"] = bool(left_follow.config.enable)
    surface_meta["left_grasp_scale_b"] = float(l_scale_b)
    surface_meta["experiment_tag"] = exp_tag

    traj: list[dict[str, Any]] = []
    control_steps = 0
    insert_ok = False
    fail_reason = "max_control_steps"
    final_phase = PipelinePhase.COMPLIANT_SEARCH.name
    search_cfg = cfg.get("compliant", {}).get("search", {})
    abort_tray = bool(search_cfg.get("abort_on_tray_lost", True))
    max_search_tilt = float(search_cfg.get("search_max_tray_tilt_deg", 10.0))
    use_hop = bool(search_cfg.get("discrete_hop", False))
    hop_period = max(3, int(search_cfg.get("hop_period_steps", 9)))
    hop_lift_step = float(search_cfg.get("hop_lift_step_m", 0.0020))
    hop_press_step = float(search_cfg.get("hop_press_step_m", 0.0004))
    hop_clear_m = float(search_cfg.get("hop_clear_m", 0.0025))
    tip_stuck_window = int(search_cfg.get("tip_stuck_window", 45))
    tip_stuck_cmd = float(search_cfg.get("tip_stuck_cmd_m", 0.012))
    tip_stuck_move = float(search_cfg.get("tip_stuck_move_m", 0.0004))
    stuck_cmd_acc = 0.0
    stuck_steps = 0
    resync_stuck = 0
    # CRITICAL: accumulate on mocap. Re-syncing to site every step cancels lift/spiral.
    hold_right_wrist6 = actual_action44_from_sites(raw)[0:6].copy()
    along0 = float(features_from_raw(raw).along_m)
    tip_clear = False
    lift_budget_m = 0.0
    max_lift_budget = float(search_cfg.get("hop_lift_cmd_budget_m", 0.020))
    left_admit_cfg_enable = bool(left_admit.config.enable)
    # Preserve cfg enable; SEARCH applies search_left_admit_scale (may be 0).
    left_admit.config.enable = False
    insert_tilt_log_deg = float(search_cfg.get("insert_tilt_warn_deg", 12.0))
    from scipy.spatial.transform import Rotation as R

    tray_R_search0 = read_priv_grasp_geom(raw).tray_rot.copy()
    mouth_ok = False
    mouth_hold_streak = 0
    peg_lost_streak = 0
    peg_grace = int(search_cfg.get("peg_ok_grace_frames", 3))

    def _search_tray_tilt_deg() -> float:
        R_now = read_priv_grasp_geom(raw).tray_rot
        return float(
            np.linalg.norm(R.from_matrix(tray_R_search0.T @ R_now).as_rotvec())
            * 180.0
            / np.pi
        )

    print(
        f"pci: search mode hop={use_hop} clear={hop_clear_m*1e3:.1f}mm "
        f"mocap_accumulate=True (no per-step site reset) "
        f"priv_seek={float(search_cfg.get('priv_seek_step_m', 0.0))*1e3:.2f}mm "
        f"left_share={float(search_cfg.get('near_hole_left_share', 0.0)):.2f} "
        f"tip_soft={float(search_cfg.get('search_tip_soft_deg', 999)):.1f} "
        f"mouth_stop={bool(search_cfg.get('mouth_hold_stop', False))} "
        f"insert_left_admit={left_admit_cfg_enable}",
        flush=True,
    )

    # Post-settle QP-only squeeze: freeze wrists, tighten grasp before spiral (ep6).
    pre_qp = int(search_cfg.get("pre_search_qp_frames", 0))
    pre_qp_rel_peak = 0.0
    settle_soft_rel = float(search_cfg.get("settle_max_rel_rot_rad", 0.12))
    settle_abort_rel = float(search_cfg.get("settle_rel_rot_abort_rad", 0.20))
    settle_abort_tilt = float(search_cfg.get("settle_rel_rot_abort_tilt_deg", 15.0))
    recovery_frames = int(search_cfg.get("pre_search_rel_recovery_frames", 60))
    recovery_follow_scale = float(search_cfg.get("pre_search_rel_recovery_follow_scale", 0.30))
    pre_qp_left_scale = float(search_cfg.get("settle_admit_left_scale", 0.0))

    def _run_pre_search_qp(frames: int, *, tag: str) -> float:
        nonlocal pre_qp_rel_peak
        if not (use_priv_grasp and priv_ctrl and frames > 0):
            return pre_qp_rel_peak
        print(f"pci: {tag} QP squeeze {frames} frames (wrists frozen)", flush=True)
        pre_admit: LeftWristAdmitController | None = None
        if pre_qp_left_scale > 1e-9:
            pre_admit = LeftWristAdmitController(_left_wrist_admit_config(cfg))
            pre_admit.reset(read_wrist_wrench_world(raw)[1])
            pre_admit.config.enable = True
        wrist_ax_pre = _wrist_approach_axis(raw)
        for _qi in range(frames):
            wrench_q = read_wrist_wrench_world(raw)
            finger12_q, left_finger12_q = read_dual_finger_force24(raw, force_labeler)
            geom_q = read_priv_grasp_geom(raw)
            po = pipeline.priv_grasp.step(geom_q, finger12_q, left_finger12_q)
            pre_qp_rel_peak = max(pre_qp_rel_peak, float(po.rel_rot_err_rad))
            cmd = np.zeros(44, dtype=np.float64)
            cmd[0:6] = hold_right_wrist6
            lw = hold_left_wrist28.copy()
            if pre_admit is not None:
                d_left = pre_admit.step(wrench_q[1], approach_axis=wrist_ax_pre) * pre_qp_left_scale
                lw[0:3] = lw[0:3] + d_left
            cmd[22:28] = lw
            cmd[6:22] = hold_fingers + po.delta_right_hand16
            left_q = hold_left_fingers + po.delta_left_hand16
            cmd[28:44] = np.maximum(left_q, hold_left_fingers * 0.98)
            step_action44(gym_env, cmd, ego_recorder=ego_recorder)
            out_q = env._labeler.compute(raw)
            if bool(out_q.peg_ok) and bool(out_q.tray_ok):
                surface_meta["_pre_search_qp_stable"] = (
                    int(surface_meta.get("_pre_search_qp_stable", 0)) + 1
                )
            else:
                surface_meta["_pre_search_qp_stable"] = 0
        surface_meta["pre_search_qp_frames"] = int(surface_meta.get("pre_search_qp_frames", 0)) + frames
        print(
            f"pci: {tag} QP done rel_peak={pre_qp_rel_peak:.3f}rad "
            f"peg∧tray streak={int(surface_meta.get('_pre_search_qp_stable', 0))}",
            flush=True,
        )
        return pre_qp_rel_peak

    def _run_rel_rot_recovery(frames: int) -> float:
        """r14: QP + left_admit + light left_tray_follow to pull rel_rot down."""
        nonlocal pre_qp_rel_peak
        if not (use_priv_grasp and priv_ctrl and frames > 0):
            return pre_qp_rel_peak
        print(
            f"pci: rel_rot recovery {frames} frames "
            f"(QP+admit+follow scale={recovery_follow_scale:.2f})",
            flush=True,
        )
        rec_admit: LeftWristAdmitController | None = None
        if pre_qp_left_scale > 1e-9:
            rec_admit = LeftWristAdmitController(_left_wrist_admit_config(cfg))
            rec_admit.reset(read_wrist_wrench_world(raw)[1])
            rec_admit.config.enable = True
        rec_follow = LeftTrayFollowController(_left_tray_follow_config(cfg))
        rec_follow.reset(priv_geom_latch)
        wrist_ax_pre = _wrist_approach_axis(raw)
        used = 0
        for used in range(1, frames + 1):
            wrench_q = read_wrist_wrench_world(raw)
            finger12_q, left_finger12_q = read_dual_finger_force24(raw, force_labeler)
            geom_q = read_priv_grasp_geom(raw)
            po = pipeline.priv_grasp.step(geom_q, finger12_q, left_finger12_q)
            pre_qp_rel_peak = max(pre_qp_rel_peak, float(po.rel_rot_err_rad))
            cmd = np.zeros(44, dtype=np.float64)
            cmd[0:6] = hold_right_wrist6
            lw = hold_left_wrist28.copy()
            if rec_admit is not None:
                d_left = rec_admit.step(wrench_q[1], approach_axis=wrist_ax_pre) * pre_qp_left_scale
                lw[0:3] = lw[0:3] + d_left
            if rec_follow.config.enable and recovery_follow_scale > 1e-9:
                lw_before = lw[0:3].copy()
                lw, _ = rec_follow.step_hold6(lw, geom_q)
                d_follow = lw[0:3] - lw_before
                lw[0:3] = lw_before + d_follow * recovery_follow_scale
            hold_left_wrist28[0:3] = lw[0:3]
            cmd[22:28] = lw
            cmd[6:22] = hold_fingers + po.delta_right_hand16
            left_q = hold_left_fingers + po.delta_left_hand16
            cmd[28:44] = np.maximum(left_q, hold_left_fingers * 0.98)
            step_action44(gym_env, cmd, ego_recorder=ego_recorder)
            if pre_qp_rel_peak <= settle_soft_rel:
                break
        surface_meta["pre_search_rel_recovery_used"] = used
        print(
            f"pci: rel_rot recovery done used={used}/{frames} rel_peak={pre_qp_rel_peak:.3f}rad",
            flush=True,
        )
        return pre_qp_rel_peak

    if use_priv_grasp and pre_qp > 0 and priv_ctrl and _phase_a_qp_enable(cfg):
        rel_contact_ok = float(surface_meta.get("rel_rot_at_contact_rad", 999.0))
        pre_qp_run = pre_qp
        skip_rel_thresh = max(settle_soft_rel, 0.20)
        if rel_contact_ok <= skip_rel_thresh:
            pre_qp_run = 0
            print(
                f"pci: skip pre-search QP — rel_at_contact={rel_contact_ok:.3f}rad "
                f"≤ {skip_rel_thresh:.3f}",
                flush=True,
            )
            pre_qp_rel_peak = rel_contact_ok
            surface_meta["pre_search_qp_skipped"] = True
        if pre_qp_run > 0:
            _run_pre_search_qp(pre_qp_run, tag="pre-search")
        surface_meta["pre_search_rel_rot_peak_rad"] = pre_qp_rel_peak
        surface_meta["rel_rot_at_search_start_rad"] = pre_qp_rel_peak
        if pre_qp_rel_peak > settle_soft_rel and recovery_frames > 0 and pre_qp_run > 0:
            _run_rel_rot_recovery(recovery_frames)
            surface_meta["pre_search_rel_rot_peak_rad"] = pre_qp_rel_peak
            surface_meta["rel_rot_at_search_start_rad"] = pre_qp_rel_peak
        tilt_pre_search = _search_tray_tilt_deg()
        if pre_qp_rel_peak > settle_soft_rel:
            surface_meta["settle_rel_rot_soft_continue"] = True
            print(
                f"pci: settle rel_rot soft-continue SEARCH "
                f"rel={pre_qp_rel_peak:.3f}rad tilt={tilt_pre_search:.1f}deg",
                flush=True,
            )

    for _ in range(max_ctrl):
        outcome = env._labeler.compute(raw)
        if outcome.insert_ok:
            insert_ok = True
        in_search_gate = final_phase == PipelinePhase.COMPLIANT_SEARCH.name
        # Hard PRIV aborts only during SEARCH; INSERT keeps going (tilt logged).
        if abort_tray and in_search_gate and not outcome.tray_ok:
            # Near-hole seating: soft-continue (tray flicker); far: hard abort.
            # Early SEARCH grace: allow left_tray_follow / QP to recover tip (ep6).
            tray_grace = int(search_cfg.get("search_tray_lost_grace_steps", 0))
            lat_gate = float(search_cfg.get("priv_far_lat_m", 0.010))
            feat_gate = features_from_raw(raw)
            if control_steps < tray_grace:
                if control_steps % 20 == 0:
                    print(
                        f"pci: SEARCH tray_ok lost early grace "
                        f"step={control_steps}/{tray_grace} — continue",
                        flush=True,
                    )
            elif float(feat_gate.lateral_m) > lat_gate:
                print("pci: PRIV GATE abort — tray_ok lost during search", flush=True)
                fail_reason = "search_tray_lost"
                break
            elif control_steps % 20 == 0:
                print("pci: SEARCH tray_ok flicker near-hole — continue", flush=True)
        if abort_tray and in_search_gate and not outcome.peg_ok:
            peg_lost_streak += 1
            if peg_lost_streak > peg_grace:
                print("pci: PRIV GATE abort — peg_ok lost during search", flush=True)
                fail_reason = "search_peg_lost"
                break
        else:
            peg_lost_streak = 0
        tilt_now = _search_tray_tilt_deg()
        # Layered tip: soft freeze/continue below hard threshold; only extreme tip aborts.
        search_hard_tilt = float(
            search_cfg.get("search_hard_tray_tilt_deg", max(max_search_tilt, 28.0))
        )
        if abort_tray and in_search_gate and tilt_now > search_hard_tilt:
            # Sensor mode: hard tilt always aborts (no soft-continue via mouth_hold_stop).
            # Priv mouth campaign may soft-continue hard tip for diagnostic seating.
            if (not priv_ctrl) or (not bool(search_cfg.get("mouth_hold_stop", False))):
                print(
                    f"pci: PRIV GATE abort — tray tilt {tilt_now:.1f}deg > {search_hard_tilt:.1f} "
                    "(hard layer)",
                    flush=True,
                )
                fail_reason = "search_tray_tilt"
                break
            if control_steps % 40 == 0:
                print(
                    f"pci: SEARCH tip hard {tilt_now:.1f}deg — soft continue (mouth)",
                    flush=True,
                )
        if (
            abort_tray
            and in_search_gate
            and tilt_now > max_search_tilt
            and tilt_now <= search_hard_tilt
        ):
            # Soft layer: priv may freeze/unload below; sensor ignores soft continue.
            if priv_ctrl and control_steps % 30 == 0:
                print(
                    f"pci: SEARCH tip soft {tilt_now:.1f}deg — freeze/unload continue",
                    flush=True,
                )
        if (not in_search_gate) and tilt_now > insert_tilt_log_deg:
            if control_steps % 20 == 0:
                print(
                    f"pci: INSERT tilt warn {tilt_now:.1f}deg (soft compliance)",
                    flush=True,
                )
        insert_max_tilt = float(search_cfg.get("insert_max_tray_tilt_deg", 25.0))
        insert_hard_tilt = float(
            search_cfg.get("insert_hard_tray_tilt_deg", max(insert_max_tilt + 4.0, 40.0))
        )
        if (not in_search_gate) and tilt_now > insert_max_tilt:
            # r10: tip 高先卸力/降压，勿立刻硬 abort；仅超 hard 才 insert_tray_tilt。
            if final_phase != PipelinePhase.RELEASE.name:
                surface_meta["_insert_tilt_unload"] = True
                if tilt_now > insert_hard_tilt:
                    print(
                        f"pci: INSERT abort — tray tilt {tilt_now:.1f}deg > "
                        f"{insert_hard_tilt:.1f} (hard) "
                        f"tray_ok={int(bool(outcome.tray_ok))}",
                        flush=True,
                    )
                    fail_reason = "insert_tray_tilt"
                    break
                if control_steps % 15 == 0:
                    print(
                        f"pci: INSERT tip high {tilt_now:.1f}deg — unload/reduce press",
                        flush=True,
                    )
        elif (not in_search_gate) and tilt_now <= insert_tilt_log_deg:
            surface_meta.pop("_insert_tilt_unload", None)

        site44 = actual_action44_from_sites(raw)
        action44 = current_action44(raw)
        wrench = read_wrist_wrench_world(raw)
        finger12, left_finger12 = read_dual_finger_force24(raw, force_labeler)
        feat_priv = features_from_raw(raw)
        tip_before = np.asarray(feat_priv.tip_pos, dtype=np.float64).reshape(3).copy()
        tip = tip_before
        sock = np.asarray(feat_priv.socket_pos, dtype=np.float64).reshape(3)
        hole_u = feat_priv.hole_axis / (np.linalg.norm(feat_priv.hole_axis) + 1e-12)
        lat_vec = tip - sock
        lat_vec = lat_vec - hole_u * float(np.dot(lat_vec, hole_u))
        along_now = float(feat_priv.along_m)
        if (not in_search_gate) and float(feat_priv.lateral_m) > float(
            search_cfg.get("priv_far_lat_m", 0.010)
        ) * 2.5:
            print(
                f"pci: INSERT abort — lat={feat_priv.lateral_m*1e3:.1f}mm escaped mouth",
                flush=True,
            )
            fail_reason = "insert_lat_escape"
            break
        # Tip cleared surface only if along rose AND already near hole (avoid false clear).
        # Privileged hop clear only — sensor hop is off / must not gate on tip+along.
        enter_lat_clear = float(search_cfg.get("priv_enter_lat_m", 0.0045))
        if (
            priv_ctrl
            and (along_now - along0) >= hop_clear_m
            and float(feat_priv.lateral_m) <= enter_lat_clear * 3.0
        ):
            tip_clear = True
        priv_geom = read_priv_grasp_geom(raw) if use_priv_grasp else None
        # Early RELEASE: once tip is deep+centered, ignore tray tilt (was stuck seated @19°).
        early_along = float(
            cfg.get("compliant", {}).get("insert", {}).get("early_release_along_m", 0.018)
        )
        if (
            (not in_search_gate)
            and along_now < early_along
            and float(feat_priv.lateral_m) < 0.0045
        ):
            surface_meta["_near_seat_frames"] = int(surface_meta.get("_near_seat_frames", 0)) + 1
        else:
            surface_meta["_near_seat_frames"] = 0
        seat_wait = int(
            cfg.get("compliant", {}).get("insert", {}).get("early_release_wait_frames", 12)
        )
        deep_seat = along_now < 0.010 and float(feat_priv.lateral_m) < 0.0045
        early_release = bool(priv_ctrl) and int(surface_meta.get("_near_seat_frames", 0)) >= seat_wait and (
            tilt_now < 22.0 or deep_seat
        )
        if early_release and not surface_meta.get("_early_release_logged"):
            surface_meta["_early_release_logged"] = True
            print(
                f"pci: early RELEASE along={along_now*1e3:.1f}mm after "
                f"{seat_wait} seated frames (spring bled)",
                flush=True,
            )
        # search.step 进孔：仅 priv_assist 可读 along/vec（assist seek）；
        # priv_lat 在 priv_ctrl 下仍传入作监考拒识（reject_hole_if_priv_lat_m）；
        # recovery-gated seek 额外传 lat_vec（priv_recovery_lat_m 触发，非全程开挂）。
        priv_assist = bool(search_cfg.get("priv_assist", False))
        search_priv = bool(priv_ctrl) and priv_assist
        recovery_gate_on = (
            bool(priv_ctrl)
            and float(search_cfg.get("priv_recovery_lat_m", 0.0)) > 0.0
            and float(search_cfg.get("priv_seek_step_m", 0.0)) > 0.0
        )
        priv_lat_for_search = (
            float(feat_priv.lateral_m) if (search_priv or priv_ctrl) else None
        )
        pr = pipeline.step(
            wrench[0],
            hold_right_wrist6[0:3],
            finger12,
            insert_ok_eval=bool(insert_ok) or early_release,
            dt=_sim_dt(cfg),
            priv_lat_m=priv_lat_for_search,
            priv_along_m=along_now if search_priv else None,
            priv_lat_vec=lat_vec if (search_priv or recovery_gate_on) else None,
            priv_geom=priv_geom,
            left_finger_force12=left_finger12,
        )
        final_phase = pr.phase.name

        delta = np.asarray(pr.delta_xyz, dtype=np.float64).reshape(3)
        ax = task_frame.approach_axis
        in_search = final_phase == PipelinePhase.COMPLIANT_SEARCH.name
        # Privileged diagnostic only: refresh task frame to hole axis on seat enter.
        if (
            priv_ctrl
            and not in_search
            and not surface_meta.get("_insert_frame_refreshed")
            and pr.reason in ("priv_along_seat", "hole_detected")
        ):
            feat_ref = features_from_raw(raw)
            task_frame = TaskFrame.from_hole_axis(
                hold_right_wrist6[0:3],
                feat_ref.hole_axis,
                peg_axis_world=feat_ref.peg_axis,
            )
            pipeline._frame = task_frame  # noqa: SLF001
            pipeline.insert.reset(task_frame, wrench[0], hold_right_wrist6[0:3])
            surface_meta["_insert_frame_refreshed"] = True
            ax = task_frame.approach_axis
            print("pci: INSERT frame refreshed to current hole axis", flush=True)
        if (
            not in_search
            and surface_meta.get("_insert_phase_entered") is None
        ):
            surface_meta["_insert_phase_entered"] = True
            surface_meta["_last_tip_along_delta"] = 0.0
            surface_meta["_tip_stall_frames"] = 0
            # Re-sync left wrist pose at INSERT entry (SEARCH may have drifted).
            hold_left_wrist28 = actual_action44_from_sites(raw)[22:28].copy()
            # Theory path: keep surface latch; do not re-lock relative pose at INSERT.
            if priv_ctrl and left_follow.config.enable and not _theory_pose_qp(cfg):
                left_follow.reset(read_priv_grasp_geom(raw))
                print("pci: left tray-follow re-latch at INSERT entry", flush=True)
            elif priv_ctrl and left_follow.config.enable and _theory_pose_qp(cfg):
                print("pci: theory_pose_qp — keep surface latch (no INSERT re-lock)", flush=True)
            left_admit.reset(wrench[1])
        fz_base = surface_meta.get("fz_baseline")
        abs_fz_now = abs(float(_right_fz_sensor(raw, task_frame)))
        xy_blocked = False
        hop_mode = "insert"
        frame_left_lat_cmd = np.zeros(3, dtype=np.float64)
        site_before = site44[0:3].copy()
        if in_search:
            # Theory soft-surface spiral: command from actual site each step.
            # Mocap accumulate drifts into air (r→50mm while tip lat stuck) and yanks peg.
            site_rel = bool(search_cfg.get("search_site_relative_spiral", False)) or _theory_pose_qp(
                cfg
            )
            if site_rel:
                action44[0:6] = actual_action44_from_sites(raw)[0:6]
                hold_right_wrist6 = action44[0:6].copy()
            else:
                # Legacy: accumulate on held mocap wrist.
                action44[0:6] = hold_right_wrist6
            planar = delta - ax * float(np.dot(delta, ax))
            axial = ax * float(np.dot(delta, ax))
            unload_acc = float(surface_meta.get("_search_unload_acc_m", 0.0))
            unload_cap = float(search_cfg.get("search_unload_cap_m", 0.003))
            unload_step = float(search_cfg.get("search_unload_step_m", 0.0004))
            resid_gate = float(search_cfg.get("search_unload_if_abs_fz_n", 0.55))
            resid_now_pre = None
            if fz_base is not None:
                resid_now_pre = float(_right_fz_sensor(raw, task_frame)) - float(fz_base)
            resid_blocking = resid_now_pre is not None and abs(float(resid_now_pre)) > resid_gate
            # Prefer search_hold_xy_steps (guard was accidentally preferred and froze spiral).
            hold_xy_steps = int(
                search_cfg.get(
                    "search_hold_xy_steps",
                    search_cfg.get("search_guard_steps", 10),
                )
            )
            start_stable_need = int(search_cfg.get("search_start_stable_frames", 0))
            start_max_hold = int(search_cfg.get("search_start_max_hold_steps", 0))
            if start_stable_need > 0:
                if bool(outcome.peg_ok) and bool(outcome.tray_ok):
                    surface_meta["_search_start_stable"] = (
                        int(surface_meta.get("_search_start_stable", 0)) + 1
                    )
                else:
                    surface_meta["_search_start_stable"] = 0
            start_stable_ok = (
                start_stable_need <= 0
                or int(surface_meta.get("_search_start_stable", 0)) >= start_stable_need
                or (start_max_hold > 0 and control_steps >= start_max_hold)
            )
            # ep6: if tray already tipping during hold, escape to spiral (don't wait).
            tip_soft_hold = float(search_cfg.get("search_tip_soft_deg", 22.0))
            if (not start_stable_ok) and float(tilt_now) > tip_soft_hold * 0.5:
                start_stable_ok = True
                surface_meta["_search_start_escape_tilt"] = float(tilt_now)
            enter_lat = float(search_cfg.get("priv_enter_lat_m", 0.0045))
            # near_hole recenter is privileged tip→hole seek — sensor mode never takes it.
            # Recovery-gated mode: seek only inside search.py when lat>priv_recovery_lat_m;
            # do NOT enable always-on near_recenter / tip_soft 朝孔拽.
            seek_m_cfg = float(search_cfg.get("priv_seek_step_m", 0.0))
            recovery_lat_m = float(search_cfg.get("priv_recovery_lat_m", 0.0))
            if recovery_lat_m > 0.0 and seek_m_cfg > 0.0:
                seek_m = 0.0
            else:
                seek_m = seek_m_cfg
            # Optional sim_runner-side recovery pull when search controller rebias active.
            sc_rec = getattr(pipeline, "search", None)
            recovery_active = bool(
                recovery_lat_m > 0.0
                and seek_m_cfg > 0.0
                and sc_rec is not None
                and bool(getattr(sc_rec, "_recovery_active", False))
            )
            near_hole = (
                bool(priv_ctrl)
                and seek_m > 0.0
                and float(feat_priv.lateral_m) <= enter_lat * 1.5
            )
            # Near hole: prefer axial seat over resid unload (unload caused search_timeout).
            if (
                resid_blocking
                and unload_acc < unload_cap
                and unload_step > 0.0
                and not near_hole
            ):
                step_u = min(unload_step, unload_cap - unload_acc)
                delta = -ax * step_u
                surface_meta["_search_unload_acc_m"] = unload_acc + step_u
                xy_blocked = True
                hop_mode = "unload"
            elif control_steps < hold_xy_steps or not start_stable_ok:
                delta = np.zeros(3, dtype=np.float64)
                xy_blocked = True
                hop_mode = "hold"
            elif use_hop and not tip_clear:
                if lift_budget_m >= max_lift_budget:
                    fail_reason = "search_lift_fail"
                    print(
                        f"pci: tip lift fail — alongΔ={(along_now-along0)*1e3:.2f}mm "
                        f"< clear {hop_clear_m*1e3:.1f}mm after cmd_lift={lift_budget_m*1e3:.1f}mm",
                        flush=True,
                    )
                    break
                delta = -ax * hop_lift_step
                lift_budget_m += hop_lift_step
                xy_blocked = True
                hop_mode = "hop_lift"
            elif use_hop:
                phase = control_steps % hop_period
                third = max(1, hop_period // 3)
                if phase < third:
                    delta = -ax * (hop_lift_step * 0.5)
                    hop_mode = "hop_lift"
                elif phase < 2 * third:
                    delta = planar.copy()
                    hop_mode = "hop_slide"
                else:
                    delta = planar * 0.35 + ax * hop_press_step
                    hop_mode = "hop_press"
            elif near_hole:
                # On rim (along still above seat): recenter first, don't jam-press.
                press_keep = float(search_cfg.get("near_hole_press_m", 0.00050))
                f_hi_nh = float(search_cfg.get("contact_f_max_n", 1.2))
                if resid_now_pre is not None and abs(float(resid_now_pre)) >= f_hi_nh:
                    press_keep = 0.0  # residual heavy → no axial floor
                seat_along = float(search_cfg.get("priv_enter_along_max_m", 0.100))
                lat_now = float(feat_priv.lateral_m)
                on_rim = along_now > seat_along
                left_share = float(search_cfg.get("near_hole_left_share", 0.0))
                # Only engage left socket pull when tray upright enough.
                tip_soft = float(search_cfg.get("search_tip_soft_deg", 14.0))
                if tilt_now > tip_soft:
                    left_share = 0.0
                left_lat_cmd = np.zeros(3, dtype=np.float64)

                def _dual_seek_toward_hole(step: float) -> tuple[np.ndarray, np.ndarray]:
                    """tip −(1−α)ê·s , socket +α ê·s  (relative lat ↓)."""
                    v = np.asarray(lat_vec, dtype=np.float64).reshape(3)
                    v = v - ax * float(np.dot(v, ax))
                    vn = float(np.linalg.norm(v))
                    if vn <= 1e-9:
                        return np.zeros(3), np.zeros(3)
                    u = v / vn
                    s = min(float(step), lat_now * 0.50)
                    a = float(np.clip(left_share, 0.0, 0.6))
                    return -u * (s * (1.0 - a)), u * (s * a)

                if on_rim and lat_now > enter_lat:
                    seek_r, left_lat_cmd = _dual_seek_toward_hole(seek_m)
                    delta = seek_r
                    if lat_now <= enter_lat * 1.5:
                        delta = delta + ax * (press_keep * 0.30)
                    xy_blocked = True
                    hop_mode = "near_recenter"
                elif lat_now <= enter_lat * 1.25 and along_now <= seat_along + 0.030:
                    # NEVER pure-axial while lat still open — that wedges tip off-axis.
                    # Also NEVER axial-press when tray already tipped (歪了还怼).
                    keep_lat = float(search_cfg.get("near_axial_keep_lat_m", 0.0020))
                    tip_soft = float(search_cfg.get("search_tip_soft_deg", 15.0))
                    if tilt_now > tip_soft:
                        seek_r, left_lat_cmd = _dual_seek_toward_hole(seek_m)
                        # Lat already good but tipped & above mouth: re-approach, don't unload.
                        if lat_now <= enter_lat and along_now > seat_along:
                            delta = seek_r + ax * press_keep
                            hop_mode = "near_tip_reapproach"
                        else:
                            unload = float(search_cfg.get("search_unload_step_m", 0.0004))
                            delta = seek_r - ax * unload
                            hop_mode = "near_tip_hold"
                    else:
                        press = ax * max(float(np.dot(delta, ax)), press_keep * 1.4)
                        if lat_now > keep_lat:
                            seek_r, left_lat_cmd = _dual_seek_toward_hole(seek_m)
                            delta = press + seek_r
                        else:
                            delta = press
                        hop_mode = "near_axial"
                    xy_blocked = True
                elif lat_now <= enter_lat * 2.0 and along_now <= seat_along + 0.040:
                    seek_r, left_lat_cmd = _dual_seek_toward_hole(seek_m * 0.6)
                    delta = planar * 0.25 + seek_r + ax * max(
                        float(np.dot(delta, ax)), press_keep
                    )
                    hop_mode = "near_press"
                else:
                    delta = planar + ax * max(float(np.dot(delta, ax)), press_keep * 0.8)
                    hop_mode = "near_press"
                frame_left_lat_cmd = left_lat_cmd
            else:
                # Spiral with light axial so tip slides on surface (planar-only = mocap air).
                # Force already at contact target → no press_keep floor (search axial only).
                press_keep = float(search_cfg.get("near_hole_press_m", 0.00035)) * float(
                    search_cfg.get("spiral_surface_press_scale", 0.35)
                )
                f_des_contact = float(search_cfg.get("contact_f_des_n", 0.45))
                force_ok_press = (
                    (
                        resid_now_pre is not None
                        and abs(float(resid_now_pre)) >= f_des_contact
                    )
                    or abs_fz_now >= f_des_contact
                )
                if force_ok_press:
                    press_keep = 0.0
                ax_from_search = float(np.dot(delta, ax))
                delta = planar + ax * max(ax_from_search, press_keep)
                hop_mode = "spiral"
                if recovery_active and float(feat_priv.lateral_m) > recovery_lat_m:
                    hop_mode = "priv_recovery_seek"

            # Soft tip: priv may seek/unload; sensor ignores soft continue (hard abort above).
            # priv_seek_step_m==0 → no tip_soft 朝孔拽; brief unload + keep spiral planar.
            tip_soft_deg = float(search_cfg.get("search_tip_soft_deg", 14.0))
            priv_far = float(search_cfg.get("priv_far_lat_m", 0.010))
            lat_for_soft = float(feat_priv.lateral_m)
            seat_along_soft = float(search_cfg.get("priv_enter_along_max_m", 0.100))
            floating = along_now > seat_along_soft + 0.020
            if (
                priv_ctrl
                and seek_m > 0.0
                and tilt_now > tip_soft_deg
                and hop_mode not in ("unload", "hold", "near_tip_hold", "near_tip_reapproach")
            ):
                step_u = float(search_cfg.get("search_unload_step_m", 0.0004))
                press_keep = float(search_cfg.get("near_hole_press_m", 0.00040))
                seek = np.zeros(3, dtype=np.float64)
                v = np.asarray(lat_vec, dtype=np.float64).reshape(3)
                v = v - ax * float(np.dot(v, ax))
                vn = float(np.linalg.norm(v))
                if vn > 1e-9 and lat_for_soft > 1e-4:
                    u = v / vn
                    s = min(seek_m, lat_for_soft * 0.45)
                    seek = -u * s
                if floating:
                    # Floating above mouth while tipped: re-approach, do NOT unload further.
                    delta = seek + ax * (press_keep * 0.8)
                    hop_mode = "tip_reapproach"
                elif lat_for_soft <= priv_far:
                    delta = seek - ax * step_u
                    hop_mode = "tip_soft"
                    xy_blocked = True
                else:
                    delta = seek + (delta - ax * float(np.dot(delta, ax)))
                    hop_mode = "spiral_no_press"
                frame_left_lat_cmd = np.zeros(3, dtype=np.float64)
                sc_rebias = getattr(pipeline, "search", None)
                if sc_rebias is not None:
                    setattr(sc_rebias, "_rebias_spiral", True)
            elif (
                priv_ctrl
                and seek_m <= 0.0
                and tilt_now > tip_soft_deg
                and hop_mode == "spiral"
            ):
                # Theory path: light unload + keep planar spiral (no freeze / no theta reset).
                step_u = float(search_cfg.get("search_unload_step_m", 0.0003))
                press = float(np.dot(delta, ax))
                planar_keep = delta - ax * press
                delta = planar_keep - ax * max(step_u, 0.0) * 0.5
                hop_mode = "spiral_soft"

            max_step = float(cfg.get("sim", {}).get("max_pos_step_m", 0.004))
            if hop_mode in (
                "hop_slide",
                "spiral",
                "spiral_soft",
                "near_press",
                "near_recenter",
                "priv_recovery_seek",
            ):
                max_step = min(max_step, float(search_cfg.get("max_lat_step_m", max_step)))
            elif hop_mode in ("hop_lift", "hop_press", "unload", "tip_soft", "near_axial"):
                max_step = min(max_step, max(hop_lift_step, hop_press_step, unload_step) * 2.0)
            else:
                max_step = min(max_step, float(search_cfg.get("max_lat_step_m", max_step)))
            dn = float(np.linalg.norm(delta))
            if dn > max_step > 0.0:
                scale = max_step / dn
                delta = delta * scale
                frame_left_lat_cmd = frame_left_lat_cmd * scale
        else:
            action44[0:6] = hold_right_wrist6
            max_step = float(cfg.get("sim", {}).get("max_pos_step_m", 0.004))
            if final_phase == PipelinePhase.RELEASE.name:
                # Clip axial spring only; allow micro-creep to accumulate for insert_ok.
                sites_xyz = actual_action44_from_sites(raw)[0:3]
                err = hold_right_wrist6[0:3] - sites_xyz
                along_e = float(np.dot(err, ax))
                lat_e = err - ax * along_e
                if along_e > 0.005:
                    hold_right_wrist6[0:3] = sites_xyz + ax * 0.005 + lat_e
                    action44[0:6] = hold_right_wrist6
                elif along_e < -0.0003:
                    hold_right_wrist6[0:3] = sites_xyz + lat_e
                    action44[0:6] = hold_right_wrist6
                rel_i = int(surface_meta.get("_release_steps", 0))
                surface_meta["_release_steps"] = rel_i + 1
                creep_after = int(
                    cfg.get("compliant", {}).get("release", {}).get("creep_after_steps", 10)
                )
                creep_m = float(
                    cfg.get("compliant", {}).get("release", {}).get("creep_step_m", 0.00045)
                )
                # Creep until insert_ok; along gate only in priv diagnostic.
                creep_ok = rel_i >= creep_after and (not insert_ok)
                if priv_ctrl:
                    creep_ok = creep_ok and along_now > -0.016
                if creep_ok:
                    delta = ax * creep_m
                    hop_mode = "release_creep"
                else:
                    delta = np.zeros(3, dtype=np.float64)
                    hop_mode = "release_hold"
                if not surface_meta.get("_release_freeze"):
                    surface_meta["_release_freeze"] = True
                    print(
                        "pci: RELEASE — clip axial spring, creep for insert_ok, "
                        "keep grasp",
                        flush=True,
                    )
            else:
                if not priv_ctrl:
                    # Sensor INSERT: keep pipeline force-admittance delta (no priv seek/mouth_lock).
                    hop_mode = "insert_admit"
                    frame_left_lat_cmd = np.zeros(3, dtype=np.float64)
                else:
                    # Privileged diagnostic insert: axial press + lateral seek to hole axis.
                    press = float(
                        cfg.get("compliant", {}).get("insert", {}).get("hold_press_m", 0.00028)
                    )
                    lat_m = float(feat_priv.lateral_m)
                    # Tight-lat mouth lock: strong axial only when near axis.
                    mouth_lock = along_now >= 0.025 and along_now > 0.055 and lat_m < 0.0032
                    if along_now < 0.025:
                        # Bleed mocap spring NOW (before RELEASE) so tip stop plunging.
                        press = 0.0
                        hop_mode = "insert_seated"
                        sites_xyz = actual_action44_from_sites(raw)[0:3]
                        err = hold_right_wrist6[0:3] - sites_xyz
                        along_e = float(np.dot(err, ax))
                        lat_e = err - ax * along_e
                        keep = min(max(along_e, 0.0), 0.001)
                        hold_right_wrist6[0:3] = sites_xyz + ax * keep + lat_e
                        action44[0:6] = hold_right_wrist6
                    elif mouth_lock:
                        # Cap press if tray already tipping.
                        scale = 3.0 if lat_m < 0.0020 else 2.4
                        if tilt_now > 16.0:
                            scale = min(scale, 1.2)
                        press *= scale
                        hop_mode = "insert_bottom"
                    elif along_now < 0.100 and lat_m < 0.0055 and tilt_now < 24.0:
                        # Soften when lat high — hard press @4–7mm wedges/pops.
                        if lat_m < 0.0025:
                            press *= 3.2 if tilt_now < 16.0 else 1.5
                        elif lat_m < 0.0035:
                            press *= 2.2 if tilt_now < 16.0 else 1.2
                        else:
                            press *= 1.1 if tilt_now < 16.0 else 0.6
                        hop_mode = "insert_bottom"
                    elif along_now < 0.065 and tilt_now < 18.0:
                        if lat_m < 0.0020:
                            press *= 3.2
                        elif lat_m < 0.0040:
                            press *= 2.2
                        else:
                            press *= 0.9
                        hop_mode = "insert_bottom"
                    elif (
                        tilt_now > float(search_cfg.get("insert_tilt_warn_deg", 12.0))
                        and along_now
                        >= float(search_cfg.get("insert_mouth_along_min_m", 0.085))
                    ):
                        if lat_m < 0.0055:
                            press *= 0.35
                            hop_mode = "insert_soft"
                        else:
                            press = 0.0
                            hop_mode = "insert_mouth_tilt"
                    elif tilt_now > float(search_cfg.get("insert_tilt_warn_deg", 12.0)):
                        if lat_m < 0.0040 and along_now < 0.085:
                            press *= 0.55
                        else:
                            press *= 0.15
                        hop_mode = "insert_soft"
                    else:
                        hop_mode = "insert_priv"
                    seek = np.zeros(3, dtype=np.float64)
                    chamfer_lo = float(search_cfg.get("insert_chamfer_lo_m", 0.075))
                    chamfer_hi = float(search_cfg.get("insert_chamfer_hi_m", 0.095))
                    chamfer_hold = int(search_cfg.get("insert_chamfer_stall_hold", 8))
                    tip_stall = float(surface_meta.get("_last_tip_along_delta", 0.0))
                    stall_n = int(surface_meta.get("_tip_stall_frames", 0))
                    if abs(tip_stall) < 0.00025:
                        stall_n += 1
                    else:
                        stall_n = 0
                    surface_meta["_tip_stall_frames"] = stall_n
                    in_chamfer = chamfer_lo <= along_now <= chamfer_hi
                    if in_chamfer and stall_n >= 3 and lat_m > 0.0010:
                        surface_meta["_chamfer_latched"] = True
                    if along_now < chamfer_lo - 0.002:
                        surface_meta.pop("_chamfer_latched", None)
                    if (
                        (not mouth_lock)
                        and surface_meta.get("_chamfer_latched")
                        and chamfer_lo - 0.002 <= along_now <= chamfer_hi + 0.015
                        and hop_mode != "insert_bottom"
                    ):
                        hold_press_base = float(
                            cfg.get("compliant", {})
                            .get("insert", {})
                            .get("hold_press_m", 0.00028)
                        )
                        deep_centered = along_now < 0.095 and lat_m < 0.0080
                        sink_lat = 0.0080 if along_now < 0.095 else 0.0030
                        if lat_m < sink_lat or deep_centered:
                            if along_now < 0.100 and lat_m < 0.0055:
                                sink_scale = 2.2 if along_now < 0.095 else 1.8
                            else:
                                sink_scale = 1.6 if along_now < 0.090 else 1.0
                            press = max(float(press) * 0.80, hold_press_base * sink_scale)
                            hop_mode = "insert_chamfer_sink"
                        elif stall_n >= chamfer_hold:
                            if lat_m < 0.0060 and along_now < 0.095:
                                press = hold_press_base * 1.8
                                hop_mode = "insert_chamfer_sink"
                            elif along_now < 0.095:
                                press = hold_press_base * 0.55
                                hop_mode = "insert_chamfer_sink"
                            else:
                                press = 0.0
                                hop_mode = "insert_chamfer_hold"
                        elif hop_mode not in ("insert_seated", "insert_mouth_tilt"):
                            press *= 0.35
                            hop_mode = "insert_chamfer"
                    # Keep right seek while lat open; pure axial only when nearly centered.
                    # Left share 0 by default: move tip to hole (avoid dragging socket).
                    left_share_i = float(search_cfg.get("insert_left_share", 0.0))
                    left_lat_cmd = np.zeros(3, dtype=np.float64)
                    pure_axial_lat = float(search_cfg.get("insert_pure_axial_lat_m", 0.0020))
                    if mouth_lock or (
                        hop_mode in ("insert_bottom", "insert_chamfer_sink")
                        and lat_m < pure_axial_lat
                    ):
                        seek = np.zeros(3, dtype=np.float64)
                    elif hop_mode not in ("insert_seated", "insert_soft") and lat_m > 0.0004:
                        v = np.asarray(lat_vec, dtype=np.float64).reshape(3)
                        v = v - ax * float(np.dot(v, ax))
                        vn = float(np.linalg.norm(v))
                        if vn > 1e-9:
                            if hop_mode in (
                                "insert_chamfer",
                                "insert_chamfer_hold",
                                "insert_mouth_tilt",
                            ):
                                step = 0.00055
                            elif hop_mode == "insert_bottom":
                                step = 0.00045
                            else:
                                step = 0.00065 if along_now < 0.060 else 0.00045
                            s = min(step, lat_m * 0.50)
                            u = v / vn
                            a = float(np.clip(left_share_i, 0.0, 0.55))
                            seek = -u * (s * (1.0 - a))
                            left_lat_cmd = u * (s * a)
                    frame_left_lat_cmd = left_lat_cmd
                    if hop_mode == "insert_mouth_tilt":
                        unload = float(search_cfg.get("insert_mouth_unload_m", 0.00020))
                        delta = seek - ax * unload
                    else:
                        # If lat still open, cut press so seek can recenter (anti wrong-way jam).
                        if lat_m > pure_axial_lat * 1.5:
                            press = min(press, float(
                                cfg.get("compliant", {}).get("insert", {}).get("hold_press_m", 0.00028)
                            ) * 0.6)
                        delta = ax * press + seek
                    # r10: tip 高强制卸力/零压，避免硬插顶翻。
                    if surface_meta.get("_insert_tilt_unload"):
                        unload = float(search_cfg.get("insert_mouth_unload_m", 0.00035))
                        axial = float(np.dot(delta, ax))
                        if axial > 0.0:
                            delta = delta - ax * axial
                        delta = delta - ax * unload
                        press = 0.0
                        hop_mode = "insert_mouth_tilt"
                        frame_left_lat_cmd = np.zeros(3, dtype=np.float64)
                    if hop_mode == "insert_soft":
                        delta = ax * float(np.dot(delta, ax))
                        frame_left_lat_cmd = np.zeros(3, dtype=np.float64)
                    tilt_warn = float(search_cfg.get("insert_tilt_warn_deg", 12.0))
                    if (
                        (not mouth_lock)
                        and hop_mode in ("insert_chamfer_sink", "insert_repress", "insert_soft")
                        and tilt_now > tilt_warn + 6.0
                        and lat_m > 0.0035
                    ):
                        unload = float(search_cfg.get("insert_mouth_unload_m", 0.00020))
                        axial = float(np.dot(delta, ax))
                        if axial > 0.0:
                            cut = min(axial, unload * min(1.5, (tilt_now - tilt_warn) / 6.0))
                            delta = delta - ax * cut
                        if tilt_now > tilt_warn + 12.0 and lat_m > 0.0050:
                            delta = seek - ax * unload
                            hop_mode = "insert_mouth_tilt"
                    best = surface_meta.get("_insert_best_along_m")
                    pop_margin = 0.022 if along_now >= chamfer_lo else 0.040
                    if best is None or along_now < float(best):
                        surface_meta["_insert_best_along_m"] = along_now
                    elif along_now > float(best) + pop_margin:
                        if lat_m > 0.008:
                            print(
                                f"pci: INSERT abort — along popped {along_now*1e3:.1f}mm "
                                f"(best {float(best)*1e3:.1f}mm lat={lat_m*1e3:.1f}mm)",
                                flush=True,
                            )
                            fail_reason = "insert_along_pop"
                            break
                        if mouth_lock or (lat_m < 0.0035 and tilt_now <= tilt_warn + 6.0):
                            repress = float(
                                cfg.get("compliant", {})
                                .get("insert", {})
                                .get("hold_press_m", 0.00028)
                            )
                            if mouth_lock:
                                repress *= 2.4
                            delta = ax * repress
                            hop_mode = "insert_repress"
                            press = repress
                        else:
                            unload = float(search_cfg.get("insert_mouth_unload_m", 0.00020))
                            delta = seek - ax * unload
                            hop_mode = "insert_mouth_tilt"
                            press = 0.0
                    if surface_meta.get("_seat_along_m") is None and pr.reason in (
                        "priv_along_seat",
                        "hole_detected",
                    ):
                        surface_meta["_seat_along_m"] = along_now
            dn = float(np.linalg.norm(delta))
            if dn > max_step:
                scale = max_step / dn
                delta = delta * scale
                frame_left_lat_cmd = frame_left_lat_cmd * scale

        left_delta = np.zeros(3, dtype=np.float64)
        left_follow_step_m = 0.0
        left_lat_cmd = frame_left_lat_cmd
        # Privileged dual-arm recenter overrides blind tray-follow this frame.
        use_dual_recenter = bool(priv_ctrl) and float(np.linalg.norm(left_lat_cmd)) > 1e-9
        geom_left = read_priv_grasp_geom(raw) if (priv_ctrl or left_follow.config.enable) else None
        # Theory/ep6: SEARCH may follow tray (anti-tip); INSERT always may.
        search_left_follow_steps = int(search_cfg.get("search_left_follow_steps", 0))
        search_left_follow_scale = float(search_cfg.get("search_left_follow_scale", 1.0))
        follow_tilt_cap = float(search_cfg.get("search_left_follow_max_tilt_deg", 99.0))
        # Follow throughout SEARCH when steps>0 (anti-tip); tilt_cap is upper abort soft.
        early_search_follow = (
            bool(in_search)
            and bool(priv_ctrl)
            and bool(left_follow.config.enable)
            and search_left_follow_steps > 0
            and (
                search_left_follow_steps >= 9999
                or control_steps < search_left_follow_steps
            )
            and float(tilt_now) <= follow_tilt_cap
        )
        follow_now = (
            bool(left_follow.config.enable)
            and bool(priv_ctrl)
            and (not use_dual_recenter)
            and ((not in_search) or early_search_follow)
        )
        if use_dual_recenter:
            left_admit.config.enable = False
            hold_left_wrist28[0:3] = hold_left_wrist28[0:3] + left_lat_cmd
            left_delta = left_lat_cmd.copy()
        elif follow_now:
            hold_before = hold_left_wrist28[0:3].copy()
            hold_left_wrist28, left_follow_step_m = left_follow.step_hold6(
                hold_left_wrist28, geom_left
            )
            if early_search_follow and search_left_follow_scale < 1.0 - 1e-9:
                d = hold_left_wrist28[0:3] - hold_before
                hold_left_wrist28[0:3] = hold_before + d * search_left_follow_scale
                left_follow_step_m = float(np.linalg.norm(d) * search_left_follow_scale)
            if (
                left_admit_cfg_enable
                and (not in_search)
                and tilt_now < float(search_cfg.get("insert_tilt_warn_deg", 12.0)) + 8.0
            ):
                left_admit.config.enable = True
                admit_d = left_admit.step(wrench[1], approach_axis=ax) * 0.25
                hold_left_wrist28[0:3] = hold_left_wrist28[0:3] + admit_d
                left_delta = admit_d
            else:
                left_admit.config.enable = False
        elif in_search and left_admit_cfg_enable and (not use_dual_recenter):
            left_admit.config.enable = True
            left_delta = left_admit.step(wrench[1], approach_axis=ax) * float(
                search_cfg.get("search_left_admit_scale", 0.0)
            )
            hold_left_wrist28[0:3] = hold_left_wrist28[0:3] + left_delta
        elif (
            (not in_search)
            and left_admit_cfg_enable
            and final_phase != PipelinePhase.RELEASE.name
        ):
            tilt_warn = float(search_cfg.get("insert_tilt_warn_deg", 12.0))
            mouth_min = float(search_cfg.get("insert_mouth_along_min_m", 0.085))
            # Sensor: left FT admit only; no along/mouth gate from privileged depth.
            if not priv_ctrl:
                left_admit.config.enable = True
                left_delta = left_admit.step(wrench[1], approach_axis=ax)
            elif tilt_now > tilt_warn + 10.0 and along_now >= mouth_min:
                left_admit.config.enable = False
            else:
                left_admit.config.enable = True
                left_delta = left_admit.step(wrench[1], approach_axis=ax)
                if along_now >= mouth_min:
                    left_delta = left_delta * float(
                        search_cfg.get("insert_mouth_left_admit_scale", 0.35)
                    )
            hold_left_wrist28[0:3] = hold_left_wrist28[0:3] + left_delta
        else:
            left_admit.config.enable = False
        # Right delta only here; left xyz already folded into hold_left_wrist28.
        action44 = apply_dual_wrist_delta44(action44, delta, None)
        hold_right_wrist6 = action44[0:6].copy()
        # Soft-cap mocap↔site drift so peg is not yanked out of the hand.
        drift_cap = float(search_cfg.get("mocap_site_drift_cap_m", 0.012))
        site_now = actual_action44_from_sites(raw)[0:3]
        drift_vec = hold_right_wrist6[0:3] - site_now
        # Deep seat DURING INSERT only: clamp axial spring. Never during RELEASE creep.
        # Privileged along gate only — sensor keeps isotropic drift_cap.
        if (
            priv_ctrl
            and (not in_search)
            and final_phase != PipelinePhase.RELEASE.name
            and along_now < 0.025
        ):
            ax_cap = float(search_cfg.get("near_bottom_axial_drift_cap_m", 0.002))
            along_e = float(np.dot(drift_vec, ax))
            lat_e = drift_vec - ax * along_e
            if abs(along_e) > ax_cap > 0.0:
                along_e = float(np.sign(along_e)) * ax_cap
            drift_vec = ax * along_e + lat_e
            hold_right_wrist6[0:3] = site_now + drift_vec
            action44[0:3] = hold_right_wrist6[0:3]
        else:
            drift_n = float(np.linalg.norm(drift_vec))
            if drift_n > drift_cap > 0.0:
                hold_right_wrist6[0:3] = site_now + drift_vec * (drift_cap / drift_n)
                action44[0:3] = hold_right_wrist6[0:3]
        action44[22:28] = hold_left_wrist28
        action44[6:22] = hold_fingers
        action44[28:44] = hold_left_fingers
        # Wire insert.finger_open.
        open_amt = float(getattr(pr, "finger_open", 0.0) or 0.0)
        if final_phase == PipelinePhase.RELEASE.name and open_amt > 0.0:
            # Do NOT open right fingers: opening at ~1mm along lets tip plunge
            # past bottom (−11mm) and tips the tray (privileged: R0–19 tilt≈13°
            # stable; R20 open → tip_ad−6mm → tilt→30°).
            pass
        # Keep *0.82 near-bottom / creep — tip must slip slightly to reach bottom contact.
        # Privileged along depth only; sensor uses release_creep hop tag only.
        elif (not in_search) and hop_mode in (
            "insert_bottom",
            "insert_priv",
            "release_creep",
        ) and (
            (priv_ctrl and along_now < 0.060)
            or ((not priv_ctrl) and hop_mode == "release_creep")
        ):
            action44[6:22] = np.clip(hold_fingers * 0.82, -1.5, 1.5)
        # Re-assert left grasp clamp (never loosen left during Phase B).
        action44[28:44] = hold_left_fingers
        use_fingers = bool(cfg.get("compliant", {}).get("fingers", {}).get("enable", False))
        # QP deltas must apply when priv_grasp_opt is on (not gated on fingers.enable).
        use_hand_delta = bool(use_fingers or use_priv_grasp)
        resid_now = None
        if fz_base is not None:
            resid_now = float(_right_fz_sensor(raw, task_frame)) - float(fz_base)
        resid_soft = resid_now is None or abs(float(resid_now)) <= float(
            search_cfg.get("search_unload_if_abs_fz_n", 0.5)
        )
        hold_xy_steps = int(search_cfg.get("search_hold_xy_steps", 10))
        # fingers.enable / priv_grasp_opt: Δq from step 0 (hold_xy must not block).
        finger_gate_ok = (
            use_fingers
            or use_priv_grasp
            or (not in_search)
            or (control_steps >= hold_xy_steps and resid_soft)
        )
        if use_hand_delta and finger_gate_ok:
            action44[6:22] = hold_fingers + pr.delta_hand16
            # Wrist tray-follow and finger QP are orthogonal; do not block left Δq.
            if pr.delta_left_hand16 is not None:
                action44[28:44] = hold_left_fingers + pr.delta_left_hand16
                # Still never open left beyond hold.
                action44[28:44] = np.maximum(action44[28:44], hold_left_fingers * 0.98)
        step_action44(gym_env, action44, ego_recorder=ego_recorder)
        control_steps += 1

        feat_after = features_from_raw(raw)
        tip_after = np.asarray(feat_after.tip_pos, dtype=np.float64).reshape(3)
        site_after = actual_action44_from_sites(raw)[0:3]
        tip_d = tip_after - tip_before
        tip_planar = tip_d - hole_u * float(np.dot(tip_d, hole_u))
        tip_move = float(np.linalg.norm(tip_planar))
        tip_along_delta = float(np.dot(tip_d, hole_u))
        surface_meta["_last_tip_along_delta"] = tip_along_delta
        site_move = float(np.linalg.norm(site_after - site_before))
        dxy = float(np.linalg.norm(delta - ax * float(np.dot(delta, ax))))
        mocap_site_drift = float(np.linalg.norm(hold_right_wrist6[0:3] - site_after))

        # tip_resync uses privileged tip motion vs hole axis — sensor mode off.
        if (
            priv_ctrl
            and in_search
            and hop_mode in ("spiral", "spiral_soft", "near_press", "hop_slide", "priv_recovery_seek")
            and not xy_blocked
        ):
            resync_win = int(search_cfg.get("tip_resync_window", 12))
            resync_move = float(search_cfg.get("tip_resync_move_m", 0.00015))
            if tip_move < resync_move and dxy > 1e-5:
                resync_stuck += 1
            else:
                resync_stuck = 0
            if resync_stuck >= resync_win:
                # Tip stuck while cmd moves: re-sync mocap to site, break air-spiral.
                # Keep spiral theta — zeroing it every window froze r_cmd near ~2mm (r2).
                site_sync = actual_action44_from_sites(raw)[0:6].copy()
                # Micro re-contact along hole if residual under target (not slam).
                f_des_rc = float(search_cfg.get("contact_f_des_n", 0.55))
                if resid_now_pre is not None and abs(float(resid_now_pre)) < f_des_rc * 0.85:
                    micro = float(search_cfg.get("hold_press_m", 0.0001))
                    site_sync[0:3] = site_sync[0:3] + ax * micro
                hold_right_wrist6 = site_sync
                action44[0:6] = site_sync
                resync_stuck = 0
                sc_rebias = getattr(pipeline, "search", None)
                if sc_rebias is not None:
                    setattr(sc_rebias, "_rebias_spiral", True)
                    setattr(sc_rebias, "_rebias_keep_theta", True)
                print(
                    f"pci: tip resync mocap→site (stuck {resync_win} steps, keep θ) "
                    f"lat={feat_after.lateral_m*1e3:.1f}mm",
                    flush=True,
                )

        if in_search and hop_mode == "hop_slide" and tip_clear:
            stuck_cmd_acc += dxy
            if tip_move < tip_stuck_move:
                stuck_steps += 1
            else:
                stuck_steps = 0
                stuck_cmd_acc = 0.0
            if stuck_steps >= tip_stuck_window and stuck_cmd_acc >= tip_stuck_cmd:
                fail_reason = "search_tip_stuck"
                print(
                    f"pci: tip stuck after clear — cmd_xy_acc={stuck_cmd_acc*1e3:.1f}mm "
                    f"tip_move<{tip_stuck_move*1e3:.2f}mm over {stuck_steps} steps "
                    f"mocap_site_drift={mocap_site_drift*1e3:.1f}mm",
                    flush=True,
                )
                break

        sc = pipeline.search
        theta = float(getattr(sc, "_theta", 0.0))
        r_cmd = float(sc._spiral_radius(theta)) if hasattr(sc, "_spiral_radius") else 0.0
        traj.append(
            {
                "phase": final_phase,
                "reason": pr.reason,
                "insert_ok": insert_ok,
                "peg_ok": bool(outcome.peg_ok),
                "tray_ok": bool(outcome.tray_ok),
                "wrist_fz": float(task_frame.wrench_tool(wrench[0])[2]),
                "abs_fz": abs_fz_now,
                "left_fz": float(wrench[1][2]),
                "resid_fz": float(resid_now) if resid_now is not None else None,
                "tray_tilt_deg": tilt_now,
                "priv_lat_m": float(feat_after.lateral_m),
                "priv_along_m": float(feat_after.along_m),
                "along_delta_m": float(feat_after.along_m) - along0,
                "spiral_theta": theta,
                "spiral_r_cmd_m": r_cmd,
                "delta_xy_m": dxy,
                "tip_move_m": tip_move,
                "tip_along_delta_m": tip_along_delta,
                "site_move_m": site_move,
                "mocap_site_drift_m": mocap_site_drift,
                "tip_clear": bool(tip_clear),
                "xy_blocked": bool(xy_blocked),
                "hop_mode": hop_mode,
                "left_admit_step_m": float(np.linalg.norm(left_delta)),
                "left_follow_step_m": float(left_follow_step_m),
                "left_lat_cmd_m": float(np.linalg.norm(left_lat_cmd)),
                "mouth_ok": bool(mouth_ok),
                "priv_rel_rot_err_rad": float(getattr(pr, "priv_rel_rot_err_rad", 0.0)),
                "compliance": compliance_tag,
                "priv_gate": True,
                "priv_in_control": bool(priv_ctrl),
                "experiment_tag": exp_tag,
            }
        )

        # Privileged mouth-hold: lat/along/tilt/peg/tray all OK → campaign success stop.
        mouth_lat = float(search_cfg.get("mouth_max_lat_m", 0.0045))
        mouth_along = float(search_cfg.get("mouth_max_along_m", 0.100))
        mouth_tilt = float(search_cfg.get("mouth_max_tilt_deg", 16.0))
        mouth_need = int(search_cfg.get("mouth_hold_confirm_frames", 8))
        geom_mouth = (
            bool(outcome.tray_ok)
            and bool(outcome.peg_ok)
            and float(feat_after.lateral_m) <= mouth_lat
            and float(feat_after.along_m) <= mouth_along
            and float(tilt_now) <= mouth_tilt
        )
        if geom_mouth:
            mouth_hold_streak += 1
        else:
            mouth_hold_streak = 0
        if mouth_hold_streak >= mouth_need:
            mouth_ok = True
            if surface_meta.get("rel_rot_at_mouth_rad") is None:
                surface_meta["rel_rot_at_mouth_rad"] = float(
                    getattr(pr, "priv_rel_rot_err_rad", 0.0)
                )
            if bool(search_cfg.get("mouth_hold_stop", False)):
                fail_reason = ""
                print(
                    f"pci: MOUTH HOLD OK lat={feat_after.lateral_m*1e3:.1f}mm "
                    f"along={feat_after.along_m*1e3:.1f}mm tilt={tilt_now:.1f}deg "
                    f"confirm={mouth_hold_streak} — stop (mouth campaign)",
                    flush=True,
                )
                break

        if fail_reason == "search_lift_fail":
            break

        outcome2 = env._labeler.compute(raw)
        still_search = final_phase == PipelinePhase.COMPLIANT_SEARCH.name
        if abort_tray and still_search and not outcome2.tray_ok:
            tray_grace = int(search_cfg.get("search_tray_lost_grace_steps", 0))
            lat_gate = float(search_cfg.get("priv_far_lat_m", 0.010))
            if control_steps < tray_grace:
                pass
            elif float(feat_after.lateral_m) > lat_gate:
                fail_reason = "search_tray_lost"
                print("pci: PRIV GATE abort after step — tray_ok lost", flush=True)
                break
        if abort_tray and still_search and not outcome2.peg_ok:
            peg_lost_streak += 1
            if peg_lost_streak > peg_grace:
                fail_reason = "search_peg_lost"
                print("pci: PRIV GATE abort after step — peg_ok lost", flush=True)
                break
        elif still_search:
            peg_lost_streak = 0
        tilt_after = _search_tray_tilt_deg()
        hard_tilt_after = float(
            search_cfg.get("search_hard_tray_tilt_deg", max(max_search_tilt, 28.0))
        )
        mouth_campaign = bool(search_cfg.get("mouth_hold_stop", False))
        if (
            abort_tray
            and still_search
            and tilt_after > hard_tilt_after
            and ((not priv_ctrl) or (not mouth_campaign))
        ):
            fail_reason = "search_tray_tilt"
            print("pci: PRIV GATE abort after step — tray tilt (hard)", flush=True)
            break

        if pr.done:
            fail_reason = "" if pr.success or insert_ok else pr.reason
            break
        if insert_ok and pr.phase.name == "DONE":
            fail_reason = ""
            break

    feat_eval = features_from_raw(raw)
    outcome_final = env._labeler.compute(raw)
    tray_ok_final = bool(outcome_final.tray_ok)
    peg_ok_final = bool(outcome_final.peg_ok)
    # Mouth campaign: success = mouth_ok (peg+tray held, not tipped).
    # Full insert still requires insert_ok ∧ tray_ok when mouth_hold_stop is off.
    if bool(search_cfg.get("mouth_hold_stop", False)):
        success = bool(mouth_ok) and tray_ok_final and peg_ok_final
        if mouth_ok and success:
            fail_reason = ""
        elif not mouth_ok and fail_reason in ("", "max_control_steps"):
            fail_reason = fail_reason or "mouth_not_reached"
    else:
        success = bool(insert_ok) and tray_ok_final
        if insert_ok and not tray_ok_final:
            fail_reason = "insert_ok_but_tray_lost"
            print(
                "pci: PRIV DIAG — insert_ok but tray_ok=0 → not counted as success",
                flush=True,
            )
    # Drop non-JSON scratch keys from surface_meta before serialize.
    clean_meta = {
        k: v
        for k, v in dict(surface_meta).items()
        if not str(k).startswith("_")
        and not hasattr(v, "dtype")
        and not isinstance(v, PrivGraspGeom)
    }
    clean_meta["mouth_ok"] = bool(mouth_ok)
    clean_meta["mouth_hold_streak"] = int(mouth_hold_streak)
    clean_meta["latch_priv_geom_ok"] = bool(surface_meta.get("latch_priv_geom_ok"))
    clean_meta["theory_pose_qp"] = bool(_theory_pose_qp(cfg))
    phase_a_rel = float(surface_meta.get("phase_a_rel_rot_peak_rad", 0.0))
    pre_rel = float(surface_meta.get("pre_search_rel_rot_peak_rad", 0.0))
    traj_rel = max(
        (float(t.get("priv_rel_rot_err_rad", 0.0)) for t in traj),
        default=0.0,
    )
    max_rel_rot = max(phase_a_rel, pre_rel, traj_rel)
    clean_meta["max_rel_rot_err_rad"] = float(max_rel_rot)
    return {
        "success": success,
        "mouth_ok": bool(mouth_ok),
        "insert_ok": insert_ok,
        "tray_ok": tray_ok_final,
        "peg_ok": peg_ok_final,
        "fail_reason": fail_reason,
        "align_phase": "SURFACE",
        "align_steps": align_steps,
        "control_steps": control_steps,
        "final_phase": final_phase,
        "final_tip_dist_m": feat_eval.tip_socket_dist_m,
        "final_lat_m": feat_eval.lateral_m,
        "final_along_m": feat_eval.along_m,
        "eval_only": True,
        "hybrid_summary": hybrid.episode_summary(),
        "traj_tail": traj[-5:],
        "traj": traj,
        "phase_b_sensors": (
            "wrist_ft_world+dual_fingertip_force+left_wrist_admit"
            if not priv_ctrl
            else "wrist_ft_world+dual_fingertip_force+left_tray_follow"
        ),
        "phase_a_mode": "pbvs_biased_surface_press",
        "approach_noise": clean_meta,
        "surface_meta": clean_meta,
        "surface_reason": align_reason,
        "compliance": compliance_tag,
        "priv_gate": True,
        "priv_in_control": bool(priv_ctrl),
        "priv_grasp_in_loop": bool(use_priv_grasp),
        "settle_tilt_peak_deg": float(settle_tilt_peak),
        "max_rel_rot_err_rad": float(max_rel_rot),
        "theory_pose_qp": bool(_theory_pose_qp(cfg)),
        "experiment_tag": exp_tag,
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
        priv_grasp_config=_priv_grasp_config(cfg),
    )
    return env, hybrid, pipeline, force_labeler


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _jsonable(o: Any) -> Any:
        if isinstance(o, PrivGraspGeom):
            return {"_type": "PrivGraspGeom", "omitted": True}
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    path.write_text(json.dumps(summary, indent=2, default=_jsonable), encoding="utf-8")
