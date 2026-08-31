"""Privileged dual-hand grasp optimization during spiral search (PCI research).

Innovation (vs classical wrist-only PSFT / locked fingers)
---------------------------------------------------------
Keep **peg–tray relative orientation** and **in-hand object poses** near the
surface-latch freeze, while wrist XY continues spiral search.

Formulation (referenced, not copy-pasted):
- **Pfanne RA-L 2020**: object-level impedance → desired object wrench from pose error.
- **TUM wrench decomposition**: resistible wrench ≈ manipulation + internal (squeeze).
- **GraspQP (leggedrobotics/graspqp)**: friction-cone grasp matrix ``G``, bounded LS/QP
  for contact force magnitudes (numpy ``lsq_linear``, no Isaac/qpth dependency).

Decision vars: per-contact cone edge forces ``λ`` → tip ``f_des`` → Allegro Δq
(admittance). Outputs both hands' 16-D joint deltas.

合规: privileged geom in the loop → ``privileged_diagnostic`` only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation as R

from pci.priv_geom import PrivGraspGeom


@dataclass
class PrivGraspOptConfig:
    """Weights / limits for privileged relative-pose grasp QP."""

    enable: bool = True
    # Object impedance (Pfanne-style) gains.
    k_pos: float = 40.0  # N/m in-hand translation
    k_rot: float = 1.2  # N·m / rad in-hand rotation
    k_rel_rot: float = 2.0  # N·m / rad peg–tray relative rotation
    # Ignore relative translation in hole plane (search may move xy).
    regulate_rel_translation: bool = False
    k_rel_pos: float = 0.0
    # GraspQP-style cone + QP.
    friction_mu: float = 0.45
    n_cone_vecs: int = 4
    torque_weight: float = 5.0  # same scale idea as GraspQP span metric
    f_min_n: float = 0.4
    f_max_n: float = 8.0
    lambda_reg: float = 0.15  # pull λ toward nominal squeeze
    f_squeeze_n: float = 2.2  # nominal internal tip force
    # Map f_des → joint (same structure as fingers.py).
    k_admit: float = 0.00055
    b_admit: float = 0.30
    max_joint_step: float = 0.030
    hold_open_limit: float = 0.10
    hold_close_limit: float = 0.045
    # Cap object wrench demand so QP stays well-conditioned.
    w_force_max_n: float = 6.0
    w_torque_max_nm: float = 0.35
    # Tray side torque cap as fraction of w_torque_max_nm (was 0.4× — too weak).
    tray_torque_cap_frac: float = 0.70
    # Optional left squeeze boost when rel_rot_err is large (tilt-gated).
    allow_left_rel_boost: bool = False
    rel_boost_left_gain: float = 0.18
    rel_boost_rot_thresh_rad: float = 0.12
    rel_boost_tilt_gate_deg: float = 18.0


@dataclass(frozen=True, slots=True)
class PrivGraspOptResult:
    delta_right_hand16: np.ndarray
    delta_left_hand16: np.ndarray
    rel_rot_err_rad: float
    peg_inhand_rot_err_rad: float
    tray_inhand_rot_err_rad: float
    right_f_des: np.ndarray  # (4,)
    left_f_des: np.ndarray


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-10:
        return np.zeros(3, dtype=np.float64)
    return v / n


def _rotvec_err(R_ref: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """Rotation vector of R_ref^{-1} R_cur (small-angle object attitude error)."""
    d = R_ref.T @ R_cur
    return R.from_matrix(d).as_rotvec().astype(np.float64)


def _friction_cone_dirs(normal: np.ndarray, mu: float, n_vecs: int) -> np.ndarray:
    """GraspQP-style discretized friction cone edges (n_vecs, 3)."""
    n = _unit(normal)
    if float(np.linalg.norm(n)) < 1e-8:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(n @ ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    t1 = _unit(np.cross(ref, n))
    t2 = _unit(np.cross(n, t1))
    out = np.empty((n_vecs, 3), dtype=np.float64)
    for k in range(n_vecs):
        ang = 2.0 * np.pi * k / float(n_vecs)
        d = n + mu * (np.cos(ang) * t1 + np.sin(ang) * t2)
        out[k] = _unit(d)
    return out


def build_grasp_matrix(
    tip_pos: np.ndarray,
    tip_force12: np.ndarray,
    cog: np.ndarray,
    *,
    mu: float,
    n_cone: int,
    torque_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble G (6, 4*n_cone) like GraspQP span metric (numpy).

    Contact normal ≈ -measured force direction (hand pushes on object), fallback
    to tip→cog direction.
    """
    tips = np.asarray(tip_pos, dtype=np.float64).reshape(4, 3)
    f12 = np.asarray(tip_force12, dtype=np.float64).reshape(12)
    cog = np.asarray(cog, dtype=np.float64).reshape(3)
    cols: list[np.ndarray] = []
    for i in range(4):
        fi = f12[i * 3 : (i + 1) * 3]
        fn = float(np.linalg.norm(fi))
        if fn > 0.15:
            # Force on tip from object; contact normal on object ≈ +fi direction
            # from object view: hand applies -fi on object → normal from object to hand
            normal = _unit(-fi)
        else:
            normal = _unit(tips[i] - cog)
            if float(np.linalg.norm(normal)) < 1e-8:
                normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        dirs = _friction_cone_dirs(normal, mu, n_cone)
        r = tips[i] - cog
        for d in dirs:
            force = d
            torque = torque_weight * np.cross(r, d)
            cols.append(np.concatenate([force, torque]))
    G = np.stack(cols, axis=1)  # (6, 4*n_cone)
    return G, tips


def solve_contact_forces_qp(
    G: np.ndarray,
    w_des: np.ndarray,
    *,
    f_squeeze: float,
    n_cone: int,
    f_min: float,
    f_max: float,
    reg: float,
) -> np.ndarray:
    """Bounded LS: min ||Gλ - w||^2 + reg||λ - λ0||^2, box on λ.

    Returns per-finger force magnitudes (4,) by summing cone edges (GraspQP-style).
    """
    G = np.asarray(G, dtype=np.float64)
    w = np.asarray(w_des, dtype=np.float64).reshape(6)
    n_var = G.shape[1]
    assert n_var == 4 * n_cone

    # Augment for regularization toward uniform squeeze on cone edges.
    lam0 = np.full(n_var, float(f_squeeze) / float(n_cone), dtype=np.float64)
    A = np.vstack([G, np.sqrt(reg) * np.eye(n_var)])
    b = np.concatenate([w, np.sqrt(reg) * lam0])
    lo = np.full(n_var, 0.0, dtype=np.float64)
    hi = np.full(n_var, float(f_max), dtype=np.float64)
    try:
        sol = lsq_linear(A, b, bounds=(lo, hi), lsmr_tol=1e-4, max_iter=40)
        lam = np.asarray(sol.x, dtype=np.float64)
    except Exception:
        lam = np.linalg.lstsq(G, w, rcond=None)[0]
        lam = np.clip(lam, 0.0, f_max)

    mags = lam.reshape(4, n_cone).sum(axis=1)
    return np.clip(mags, f_min, f_max)


def _clip_wrench(w: np.ndarray, f_max: float, t_max: float) -> np.ndarray:
    out = np.asarray(w, dtype=np.float64).reshape(6).copy()
    fn = float(np.linalg.norm(out[:3]))
    if fn > f_max:
        out[:3] *= f_max / fn
    tn = float(np.linalg.norm(out[3:]))
    if tn > t_max:
        out[3:] *= t_max / tn
    return out


class PrivGraspOptController:
    """Freeze latch poses → each step QP redistributes dual fingertip forces."""

    def __init__(self, config: PrivGraspOptConfig | None = None) -> None:
        self.config = config or PrivGraspOptConfig()
        self._R_rel0: np.ndarray | None = None
        self._peg_in_r0_R: np.ndarray | None = None
        self._peg_in_r0_p: np.ndarray | None = None
        self._tray_in_l0_R: np.ndarray | None = None
        self._tray_in_l0_p: np.ndarray | None = None
        self._delta_r_prev = np.zeros(16, dtype=np.float64)
        self._delta_l_prev = np.zeros(16, dtype=np.float64)
        self._hold_r: np.ndarray | None = None
        self._hold_l: np.ndarray | None = None

    def reset(
        self,
        geom: PrivGraspGeom,
        hold_right16: np.ndarray,
        hold_left16: np.ndarray,
    ) -> None:
        self._hold_r = np.asarray(hold_right16, dtype=np.float64).reshape(16).copy()
        self._hold_l = np.asarray(hold_left16, dtype=np.float64).reshape(16).copy()
        self._delta_r_prev[:] = 0.0
        self._delta_l_prev[:] = 0.0
        self._R_rel0 = geom.tray_rot.T @ geom.peg_rot
        self._peg_in_r0_R = geom.right_wrist_rot.T @ geom.peg_rot
        self._peg_in_r0_p = geom.right_wrist_rot.T @ (geom.peg_pos - geom.right_wrist_pos)
        self._tray_in_l0_R = geom.left_wrist_rot.T @ geom.tray_rot
        self._tray_in_l0_p = geom.left_wrist_rot.T @ (geom.tray_pos - geom.left_wrist_pos)

    def _admit_hand(
        self,
        f_des: np.ndarray,
        f_meas12: np.ndarray,
        hold: np.ndarray,
        delta_prev: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config
        norms = np.array(
            [float(np.linalg.norm(f_meas12[i * 3 : (i + 1) * 3])) for i in range(4)],
            dtype=np.float64,
        )
        delta = np.zeros(16, dtype=np.float64)
        for i in range(4):
            j = 4 * i + 1
            delta[j] = cfg.k_admit * (float(f_des[i]) - float(norms[i])) - cfg.b_admit * float(
                delta_prev[j]
            )
        delta = np.clip(delta, -cfg.max_joint_step, cfg.max_joint_step)
        proposed = hold + delta
        delta = (
            np.clip(proposed, hold - cfg.hold_open_limit, hold + cfg.hold_close_limit) - hold
        )
        return delta

    def step(
        self,
        geom: PrivGraspGeom,
        right_force12: np.ndarray,
        left_force12: np.ndarray,
    ) -> PrivGraspOptResult:
        cfg = self.config
        zero16 = np.zeros(16, dtype=np.float64)
        if (
            not cfg.enable
            or self._R_rel0 is None
            or self._hold_r is None
            or self._hold_l is None
        ):
            return PrivGraspOptResult(
                zero16,
                zero16,
                0.0,
                0.0,
                0.0,
                np.full(4, cfg.f_squeeze_n),
                np.full(4, cfg.f_squeeze_n),
            )

        # --- Privileged pose errors (Pfanne object impedance targets) ---
        R_rel = geom.tray_rot.T @ geom.peg_rot
        e_rel = _rotvec_err(self._R_rel0, R_rel)

        R_peg_in_r = geom.right_wrist_rot.T @ geom.peg_rot
        p_peg_in_r = geom.right_wrist_rot.T @ (geom.peg_pos - geom.right_wrist_pos)
        e_rot_peg = _rotvec_err(self._peg_in_r0_R, R_peg_in_r)
        e_pos_peg = p_peg_in_r - self._peg_in_r0_p

        R_tray_in_l = geom.left_wrist_rot.T @ geom.tray_rot
        p_tray_in_l = geom.left_wrist_rot.T @ (geom.tray_pos - geom.left_wrist_pos)
        e_rot_tray = _rotvec_err(self._tray_in_l0_R, R_tray_in_l)
        e_pos_tray = p_tray_in_l - self._tray_in_l0_p

        # Map wrist-local errors to world wrenches on each object.
        # Relative couple OFF by default during spiral+frozen-left: fighting
        # search XY with tray torque tips the tray (PRIV GATE tilt abort).
        tau_rel_world = geom.tray_rot @ (cfg.k_rel_rot * e_rel)
        f_peg_w = geom.right_wrist_rot @ (cfg.k_pos * e_pos_peg)
        tau_peg_w = geom.right_wrist_rot @ (cfg.k_rot * e_rot_peg) + 0.5 * tau_rel_world
        # Tray side: squeeze + light in-hand pose only (no shared couple if k_rel≈0).
        f_tray_w = geom.left_wrist_rot @ (cfg.k_pos * e_pos_tray)
        tau_tray_w = geom.left_wrist_rot @ (cfg.k_rot * e_rot_tray) - 0.5 * tau_rel_world

        if cfg.regulate_rel_translation and cfg.k_rel_pos > 0.0:
            # Optional; default off so spiral XY is free.
            p_rel = geom.tray_rot.T @ (geom.peg_pos - geom.tray_pos)
            # no frozen p_rel0 stored — skip unless enabled later
            _ = p_rel

        w_peg = _clip_wrench(
            np.concatenate([f_peg_w, tau_peg_w]), cfg.w_force_max_n, cfg.w_torque_max_nm
        )
        # Tray wrench cap — left wrist frozen; raise torque cap vs old 0.4×.
        tray_t_cap = float(cfg.tray_torque_cap_frac) * float(cfg.w_torque_max_nm)
        w_tray = _clip_wrench(
            np.concatenate([f_tray_w, tau_tray_w]),
            0.6 * cfg.w_force_max_n,
            tray_t_cap,
        )

        G_r, _ = build_grasp_matrix(
            geom.right_tip_pos,
            right_force12,
            geom.peg_pos,
            mu=cfg.friction_mu,
            n_cone=cfg.n_cone_vecs,
            torque_weight=cfg.torque_weight,
        )
        G_l, _ = build_grasp_matrix(
            geom.left_tip_pos,
            left_force12,
            geom.tray_pos,
            mu=cfg.friction_mu,
            n_cone=cfg.n_cone_vecs,
            torque_weight=cfg.torque_weight,
        )
        f_des_r = solve_contact_forces_qp(
            G_r,
            w_peg,
            f_squeeze=cfg.f_squeeze_n,
            n_cone=cfg.n_cone_vecs,
            f_min=cfg.f_min_n,
            f_max=cfg.f_max_n,
            reg=cfg.lambda_reg,
        )
        f_des_l = solve_contact_forces_qp(
            G_l,
            w_tray,
            f_squeeze=cfg.f_squeeze_n,
            n_cone=cfg.n_cone_vecs,
            f_min=cfg.f_min_n,
            f_max=cfg.f_max_n,
            reg=cfg.lambda_reg,
        )

        rel_norm = float(np.linalg.norm(e_rel))
        # Mild peg-side squeeze boost when rel couple is enabled.
        if cfg.k_rel_rot > 1e-6:
            boost = 1.0 + 1.0 * min(rel_norm, 0.25)
            f_des_r = np.clip(f_des_r * boost, cfg.f_min_n, cfg.f_max_n)
        # Optional left boost: only when rel error large and tray not already tipped.
        if (
            cfg.allow_left_rel_boost
            and rel_norm > float(cfg.rel_boost_rot_thresh_rad)
        ):
            tray_tilt_deg = float(np.linalg.norm(e_rot_tray)) * 180.0 / np.pi
            if tray_tilt_deg <= float(cfg.rel_boost_tilt_gate_deg):
                l_boost = 1.0 + float(cfg.rel_boost_left_gain) * min(
                    rel_norm - float(cfg.rel_boost_rot_thresh_rad), 0.20
                )
                f_des_l = np.clip(f_des_l * l_boost, cfg.f_min_n, cfg.f_max_n)
        f_des_l = np.clip(f_des_l, cfg.f_min_n, cfg.f_max_n)

        d_r = self._admit_hand(f_des_r, right_force12, self._hold_r, self._delta_r_prev)
        d_l = self._admit_hand(f_des_l, left_force12, self._hold_l, self._delta_l_prev)
        self._delta_r_prev = d_r.copy()
        self._delta_l_prev = d_l.copy()

        return PrivGraspOptResult(
            delta_right_hand16=d_r,
            delta_left_hand16=d_l,
            rel_rot_err_rad=float(np.linalg.norm(e_rel)),
            peg_inhand_rot_err_rad=float(np.linalg.norm(e_rot_peg)),
            tray_inhand_rot_err_rad=float(np.linalg.norm(e_rot_tray)),
            right_f_des=f_des_r,
            left_f_des=f_des_l,
        )
