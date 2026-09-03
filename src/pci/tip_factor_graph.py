"""Track-A: Kim TEC-spirit tip/contact factor-graph estimator (skeleton).

REUSE: Kim et al. ICRA 2023 Simultaneous Tactile Estimation and Control of
Extrinsic Contact — factor catalog in refs/Tactile-Estimator-Controller/FACTORS.md
(ContactMotion, PoseDiff, DispDiff, Wrench, WrenchInc, TorqPoint/TorqLine,
EnergyElastic, PenHinge, …). Active Extrinsic / Doshi lever for wrench→contact.

Orthogonal to gated TEC-slim EKF + NIS (`tip_theory_estimator`,
`track_a_tec_joint`): this is a sliding-window NLS over fuller factors.
No GTSAM runtime dep; no episode patches (GENERAL_ALGO_MANDATE).

Design: docs/TRACK_A_FACTOR_GRAPH_TIP.md
Forbidden: tip_gt / tip_gt+noise / peg xpos in tip_hat; ep-index logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(x))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return x / n


def _planar(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    n = _unit(normal)
    x = np.asarray(v, dtype=np.float64).reshape(3)
    return x - n * float(np.dot(x, n))


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = _unit(normal)
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(a, n))) > 0.9:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    t1 = _unit(np.cross(a, n))
    t2 = np.cross(n, t1)
    return t1, t2


def _to2(v: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([float(np.dot(v, t1)), float(np.dot(v, t2))], dtype=np.float64)


def _from2(xy: np.ndarray, origin: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(2)
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    return o + xy[0] * t1 + xy[1] * t2


# ---------------------------------------------------------------------------
# Factor residuals (TEC FACTORS.md → planar / wrench slim)
# ---------------------------------------------------------------------------


def residual_contact_motion(
    tip_prev2: np.ndarray,
    tip_cur2: np.ndarray,
    c_mat: np.ndarray,
    dw_parallel2: np.ndarray,
) -> np.ndarray:
    """ContactMotion: tip_i - tip_{i-1} - C Δw_∥."""
    return (
        np.asarray(tip_cur2, dtype=np.float64).reshape(2)
        - np.asarray(tip_prev2, dtype=np.float64).reshape(2)
        - np.asarray(c_mat, dtype=np.float64).reshape(2, 2)
        @ np.asarray(dw_parallel2, dtype=np.float64).reshape(2)
    )


def residual_pose_diff(tip2: np.ndarray, geom_tip2: np.ndarray) -> np.ndarray:
    """PoseDiff spirit: tip ≈ in-hand geom tip (planar)."""
    return (
        np.asarray(tip2, dtype=np.float64).reshape(2)
        - np.asarray(geom_tip2, dtype=np.float64).reshape(2)
    )


def residual_disp_diff(
    tip_prev2: np.ndarray,
    tip_cur2: np.ndarray,
    geom_prev2: np.ndarray,
    geom_cur2: np.ndarray,
) -> np.ndarray:
    """DispDiff spirit: Δtip ≈ Δgeom."""
    return residual_pose_diff(tip_cur2, tip_prev2) - residual_pose_diff(
        geom_cur2, geom_prev2
    )


def residual_extrinsic_contact(tip2: np.ndarray, contact2: np.ndarray) -> np.ndarray:
    """Unary extrinsic contact measurement on tip (or shared tip≈c for point)."""
    return residual_pose_diff(tip2, contact2)


def residual_object_fixed_contact(
    tip_prev2: np.ndarray,
    tip_cur2: np.ndarray,
    c_prev2: np.ndarray,
    c_cur2: np.ndarray,
) -> np.ndarray:
    """TEC F_oc spirit: contact offset in tip frame sticky across time."""
    off_prev = np.asarray(c_prev2, dtype=np.float64).reshape(2) - np.asarray(
        tip_prev2, dtype=np.float64
    ).reshape(2)
    off_cur = np.asarray(c_cur2, dtype=np.float64).reshape(2) - np.asarray(
        tip_cur2, dtype=np.float64
    ).reshape(2)
    return off_cur - off_prev


def residual_env_contact(
    c_prev2: np.ndarray,
    c_cur2: np.ndarray,
    *,
    along_delta: float = 0.0,
    along_weight: float = 1.0,
) -> np.ndarray:
    """TEC F_cc spirit (planar): tangential slip soft; along handled separately.

    Returns [Δc_x, Δc_y, along_weight * along_delta].
    """
    dc = np.asarray(c_cur2, dtype=np.float64).reshape(2) - np.asarray(
        c_prev2, dtype=np.float64
    ).reshape(2)
    return np.array(
        [dc[0], dc[1], float(along_weight) * float(along_delta)],
        dtype=np.float64,
    )


def residual_torq_point(
    r_contact3: np.ndarray,
    force3: np.ndarray,
    torque3: np.ndarray,
) -> np.ndarray:
    """TorqPoint: M - r × F → 0 at extrinsic point contact."""
    r = np.asarray(r_contact3, dtype=np.float64).reshape(3)
    f = np.asarray(force3, dtype=np.float64).reshape(3)
    m = np.asarray(torque3, dtype=np.float64).reshape(3)
    return m - np.cross(r, f)


def residual_torq_line(
    r_contact3: np.ndarray,
    force3: np.ndarray,
    torque3: np.ndarray,
    line_dir3: np.ndarray,
) -> np.ndarray:
    """TorqLine: (M - r×F) · a_x → 0 along contact line."""
    tp = residual_torq_point(r_contact3, force3, torque3)
    a = _unit(line_dir3)
    return np.array([float(np.dot(tp, a))], dtype=np.float64)


def residual_wrench_lever_tip(
    tip3: np.ndarray,
    wrist3: np.ndarray,
    force3: np.ndarray,
    torque3: np.ndarray,
    plane_n: np.ndarray,
) -> np.ndarray:
    """Wrench / Active Extrinsic: planar projection of TorqPoint residual.

    Soft measurement linking tip locus to wrist F/T (Doshi / CLEP spirit).
    """
    r = np.asarray(tip3, dtype=np.float64).reshape(3) - np.asarray(
        wrist3, dtype=np.float64
    ).reshape(3)
    tp = residual_torq_point(r, force3, torque3)
    t1, t2 = _tangent_basis(plane_n)
    # Project torque residual onto plane tangents (2-DoF soft tip cue).
    return np.array([float(np.dot(tp, t1)), float(np.dot(tp, t2))], dtype=np.float64)


def residual_energy_elastic(
    wrench6: np.ndarray,
    stiffness6: np.ndarray,
) -> np.ndarray:
    """EnergyElastic: w ⊘ sqrt(K) (TEC Eq.16 spirit)."""
    w = np.asarray(wrench6, dtype=np.float64).reshape(-1)
    k = np.asarray(stiffness6, dtype=np.float64).reshape(-1)
    if w.size != k.size:
        raise ValueError("wrench and stiffness must match length")
    sqrt_k = np.sqrt(np.maximum(k, 1e-12))
    return w / sqrt_k


def residual_pen_hinge(penetration: float, d_min: float) -> np.ndarray:
    """PenHinge: hinge(d_min - d_pen) ≥ 0 inequality soft."""
    gap = float(d_min) - float(penetration)
    return np.array([max(gap, 0.0)], dtype=np.float64)


def residual_c_prior(c_mat: np.ndarray, c_prior: np.ndarray | None = None) -> np.ndarray:
    """Weak prior C ≈ I (or given prior); slip handled by cov / weight outside."""
    c = np.asarray(c_mat, dtype=np.float64).reshape(2, 2)
    p = (
        np.eye(2, dtype=np.float64)
        if c_prior is None
        else np.asarray(c_prior, dtype=np.float64).reshape(2, 2)
    )
    return (c - p).reshape(4)


# ---------------------------------------------------------------------------
# Window state + Gauss–Newton skeleton
# ---------------------------------------------------------------------------


@dataclass
class TipFGConfig:
    """Information weights (inverse variances) — theory knobs, not ep thresholds."""

    window: int = 8
    w_contact_motion: float = 1.0 / 2e-6
    w_pose_diff: float = 1.0 / 1.5e-5
    w_disp_diff: float = 1.0 / 3e-5
    w_extrinsic: float = 1.0 / 2e-7
    w_wrench_lever: float = 1.0 / 4e-5
    w_c_prior: float = 1.0 / 2e-3
    w_oc: float = 1.0 / 1e-6
    max_iters: int = 8
    step_clip: float = 0.05  # tip planar step (m)
    alpha_floor: float = 0.15
    alpha_ceil: float = 1.2
    origin_refresh_along: bool = True


@dataclass
class TipFGFrame:
    """One timestep of eligible (+ optional residual-priv) measurements."""

    wrist_pos: np.ndarray
    plane_n: np.ndarray
    geom_tip: np.ndarray | None = None
    contact_tip: np.ndarray | None = None
    contact_n: int = 0
    force_xyz: np.ndarray | None = None
    torque_xyz: np.ndarray | None = None
    tip_gt: np.ndarray | None = None  # diagnostic only; never in residual graph


@dataclass
class TipFGEstimate:
    tip_hat: np.ndarray
    C: np.ndarray = field(default_factory=lambda: np.eye(2))
    contact2: np.ndarray = field(default_factory=lambda: np.zeros(2))
    cost: float = float("nan")
    n_factors: int = 0
    source: str = "fg_init"
    tip_err_to_gt_m: float = float("nan")
    meta: dict = field(default_factory=dict)


class TipContactFactorGraph:
    """Sliding-window tip/contact NLS (TEC spirit, numpy-only skeleton).

    State vector (shared C across window):
      x = [t_0x, t_0y, …, t_{W-1}x, t_{W-1}y, c11, c12, c21, c22]
    Optional per-frame contact2 stored parallel to tip (sticky F_oc).
    """

    def __init__(self, cfg: TipFGConfig | None = None) -> None:
        self.cfg = cfg or TipFGConfig()
        self._frames: list[TipFGFrame] = []
        self._tips2: list[np.ndarray] = []
        self._contacts2: list[np.ndarray] = []
        self._C = np.eye(2, dtype=np.float64)
        self._origin = np.zeros(3, dtype=np.float64)
        self._along0 = 0.0
        self._wrist_prev: np.ndarray | None = None
        self._seed_tip2: np.ndarray | None = None
        self._initialized = False
        self._last_cost = float("nan")
        self._last_n_factors = 0
        self._last_source = "init"

    @property
    def initialized(self) -> bool:
        return bool(self._initialized)

    def reset(
        self,
        *,
        wrist_pos: np.ndarray,
        plane_n: np.ndarray,
        tip_seed: np.ndarray | None = None,
    ) -> TipFGEstimate:
        wrist = np.asarray(wrist_pos, dtype=np.float64).reshape(3)
        n = _unit(plane_n)
        tip = (
            np.asarray(tip_seed, dtype=np.float64).reshape(3)
            if tip_seed is not None
            else wrist.copy()
        )
        self._along0 = float(np.dot(tip, n))
        self._origin = n * self._along0
        t1, t2 = _tangent_basis(n)
        t2p = _to2(tip, t1, t2)
        self._frames.clear()
        self._tips2 = []
        self._contacts2 = []
        self._seed_tip2 = t2p.copy()
        self._C = np.eye(2, dtype=np.float64)
        self._wrist_prev = wrist.copy()
        self._initialized = True
        self._last_source = "seed" if tip_seed is not None else "wrist_prior"
        # Temporary single-tip view for tip_world before first step.
        self._tips2 = [t2p.copy()]
        self._contacts2 = [t2p.copy()]
        out = self._pack(n, tip_gt=None)
        self._tips2 = []
        self._contacts2 = []
        return out

    def _clip_c(self) -> None:
        u, s, vt = np.linalg.svd(self._C)
        s = np.clip(s, self.cfg.alpha_floor, self.cfg.alpha_ceil)
        self._C = u @ np.diag(s) @ vt

    def _state_dim(self) -> int:
        w = len(self._tips2)
        return 2 * w + 4

    def _pack_state(self) -> np.ndarray:
        parts = [t.reshape(2) for t in self._tips2]
        parts.append(self._C.reshape(4))
        return np.concatenate(parts)

    def _unpack_state(self, x: np.ndarray) -> None:
        w = len(self._tips2)
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        for i in range(w):
            self._tips2[i] = x[2 * i : 2 * i + 2].copy()
        self._C = x[2 * w : 2 * w + 4].reshape(2, 2).copy()
        self._clip_c()

    def _residuals_and_jac(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build stacked weighted residuals and dense Jacobian at x."""
        self._unpack_state(x)
        w = len(self._tips2)
        rows: list[np.ndarray] = []
        jacs: list[np.ndarray] = []
        dim = self._state_dim()
        cfg = self.cfg

        def add(r: np.ndarray, j: np.ndarray, weight: float) -> None:
            sw = float(np.sqrt(max(weight, 0.0)))
            rows.append(sw * np.asarray(r, dtype=np.float64).reshape(-1))
            jacs.append(sw * np.asarray(j, dtype=np.float64))

        # C prior
        r_c = residual_c_prior(self._C)
        j_c = np.zeros((4, dim))
        j_c[:, 2 * w : 2 * w + 4] = np.eye(4)
        add(r_c, j_c, cfg.w_c_prior)

        for i in range(1, w):
            fr = self._frames[i]
            fr_p = self._frames[i - 1]
            n = _unit(fr.plane_n)
            t1, t2 = _tangent_basis(n)
            dw = _to2(_planar(fr.wrist_pos - fr_p.wrist_pos, n), t1, t2)

            # ContactMotion
            r_cm = residual_contact_motion(
                self._tips2[i - 1], self._tips2[i], self._C, dw
            )
            j = np.zeros((2, dim))
            j[:, 2 * (i - 1) : 2 * (i - 1) + 2] = -np.eye(2)
            j[:, 2 * i : 2 * i + 2] = np.eye(2)
            # ∂(C dw)/∂C = kron(dw^T, I_2) on C block
            j[:, 2 * w : 2 * w + 4] = -np.array(
                [
                    [dw[0], dw[1], 0.0, 0.0],
                    [0.0, 0.0, dw[0], dw[1]],
                ],
                dtype=np.float64,
            )
            add(r_cm, j, cfg.w_contact_motion)

            # F_oc: contact sticky offset (use parallel contact list)
            r_oc = residual_object_fixed_contact(
                self._tips2[i - 1],
                self._tips2[i],
                self._contacts2[i - 1],
                self._contacts2[i],
            )
            j = np.zeros((2, dim))
            j[:, 2 * (i - 1) : 2 * (i - 1) + 2] = np.eye(2)
            j[:, 2 * i : 2 * i + 2] = -np.eye(2)
            add(r_oc, j, cfg.w_oc)

            # PoseDiff / DispDiff if geom available
            if fr.geom_tip is not None:
                zg = _to2(fr.geom_tip, t1, t2)
                r_pd = residual_pose_diff(self._tips2[i], zg)
                j = np.zeros((2, dim))
                j[:, 2 * i : 2 * i + 2] = np.eye(2)
                add(r_pd, j, cfg.w_pose_diff)
                if fr_p.geom_tip is not None:
                    zg_p = _to2(fr_p.geom_tip, t1, t2)
                    r_dd = residual_disp_diff(
                        self._tips2[i - 1], self._tips2[i], zg_p, zg
                    )
                    j = np.zeros((2, dim))
                    j[:, 2 * (i - 1) : 2 * (i - 1) + 2] = -np.eye(2)
                    j[:, 2 * i : 2 * i + 2] = np.eye(2)
                    add(r_dd, j, cfg.w_disp_diff)

        # Unary measurements on latest (and any contact frames)
        for i, fr in enumerate(self._frames):
            n = _unit(fr.plane_n)
            t1, t2 = _tangent_basis(n)
            if fr.contact_tip is not None and fr.contact_n > 0:
                zc = _to2(fr.contact_tip, t1, t2)
                self._contacts2[i] = zc.copy()
                r_ec = residual_extrinsic_contact(self._tips2[i], zc)
                j = np.zeros((2, dim))
                j[:, 2 * i : 2 * i + 2] = np.eye(2)
                add(r_ec, j, cfg.w_extrinsic)

            if fr.force_xyz is not None and fr.torque_xyz is not None:
                tip3 = _from2(self._tips2[i], self._origin, t1, t2)
                tip3 = tip3 - n * float(np.dot(tip3, n)) + n * self._along0
                r_w = residual_wrench_lever_tip(
                    tip3,
                    fr.wrist_pos,
                    fr.force_xyz,
                    fr.torque_xyz,
                    n,
                )
                # Finite-diff Jacobian for nonlinear lever (skeleton; small dense).
                j = np.zeros((2, dim))
                eps = 1e-5
                for k in range(2):
                    x_pert = x.copy()
                    x_pert[2 * i + k] += eps
                    tips_save = [t.copy() for t in self._tips2]
                    c_save = self._C.copy()
                    self._unpack_state(x_pert)
                    tip_p = _from2(self._tips2[i], self._origin, t1, t2)
                    tip_p = tip_p - n * float(np.dot(tip_p, n)) + n * self._along0
                    r_p = residual_wrench_lever_tip(
                        tip_p,
                        fr.wrist_pos,
                        fr.force_xyz,
                        fr.torque_xyz,
                        n,
                    )
                    j[:, 2 * i + k] = (r_p - r_w) / eps
                    self._tips2 = tips_save
                    self._C = c_save
                add(r_w, j, cfg.w_wrench_lever)

        if not rows:
            return np.zeros(0), np.zeros((0, dim))
        r = np.concatenate(rows)
        j = np.vstack(jacs)
        self._last_n_factors = int(len(rows))
        return r, j

    def _solve_window(self) -> float:
        x = self._pack_state()
        cost = float("nan")
        for _ in range(int(self.cfg.max_iters)):
            r, j = self._residuals_and_jac(x)
            if r.size == 0:
                break
            cost = float(0.5 * np.dot(r, r))
            # Gauss–Newton: (JᵀJ + λI) δ = -Jᵀr
            jt_j = j.T @ j
            jt_j = jt_j + 1e-8 * np.eye(jt_j.shape[0])
            try:
                delta = -np.linalg.solve(jt_j, j.T @ r)
            except np.linalg.LinAlgError:
                break
            # Clip tip steps
            w = len(self._tips2)
            for i in range(w):
                d2 = delta[2 * i : 2 * i + 2]
                nrm = float(np.linalg.norm(d2))
                if nrm > self.cfg.step_clip:
                    delta[2 * i : 2 * i + 2] *= self.cfg.step_clip / nrm
            x = x + delta
            self._unpack_state(x)
            if float(np.linalg.norm(delta)) < 1e-9:
                break
        r, _ = self._residuals_and_jac(x)
        cost = float(0.5 * np.dot(r, r)) if r.size else cost
        self._last_cost = cost
        return cost

    def tip_world(self, plane_n: np.ndarray) -> np.ndarray:
        n = _unit(plane_n)
        t1, t2 = _tangent_basis(n)
        tip2 = self._tips2[-1] if self._tips2 else np.zeros(2)
        tip = _from2(tip2, self._origin, t1, t2)
        tip = tip - n * float(np.dot(tip, n)) + n * self._along0
        return tip

    def _pack(self, plane_n: np.ndarray, *, tip_gt: np.ndarray | None) -> TipFGEstimate:
        tip = self.tip_world(plane_n)
        err = float("nan")
        if tip_gt is not None:
            err = float(
                np.linalg.norm(
                    _planar(
                        tip - np.asarray(tip_gt, dtype=np.float64).reshape(3),
                        plane_n,
                    )
                )
            )
        return TipFGEstimate(
            tip_hat=tip.copy(),
            C=self._C.copy(),
            contact2=(
                self._contacts2[-1].copy()
                if self._contacts2
                else np.zeros(2, dtype=np.float64)
            ),
            cost=float(self._last_cost),
            n_factors=int(self._last_n_factors),
            source=str(self._last_source),
            tip_err_to_gt_m=err,
            meta={
                "tip_est_backend": "tip_factor_graph",
                "tip_est_source": self._last_source,
                "tip_est_uses_peg_xpos": False,
                "tip_est_residual_privilege": (
                    "mujoco_peg_tray_contact_pos_optional+wrench_eligible"
                    "+geom_tip_posediff_prior"
                ),
                "tip_est_C": self._C.tolist(),
                "tip_fg_cost": float(self._last_cost),
                "tip_fg_n_factors": int(self._last_n_factors),
                "tip_fg_window": len(self._tips2),
            },
        )

    def step(self, frame: TipFGFrame) -> TipFGEstimate:
        """Append frame, maintain window, solve NLS, return tip estimate."""
        wrist = np.asarray(frame.wrist_pos, dtype=np.float64).reshape(3)
        n = _unit(frame.plane_n)
        frame = TipFGFrame(
            wrist_pos=wrist,
            plane_n=n,
            geom_tip=(
                np.asarray(frame.geom_tip, dtype=np.float64).reshape(3)
                if frame.geom_tip is not None
                else None
            ),
            contact_tip=(
                np.asarray(frame.contact_tip, dtype=np.float64).reshape(3)
                if frame.contact_tip is not None
                else None
            ),
            contact_n=int(frame.contact_n),
            force_xyz=(
                np.asarray(frame.force_xyz, dtype=np.float64).reshape(3)
                if frame.force_xyz is not None
                else None
            ),
            torque_xyz=(
                np.asarray(frame.torque_xyz, dtype=np.float64).reshape(3)
                if frame.torque_xyz is not None
                else None
            ),
            tip_gt=(
                np.asarray(frame.tip_gt, dtype=np.float64).reshape(3)
                if frame.tip_gt is not None
                else None
            ),
        )

        if not self._initialized:
            seed = frame.contact_tip if frame.contact_tip is not None else frame.geom_tip
            self.reset(wrist_pos=wrist, plane_n=n, tip_seed=seed)

        t1, t2 = _tangent_basis(n)
        # Predict new tip via ContactMotion before optimize.
        if self._tips2 and self._wrist_prev is not None:
            dw = _to2(_planar(wrist - self._wrist_prev, n), t1, t2)
            tip_pred = self._tips2[-1] + self._C @ dw
        elif frame.contact_tip is not None and frame.contact_n > 0:
            tip_pred = _to2(frame.contact_tip, t1, t2)
        elif frame.geom_tip is not None:
            tip_pred = _to2(frame.geom_tip, t1, t2)
        elif self._seed_tip2 is not None:
            tip_pred = self._seed_tip2.copy()
        else:
            tip_pred = np.zeros(2, dtype=np.float64)

        self._frames.append(frame)
        self._tips2.append(tip_pred.copy())
        self._contacts2.append(tip_pred.copy())
        assert len(self._frames) == len(self._tips2) == len(self._contacts2)

        # Drop oldest beyond window
        wmax = max(int(self.cfg.window), 2)
        while len(self._frames) > wmax:
            self._frames.pop(0)
            self._tips2.pop(0)
            self._contacts2.pop(0)

        if frame.contact_tip is not None and frame.contact_n > 0:
            if self.cfg.origin_refresh_along:
                self._along0 = float(np.dot(frame.contact_tip, n))
                self._origin = n * self._along0
            self._last_source = "fg_contact"
        elif frame.geom_tip is not None:
            self._last_source = "fg_geom"
        elif frame.force_xyz is not None:
            self._last_source = "fg_wrench"
        else:
            self._last_source = "fg_predict"

        self._solve_window()
        self._wrist_prev = wrist.copy()
        return self._pack(n, tip_gt=frame.tip_gt)


def is_tip_factor_graph_mode(mode: str) -> bool:
    m = str(mode or "").strip().lower()
    return m in (
        "track_a_tip_fg",
        "tip_factor_graph",
        "track_a_factor_graph",
        "tec_factor_graph",
        "kim_tec_fg",
    )


# Factor name registry (docs / tests)
TEC_FACTOR_NAMES = (
    "ContactMotion",
    "PoseDiff",
    "DispDiff",
    "ExtrinsicContact",
    "ObjectFixedContact",
    "EnvContact",
    "Wrench",
    "WrenchInc",
    "TorqPoint",
    "TorqLine",
    "EnergyElastic",
    "PenHinge",
    "PenEven",
    "DispVar",
    "CPrior",
)
