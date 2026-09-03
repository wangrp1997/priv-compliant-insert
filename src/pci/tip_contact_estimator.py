"""Track-A tip observer: contact-geometry tip_hat (no peg body xpos in estimate).

Sim stand-in for TEC-style extrinsic contact localization:
force-weighted MuJoCo contact points between peg↔tray (and optional finger↔peg
for grasp quality). Control must use tip_hat, never peg xpos / tip_gt.

Residual privilege (disclose): reading MuJoCo ``contact.pos`` for named
peg/tray geoms is simulator contact geometry, not a deployable tactile array.
Finger tip body positions are proprio/FK-eligible. Wrist F/T is eligible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

_PEG_BODY = "industreal_round_peg_8mm"
_TRAY_BODY = "industreal_tray_insert_round_peg_8mm"
_FINGER_TIP_BODIES_RIGHT = (
    "ff_tip_right",
    "mf_tip_right",
    "rf_tip_right",
    "th_tip_right",
)


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


def _subtree_bodies(model, root_id: int) -> set[int]:
    bodies = {int(root_id)}
    changed = True
    while changed:
        changed = False
        for bid in range(model.nbody):
            if int(model.body_parentid[bid]) in bodies and bid not in bodies:
                bodies.add(bid)
                changed = True
    return bodies


def _geoms_for_bodies(model, body_ids: set[int]) -> set[int]:
    return {
        g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in body_ids
    }


@dataclass
class ContactTipEstimate:
    tip_hat: np.ndarray
    contact_n: int = 0
    contact_force_n: float = 0.0
    seated: bool = False
    source: str = "hold"
    tip_err_to_gt_m: float = float("nan")  # fill only for logging if GT passed
    finger_grasp_force_n: float = 0.0
    meta: dict = field(default_factory=dict)


class ContactTipEstimator:
    """Contact-geometry tip observer for DexJoCo / PCI (Track-A).

    Primary: peg↔tray contacts only; tip = far-from-grasp locus (force-weighted
    local cluster; optional PCA free-end gated off by default). Project to tray
    plane, EMA, planar outlier reject.

    Lost contact (non-rigid grasp): do **not** treat wristΔ as rigid tipΔ.
    Default ``holdover_mode=freeze_planar`` freezes last **force-seated** planar
    tip_hat until extrinsic contact returns (TEC / active-extrinsic spirit).
    Freeze anchors only after peg↔tray ``fsum ≥ seat_force_n`` (v4: never freeze
    on weak/any contact — smoke_v3 ep03 peak~400 mm pathology). Until first
    strong seat, bridge with wrist+offset. First strong seat hard-snaps tip_hat.
    Offset refreshed only when seated.
    """

    def __init__(
        self,
        raw,
        *,
        ema: float = 0.55,
        min_force_n: float = 0.008,
        seat_force_n: float = 0.015,
        outlier_gate_m: float = 0.012,
        holdover_frames: int = 12,
        cluster_radius_m: float = 0.008,
        force_pow: float = 1.5,
        use_pca_locus: bool = False,
        pca_min_contacts: int = 3,
        holdover_mode: str = "freeze_planar",
        holdover_decay: float = 0.7,
        seat_only_offset: bool = True,
        freeze_require_seat: bool = True,
    ) -> None:
        model = raw._model
        peg_id = int(model.body(_PEG_BODY).id)
        tray_id = int(model.body(_TRAY_BODY).id)
        self._peg_geoms = _geoms_for_bodies(model, _subtree_bodies(model, peg_id))
        self._tray_geoms = _geoms_for_bodies(model, _subtree_bodies(model, tray_id))
        self._finger_tip_ids = tuple(
            int(model.body(n).id) for n in _FINGER_TIP_BODIES_RIGHT
        )
        self._ema = float(np.clip(ema, 0.0, 0.99))
        self._min_force_n = float(min_force_n)
        self._seat_force_n = float(seat_force_n)
        self._outlier_gate_m = float(max(outlier_gate_m, 1e-4))
        self._holdover_frames = int(max(holdover_frames, 0))
        self._cluster_radius_m = float(max(cluster_radius_m, 1e-4))
        self._force_pow = float(max(force_pow, 0.5))
        self._use_pca_locus = bool(use_pca_locus)
        self._pca_min_contacts = int(max(pca_min_contacts, 2))
        mode = str(holdover_mode or "freeze_planar").strip().lower()
        if mode not in ("freeze_planar", "wrist_delta", "decay", "offset_only"):
            mode = "freeze_planar"
        self._holdover_mode = mode
        self._holdover_decay = float(np.clip(holdover_decay, 0.0, 1.0))
        self._seat_only_offset = bool(seat_only_offset)
        # v4 default: freeze anchor only after true force-seat (not weak contact).
        self._freeze_require_seat = bool(freeze_require_seat)
        self._tip_hat = np.zeros(3, dtype=np.float64)
        self._offset_w = np.zeros(3, dtype=np.float64)  # tip - wrist (world)
        self._initialized = False
        self._last_force_n = 0.0
        self._last_ncon = 0
        self._lost_streak = 0
        self._reject_streak = 0
        self._wrist_prev: np.ndarray | None = None
        self._last_source = "init"
        self._last_seated_tip = np.zeros(3, dtype=np.float64)
        self._have_seated_tip = False

    def reset(
        self,
        raw,
        *,
        wrist_pos: np.ndarray,
        plane_n: np.ndarray,
        tip_seed: np.ndarray | None = None,
    ) -> ContactTipEstimate:
        """Seed tip_hat from current peg↔tray contacts (preferred) or tip_seed.

        ``tip_seed`` may be used only as fallback when no contact exists at
        spiral start (e.g. already in hole). Prefer contact; if seed is the
        privileged tip, callers must disclose that one-shot seed.
        """
        self._initialized = False
        self._lost_streak = 0
        self._reject_streak = 0
        self._wrist_prev = None
        self._have_seated_tip = False
        self._last_seated_tip = np.zeros(3, dtype=np.float64)
        est = self.update(raw, wrist_pos=wrist_pos, plane_n=plane_n, tip_gt=None)
        if est.contact_n <= 0 or not est.seated:
            if tip_seed is not None:
                tip = np.asarray(tip_seed, dtype=np.float64).reshape(3).copy()
                self._tip_hat = tip
                self._offset_w = tip - np.asarray(wrist_pos, dtype=np.float64).reshape(
                    3
                )
                self._initialized = True
                self._last_source = "seed_fallback"
                est = ContactTipEstimate(
                    tip_hat=tip.copy(),
                    contact_n=0,
                    contact_force_n=0.0,
                    seated=False,
                    source="seed_fallback",
                    meta={"seed_fallback": True},
                )
            else:
                # Last resort: fingertip force-weighted centroid (grasp, not tip).
                tip = self._finger_centroid(raw)
                if tip is None:
                    tip = np.asarray(wrist_pos, dtype=np.float64).reshape(3).copy()
                self._tip_hat = tip
                self._offset_w = tip - np.asarray(wrist_pos, dtype=np.float64).reshape(
                    3
                )
                self._initialized = True
                self._last_source = "finger_centroid_fallback"
                est = ContactTipEstimate(
                    tip_hat=tip.copy(),
                    source="finger_centroid_fallback",
                    meta={"finger_fallback": True},
                )
        else:
            self._initialized = True
        self._wrist_prev = np.asarray(wrist_pos, dtype=np.float64).reshape(3).copy()
        return est

    def _finger_centroid(self, raw) -> np.ndarray | None:
        data = raw._data
        pts = []
        wts = []
        for bid in self._finger_tip_ids:
            f = np.asarray(data.cfrc_ext[bid, :3], dtype=np.float64)
            fn = float(np.linalg.norm(f))
            if fn < self._min_force_n:
                continue
            pts.append(np.asarray(data.xpos[bid], dtype=np.float64).reshape(3))
            wts.append(fn)
        if not pts:
            return None
        w = np.asarray(wts, dtype=np.float64)
        p = np.asarray(pts, dtype=np.float64)
        return (w[:, None] * p).sum(axis=0) / (float(w.sum()) + 1e-12)

    def _select_tip_locus(
        self,
        pts: np.ndarray,
        w: np.ndarray,
        grasp: np.ndarray | None,
        plane_n: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        """Pick tip-like contact among peg↔tray points.

        Primary: force^pow-weighted farthest-from-grasp local cluster.
        Optional PCA free-end blend only when enabled and cloud is elongated.
        """
        n = _unit(plane_n)
        p_plan = pts - n.reshape(1, 3) * (pts @ n).reshape(-1, 1)
        ww = np.power(np.maximum(w, 1e-9), self._force_pow)

        if grasp is None:
            cen = (ww[:, None] * pts).sum(axis=0) / (float(ww.sum()) + 1e-12)
            return cen, "force_centroid"

        g_plan = _planar(grasp, n).reshape(1, 3)
        d = p_plan - g_plan
        dist2 = np.sum(d * d, axis=1)
        score = dist2 * ww
        k = int(np.argmax(score))
        cen0 = pts[k]
        near = np.linalg.norm(pts - cen0.reshape(1, 3), axis=1) <= (
            self._cluster_radius_m
        )
        ww_loc = ww.copy()
        ww_loc[~near] = 0.0
        if float(ww_loc.sum()) < 1e-12:
            ww_loc = ww
        far = (ww_loc[:, None] * pts).sum(axis=0) / (float(ww_loc.sum()) + 1e-12)
        method = "far_grasp_cluster"

        if (
            self._use_pca_locus
            and pts.shape[0] >= self._pca_min_contacts
        ):
            mu = (ww[:, None] * p_plan).sum(axis=0) / (float(ww.sum()) + 1e-12)
            x = p_plan - mu.reshape(1, 3)
            c = (x * ww[:, None]).T @ x / (float(ww.sum()) + 1e-12)
            try:
                evals, evecs = np.linalg.eigh(c)
                order = np.argsort(evals)
                lam_hi = float(evals[order[-1]])
                lam_mid = float(evals[order[-2]]) if evals.size > 1 else 0.0
                if lam_hi > 1e-10 and lam_hi > 3.0 * max(lam_mid, 1e-12):
                    axis = _planar(evecs[:, order[-1]], n)
                    an = float(np.linalg.norm(axis))
                    if an > 1e-9:
                        axis = axis / an
                        g3 = _planar(grasp, n)
                        if float(np.dot(mu - g3, axis)) < 0.0:
                            axis = -axis
                        scores = (p_plan - mu.reshape(1, 3)) @ axis
                        free = scores >= (0.4 * float(np.max(scores)) - 1e-9)
                        if np.any(free):
                            score_f = np.where(free, scores * ww, -1.0)
                            k2 = int(np.argmax(score_f))
                            cen2 = pts[k2]
                            near2 = (
                                np.linalg.norm(pts - cen2.reshape(1, 3), axis=1)
                                <= self._cluster_radius_m
                            ) & free
                            ww2 = ww.copy()
                            ww2[~near2] = 0.0
                            if float(ww2.sum()) > 1e-12:
                                pca_tip = (ww2[:, None] * pts).sum(axis=0) / float(
                                    ww2.sum()
                                )
                                d_pca = float(np.linalg.norm(_planar(pca_tip - grasp, n)))
                                d_far = float(np.linalg.norm(_planar(far - grasp, n)))
                                if d_pca >= d_far - 1e-4:
                                    far = 0.5 * pca_tip + 0.5 * far
                                    method = "pca_blend_far"
            except np.linalg.LinAlgError:
                pass

        return far, method

    def measure_peg_tray_tip(
        self, raw, plane_n: np.ndarray
    ) -> tuple[np.ndarray | None, float, int]:
        """Intermittent extrinsic contact point for theory observers.

        Returns ``(contact_pos, force_sum, ncon)``. Does **not** apply
        holdover / freeze tip_hat — absence means no measurement.
        Residual privilege: MuJoCo peg↔tray ``contact.pos``.
        """
        cen, fsum, ncon, _, _ = self._peg_tray_contacts(raw, plane_n)
        return cen, float(fsum), int(ncon)

    def measure_peg_tray_tip_ext(
        self, raw, plane_n: np.ndarray
    ) -> tuple[np.ndarray | None, float, int, np.ndarray | None]:
        """Like ``measure_peg_tray_tip`` plus force-weighted contact normal.

        Normal is residual-priv extrinsic geometry (sim stand-in for tactile).
        """
        cen, fsum, ncon, n_c, _ = self._peg_tray_contacts(raw, plane_n)
        return cen, float(fsum), int(ncon), n_c

    def finger_grasp_centroid(self, raw) -> np.ndarray | None:
        """Force-weighted right fingertip centroid (proprio/FK eligible)."""
        return self._finger_centroid(raw)

    def _peg_tray_contacts(
        self, raw, plane_n: np.ndarray
    ) -> tuple[np.ndarray | None, float, int, np.ndarray | None, str]:
        """Return (tip-like contact point, force_sum, ncon, normal, method).

        Peg↔tray only. Prefer free-end / far-from-grasp locus.
        """
        model = raw._model
        data = raw._data
        pts: list[np.ndarray] = []
        wts: list[float] = []
        normals: list[np.ndarray] = []
        force = np.zeros(6, dtype=np.float64)
        for i in range(int(data.ncon)):
            con = data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            pair = {g1, g2}
            if not (self._peg_geoms & pair and self._tray_geoms & pair):
                continue
            mujoco.mj_contactForce(model, data, i, force)
            fn = float(np.linalg.norm(force[:3]))
            if fn < self._min_force_n * 0.25:
                continue
            pos = np.asarray(con.pos, dtype=np.float64).reshape(3).copy()
            frame = np.asarray(con.frame, dtype=np.float64).reshape(3, 3)
            n_c = frame[:, 0].copy()
            pts.append(pos)
            wts.append(max(fn, 1e-9))
            normals.append(n_c)
        if not pts:
            return None, 0.0, 0, None, "none"
        w = np.asarray(wts, dtype=np.float64)
        p = np.asarray(pts, dtype=np.float64)
        grasp = self._finger_centroid(raw)
        cen, method = self._select_tip_locus(p, w, grasp, plane_n)
        n_w = (w[:, None] * np.asarray(normals, dtype=np.float64)).sum(axis=0)
        n_w = _unit(n_w) if float(np.linalg.norm(n_w)) > 1e-12 else None
        return cen, float(w.sum()), int(len(pts)), n_w, method

    def _apply_holdover(
        self,
        wrist: np.ndarray,
        n: np.ndarray,
        wrist_delta: np.ndarray | None,
    ) -> np.ndarray:
        """Propagate tip_hat without peg xpos when extrinsic contact is lost.

        Non-rigid grasp ⇒ wristΔ ≠ tipΔ. Modes:
        - ``freeze_planar`` (default): after a seated measure, freeze last seated
          planar tip (TEC / extrinsic: tip is a contact observation). If **never**
          seated, fall back to ``offset_only`` so tip_hat is not stuck on a bad
          finger-centroid init (smoke_v3 ep03 pathology).
        - ``decay``: decaying planar wristΔ.
        - ``wrist_delta``: legacy rigid push (smoke_v2 regression).
        - ``offset_only``: tip ≈ wrist + last offset (no incremental Δ).
        """
        mode = self._holdover_mode
        along0 = float(np.dot(self._tip_hat, n))

        if mode == "freeze_planar":
            if self._have_seated_tip:
                tip_hold = self._last_seated_tip.copy()
            else:
                # Never seated: do not freeze at init; weak wrist+offset bridge.
                tip_hold = wrist + self._offset_w
        elif mode == "offset_only":
            tip_hold = wrist + self._offset_w
        elif mode == "decay":
            gain = self._holdover_decay ** float(max(self._lost_streak, 1))
            if wrist_delta is not None:
                d = _planar(np.asarray(wrist_delta, dtype=np.float64).reshape(3), n)
            elif self._wrist_prev is not None:
                d = _planar(wrist - self._wrist_prev, n)
            else:
                d = np.zeros(3, dtype=np.float64)
            tip_hold = self._tip_hat + gain * d
        else:  # wrist_delta (legacy)
            if wrist_delta is not None:
                tip_hold = self._tip_hat + _planar(
                    np.asarray(wrist_delta, dtype=np.float64).reshape(3), n
                )
            elif self._wrist_prev is not None:
                tip_hold = self._tip_hat + _planar(wrist - self._wrist_prev, n)
            else:
                tip_hold = wrist + self._offset_w

        # Keep along-normal from previous tip_hat (seat height / freeze Z).
        tip_hold = tip_hold - n * float(np.dot(tip_hold, n)) + n * along0
        return tip_hold

    def update(
        self,
        raw,
        *,
        wrist_pos: np.ndarray,
        plane_n: np.ndarray,
        tip_gt: np.ndarray | None = None,
        wrist_delta: np.ndarray | None = None,
    ) -> ContactTipEstimate:
        wrist = np.asarray(wrist_pos, dtype=np.float64).reshape(3)
        n = _unit(plane_n)
        if wrist_delta is None and self._wrist_prev is not None:
            wrist_delta = wrist - self._wrist_prev

        cen, fsum, ncon, n_c, pick_method = self._peg_tray_contacts(raw, n)
        self._last_force_n = float(fsum)
        self._last_ncon = int(ncon)
        seated = bool(ncon > 0 and fsum >= self._seat_force_n)

        finger_f = 0.0
        data = raw._data
        for bid in self._finger_tip_ids:
            finger_f += float(np.linalg.norm(data.cfrc_ext[bid, :3]))

        source = "hold"
        rejected = False
        # First true force-seat: hard-snap tip_hat (leave finger/seed init).
        first_strong_seat = bool(seated and not self._have_seated_tip)
        if cen is not None and ncon > 0:
            tip_meas = cen.copy()
            # Keep height consistent with plane through previous tip_hat if any.
            if self._initialized and not first_strong_seat:
                along0 = float(np.dot(self._tip_hat, n))
                tip_meas = tip_meas - n * float(np.dot(tip_meas, n)) + n * along0
                # Planar outlier reject: large jumps from mid-shaft / wrong cluster.
                jump = float(np.linalg.norm(_planar(tip_meas - self._tip_hat, n)))
                gate = self._outlier_gate_m
                if fsum < self._seat_force_n:
                    gate *= 0.75
                # Strong force + seated: trust more (wider gate).
                if fsum >= 2.0 * self._seat_force_n:
                    gate *= 1.35
                # After several rejects, force-accept to avoid permanent freeze.
                if jump > gate and self._reject_streak < 5:
                    rejected = True
                    self._reject_streak += 1
            if not rejected:
                self._reject_streak = 0
                self._lost_streak = 0
                if first_strong_seat:
                    # Init from first strong peg↔tray seat (bypass EMA / gate).
                    tip_new = tip_meas
                    source = f"peg_tray_first_seat_{pick_method}"
                elif self._initialized and self._ema > 0.0:
                    ema = self._ema
                    if fsum >= 2.0 * self._seat_force_n:
                        ema = max(0.30, self._ema - 0.20)
                    tip_new = (1.0 - ema) * tip_meas + ema * self._tip_hat
                    source = (
                        f"peg_tray_{pick_method}"
                        if seated
                        else f"peg_tray_weak_{pick_method}"
                    )
                else:
                    tip_new = tip_meas
                    source = (
                        f"peg_tray_{pick_method}"
                        if seated
                        else f"peg_tray_weak_{pick_method}"
                    )
                # Offset refresh: seated (force gate) only when seat_only_offset.
                refresh_off = (not self._seat_only_offset) or seated
                if refresh_off:
                    off = tip_new - wrist
                    self._offset_w = _planar(off, n) + n * float(np.dot(off, n))
                self._tip_hat = tip_new
                self._initialized = True
                # v4: freeze anchor only on true force-seat (gated freeze).
                # Legacy (freeze_require_seat=false): any accepted measure anchors.
                if seated or not self._freeze_require_seat:
                    self._last_seated_tip = tip_new.copy()
                    self._have_seated_tip = True
            else:
                # Rejected outlier: freeze / soft holdover (no junk pull).
                tip_hold = self._apply_holdover(wrist, n, wrist_delta)
                if (
                    self._holdover_mode != "freeze_planar"
                    and self._initialized
                    and self._ema > 0.0
                ):
                    self._tip_hat = (
                        (1.0 - self._ema) * tip_hold + self._ema * self._tip_hat
                    )
                else:
                    self._tip_hat = tip_hold
                    self._initialized = True
                source = (
                    "outlier_freeze"
                    if self._holdover_mode == "freeze_planar"
                    else "outlier_hold"
                )
        else:
            self._lost_streak += 1
            tip_hold = self._apply_holdover(wrist, n, wrist_delta)
            freeze_seated = (
                self._holdover_mode == "freeze_planar" and self._have_seated_tip
            )
            # After seated contact: keep freeze (no wristΔ push). Before seat:
            # short hard holdover then soft blend toward wrist+offset bridge.
            if freeze_seated or self._lost_streak <= self._holdover_frames:
                self._tip_hat = tip_hold
                if freeze_seated:
                    source = "lost_freeze"
                elif self._holdover_mode == "freeze_planar":
                    source = "lost_offset_bridge"
                else:
                    source = "lost_holdover"
            elif self._initialized and self._ema > 0.0:
                self._tip_hat = (1.0 - self._ema) * tip_hold + self._ema * self._tip_hat
                source = "wrist_hold"
            else:
                self._tip_hat = tip_hold
                self._initialized = True
                source = "wrist_hold"

        self._last_source = source
        self._wrist_prev = wrist.copy()
        tip_hat = self._tip_hat.copy()
        err = float("nan")
        if tip_gt is not None:
            err = float(
                np.linalg.norm(
                    _planar(
                        tip_hat - np.asarray(tip_gt, dtype=np.float64).reshape(3), n
                    )
                )
            )
        # Log tip_est error even when not seated (float/hold) for diagnosis.
        return ContactTipEstimate(
            tip_hat=tip_hat,
            contact_n=int(ncon),
            contact_force_n=float(fsum),
            seated=bool(seated),
            source=source,
            tip_err_to_gt_m=err,
            finger_grasp_force_n=float(finger_f),
            meta={
                "tip_est_source": source,
                "tip_est_ncon": int(ncon),
                "tip_est_force_n": float(fsum),
                "tip_est_seated": bool(seated),
                "tip_est_pick": pick_method if cen is not None else "none",
                "tip_est_outlier_reject": bool(rejected),
                "tip_est_lost_streak": int(self._lost_streak),
                "tip_est_holdover_mode": str(self._holdover_mode),
                "tip_est_contact_normal": None if n_c is None else n_c.tolist(),
                "tip_est_uses_peg_xpos": False,
                "tip_est_residual_privilege": "mujoco_peg_tray_contact_pos",
            },
        )
