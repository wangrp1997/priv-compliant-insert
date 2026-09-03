"""Track-A: Kim TEC joint estimation–control layered on GMHQP (general).

REUSE (METHOD_GATE + GENERAL_ALGO_MANDATE):
  Kim ICRA 2023 Simultaneous Tactile Estimation and Control of Extrinsic Contact
  (`refs/Tactile-Estimator-Controller`) + Active Extrinsic (ICRA 2022)
  + existing GMHQP tip-referenced Escande+C⁺.

General law (no episode indices):
  Estimate extrinsic tip/contact under non-rigid grasp; feed ˆt into GMHQP tip
  task only when contact mode is information-consistent (NIS + PoseDiff +
  contact meas); otherwise tip_obj = in-hand geom tip (GMHQP default).

Prior free-run tip_obs/tip_fuse ignored TEC “control when estimable” → drift
became command. Gate is theory (Kalman NIS / contact observability), not
keep-set or fail-subset structure.

Design: docs/TRACK_A_TEC_JOINT_ON_GMHQP.md
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pci.tip_theory_estimator import ExtrinsicTipStateEKF, TheoryEstConfig, TheoryTipEstimate


# χ² critical value, 2 DoF, 95% — standard Kalman consistency gate (Bar-Shalom).
NIS_CHI2_95_2DOF = 5.991


def _unit(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(x))
    return x / n if n > 1e-12 else x


def _planar(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    n = _unit(normal)
    x = np.asarray(v, dtype=np.float64).reshape(3)
    return x - n * float(np.dot(x, n))


@dataclass
class TecJointConfig:
    """Information / observability gates — fixed from filter theory, not retuned."""

    # Kalman NIS accept (χ² 95%, 2-DoF).
    nis_chi2_accept: float = NIS_CHI2_95_2DOF
    # Require intermittent extrinsic contact this step (ContactMotion observability).
    require_contact: bool = True
    # Reject if filter just inflated C (slip / mode break).
    reject_on_slip: bool = True
    # PoseDiff 3σ from meas_geom_tip_var.
    geom_disagree_sigma: float = 3.0
    # Uncert 3σ from meas_geom_tip_var.
    uncert_sigma: float = 3.0
    # Hard gate: α∈{0,1} — estimable → ˆt, else geom (TEC control-when-estimable).
    hard_gate: bool = True


@dataclass
class TecJointState:
    frames: int = 0
    gated_on_frames: int = 0
    last_use_tip_hat: bool = False
    last_reason: str = "init"
    last_nis: float = float("nan")
    last_geom_disagree_m: float = float("nan")
    meta: dict = field(default_factory=dict)


def is_tec_joint_mode(mode: str) -> bool:
    m = str(mode).strip().lower()
    return m in (
        "track_a_tec_joint_gmhqp",
        "tec_joint_gmhqp",
        "tec_joint",
        "track_a_tec_joint",
        "gmhqp_tec_joint",
    )


def derived_geom_disagree_m(cfg: TheoryEstConfig, sigma: float = 3.0) -> float:
    """3σ PoseDiff consistency radius from geom measurement variance."""
    return float(sigma) * float(np.sqrt(2.0 * max(float(cfg.meas_geom_tip_var), 1e-18)))


def derived_uncert_max_m(cfg: TheoryEstConfig, sigma: float = 3.0) -> float:
    """3σ uncert bound from geom prior variance."""
    return float(sigma) * float(np.sqrt(2.0 * max(float(cfg.meas_geom_tip_var), 1e-18)))


def select_tip_for_gmhqp(
    *,
    tip_hat: np.ndarray,
    tip_geom: np.ndarray,
    plane_n: np.ndarray,
    est: TheoryTipEstimate,
    ekf: ExtrinsicTipStateEKF,
    joint_cfg: TecJointConfig | None = None,
    state: TecJointState | None = None,
) -> tuple[np.ndarray, TecJointState]:
    """General tip argument for GMHQP: ˆt iff TEC consistency; else geom tip.

    Observability / consistency (theory, any episode):
      1. contact measurement this step (extrinsic mode estimable)
      2. contact NIS ≤ χ²_{0.95,2}
      3. ‖Π(ˆt − geom)‖ ≤ 3σ(PoseDiff R)
      4. filter uncert ≤ 3σ(PoseDiff R)
      5. not just_slipped
    """
    joint_cfg = joint_cfg or TecJointConfig()
    state = state or TecJointState()
    state.frames += 1
    th = np.asarray(tip_hat, dtype=np.float64).reshape(3)
    tg = np.asarray(tip_geom, dtype=np.float64).reshape(3)
    n = _unit(plane_n)
    geom_disagree = float(np.linalg.norm(_planar(th - tg, n)))
    state.last_geom_disagree_m = geom_disagree
    nis = float(ekf.last_contact_nis)
    state.last_nis = nis

    max_geom = derived_geom_disagree_m(ekf.cfg, joint_cfg.geom_disagree_sigma)
    max_uncert = derived_uncert_max_m(ekf.cfg, joint_cfg.uncert_sigma)
    uncert = float(est.uncert_m) if np.isfinite(est.uncert_m) else 1e9

    reason = "ok"
    use = True
    if joint_cfg.require_contact and not bool(ekf.had_contact_meas):
        use = False
        reason = "no_contact"
    elif joint_cfg.reject_on_slip and bool(ekf.just_slipped):
        use = False
        reason = "slip"
    elif not np.isfinite(nis) or nis > float(joint_cfg.nis_chi2_accept):
        use = False
        reason = "nis_reject"
    elif geom_disagree > max_geom:
        use = False
        reason = "geom_disagree"
    elif uncert > max_uncert:
        use = False
        reason = "uncert"

    state.last_use_tip_hat = bool(use)
    state.last_reason = reason
    if use:
        state.gated_on_frames += 1
        tip_out = th.copy()
    else:
        tip_out = tg.copy()

    state.meta = {
        "tec_joint_use_tip_hat": bool(use),
        "tec_joint_reason": reason,
        "tec_joint_nis": float(nis) if np.isfinite(nis) else -1.0,
        "tec_joint_nis_lim": float(joint_cfg.nis_chi2_accept),
        "tec_joint_geom_disagree_mm": geom_disagree * 1000.0,
        "tec_joint_geom_lim_mm": max_geom * 1000.0,
        "tec_joint_uncert_mm": uncert * 1000.0 if np.isfinite(uncert) else -1.0,
        "tec_joint_uncert_lim_mm": max_uncert * 1000.0,
        "tec_joint_gated_on_frac": (
            float(state.gated_on_frames) / float(max(state.frames, 1))
        ),
        "tec_joint_cite": "Kim_TEC_ICRA2023+Active_Extrinsic+GMHQP",
    }
    return tip_out, state
