"""One-time privileged misalignment at Phase A→B (sim setup only, not in Phase B loop)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from pci.features import features_from_raw
from pci.wrist import hole_task_basis


@dataclass(frozen=True, slots=True)
class ApproachNoiseConfig:
    lat_std_m: float = 0.004
    along_std_m: float = 0.002
    rot_std_rad: float = 0.025
    seed: int | None = None


def apply_approach_noise(
    action44: np.ndarray,
    raw,
    *,
    cfg: ApproachNoiseConfig,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Perturb right wrist in hole plane so Phase B does not start perfectly centered."""
    out = np.asarray(action44, dtype=np.float64).reshape(44).copy()
    feat = features_from_raw(raw)
    t1, t2, axis = hole_task_basis(feat.hole_axis)
    gen = rng if rng is not None else np.random.default_rng(cfg.seed)

    d_lat1 = float(gen.normal(0.0, cfg.lat_std_m))
    d_lat2 = float(gen.normal(0.0, cfg.lat_std_m))
    d_along = float(gen.normal(0.0, cfg.along_std_m))
    delta = t1 * d_lat1 + t2 * d_lat2 + axis * d_along
    out[0:3] = out[0:3] + delta

    r = R.from_rotvec(out[3:6])
    drot = (
        R.from_rotvec(t1 * float(gen.normal(0.0, cfg.rot_std_rad)))
        * R.from_rotvec(t2 * float(gen.normal(0.0, cfg.rot_std_rad)))
    )
    out[3:6] = (drot * r).as_rotvec()

    meta = {
        "noise_lat1_m": d_lat1,
        "noise_lat2_m": d_lat2,
        "noise_along_m": d_along,
    }
    return out, meta
