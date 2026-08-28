"""Pre-PBVS Gaussian noise on wrist (legacy); prefer pbvs_socket_bias."""

from pci.approach_noise import ApproachNoiseConfig, apply_approach_noise

__all__ = ["ApproachNoiseConfig", "apply_pre_pbvs_noise"]


def apply_pre_pbvs_noise(action44, raw, *, cfg: ApproachNoiseConfig, rng=None):
    return apply_approach_noise(action44, raw, cfg=cfg, rng=rng)
