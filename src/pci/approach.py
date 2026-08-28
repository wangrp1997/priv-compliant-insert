"""Phase A: hybrid_insert ALIGN-only wrapper (no snap / no priv_snap_insert)."""

from __future__ import annotations

from dataclasses import dataclass

from pci.features import InsertFeatures, approach_gate_ok


@dataclass
class ApproachConfig:
    """Gate thresholds; motion from HybridInsertController ALIGN."""

    standoff_lo_m: float = 0.040
    standoff_hi_m: float = 0.070
    lat_gate_m: float = 0.008
    ang_gate_rad: float = 0.14
    max_steps: int = 1200


@dataclass(frozen=True, slots=True)
class ApproachStatus:
    done: bool
    success: bool
    steps: int
    last: InsertFeatures | None
    reason: str


class ApproachController:
    """Phase A state; commands come from hybrid_insert HybridInsertController ALIGN."""

    def __init__(self, config: ApproachConfig | None = None) -> None:
        self.config = config or ApproachConfig()
        self._steps = 0
        # P0: self._hybrid: HybridInsertController | None = None

    def reset(self) -> None:
        self._steps = 0

    def observe(self, feat: InsertFeatures) -> ApproachStatus:
        self._steps += 1
        cfg = self.config
        if approach_gate_ok(
            feat,
            lat_gate_m=cfg.lat_gate_m,
            ang_gate_rad=cfg.ang_gate_rad,
            standoff_lo_m=cfg.standoff_lo_m,
            standoff_hi_m=cfg.standoff_hi_m,
        ):
            return ApproachStatus(
                done=True,
                success=True,
                steps=self._steps,
                last=feat,
                reason="gate_ok",
            )
        if self._steps >= cfg.max_steps:
            return ApproachStatus(
                done=True,
                success=False,
                steps=self._steps,
                last=feat,
                reason="timeout",
            )
        return ApproachStatus(
            done=False,
            success=False,
            steps=self._steps,
            last=feat,
            reason="approaching",
        )
