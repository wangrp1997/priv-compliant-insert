"""Phase A → Phase B orchestration (Phase B: sensor-only)."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from pci.compliant.fingers import FingerCompliantController, FingerCompliantConfig
from pci.compliant.insert import CompliantInsertController, CompliantInsertConfig, InsertStepResult
from pci.compliant.search import CompliantSearchController, CompliantSearchConfig, SearchStepResult
from pci.task_frame import TaskFrame


class PipelinePhase(Enum):
    APPROACH = auto()
    COMPLIANT_SEARCH = auto()
    COMPLIANT_INSERT = auto()
    RELEASE = auto()
    DONE = auto()


@dataclass(frozen=True, slots=True)
class PipelineStepResult:
    phase: PipelinePhase
    done: bool
    success: bool
    reason: str
    delta_xyz: np.ndarray
    delta_hand16: np.ndarray
    finger_open: float = 0.0


@dataclass
class PipelineConfig:
    max_jam_recoveries: int = 5
    default_dt: float = 1.0 / 30.0


class InsertPipeline:
    """Hybrid ALIGN (privileged, external) → sensor-only B1/B2 + finger compliance."""

    def __init__(
        self,
        *,
        search_config: CompliantSearchConfig | None = None,
        insert_config: CompliantInsertConfig | None = None,
        finger_config: FingerCompliantConfig | None = None,
        pipeline_config: PipelineConfig | None = None,
    ) -> None:
        self.search = CompliantSearchController(search_config)
        self.insert = CompliantInsertController(insert_config)
        self.fingers = FingerCompliantController(finger_config)
        self.pipeline_config = pipeline_config or PipelineConfig()
        self._frame: TaskFrame | None = None
        self._hold_hand16: np.ndarray | None = None
        self._prev_delta_xyz = np.zeros(3, dtype=np.float64)
        self._dt = self.pipeline_config.default_dt
        self._jam_recoveries = 0
        self.phase = PipelinePhase.APPROACH

    def reset(self) -> None:
        self.phase = PipelinePhase.APPROACH
        self._frame = None
        self._hold_hand16 = None
        self._prev_delta_xyz = np.zeros(3, dtype=np.float64)
        self._dt = self.pipeline_config.default_dt
        self._jam_recoveries = 0
        self.search.reset_steps_only()
        self.insert._steps = 0
        self.insert._stall = 0
        self.insert._releasing = False

    def begin_compliant(
        self,
        frame: TaskFrame,
        wrench_right6: np.ndarray,
        finger_force12: np.ndarray,
        hold_hand16: np.ndarray,
        wrist_xyz: np.ndarray,
        *,
        along_at_b_m: float | None = None,
    ) -> None:
        self.phase = PipelinePhase.COMPLIANT_SEARCH
        self._frame = frame
        self._hold_hand16 = np.asarray(hold_hand16, dtype=np.float64).reshape(16).copy()
        self._prev_delta_xyz = np.zeros(3, dtype=np.float64)
        self._jam_recoveries = 0
        self.search.reset(
            frame,
            wrench_right6,
            wrist_xyz=wrist_xyz,
            along_at_b_m=along_at_b_m,
        )
        self.insert.reset(frame, wrench_right6, wrist_xyz)
        self.fingers.reset(finger_force12)

    def _call_controller_step(
        self,
        controller: Any,
        frame: TaskFrame,
        wrench_right6: np.ndarray,
        wrist_xyz: np.ndarray,
        *,
        extra: dict[str, Any] | None = None,
    ) -> Any:
        """Invoke sub-controller step; pass prev_delta/dt only if signature accepts them."""
        payload = {
            "frame": frame,
            "wrench_right6": wrench_right6,
            "wrist_xyz": wrist_xyz,
            "prev_delta_xyz": self._prev_delta_xyz,
            "dt": self._dt,
            **(extra or {}),
        }
        sig = inspect.signature(controller.step)
        kwargs = {k: v for k, v in payload.items() if k in sig.parameters}
        return controller.step(**kwargs)

    def _recover_to_search(
        self,
        wrench_right6: np.ndarray,
        wrist_xyz: np.ndarray,
    ) -> None:
        assert self._frame is not None
        self.search.recover_from_jam(self._frame, wrench_right6, wrist_xyz)
        self.insert.reset(self._frame, wrench_right6, wrist_xyz)
        self.phase = PipelinePhase.COMPLIANT_SEARCH

    def _remember_delta(self, delta_xyz: np.ndarray) -> None:
        self._prev_delta_xyz = np.asarray(delta_xyz, dtype=np.float64).reshape(3).copy()

    def step(
        self,
        wrench_right6: np.ndarray,
        wrist_xyz: np.ndarray,
        finger_force12: np.ndarray,
        *,
        insert_ok_eval: bool,
        dt: float | None = None,
    ) -> PipelineStepResult:
        if self._frame is None or self._hold_hand16 is None:
            raise RuntimeError("call begin_compliant() after hybrid ALIGN")

        if dt is not None:
            self._dt = float(dt)

        frame = self._frame
        hold = self._hold_hand16
        delta_hand = self.fingers.step(finger_force12, hold)

        if self.phase == PipelinePhase.COMPLIANT_SEARCH:
            sr: SearchStepResult = self._call_controller_step(
                self.search, frame, wrench_right6, wrist_xyz
            )
            self._remember_delta(sr.delta_xyz)
            if sr.done:
                if sr.success:
                    self.phase = PipelinePhase.COMPLIANT_INSERT
                    self.insert.reset(frame, wrench_right6, wrist_xyz)
                    self._prev_delta_xyz = np.zeros(3, dtype=np.float64)
                    return PipelineStepResult(
                        self.phase, False, False, sr.reason, np.zeros(3), delta_hand
                    )
                self.phase = PipelinePhase.DONE
                return PipelineStepResult(
                    PipelinePhase.DONE, True, False, sr.reason, sr.delta_xyz, delta_hand
                )
            return PipelineStepResult(
                PipelinePhase.COMPLIANT_SEARCH, False, False, sr.reason, sr.delta_xyz, delta_hand
            )

        if self.phase in (PipelinePhase.COMPLIANT_INSERT, PipelinePhase.RELEASE):
            ir: InsertStepResult = self._call_controller_step(
                self.insert,
                frame,
                wrench_right6,
                wrist_xyz,
                extra={"release_trigger": insert_ok_eval},
            )
            phase = (
                PipelinePhase.RELEASE
                if ir.phase == "release"
                else PipelinePhase.COMPLIANT_INSERT
            )
            finger_open = ir.finger_open
            if finger_open > 0.0:
                delta_hand = np.zeros(16, dtype=np.float64)

            if not ir.done and ir.reason == "retreat":
                self._remember_delta(ir.delta_xyz)
                if self._jam_recoveries < self.pipeline_config.max_jam_recoveries:
                    self._jam_recoveries += 1
                    self._recover_to_search(wrench_right6, wrist_xyz)
                    return PipelineStepResult(
                        PipelinePhase.COMPLIANT_SEARCH,
                        False,
                        False,
                        f"jam_recover_{self._jam_recoveries}",
                        ir.delta_xyz,
                        delta_hand,
                    )
                self.phase = PipelinePhase.DONE
                return PipelineStepResult(
                    PipelinePhase.DONE,
                    True,
                    False,
                    "jam_recovery_exhausted",
                    ir.delta_xyz,
                    delta_hand,
                )

            self._remember_delta(ir.delta_xyz)
            self.phase = PipelinePhase.DONE if ir.done else phase
            return PipelineStepResult(
                phase,
                ir.done,
                ir.success,
                ir.reason,
                ir.delta_xyz,
                delta_hand,
                finger_open=finger_open,
            )

        return PipelineStepResult(
            PipelinePhase.DONE, True, False, "already_done", np.zeros(3), delta_hand
        )
