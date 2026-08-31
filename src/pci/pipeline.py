"""Phase A → Phase B orchestration (Phase B: sensor-only)."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from pci.compliant.fingers import FingerCompliantController, FingerCompliantConfig
from pci.compliant.insert import CompliantInsertController, CompliantInsertConfig, InsertStepResult
from pci.compliant.priv_grasp_opt import PrivGraspOptController, PrivGraspOptConfig
from pci.compliant.search import CompliantSearchController, CompliantSearchConfig, SearchStepResult
from pci.priv_geom import PrivGraspGeom
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
    delta_left_xyz: np.ndarray | None = None
    delta_left_hand16: np.ndarray | None = None
    priv_rel_rot_err_rad: float = 0.0


@dataclass
class PipelineConfig:
    max_jam_recoveries: int = 5
    default_dt: float = 1.0 / 30.0


class InsertPipeline:
    """Hybrid ALIGN → B1/B2.

    Default control: wrist F/T + fingertip forces (deployable).
    Optional ``priv_grasp_opt`` is research-only and off by default.
    """

    def __init__(
        self,
        *,
        search_config: CompliantSearchConfig | None = None,
        insert_config: CompliantInsertConfig | None = None,
        finger_config: FingerCompliantConfig | None = None,
        pipeline_config: PipelineConfig | None = None,
        priv_grasp_config: PrivGraspOptConfig | None = None,
    ) -> None:
        self.search = CompliantSearchController(search_config)
        self.insert = CompliantInsertController(insert_config)
        fcfg = finger_config or FingerCompliantConfig()
        self.fingers = FingerCompliantController(fcfg)
        # Left uses same tactile admittance gains (separate state).
        self.fingers_left = FingerCompliantController(fcfg)
        self.priv_grasp = PrivGraspOptController(priv_grasp_config)
        self.pipeline_config = pipeline_config or PipelineConfig()
        self._frame: TaskFrame | None = None
        self._hold_hand16: np.ndarray | None = None
        self._hold_left_hand16: np.ndarray | None = None
        self._prev_delta_xyz = np.zeros(3, dtype=np.float64)
        self._dt = self.pipeline_config.default_dt
        self._jam_recoveries = 0
        self.phase = PipelinePhase.APPROACH
        self._use_priv_grasp = bool(
            (priv_grasp_config or PrivGraspOptConfig()).enable
        )

    def reset(self) -> None:
        self.phase = PipelinePhase.APPROACH
        self._frame = None
        self._hold_hand16 = None
        self._hold_left_hand16 = None
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
        already_on_surface: bool = False,
        hold_left_hand16: np.ndarray | None = None,
        priv_geom: PrivGraspGeom | None = None,
        left_finger_force12: np.ndarray | None = None,
    ) -> None:
        self.phase = PipelinePhase.COMPLIANT_SEARCH
        self._frame = frame
        self._hold_hand16 = np.asarray(hold_hand16, dtype=np.float64).reshape(16).copy()
        if hold_left_hand16 is not None:
            self._hold_left_hand16 = np.asarray(hold_left_hand16, dtype=np.float64).reshape(16).copy()
        else:
            self._hold_left_hand16 = None
        self._prev_delta_xyz = np.zeros(3, dtype=np.float64)
        self._jam_recoveries = 0
        self.search.reset(
            frame,
            wrench_right6,
            wrist_xyz=wrist_xyz,
            already_on_surface=already_on_surface,
        )
        self.insert.reset(frame, wrench_right6, wrist_xyz)
        self.fingers.reset(finger_force12)
        if left_finger_force12 is not None:
            self.fingers_left.reset(left_finger_force12)
        if self._use_priv_grasp and priv_geom is not None and self._hold_left_hand16 is not None:
            self.priv_grasp.reset(priv_geom, self._hold_hand16, self._hold_left_hand16)
        _ = left_finger_force12

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
        priv_lat_m: float | None = None,
        priv_along_m: float | None = None,
        priv_lat_vec: np.ndarray | None = None,
        priv_geom: PrivGraspGeom | None = None,
        left_finger_force12: np.ndarray | None = None,
    ) -> PipelineStepResult:
        if self._frame is None or self._hold_hand16 is None:
            raise RuntimeError("call begin_compliant() after hybrid ALIGN")

        if dt is not None:
            self._dt = float(dt)

        frame = self._frame
        hold = self._hold_hand16
        delta_left_hand: np.ndarray | None = None
        rel_err = 0.0

        if (
            self._use_priv_grasp
            and priv_geom is not None
            and left_finger_force12 is not None
            and self.phase
            in (PipelinePhase.COMPLIANT_SEARCH, PipelinePhase.COMPLIANT_INSERT)
        ):
            po = self.priv_grasp.step(priv_geom, finger_force12, left_finger_force12)
            delta_hand = po.delta_right_hand16
            delta_left_hand = po.delta_left_hand16
            rel_err = float(po.rel_rot_err_rad)
        else:
            # Deployable path: dual fingertip force admittance (no object pose).
            delta_hand = self.fingers.step(finger_force12, hold)
            if self._hold_left_hand16 is not None and left_finger_force12 is not None:
                delta_left_hand = self.fingers_left.step(
                    left_finger_force12, self._hold_left_hand16
                )
            else:
                delta_left_hand = None

        if self.phase == PipelinePhase.COMPLIANT_SEARCH:
            sr: SearchStepResult = self._call_controller_step(
                self.search,
                frame,
                wrench_right6,
                wrist_xyz,
                extra={
                    "priv_lat_m": priv_lat_m,
                    "priv_along_m": priv_along_m,
                    "priv_lat_vec": priv_lat_vec,
                },
            )
            self._remember_delta(sr.delta_xyz)
            if sr.done:
                if sr.success:
                    self.phase = PipelinePhase.COMPLIANT_INSERT
                    self.insert.reset(frame, wrench_right6, wrist_xyz)
                    self._prev_delta_xyz = np.zeros(3, dtype=np.float64)
                    return PipelineStepResult(
                        self.phase,
                        False,
                        False,
                        sr.reason,
                        np.zeros(3),
                        delta_hand,
                        delta_left_hand16=delta_left_hand,
                        priv_rel_rot_err_rad=rel_err,
                    )
                self.phase = PipelinePhase.DONE
                return PipelineStepResult(
                    PipelinePhase.DONE,
                    True,
                    False,
                    sr.reason,
                    sr.delta_xyz,
                    delta_hand,
                    delta_left_hand16=delta_left_hand,
                    priv_rel_rot_err_rad=rel_err,
                )
            return PipelineStepResult(
                PipelinePhase.COMPLIANT_SEARCH,
                False,
                False,
                sr.reason,
                sr.delta_xyz,
                delta_hand,
                delta_left_xyz=getattr(sr, "delta_left_xyz", None),
                delta_left_hand16=delta_left_hand,
                priv_rel_rot_err_rad=rel_err,
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
                delta_left_hand = (
                    np.zeros(16, dtype=np.float64) if delta_left_hand is not None else None
                )

            if not ir.done and ir.reason == "retreat":
                self._remember_delta(ir.delta_xyz)
                near = priv_lat_m is not None and float(priv_lat_m) <= 0.010
                if near:
                    return PipelineStepResult(
                        PipelinePhase.COMPLIANT_INSERT,
                        False,
                        False,
                        "soft_retreat_near_hole",
                        ir.delta_xyz,
                        delta_hand,
                        delta_left_xyz=getattr(ir, "delta_left_xyz", None),
                        delta_left_hand16=delta_left_hand,
                        priv_rel_rot_err_rad=rel_err,
                    )
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
                        delta_left_xyz=getattr(ir, "delta_left_xyz", None),
                        delta_left_hand16=delta_left_hand,
                        priv_rel_rot_err_rad=rel_err,
                    )
                self.phase = PipelinePhase.DONE
                return PipelineStepResult(
                    PipelinePhase.DONE,
                    True,
                    False,
                    "jam_recovery_exhausted",
                    ir.delta_xyz,
                    delta_hand,
                    delta_left_xyz=getattr(ir, "delta_left_xyz", None),
                    delta_left_hand16=delta_left_hand,
                    priv_rel_rot_err_rad=rel_err,
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
                delta_left_xyz=getattr(ir, "delta_left_xyz", None),
                delta_left_hand16=delta_left_hand,
                priv_rel_rot_err_rad=rel_err,
            )

        return PipelineStepResult(
            PipelinePhase.DONE,
            True,
            False,
            "already_done",
            np.zeros(3),
            delta_hand,
            delta_left_hand16=delta_left_hand,
            priv_rel_rot_err_rad=rel_err,
        )
