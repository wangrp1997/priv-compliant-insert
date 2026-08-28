"""Ego mp4 recording for PCI sim smoke / eval."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


class EgoVideoRecorder:
    def __init__(self, path: Path, *, fps: int = 30) -> None:
        self.path = path
        self.fps = fps
        self._frames: list[np.ndarray] = []

    def write_rgb(self, frame: np.ndarray) -> None:
        rgb = np.asarray(frame)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if rgb.ndim == 3 and rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))
        if rgb.shape[:2] != (640, 640):
            rgb = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        self._frames.append(np.ascontiguousarray(rgb))

    def close(self) -> None:
        if not self._frames:
            self.path.unlink(missing_ok=True)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        import imageio.v3 as iio

        iio.imwrite(
            self.path,
            np.stack(self._frames, axis=0),
            fps=self.fps,
            codec="libx264",
            plugin="pyav",
        )
        self._frames.clear()


def ego_from_gym_obs(obs: dict[str, Any]) -> np.ndarray | None:
    images = obs.get("images")
    if isinstance(images, dict):
        for key in ("ego", "random_camera"):
            if key in images:
                return np.asarray(images[key])
    if "ego" in obs:
        return np.asarray(obs["ego"])
    return None


def make_gym_ego_video_cb(recorder: EgoVideoRecorder | None) -> Callable[[dict[str, Any]], None] | None:
    if recorder is None:
        return None

    def _cb(obs: dict[str, Any]) -> None:
        frame = ego_from_gym_obs(obs)
        if frame is not None:
            recorder.write_rgb(frame)

    return _cb
