"""Continuous video-stream denoising pipeline.

The project brief specifically calls out checking "if the selected
method works well for continuous video stream" and measuring
performance (i.e. achievable FPS), not just single-image quality. This
module wraps any of the per-frame spatial filters in this package with:

* an optional **motion-adaptive temporal filter** -- an exponential
  moving average across frames, blended in only where the scene is
  locally static (small inter-frame difference), which further reduces
  low-light temporal noise ("shimmer") without motion blur/ghosting on
  moving content; and
* per-frame and aggregate **runtime/FPS instrumentation**.

This is a lightweight version of the classic noise-adaptive
spatio-temporal filtering approach used in real-time low-light video
pipelines (temporal recursive filtering where motion is absent, spatial
filtering where it is present).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional

import cv2
import numpy as np

SpatialFilter = Callable[[np.ndarray], np.ndarray]


@dataclass
class FrameStats:
    frame_index: int
    spatial_ms: float
    total_ms: float


@dataclass
class VideoStats:
    frame_count: int = 0
    total_seconds: float = 0.0
    per_frame: List[FrameStats] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.frame_count / self.total_seconds if self.total_seconds > 0 else float("inf")

    @property
    def avg_frame_ms(self) -> float:
        return (self.total_seconds / self.frame_count * 1000.0) if self.frame_count else 0.0


class TemporalDenoiser:
    """Motion-adaptive exponential-moving-average temporal filter.

    For each pixel, if the (spatially denoised) current frame is close
    to the running estimate, blend it in with weight ``alpha`` (reduces
    residual temporal noise on static content). Where the difference
    exceeds ``motion_threshold`` (real motion, not noise), the pixel is
    passed through unblended to avoid ghosting/motion blur.
    """

    def __init__(self, alpha: float = 0.35, motion_threshold: float = 25.0):
        self.alpha = alpha
        self.motion_threshold = motion_threshold
        self._running: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._running = None

    def apply(self, spatial_denoised_frame: np.ndarray) -> np.ndarray:
        frame = spatial_denoised_frame.astype(np.float32)
        if self._running is None:
            self._running = frame.copy()
            return spatial_denoised_frame

        diff = np.abs(frame - self._running)
        if diff.ndim == 3:
            diff = diff.mean(axis=2, keepdims=True)
        static_mask = (diff < self.motion_threshold).astype(np.float32)

        blended = self.alpha * self._running + (1 - self.alpha) * frame
        output = static_mask * blended + (1 - static_mask) * frame

        self._running = output
        return np.clip(output, 0, 255).astype(np.uint8)


def denoise_video(
    input_path: str,
    output_path: str,
    spatial_filter: SpatialFilter,
    use_temporal: bool = True,
    temporal_alpha: float = 0.35,
    temporal_motion_threshold: float = 25.0,
    max_frames: Optional[int] = None,
    fourcc: str = "mp4v",
) -> VideoStats:
    """Denoise every frame of ``input_path`` and write to ``output_path``.

    ``spatial_filter`` is any single-frame denoiser with signature
    ``frame -> frame`` (e.g. ``noise_reduction.bilateral.bilateral_optimized``).
    Returns a :class:`VideoStats` with per-frame and aggregate timing so
    the achievable FPS of a given method can be reported.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {input_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*fourcc), fps_in, (width, height))

    temporal = TemporalDenoiser(alpha=temporal_alpha, motion_threshold=temporal_motion_threshold) if use_temporal else None
    stats = VideoStats()

    frame_index = 0
    overall_start = time.perf_counter()
    while True:
        if max_frames is not None and frame_index >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        denoised = spatial_filter(frame)
        t1 = time.perf_counter()

        if temporal is not None:
            denoised = temporal.apply(denoised)
        t2 = time.perf_counter()

        writer.write(denoised)
        stats.per_frame.append(FrameStats(frame_index, (t1 - t0) * 1000.0, (t2 - t0) * 1000.0))
        frame_index += 1

    stats.total_seconds = time.perf_counter() - overall_start
    stats.frame_count = frame_index

    cap.release()
    writer.release()
    return stats


def iter_frames(video_path: str, max_frames: Optional[int] = None) -> Iterable[np.ndarray]:
    """Yield BGR frames from a video file (helper for scripts/tests)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    count = 0
    try:
        while True:
            if max_frames is not None and count >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
            count += 1
    finally:
        cap.release()
