"""Quality and performance measurement helpers.

Used by the benchmark script to compare the noise-reduction methods
against each other on the two axes called out in the project brief:
*quality* (PSNR / SSIM against a clean reference) and *performance*
(wall-clock runtime / achievable FPS for continuous video streams).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(clean: np.ndarray, test: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB (higher is better)."""
    return float(peak_signal_noise_ratio(clean, test, data_range=255))


def ssim(clean: np.ndarray, test: np.ndarray) -> float:
    """Structural Similarity Index (0-1, higher is better).

    Handles both grayscale and multi-channel (BGR/RGB) images.
    """
    channel_axis = -1 if clean.ndim == 3 else None
    win_size = min(7, min(clean.shape[:2]) - (1 - min(clean.shape[:2]) % 2))
    win_size = max(3, win_size if win_size % 2 == 1 else win_size - 1)
    return float(
        structural_similarity(
            clean, test, data_range=255, channel_axis=channel_axis, win_size=win_size
        )
    )


@dataclass
class Timing:
    """Wall-clock timing result for a single call."""

    seconds: float

    @property
    def ms(self) -> float:
        return self.seconds * 1000.0

    def fps(self, num_frames: int = 1) -> float:
        if self.seconds <= 0:
            return float("inf")
        return num_frames / self.seconds


def measure_runtime(func: Callable, *args, **kwargs) -> Tuple[np.ndarray, Timing]:
    """Run ``func(*args, **kwargs)`` once and time it.

    Returns ``(result, Timing)``. A perf_counter is used (monotonic,
    highest available resolution) rather than ``time.time()``.
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, Timing(elapsed)


def evaluate(clean: np.ndarray, test: np.ndarray, elapsed_seconds: float | None = None) -> dict:
    """Convenience bundle: PSNR + SSIM (+ optional timing) as a dict row."""
    row = {"psnr_db": psnr(clean, test), "ssim": ssim(clean, test)}
    if elapsed_seconds is not None:
        row["runtime_ms"] = elapsed_seconds * 1000.0
    return row
