"""Wavelet-shrinkage denoising (VisuShrink / BayesShrink).

Wavelet denoising exploits the fact that a discrete wavelet transform
concentrates a natural image's energy into a few large coefficients
while additive Gaussian noise spreads roughly evenly across all
coefficients at all scales. Shrinking (soft-thresholding) small
coefficients towards zero therefore removes most of the noise while
leaving genuine image structure largely intact -- and, unlike a fixed
spatial-kernel filter, it does so at multiple scales simultaneously.

Two classic threshold selection rules are exposed:

* **VisuShrink** (Donoho & Johnstone, 1994): a single universal
  threshold ``sigma * sqrt(2 * log(N))`` for the whole image. Simple
  and robust, but tends to over-smooth because it uses one global
  threshold for every sub-band.
* **BayesShrink** (Chang, Yu & Vetterli, 2000): a Bayesian,
  sub-band-adaptive threshold that generally preserves more detail
  than VisuShrink for natural images -- the literature survey performed
  for this project (see docs/LITERATURE_SURVEY.md) found BayesShrink
  and other adaptive shrinkage rules consistently outperform the
  non-adaptive VisuShrink baseline.

This module is a thin, denoising-focused wrapper around
``skimage.restoration.denoise_wavelet`` (which implements both rules)
so it shares a common ``(image, **kwargs) -> image`` interface with the
other filters in this package.
"""
from __future__ import annotations

import numpy as np
from skimage.restoration import denoise_wavelet, estimate_sigma


def wavelet_denoise(
    image: np.ndarray,
    method: str = "BayesShrink",
    wavelet: str = "db8",
    mode: str = "soft",
    levels: int | None = None,
    sigma: float | None = None,
) -> np.ndarray:
    """Denoise a grayscale or BGR image via wavelet coefficient shrinkage.

    Parameters
    ----------
    image: grayscale or BGR uint8 image.
    method: ``"BayesShrink"`` (sub-band adaptive, default, generally
        best quality) or ``"VisuShrink"`` (single global threshold).
    wavelet: mother wavelet (``"db8"`` -- Daubechies-8 -- is a common
        general-purpose choice balancing smoothness and locality).
    mode: ``"soft"`` (shrink towards zero, fewer artifacts) or
        ``"hard"`` (zero below threshold, keep above unchanged).
    levels: number of decomposition levels; ``None`` lets skimage pick
        based on image size.
    sigma: known/assumed noise sigma (0-1 scale internally handled by
        this wrapper as 0-255); if ``None``, skimage estimates it from
        the image itself (robust wavelet-domain estimator, similar in
        spirit to the one in :mod:`noise_reduction.low_light`).
    """
    is_color = image.ndim == 3
    img_float = image.astype(np.float64) / 255.0

    kwargs = dict(
        method=method,
        mode=mode,
        wavelet=wavelet,
        rescale_sigma=True,
    )
    if levels is not None:
        kwargs["wavelet_levels"] = levels
    if sigma is not None:
        kwargs["sigma"] = sigma / 255.0

    if is_color:
        # BGR -> RGB is irrelevant for a per-channel transform, but keep
        # channel_axis explicit for clarity / forward-compat with skimage.
        denoised = denoise_wavelet(img_float, channel_axis=-1, **kwargs)
    else:
        denoised = denoise_wavelet(img_float, channel_axis=None, **kwargs)

    return np.clip(denoised * 255.0, 0, 255).astype(np.uint8)


def estimate_wavelet_sigma(image: np.ndarray) -> float:
    """Return skimage's own robust per-channel-averaged noise estimate
    (0-255 scale), useful for reporting/comparison against the
    MAD-based estimator in :mod:`noise_reduction.low_light`.
    """
    img_float = image.astype(np.float64) / 255.0
    if image.ndim == 3:
        sigma = estimate_sigma(img_float, channel_axis=-1, average_sigmas=True)
    else:
        sigma = estimate_sigma(img_float, channel_axis=None)
    return float(sigma) * 255.0
