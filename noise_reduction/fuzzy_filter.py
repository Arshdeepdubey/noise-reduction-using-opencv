"""Fuzzy-logic weighted averaging filter for image denoising.

Classical fuzzy denoising filters (e.g. Farbiz & Menhaj 1998, Kwan
2003, Van De Ville et al.'s GOA fuzzy filter/Lukac's fuzzy vector
filters) replace the hard/statistical weighting used by mean, Gaussian
or bilateral filters with a *fuzzy membership function* that expresses,
in [0, 1], "how similar" a neighboring pixel is to the center pixel and
"how large" its intensity gradient is -- without needing an exact noise
model. This makes them a reasonable choice when the noise
characteristics (impulse + Gaussian mixtures typical of compressed,
low-light sensor video) are not well described by a single Gaussian
range kernel.

This module implements a vectorized fuzzy weighted-mean filter:

For every pixel and each neighbor in an ``N x N`` window, a *similarity
membership* mu_s = triangular_membership(|center - neighbor|) grades how
alike the two pixels are (1 when identical, 0 beyond a threshold), and a
*directional-gradient membership* mu_g down-weights neighbors that sit
across a strong local edge (estimated by the local gradient magnitude at
the neighbor). The output is the gradient- and similarity-weighted
average of the neighborhood, which suppresses noise while still
respecting edges (a fuzzy analogue of the bilateral filter's Gaussian
range kernel).
"""
from __future__ import annotations

import cv2
import numpy as np


def _triangular_membership(diff: np.ndarray, threshold: float) -> np.ndarray:
    """mu(x) = max(0, 1 - x/threshold) -- a standard triangular fuzzy
    membership function used for "closeness" in fuzzy filters."""
    return np.clip(1.0 - diff / max(threshold, 1e-6), 0.0, 1.0)


def _fuzzy_denoise_channel(channel: np.ndarray, window: int, similarity_threshold: float, gradient_threshold: float) -> np.ndarray:
    ch = channel.astype(np.float64)
    h, w = ch.shape
    r = window // 2

    padded = cv2.copyMakeBorder(ch, r, r, r, r, borderType=cv2.BORDER_REFLECT)
    # Local gradient magnitude (Sobel) used for the edge-aware membership.
    # Computed on a lightly median-smoothed proxy, NOT the raw noisy
    # channel: in a noisy low-light frame, pixel-to-pixel Poisson/read
    # noise itself produces large Sobel responses everywhere, which
    # would make mu_gradient reject almost every neighbor and leave the
    # image nearly unfiltered. Median blur suppresses that noise-driven
    # gradient while still tracking genuine edges, which is what the
    # membership function is meant to respond to.
    gradient_proxy = cv2.medianBlur(channel.astype(np.uint8), 5).astype(np.float64)
    gx = cv2.Sobel(gradient_proxy, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gradient_proxy, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = cv2.copyMakeBorder(np.hypot(gx, gy), r, r, r, r, borderType=cv2.BORDER_REFLECT)

    center = padded[r : r + h, r : r + w]
    center_grad = grad_mag[r : r + h, r : r + w]

    weighted_sum = np.zeros((h, w), dtype=np.float64)
    weight_total = np.zeros((h, w), dtype=np.float64)

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            neighbor = padded[r + dy : r + dy + h, r + dx : r + dx + w]
            neighbor_grad = grad_mag[r + dy : r + dy + h, r + dx : r + dx + w]

            intensity_diff = np.abs(neighbor - center)
            mu_similarity = _triangular_membership(intensity_diff, similarity_threshold)

            grad_diff = np.abs(neighbor_grad - center_grad)
            mu_gradient = _triangular_membership(grad_diff, gradient_threshold)

            weight = mu_similarity * mu_gradient
            weighted_sum += weight * neighbor
            weight_total += weight

    # Guard against fully-zero weight neighborhoods (extremely rare,
    # e.g. flat thresholds): fall back to the original pixel.
    safe_total = np.where(weight_total <= 1e-8, 1.0, weight_total)
    result = np.where(weight_total <= 1e-8, center, weighted_sum / safe_total)
    return result


def fuzzy_filter(
    image: np.ndarray,
    window: int = 7,
    similarity_threshold: float = 200.0,
    gradient_threshold: float = 240.0,
) -> np.ndarray:
    """Apply the fuzzy weighted-mean filter to a grayscale or BGR image.

    Parameters
    ----------
    image: grayscale or BGR uint8 image.
    window: neighborhood size (odd, e.g. 3 or 5). Larger windows remove
        more noise but cost more and risk softening fine detail.
    similarity_threshold: intensity-difference scale (0-255) at which
        the similarity membership reaches 0; smaller = stricter
        (less smoothing, more noise left; better edge preservation).
    gradient_threshold: local-gradient-difference scale at which the
        edge-aware membership reaches 0.
    """
    if window % 2 == 0:
        raise ValueError("window must be odd")

    if image.ndim == 2:
        out = _fuzzy_denoise_channel(image, window, similarity_threshold, gradient_threshold)
        return np.clip(out, 0, 255).astype(np.uint8)

    channels = cv2.split(image)
    filtered = [
        _fuzzy_denoise_channel(ch, window, similarity_threshold, gradient_threshold) for ch in channels
    ]
    merged = cv2.merge([np.clip(c, 0, 255).astype(np.uint8) for c in filtered])
    return merged
