"""Block-DCT hard-thresholding denoising.

Implements the classical, effective DCT image-denoising method
popularized by Yaroslavsky and formalized as a simple published
algorithm by Guoshen Yu & Guillermo Sapiro, "DCT image denoising: a
simple and effective image denoising algorithm" (IPOL, 2011):

1. Slide an ``N x N`` (default 8x8) window over the image with a small
   stride (overlapping blocks -- the key to quality: overlap gives many
   independent noisy estimates of each pixel to average over).
2. Take the 2D DCT of each block.
3. Hard-threshold coefficients: zero out any coefficient with magnitude
   below ``threshold = k * sigma`` (k ~= 3, the classic choice, since
   under a Gaussian noise model this keeps signal energy while removing
   almost all pure-noise coefficients).
4. Inverse DCT each block back to the spatial domain.
5. Aggregate overlapping block estimates for each pixel by averaging
   (equivalent to the "aggregation" step BM3D also uses, minus the
   grouping/collaborative-filtering step -- DCT denoising is a fast,
   single-image-patch-only special case of that family of methods).

This is a *transform-domain* method, complementary to the *spatial*
bilateral/fuzzy filters: it is particularly effective against additive
Gaussian read-noise (the noise floor low-light sensors add regardless
of scene content) because that noise energy is spread near-uniformly
across DCT coefficients while natural-image energy concentrates in the
low-frequency ones.
"""
from __future__ import annotations

import cv2
import numpy as np


def _dct_denoise_channel(channel: np.ndarray, block_size: int, stride: int, sigma: float, k: float) -> np.ndarray:
    ch = channel.astype(np.float32)
    h, w = ch.shape

    # Pad so an integer number of (possibly overlapping) blocks tile
    # the image; reflect padding avoids introducing a hard border.
    if h >= block_size:
        pad_h = (stride - (h - block_size) % stride) % stride
    else:
        pad_h = block_size - h
    if w >= block_size:
        pad_w = (stride - (w - block_size) % stride) % stride
    else:
        pad_w = block_size - w
    padded = cv2.copyMakeBorder(ch, 0, max(pad_h, 0), 0, max(pad_w, 0), borderType=cv2.BORDER_REFLECT)
    ph, pw = padded.shape

    accum = np.zeros((ph, pw), dtype=np.float64)
    weight = np.zeros((ph, pw), dtype=np.float64)
    threshold = k * sigma

    ys = list(range(0, ph - block_size + 1, stride))
    xs = list(range(0, pw - block_size + 1, stride))
    if ys[-1] != ph - block_size:
        ys.append(ph - block_size)
    if xs[-1] != pw - block_size:
        xs.append(pw - block_size)

    for y in ys:
        for x in xs:
            block = padded[y : y + block_size, x : x + block_size]
            coeffs = cv2.dct(block)
            coeffs[np.abs(coeffs) < threshold] = 0.0
            denoised_block = cv2.idct(coeffs)
            accum[y : y + block_size, x : x + block_size] += denoised_block
            weight[y : y + block_size, x : x + block_size] += 1.0

    weight[weight == 0] = 1.0
    result = (accum / weight)[:h, :w]
    return result


def dct_denoise(
    image: np.ndarray,
    block_size: int = 8,
    stride: int | None = None,
    sigma: float | None = None,
    k: float = 3.0,
) -> np.ndarray:
    """Denoise a grayscale or BGR image with overlapping block-DCT
    hard-thresholding.

    Parameters
    ----------
    image: grayscale or BGR uint8 image.
    block_size: DCT block side length (8 is the classic choice, matches
        JPEG-scale frequency granularity).
    stride: block step; smaller = more overlap = better quality but
        more compute. Defaults to ``block_size // 2`` (50% overlap,
        the standard quality/speed compromise).
    sigma: assumed noise standard deviation (0-255 scale). If ``None``,
        it is auto-estimated per-channel via the robust wavelet-MAD
        estimator so the threshold adapts to the actual frame noise.
    k: threshold multiplier (``threshold = k * sigma``); 3.0 is the
        standard value from the denoising literature.
    """
    if stride is None:
        stride = max(1, block_size // 2)

    from .low_light import estimate_noise_sigma  # local import avoids a cycle at module load

    if image.ndim == 2:
        sig = sigma if sigma is not None else estimate_noise_sigma(image)
        out = _dct_denoise_channel(image, block_size, stride, sig, k)
        return np.clip(out, 0, 255).astype(np.uint8)

    channels = cv2.split(image)
    filtered = []
    for ch in channels:
        sig = sigma if sigma is not None else estimate_noise_sigma(ch)
        filtered.append(np.clip(_dct_denoise_channel(ch, block_size, stride, sig, k), 0, 255).astype(np.uint8))
    return cv2.merge(filtered)
