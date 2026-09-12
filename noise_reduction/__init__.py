"""
noise_reduction
===============

Image-processing based noise reduction toolkit developed for the
"Optimized noise reduction filter implementation" work-let
(Image processing / C++ & OpenCV area, low-light video use case).

The package implements and benchmarks four families of denoising
algorithms surveyed from the literature:

* ``bilateral``  -- baseline OpenCV bilateral filter and an *optimized*,
  noise-adaptive, luma/chroma-split, multi-resolution bilateral filter
  tuned for low-light video.
* ``fuzzy_filter`` -- a fuzzy-logic weighted averaging filter that uses
  triangular membership functions on local intensity differences.
* ``dct_denoise`` -- block-DCT hard-thresholding denoising (Yaroslavsky /
  Guoshen Yu & Sapiro style) with overlapping-block aggregation.
* ``wavelet_denoise`` -- wavelet-coefficient shrinkage denoising
  (VisuShrink / BayesShrink) built on PyWavelets / scikit-image.

Supporting modules:

* ``low_light`` -- low-light simulation (gamma darkening + Poisson-Gaussian
  sensor noise) and a robust noise-level estimator used to auto-tune the
  optimized bilateral filter.
* ``metrics`` -- PSNR / SSIM / runtime measurement helpers.
* ``video_pipeline`` -- frame-by-frame + motion-adaptive temporal
  denoising for continuous video streams, with FPS measurement.
"""

from . import bilateral, dct_denoise, fuzzy_filter, low_light, metrics, video_pipeline, wavelet_denoise

__all__ = [
    "bilateral",
    "fuzzy_filter",
    "dct_denoise",
    "wavelet_denoise",
    "low_light",
    "metrics",
    "video_pipeline",
]

__version__ = "0.1.0"
