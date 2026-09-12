import numpy as np

from noise_reduction import metrics
from noise_reduction.bilateral import bilateral_baseline, bilateral_optimized


def test_bilateral_baseline_shape_dtype(noisy_image):
    out = bilateral_baseline(noisy_image)
    assert out.shape == noisy_image.shape
    assert out.dtype == np.uint8


def test_bilateral_optimized_shape_dtype(noisy_image):
    out = bilateral_optimized(noisy_image)
    assert out.shape == noisy_image.shape
    assert out.dtype == np.uint8


def test_bilateral_baseline_improves_psnr(dark_reference, noisy_image):
    out = bilateral_baseline(noisy_image, sigma_color=90, sigma_space=90)
    assert metrics.psnr(dark_reference, out) > metrics.psnr(dark_reference, noisy_image)


def test_bilateral_optimized_improves_psnr(dark_reference, noisy_image):
    out = bilateral_optimized(noisy_image)
    assert metrics.psnr(dark_reference, out) > metrics.psnr(dark_reference, noisy_image)


def test_bilateral_optimized_debug_info(noisy_image):
    out, debug = bilateral_optimized(noisy_image, return_debug=True)
    assert out.shape == noisy_image.shape
    assert debug["noise_sigma"] > 0
    assert debug["sigma_color"] > 0


def test_bilateral_optimized_downscale_runs_and_is_valid(noisy_image):
    out = bilateral_optimized(noisy_image, downscale=2)
    assert out.shape == noisy_image.shape
    assert out.dtype == np.uint8


def test_bilateral_optimized_adapts_sigma_to_noise_level(dark_reference):
    from noise_reduction.low_light import add_poisson_gaussian_noise

    # Vary only the independent read-noise term for an unambiguous
    # low-vs-high comparison (see test_low_light.py for why mixing the
    # signal-dependent shot_noise_scale knob in too can be non-monotonic).
    low_noise = add_poisson_gaussian_noise(dark_reference, shot_noise_scale=10.0, read_noise_sigma=2.0, seed=1)
    high_noise = add_poisson_gaussian_noise(dark_reference, shot_noise_scale=10.0, read_noise_sigma=20.0, seed=1)

    _, low_debug = bilateral_optimized(low_noise, return_debug=True)
    _, high_debug = bilateral_optimized(high_noise, return_debug=True)

    # A noisier frame should be assigned a larger range-sigma (stronger smoothing).
    assert high_debug["sigma_color"] > low_debug["sigma_color"]
