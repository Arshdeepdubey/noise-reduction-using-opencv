import cv2
import numpy as np
import pytest

from noise_reduction import metrics
from noise_reduction.wavelet_denoise import estimate_wavelet_sigma, wavelet_denoise


@pytest.mark.parametrize("method", ["BayesShrink", "VisuShrink"])
def test_wavelet_denoise_shape_dtype(noisy_image, method):
    out = wavelet_denoise(noisy_image, method=method)
    assert out.shape == noisy_image.shape
    assert out.dtype == np.uint8


def test_wavelet_denoise_grayscale(noisy_image):
    gray = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
    out = wavelet_denoise(gray)
    assert out.shape == gray.shape


def test_wavelet_denoise_improves_psnr(dark_reference, noisy_image):
    out = wavelet_denoise(noisy_image)
    assert metrics.psnr(dark_reference, out) > metrics.psnr(dark_reference, noisy_image)


def test_estimate_wavelet_sigma_positive(noisy_image):
    assert estimate_wavelet_sigma(noisy_image) > 0
