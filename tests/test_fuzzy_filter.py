import cv2
import numpy as np
import pytest

from noise_reduction import metrics
from noise_reduction.fuzzy_filter import fuzzy_filter


def test_fuzzy_filter_shape_dtype_color(noisy_image):
    out = fuzzy_filter(noisy_image, window=5)
    assert out.shape == noisy_image.shape
    assert out.dtype == np.uint8


def test_fuzzy_filter_shape_dtype_grayscale(noisy_image):
    gray = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
    out = fuzzy_filter(gray, window=5)
    assert out.shape == gray.shape
    assert out.dtype == np.uint8


def test_fuzzy_filter_rejects_even_window(noisy_image):
    with pytest.raises(ValueError):
        fuzzy_filter(noisy_image, window=4)


def test_fuzzy_filter_improves_psnr(dark_reference, noisy_image):
    out = fuzzy_filter(noisy_image)
    assert metrics.psnr(dark_reference, out) > metrics.psnr(dark_reference, noisy_image)
