import cv2
import numpy as np

from noise_reduction import metrics
from noise_reduction.dct_denoise import dct_denoise


def test_dct_denoise_shape_dtype_color(noisy_image):
    out = dct_denoise(noisy_image)
    assert out.shape == noisy_image.shape
    assert out.dtype == np.uint8


def test_dct_denoise_shape_dtype_grayscale(noisy_image):
    gray = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
    out = dct_denoise(gray)
    assert out.shape == gray.shape
    assert out.dtype == np.uint8


def test_dct_denoise_improves_psnr(dark_reference, noisy_image):
    out = dct_denoise(noisy_image, k=4.0)
    assert metrics.psnr(dark_reference, out) > metrics.psnr(dark_reference, noisy_image)


def test_dct_denoise_handles_non_multiple_block_size(dark_reference):
    # Image dimensions that are not a clean multiple of block_size should
    # still be handled correctly via the padding logic.
    odd_sized = dark_reference[:97, :83]
    out = dct_denoise(odd_sized, block_size=8, stride=4)
    assert out.shape == odd_sized.shape
