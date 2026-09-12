import numpy as np
import pytest

from noise_reduction import metrics


def test_psnr_identical_images_is_very_high(clean_image):
    assert metrics.psnr(clean_image, clean_image) > 60


def test_ssim_identical_images_is_one(clean_image):
    assert metrics.ssim(clean_image, clean_image) == pytest.approx(1.0, abs=1e-6)


def test_psnr_decreases_with_noise(clean_image, noisy_image):
    # Noise should measurably reduce PSNR/SSIM relative to a clean image.
    assert metrics.psnr(clean_image, noisy_image) < metrics.psnr(clean_image, clean_image)
    assert metrics.ssim(clean_image, noisy_image) < 1.0


def test_measure_runtime_returns_result_and_timing():
    result, timing = metrics.measure_runtime(lambda x: x + 1, np.array([1, 2, 3]))
    np.testing.assert_array_equal(result, np.array([2, 3, 4]))
    assert timing.seconds >= 0
    assert timing.fps(10) > 0


def test_evaluate_bundle_has_expected_keys(clean_image, noisy_image):
    row = metrics.evaluate(clean_image, noisy_image, elapsed_seconds=0.01)
    assert set(row.keys()) == {"psnr_db", "ssim", "runtime_ms"}
    assert row["runtime_ms"] == 10.0
