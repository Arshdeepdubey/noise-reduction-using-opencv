import cv2
import numpy as np
import pytest

from noise_reduction.low_light import add_poisson_gaussian_noise, simulate_low_light


def _synthetic_scene(size: int = 96) -> np.ndarray:
    """A small, deterministic, structured test image (not random noise)
    so denoising filters have real edges/gradients to preserve. Kept
    small so the whole test suite (including the slower DCT/fuzzy
    filters) runs quickly.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (size - 10, size - 10), (200, 160, 90), -1)
    cv2.circle(img, (size // 2, size // 2), size // 4, (40, 220, 180), -1)
    cv2.line(img, (0, 0), (size, size), (255, 255, 255), 2)
    # smooth gradient background band so flat-region noise reduction is exercised too
    for y in range(size):
        img[y, : max(1, size // 6)] = (int(255 * y / size), 30, 30)
    return img


@pytest.fixture
def clean_image() -> np.ndarray:
    return _synthetic_scene()


@pytest.fixture
def dark_reference(clean_image) -> np.ndarray:
    return simulate_low_light(clean_image, exposure_factor=0.3)


@pytest.fixture
def noisy_image(dark_reference) -> np.ndarray:
    return add_poisson_gaussian_noise(dark_reference, shot_noise_scale=8.0, read_noise_sigma=6.0, seed=123)
