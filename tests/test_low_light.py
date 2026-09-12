import numpy as np

from noise_reduction.low_light import (
    add_poisson_gaussian_noise,
    enhance_low_light,
    estimate_noise_sigma,
    make_low_light_noisy,
    simulate_low_light,
)


def test_simulate_low_light_darkens_image(clean_image):
    dark = simulate_low_light(clean_image, exposure_factor=0.25)
    assert dark.shape == clean_image.shape
    assert dark.dtype == np.uint8
    assert dark.mean() < clean_image.mean()


def test_simulate_low_light_noop_at_full_exposure(clean_image):
    same = simulate_low_light(clean_image, exposure_factor=1.0)
    # Round-tripping through the (de)gamma should be very close to identity.
    assert np.mean(np.abs(same.astype(int) - clean_image.astype(int))) < 3


def test_add_noise_preserves_shape_and_dtype(dark_reference):
    noisy = add_poisson_gaussian_noise(dark_reference, seed=1)
    assert noisy.shape == dark_reference.shape
    assert noisy.dtype == np.uint8


def test_add_noise_increases_variance(dark_reference):
    noisy = add_poisson_gaussian_noise(dark_reference, shot_noise_scale=6.0, read_noise_sigma=10.0, seed=1)
    diff_std = np.std(noisy.astype(np.float64) - dark_reference.astype(np.float64))
    assert diff_std > 1.0  # noise was actually added


def test_estimate_noise_sigma_tracks_injected_noise_level(dark_reference):
    # Vary only the (independent, additive) read-noise term so the
    # comparison is unambiguous: shot noise is signal-dependent and its
    # `shot_noise_scale` knob interacts with image content, so mixing
    # both knobs in one comparison can be non-monotonic.
    low_noise = add_poisson_gaussian_noise(dark_reference, shot_noise_scale=10.0, read_noise_sigma=2.0, seed=1)
    high_noise = add_poisson_gaussian_noise(dark_reference, shot_noise_scale=10.0, read_noise_sigma=20.0, seed=1)

    sigma_low = estimate_noise_sigma(low_noise)
    sigma_high = estimate_noise_sigma(high_noise)

    assert sigma_low > 0
    assert sigma_high > sigma_low


def test_estimate_noise_sigma_handles_grayscale():
    gray = np.random.default_rng(0).normal(128, 15, size=(64, 64)).astype(np.uint8)
    sigma = estimate_noise_sigma(gray)
    assert sigma > 0


def test_make_low_light_noisy_convenience_wrapper(clean_image):
    out = make_low_light_noisy(clean_image, seed=5)
    assert out.shape == clean_image.shape
    assert out.dtype == np.uint8


def test_enhance_low_light_brightens_dark_image(dark_reference):
    enhanced = enhance_low_light(dark_reference)
    assert enhanced.shape == dark_reference.shape
    assert enhanced.mean() >= dark_reference.mean()
