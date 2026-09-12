#!/usr/bin/env python3
"""Generate synthetic low-light noisy test data for the benchmark.

Real low-light footage has no clean reference frame, so PSNR/SSIM
cannot be computed against "ground truth" for it. Instead we take
well-lit, noise-free sample images (bundled with scikit-image, so this
works fully offline / with no external downloads), and *synthetically*
darken them and add a realistic Poisson-Gaussian sensor-noise model
(see ``noise_reduction.low_light``). This gives a known-clean reference
for every noisy image/frame, which is what the benchmark's PSNR/SSIM
numbers are computed against.

Important methodology note: the correct ground truth for evaluating a
*denoiser* is the darkened-but-noise-free image at the SAME exposure as
the noisy input -- not the original well-lit photo. A denoiser is not
expected to relight the scene (that is a separate low-light
*enhancement* problem, see ``noise_reduction.low_light.enhance_low_light``,
applied only for visualization here). Comparing against the bright
original would conflate "noise removed" with "scene relit" and make the
PSNR/SSIM numbers meaningless. So for every noisy image/frame this
script also writes the matching noise-free-but-still-dark reference,
and the benchmark evaluates against that.

Outputs
-------
data/clean/<name>.png                     -- original well-lit images (for display only)
data/noisy/<name>_<level>.png             -- darkened + noisy (denoiser input)
data/noisy/<name>_<level>_reference.png   -- darkened, NOISE-FREE (denoiser target / PSNR ground truth)
data/video/clean_frames/*.png             -- the short synthetic sequence, well-lit (jittered, for display)
data/video/reference_frames/*.png         -- same sequence, darkened, NOISE-FREE (video PSNR ground truth)
data/video/low_light_noisy.mp4            -- same sequence, darkened + noisy (denoiser input)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from skimage import data as skdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noise_reduction.low_light import add_poisson_gaussian_noise, simulate_low_light  # noqa: E402

NOISE_LEVELS = {
    "moderate": dict(exposure_factor=0.35, shot_noise_scale=10.0, read_noise_sigma=5.0),
    "severe": dict(exposure_factor=0.15, shot_noise_scale=4.0, read_noise_sigma=10.0),
}

SAMPLE_IMAGES = {
    "astronaut": skdata.astronaut,
    "coffee": skdata.coffee,
    "chelsea": skdata.chelsea,
    "camera": lambda: cv2.cvtColor(skdata.camera(), cv2.COLOR_GRAY2RGB),
}

VIDEO_FRAMES = 40
VIDEO_SIZE = (480, 320)  # width, height -- keeps benchmark runtime reasonable


def to_bgr_uint8(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype != np.uint8:
        rgb = (255 * (rgb.astype(np.float64) / rgb.max())).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def generate_images(clean_dir: Path, noisy_dir: Path) -> None:
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    for name, loader in SAMPLE_IMAGES.items():
        rgb = loader()
        bgr = to_bgr_uint8(np.asarray(rgb))
        bgr = cv2.resize(bgr, (384, 384 * bgr.shape[0] // bgr.shape[1]))
        cv2.imwrite(str(clean_dir / f"{name}.png"), bgr)

        for level_name, params in NOISE_LEVELS.items():
            dark_reference = simulate_low_light(bgr, exposure_factor=params["exposure_factor"])
            noisy = add_poisson_gaussian_noise(
                dark_reference,
                shot_noise_scale=params["shot_noise_scale"],
                read_noise_sigma=params["read_noise_sigma"],
                seed=42,
            )
            cv2.imwrite(str(noisy_dir / f"{name}_{level_name}.png"), noisy)
            cv2.imwrite(str(noisy_dir / f"{name}_{level_name}_reference.png"), dark_reference)

        print(f"[data] {name}: clean + {len(NOISE_LEVELS)} noisy variants written")


VIDEO_NOISE = dict(exposure_factor=0.25, shot_noise_scale=6.0, read_noise_sigma=7.0)


def generate_video(video_dir: Path) -> None:
    clean_frames_dir = video_dir / "clean_frames"
    reference_frames_dir = video_dir / "reference_frames"
    clean_frames_dir.mkdir(parents=True, exist_ok=True)
    reference_frames_dir.mkdir(parents=True, exist_ok=True)

    base_rgb = np.asarray(skdata.astronaut())
    base_bgr = to_bgr_uint8(base_rgb)
    base_bgr = cv2.resize(base_bgr, VIDEO_SIZE)
    h, w = base_bgr.shape[:2]

    rng = np.random.default_rng(7)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    noisy_path = video_dir / "low_light_noisy.mp4"
    writer = cv2.VideoWriter(str(noisy_path), fourcc, 20.0, (w, h))

    for i in range(VIDEO_FRAMES):
        # Small synthetic camera jitter/pan so the sequence isn't static
        # (exercises the motion-adaptive temporal filter meaningfully).
        dx, dy = rng.uniform(-3, 3, size=2)
        angle = rng.uniform(-0.5, 0.5)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        jittered = cv2.warpAffine(base_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        cv2.imwrite(str(clean_frames_dir / f"frame_{i:03d}.png"), jittered)

        dark_reference = simulate_low_light(jittered, exposure_factor=VIDEO_NOISE["exposure_factor"])
        cv2.imwrite(str(reference_frames_dir / f"frame_{i:03d}.png"), dark_reference)

        noisy = add_poisson_gaussian_noise(
            dark_reference,
            shot_noise_scale=VIDEO_NOISE["shot_noise_scale"],
            read_noise_sigma=VIDEO_NOISE["read_noise_sigma"],
            seed=int(rng.integers(0, 1_000_000)),
        )
        writer.write(noisy)

    writer.release()
    print(f"[data] synthetic low-light video written: {noisy_path} ({VIDEO_FRAMES} frames)")


def main() -> None:
    generate_images(ROOT / "data" / "clean", ROOT / "data" / "noisy")
    generate_video(ROOT / "data" / "video")


if __name__ == "__main__":
    main()
