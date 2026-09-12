#!/usr/bin/env python3
"""Quality + performance benchmark across all four noise-reduction
methods, on both still images and continuous video, per the project
brief's "measure the quality & performance of the solution" requirement.

Usage
-----
    python scripts/benchmark.py            # full run (images + video)
    python scripts/benchmark.py --images-only
    python scripts/benchmark.py --video-only

Outputs (under results/)
-------------------------
benchmark_results.csv        -- per-image, per-method PSNR/SSIM/runtime
benchmark_summary.csv        -- averaged per method
benchmark_chart.png          -- PSNR / SSIM / runtime bar charts
video_benchmark.csv          -- per-method video FPS + avg PSNR/SSIM
sample_outputs/comparison_*.png   -- side-by-side visual comparisons
sample_outputs/<method>_denoised.mp4  -- denoised demo videos
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noise_reduction import bilateral, dct_denoise, fuzzy_filter, metrics, video_pipeline, wavelet_denoise  # noqa: E402

DATA_NOISY = ROOT / "data" / "noisy"
DATA_VIDEO = ROOT / "data" / "video"
RESULTS = ROOT / "results"
SAMPLE_OUT = RESULTS / "sample_outputs"

# Every image-benchmark method as (label, callable(image)->image).
IMAGE_METHODS = {
    "gaussian_baseline": lambda im: cv2.GaussianBlur(im, (5, 5), 0),
    "bilateral_baseline": lambda im: bilateral.bilateral_baseline(im),
    "bilateral_optimized": lambda im: bilateral.bilateral_optimized(im, d=9, sigma_space=25),
    "fuzzy_filter": lambda im: fuzzy_filter.fuzzy_filter(im),
    "dct_denoise": lambda im: dct_denoise.dct_denoise(im, k=4.0),
    "wavelet_bayesshrink": lambda im: wavelet_denoise.wavelet_denoise(im, method="BayesShrink"),
    "wavelet_visushrink": lambda im: wavelet_denoise.wavelet_denoise(im, method="VisuShrink"),
}

# Methods considered practical for continuous, real-time-ish video and
# run over the FULL synthetic clip.
VIDEO_METHODS_FULL = {
    "bilateral_baseline": lambda im: bilateral.bilateral_baseline(im),
    "bilateral_optimized": lambda im: bilateral.bilateral_optimized(im, d=9, sigma_space=25),
    "wavelet_bayesshrink": lambda im: wavelet_denoise.wavelet_denoise(im, method="BayesShrink"),
}

# Methods that are too slow per-frame for continuous video at this
# resolution; only a handful of frames are processed to still report
# their throughput/quality as reference points (see docs/BENCHMARK_REPORT.md).
VIDEO_METHODS_SAMPLE = {
    "fuzzy_filter": lambda im: fuzzy_filter.fuzzy_filter(im),
    "dct_denoise": lambda im: dct_denoise.dct_denoise(im, k=4.0),
}
VIDEO_SAMPLE_FRAMES = 8


def find_image_pairs():
    """Yield (case_name, noisy_path, reference_path) for every noisy
    image that has a matching *_reference.png ground truth."""
    for noisy_path in sorted(DATA_NOISY.glob("*.png")):
        if noisy_path.name.endswith("_reference.png"):
            continue
        ref_path = noisy_path.with_name(noisy_path.stem + "_reference.png")
        if ref_path.exists():
            yield noisy_path.stem, noisy_path, ref_path


def run_image_benchmark() -> pd.DataFrame:
    rows = []
    cases = list(find_image_pairs())
    if not cases:
        raise RuntimeError("No test images found -- run scripts/generate_test_data.py first.")

    for case_name, noisy_path, ref_path in cases:
        noisy = cv2.imread(str(noisy_path))
        ref = cv2.imread(str(ref_path))

        noisy_row = {"case": case_name, "method": "noisy_input", **metrics.evaluate(ref, noisy)}
        noisy_row["runtime_ms"] = 0.0
        rows.append(noisy_row)

        for method_name, fn in IMAGE_METHODS.items():
            result, timing = metrics.measure_runtime(fn, noisy)
            row = {"case": case_name, "method": method_name, **metrics.evaluate(ref, result, timing.seconds)}
            rows.append(row)
        print(f"[benchmark] finished images for case: {case_name}")

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "benchmark_results.csv", index=False)

    summary = df.groupby("method")[["psnr_db", "ssim", "runtime_ms"]].mean().sort_values("psnr_db", ascending=False)
    summary.to_csv(RESULTS / "benchmark_summary.csv")
    print("\n=== Average over all test images (higher PSNR/SSIM better, lower runtime better) ===")
    print(summary.round(3).to_string())
    return df


def make_chart(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = df[df["method"] != "noisy_input"].groupby("method")[["psnr_db", "ssim", "runtime_ms"]].mean()
    summary = summary.sort_values("psnr_db", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(summary)))

    summary["psnr_db"].plot(kind="bar", ax=axes[0], color=colors)
    axes[0].set_title("Average PSNR (dB, higher = better)")
    axes[0].set_ylabel("dB")
    axes[0].tick_params(axis="x", rotation=45)

    summary["ssim"].plot(kind="bar", ax=axes[1], color=colors)
    axes[1].set_title("Average SSIM (higher = better)")
    axes[1].tick_params(axis="x", rotation=45)

    summary["runtime_ms"].plot(kind="bar", ax=axes[2], color=colors)
    axes[2].set_title("Average runtime per image (ms, lower = better)")
    axes[2].set_ylabel("ms")
    axes[2].tick_params(axis="x", rotation=45)

    fig.suptitle("Noise-reduction method comparison (synthetic low-light test set)")
    fig.tight_layout()
    SAMPLE_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "benchmark_chart.png", dpi=140)
    plt.close(fig)
    print(f"[benchmark] chart written to {RESULTS / 'benchmark_chart.png'}")


def make_visual_comparison(case_name: str = "astronaut_moderate") -> None:
    noisy_path = DATA_NOISY / f"{case_name}.png"
    ref_path = DATA_NOISY / f"{case_name}_reference.png"
    if not noisy_path.exists():
        return
    noisy = cv2.imread(str(noisy_path))
    ref = cv2.imread(str(ref_path))

    tiles = [("clean (dark) reference", ref), ("noisy input", noisy)]
    for name, fn in IMAGE_METHODS.items():
        tiles.append((name, fn(noisy)))

    thumb_w = 220
    labeled = []
    for label, img in tiles:
        h, w = img.shape[:2]
        thumb = cv2.resize(img, (thumb_w, int(h * thumb_w / w)))
        thumb = cv2.copyMakeBorder(thumb, 24, 4, 4, 4, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(thumb, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        labeled.append(thumb)

    # tile into a grid, 3 per row
    per_row = 3
    rows_of_tiles = [labeled[i : i + per_row] for i in range(0, len(labeled), per_row)]
    row_imgs = []
    max_row_len = max(len(r) for r in rows_of_tiles)
    tile_h, tile_w = labeled[0].shape[:2]
    for r in rows_of_tiles:
        while len(r) < max_row_len:
            r.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
        row_imgs.append(np.hstack(r))
    collage = np.vstack(row_imgs)

    SAMPLE_OUT.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLE_OUT / f"comparison_{case_name}.png"
    cv2.imwrite(str(out_path), collage)
    print(f"[benchmark] visual comparison written to {out_path}")


def run_video_benchmark() -> pd.DataFrame:
    noisy_video = DATA_VIDEO / "low_light_noisy.mp4"
    ref_frames_dir = DATA_VIDEO / "reference_frames"
    if not noisy_video.exists():
        raise RuntimeError("No test video found -- run scripts/generate_test_data.py first.")

    ref_frames = sorted(ref_frames_dir.glob("frame_*.png"))
    rows = []

    def evaluate_output(method_name: str, output_path: Path, num_frames: int, stats: video_pipeline.VideoStats, temporal: bool):
        cap = cv2.VideoCapture(str(output_path))
        psnrs, ssims = [], []
        for i in range(num_frames):
            ok, frame = cap.read()
            if not ok or i >= len(ref_frames):
                break
            ref = cv2.imread(str(ref_frames[i]))
            psnrs.append(metrics.psnr(ref, frame))
            ssims.append(metrics.ssim(ref, frame))
        cap.release()
        rows.append(
            {
                "method": method_name,
                "temporal": temporal,
                "frames": stats.frame_count,
                "fps": stats.fps,
                "avg_frame_ms": stats.avg_frame_ms,
                "avg_psnr_db": float(np.mean(psnrs)) if psnrs else float("nan"),
                "avg_ssim": float(np.mean(ssims)) if ssims else float("nan"),
            }
        )

    SAMPLE_OUT.mkdir(parents=True, exist_ok=True)

    for method_name, fn in VIDEO_METHODS_FULL.items():
        for temporal in (False, True):
            suffix = "temporal" if temporal else "spatial_only"
            out_path = SAMPLE_OUT / f"{method_name}_{suffix}.mp4"
            stats = video_pipeline.denoise_video(str(noisy_video), str(out_path), fn, use_temporal=temporal)
            evaluate_output(method_name, out_path, stats.frame_count, stats, temporal)
            print(f"[benchmark] video method={method_name} temporal={temporal}: {stats.fps:.2f} fps")

    for method_name, fn in VIDEO_METHODS_SAMPLE.items():
        out_path = SAMPLE_OUT / f"{method_name}_spatial_only.mp4"
        stats = video_pipeline.denoise_video(
            str(noisy_video), str(out_path), fn, use_temporal=False, max_frames=VIDEO_SAMPLE_FRAMES
        )
        evaluate_output(method_name, out_path, stats.frame_count, stats, False)
        print(f"[benchmark] video method={method_name} (sample, {VIDEO_SAMPLE_FRAMES} frames): {stats.fps:.2f} fps")

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "video_benchmark.csv", index=False)
    print("\n=== Video benchmark (FPS + quality vs dark-but-clean reference frames) ===")
    print(df.round(3).to_string(index=False))
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-only", action="store_true")
    parser.add_argument("--video-only", action="store_true")
    args = parser.parse_args()

    if not args.video_only:
        df = run_image_benchmark()
        make_chart(df)
        make_visual_comparison("astronaut_moderate")
        make_visual_comparison("camera_severe")

    if not args.images_only:
        run_video_benchmark()


if __name__ == "__main__":
    main()
