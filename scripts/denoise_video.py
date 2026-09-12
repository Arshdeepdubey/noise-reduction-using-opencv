#!/usr/bin/env python3
"""CLI: denoise a video file frame-by-frame (+ optional temporal filter).

Example
-------
    python scripts/denoise_video.py input.mp4 output.mp4 \\
        --method bilateral_optimized --temporal --downscale 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noise_reduction import bilateral, dct_denoise, fuzzy_filter, wavelet_denoise  # noqa: E402
from noise_reduction.video_pipeline import denoise_video  # noqa: E402

METHODS = {
    "bilateral_baseline": lambda im, args: bilateral.bilateral_baseline(im),
    "bilateral_optimized": lambda im, args: bilateral.bilateral_optimized(im, downscale=args.downscale),
    "fuzzy_filter": lambda im, args: fuzzy_filter.fuzzy_filter(im),
    "dct_denoise": lambda im, args: dct_denoise.dct_denoise(im),
    "wavelet_denoise": lambda im, args: wavelet_denoise.wavelet_denoise(im, method=args.wavelet_method),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to the noisy input video")
    parser.add_argument("output", help="Path to write the denoised video")
    parser.add_argument("--method", choices=list(METHODS), default="bilateral_optimized")
    parser.add_argument("--downscale", type=int, default=1, help="bilateral_optimized speed/quality knob (1/2/4)")
    parser.add_argument("--wavelet-method", choices=["BayesShrink", "VisuShrink"], default="BayesShrink")
    parser.add_argument("--temporal", action="store_true", help="Enable motion-adaptive temporal denoising")
    parser.add_argument("--temporal-alpha", type=float, default=0.35)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    spatial_filter = lambda frame: METHODS[args.method](frame, args)  # noqa: E731

    stats = denoise_video(
        args.input,
        args.output,
        spatial_filter,
        use_temporal=args.temporal,
        temporal_alpha=args.temporal_alpha,
        max_frames=args.max_frames,
    )

    print(
        f"[denoise_video] method={args.method} temporal={args.temporal} "
        f"frames={stats.frame_count} fps={stats.fps:.2f} avg_frame_ms={stats.avg_frame_ms:.2f} -> {args.output}"
    )


if __name__ == "__main__":
    main()
