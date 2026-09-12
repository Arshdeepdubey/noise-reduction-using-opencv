#!/usr/bin/env python3
"""CLI: denoise a single image with any of the implemented methods.

Example
-------
    python scripts/denoise_image.py input.jpg output.png --method bilateral_optimized
    python scripts/denoise_image.py input.jpg output.png --method wavelet_denoise --wavelet-method VisuShrink
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noise_reduction import bilateral, dct_denoise, fuzzy_filter, wavelet_denoise  # noqa: E402

METHODS = {
    "bilateral_baseline": lambda im, args: bilateral.bilateral_baseline(im),
    "bilateral_optimized": lambda im, args: bilateral.bilateral_optimized(im, downscale=args.downscale),
    "fuzzy_filter": lambda im, args: fuzzy_filter.fuzzy_filter(im),
    "dct_denoise": lambda im, args: dct_denoise.dct_denoise(im),
    "wavelet_denoise": lambda im, args: wavelet_denoise.wavelet_denoise(im, method=args.wavelet_method),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to the noisy input image")
    parser.add_argument("output", help="Path to write the denoised image")
    parser.add_argument("--method", choices=list(METHODS), default="bilateral_optimized")
    parser.add_argument("--downscale", type=int, default=1, help="bilateral_optimized speed/quality knob (1/2/4)")
    parser.add_argument("--wavelet-method", choices=["BayesShrink", "VisuShrink"], default="BayesShrink")
    parser.add_argument("--enhance", action="store_true", help="Also apply CLAHE low-light brightening after denoising")
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        raise SystemExit(f"Could not read image: {args.input}")

    result = METHODS[args.method](image, args)

    if args.enhance:
        from noise_reduction.low_light import enhance_low_light

        result = enhance_low_light(result)

    cv2.imwrite(args.output, result)
    print(f"[denoise_image] method={args.method} -> {args.output}")


if __name__ == "__main__":
    main()
