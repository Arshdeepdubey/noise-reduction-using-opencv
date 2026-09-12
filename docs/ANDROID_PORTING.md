# Android Porting Strategy

The project brief's final milestone is: *"Optimize, port & deploy the
model on to an Android smartphone & measure the performance."* This
document is the planning deliverable for that milestone (per this
round's agreed scope — see `README.md`); it is a concrete, sequenced
plan rather than a working Android build, so the next engineer can pick
it up directly.

## 1. Why `bilateral_optimized` is the right filter to port

Per `docs/BENCHMARK_REPORT.md`, `bilateral_optimized` had the best
average PSNR/SSIM of every method tested *and* ran at 66-71 fps at
480x320 in pure Python + stock OpenCV on a cloud CPU core. DCT
denoising and the fuzzy filter, despite competitive still-image
quality, are 15-2500x slower and were never realistic candidates for a
phone's camera pipeline running at video frame rates. Wavelet shrinkage
was both slower and lower-quality on this project's noise model. This
makes the porting decision straightforward: port the bilateral
pipeline, not the others.

## 2. Target architecture

```
Camera2 / CameraX ImageAnalysis (YUV_420_888)
        |
        v
JNI bridge  --------------------->  C++ / OpenCV (Android NDK build)
        |                                 |
        |                        bilateral_optimized (ported)
        |                        - operates on the Y plane directly
        |                          (YUV_420_888 already separates
        |                          luma/chroma -- no BGR<->YCrCb
        |                          round-trip needed on-device)
        |                        - noise-sigma estimate on a
        |                          subsampled crop (already the
        |                          resolution-independent design
        |                          used in the Python version)
        v
Preview / Encoder surface (denoised YUV back to the pipeline)
```

**Recommended stack**: OpenCV's official **Android SDK + NDK** build
(the C++ `cv::bilateralFilter` this project's Python code already calls
via `cv2.bilateralFilter` is the same underlying OpenCV C++
implementation, so porting the *algorithm structure* is mechanical —
the noise-adaptive parameter logic and luma/chroma-split logic in
`noise_reduction/bilateral.py` translate near line-for-line into C++).
Frames are pulled from `CameraX`'s `ImageAnalysis` use case (or
`Camera2`'s equivalent) in `YUV_420_888` format and handed to native
code via JNI, avoiding an RGB conversion round-trip on every frame.
Community reference implementations confirm this is the standard,
supported pattern for real-time OpenCV-on-Android camera pipelines:
[Real-Time Image Processing using Android NDK and OpenCV](https://advaitsaravade.me/real-time-image-processing-using-android-ndk-and-opencv/),
[AndroidOpenCVCamera boilerplate (OpenCV SDK + native lib)](https://github.com/J0Nreynolds/AndroidOpenCVCamera),
[JavaCamera2View / Camera2 + OpenCV4 integration](https://github.com/onuralpszr/CvCamera-Mobile).

## 3. Staged plan

1. **Native port (1-2 weeks).** Re-implement `bilateral.py`,
   `low_light.py`'s noise estimator, and `video_pipeline.py`'s temporal
   filter in C++ against OpenCV's NDK headers. This is largely direct
   translation since the Python implementation is already thin OpenCV
   calls plus NumPy-level control logic with no Python-specific
   dependencies (no pandas/skimage in the hot path — those are
   benchmark-only). `pywt`'s single-level Haar DWT used for noise
   estimation has a two-line direct OpenCV/C++ equivalent (a 2x2
   Haar analysis filter bank), so no wavelet library dependency is
   needed on-device.
2. **JNI + CameraX integration (3-5 days).** Wire `ImageAnalysis` (or a
   `Camera2` capture session) to the native filter through a thin JNI
   boundary operating directly on the `YUV_420_888` Y-plane
   `ByteBuffer`, writing the filtered result back in place or to a
   second buffer for preview/recording.
3. **On-device profiling & tuning (3-5 days).** Measure per-frame
   latency and achievable FPS on at least two representative devices
   (a recent mid-range and a recent flagship, to bound the deployment
   target) at the resolutions the target use case actually needs (e.g.
   1080p video capture, not just a small preview). Tune the
   `downscale` knob (already exposed in the Python implementation,
   §4 of `BENCHMARK_REPORT.md`) per device tier if a flagship can
   afford full-resolution filtering but a budget device cannot hold
   30 fps.
4. **Validation against the existing benchmark (2-3 days).** Re-run
   the same PSNR/SSIM methodology from `BENCHMARK_REPORT.md` on-device
   (the synthetic test images/video in `data/` can be pushed to the
   device or bundled as test assets) to confirm the native port
   produces numerically equivalent output to the Python reference
   implementation (OpenCV's C++ and Python bindings call the identical
   underlying routines, so results should match to floating-point
   rounding).

## 4. Expected performance envelope

This project's Python benchmark already runs the spatial filter at
**66-71 fps at 480x320** on a general-purpose cloud CPU core with no
platform-specific optimization, and OpenCV's native `bilateralFilter`
is the same SIMD-optimized C++ code a native Android build would call.
On that basis:

- A **native (JNI/NDK) build removes Python/interpreter overhead
  entirely**, which should comfortably clear real-time (30 fps) at
  720p-1080p on a modern mid-range or better ARM SoC, especially with
  the `downscale=2` knob available if headroom is tight.
- The dominant remaining cost on a real device is likely to be
  **memory bandwidth / YUV<->working-format conversion**, not the
  bilateral filter's arithmetic itself — this project's own
  Section 5 finding (noise estimation, not filtering, was the actual
  bottleneck until fixed) is a reminder to profile before assuming
  which stage needs optimizing on-device, rather than optimizing the
  filter kernel by default.
- These are *projections from the desktop benchmark*, not on-device
  measurements — Step 3 above (on-device profiling) is required to
  replace them with real numbers before any performance claim is
  finalized in a project report.

## 5. Longer-term / stretch options (out of scope for this round)

- **Motion-compensated temporal filtering.** `BENCHMARK_REPORT.md` §3
  documents that the current temporal EMA filter smooths slightly over
  genuine camera-jitter motion (it lacks optical flow / motion
  compensation). A production port should consider a lightweight
  motion-compensated temporal filter (e.g. using OpenCV's
  `calcOpticalFlowFarneback` or a sparse feature-tracked warp) to
  remove this trade-off, at extra per-frame cost that would need
  re-profiling on-device.
- **Trilateral / tracking-based denoising** for extreme low light, as
  surveyed in `LITERATURE_SURVEY.md` §2 (Sensors, 2025) — a natural
  next step in quality beyond plain bilateral filtering if on-device
  performance headroom allows it after Step 3's profiling.
- **GPU/NNAPI acceleration** via OpenCV's Vulkan/OpenCL backends or a
  RenderScript/GPU compute shader implementation of the bilateral
  kernel, if CPU-only performance proves insufficient on lower-end
  target devices.
- **Learned denoising** (the CVPR 2026 raw-to-raw low-light video
  denoiser referenced in `LITERATURE_SURVEY.md`) is a plausible
  future direction once a classical baseline is shipped and its
  quality ceiling is understood, but requires a training pipeline and
  an on-device inference runtime (e.g. TensorFlow Lite / NNAPI) that
  is out of scope for this project's current stage.
