# GitHub Workflow Analysis - OpenCV Noise Reduction

## 📋 Executive Summary

This document provides a comprehensive analysis of the **`workflow.yml`** GitHub Actions CI/CD pipeline and the **9 execution stages** that thoroughly test and validate the OpenCV Noise Reduction Filter implementation. The workflow covers all **7 key implementations** in the codebase, with full benchmarking and real-world usage demonstrations.

---

## 🏗️ Workflow Architecture

### Trigger Events
- **Push**: To `main`, `develop`, and `feature/**` branches
- **Pull Request**: Against `main` and `develop`
- **Schedule**: Daily at 2 AM UTC for regression testing

### Job Execution Model
- **9 Sequential + Parallel Stages** for optimal CI/CD performance
- **Dependency management** ensures correct execution order
- **Artifact retention** for analysis (1-7 days depending on stage)
- **Conditional execution** using `if: always()` for failure resilience

---

## 🎯 The 9 Execution Stages

### **STAGE 1: Setup & Environment Verification**
**Job**: `setup`  
**Duration**: ~1-2 minutes  
**Purpose**: Initialize the CI environment and verify prerequisites

**What It Does**:
- ✅ Checks out the repository
- ✅ Sets up Python 3.11
- ✅ Verifies Python and pip installation
- ✅ Displays complete repository structure
- ✅ Generates workflow timestamp (for artifact tracking)

**Outputs**:
- Python version metadata
- Repository structure confirmation
- Workflow timestamp for logging

**Key Command**:
```bash
python --version && pip --version
```

---

### **STAGE 2: Dependency Installation & Verification**
**Job**: `dependencies`  
**Duration**: ~2-3 minutes  
**Depends On**: `setup`

**Purpose**: Ensure all Python dependencies are correctly installed and importable

**What It Does**:
- ✅ Upgrades pip, setuptools, wheel
- ✅ Installs all 8 requirements from `requirements.txt`:
  - `opencv-python-headless >= 4.9` (image processing)
  - `numpy >= 1.24` (numerical computing)
  - `scipy >= 1.11` (signal processing)
  - `scikit-image >= 0.22` (image processing utilities)
  - `PyWavelets >= 1.5` (wavelet transforms)
  - `matplotlib >= 3.8` (visualization)
  - `pandas >= 2.1` (data analysis)
  - `pytest >= 7.4` (testing framework)
- ✅ Verifies all imports and version numbers
- ✅ Tests noise_reduction package import

**Verification Steps**:
```python
import cv2; import numpy; import scipy; import skimage
import pywt; import matplotlib; import pandas; import pytest
from noise_reduction import bilateral, dct_denoise, fuzzy_filter, ...
```

**Exit Criteria**: All imports successful, all versions compatible

---

### **STAGE 3: Unit Tests - Individual Modules**
**Job**: `unit-tests`  
**Duration**: ~3-5 minutes (parallel execution across 7 modules)  
**Depends On**: `dependencies`

**Purpose**: Validate each implementation module independently with pytest

**Parallel Test Execution**:
Each module runs in parallel (matrix strategy):
1. `test_bilateral.py` - Bilateral filter variants
2. `test_dct_denoise.py` - DCT transform denoising
3. `test_fuzzy_filter.py` - Fuzzy logic filtering
4. `test_low_light.py` - Low-light simulation & noise estimation
5. `test_metrics.py` - PSNR/SSIM measurement
6. `test_video_pipeline.py` - Video processing pipeline
7. `test_wavelet_denoise.py` - Wavelet shrinkage methods

**Test Framework**: pytest with verbose output and short tracebacks

**Example Test Coverage**:
```python
# test_bilateral.py
def test_bilateral_baseline_reduces_noise()
def test_bilateral_optimized_handles_downscale()
def test_bilateral_optimized_is_faster_than_baseline()

# test_video_pipeline.py
def test_iter_frames_yields_expected_count()
def test_denoise_video_produces_matching_frame_count()
def test_temporal_denoiser_blends_static_content()
def test_temporal_denoiser_passes_through_motion()
```

---

### **STAGE 4: Complete Test Suite with Coverage**
**Job**: `test-suite`  
**Duration**: ~5-7 minutes  
**Depends On**: `dependencies`

**Purpose**: Run full test suite with aggregate results and optional coverage analysis

**What It Does**:
- ✅ Discovers all test cases (`pytest --co`)
- ✅ Runs complete pytest suite
- ✅ Generates detailed HTML/XML reports
- ✅ Fails on first error for quick feedback

**Test Output Options**:
- `-v` (verbose): Full test names and pass/fail
- `--tb=short` (traceback): Concise error messages
- `--tb=line` (minimal output): Single-line errors only

---

### **STAGE 5: Synthetic Test Data Generation**
**Job**: `generate-test-data`  
**Duration**: ~2-3 minutes  
**Depends On**: `dependencies`

**Purpose**: Create reproducible synthetic low-light test images and video

**Generated Artifacts**:
```
data/
├── noisy/
│   ├── astronaut_moderate.png
│   ├── astronaut_severe.png
│   ├── coffee_moderate.png
│   ├── coffee_severe.png
│   ├── chelsea_moderate.png
│   ├── chelsea_severe.png
│   ├── camera_moderate.png
│   └── camera_severe.png
└── video/
    └── test_video.mp4 (40 frames @ 480x320)
```

**Data Generation Script** (`scripts/generate_test_data.py`):
- **Images**: Uses scikit-image sample images (astronaut, coffee, chelsea, camera)
- **Noise Levels**:
  - `moderate`: exposure_factor=0.35, shot_noise=10.0, read_noise=5.0
  - `severe`: exposure_factor=0.15, shot_noise=4.0, read_noise=10.0
- **Video**: 40 frames synthetic video (480x320 resolution)
- **Format**: PNG for images, MP4 for video

**Key Functions Called**:
```python
simulate_low_light(image, **params)  # Apply noise model
add_poisson_gaussian_noise(image)     # Sensor noise simulation
```

**Artifact Retention**: 1 day (for debugging, minimal storage)

---

### **STAGE 6: Image Benchmark - Quality & Performance**
**Job**: `image-benchmark`  
**Duration**: ~15-30 seconds per image × 4 images × 7 methods ≈ ~8-12 minutes  
**Depends On**: `dependencies` + `generate-test-data`

**Purpose**: Measure quality (PSNR/SSIM) and performance (runtime) across all methods

**Methods Benchmarked**:
1. **gaussian_baseline** - OpenCV Gaussian blur (reference)
2. **bilateral_baseline** - Standard bilateral filter
3. **bilateral_optimized** - 3-method optimized bilateral (⭐ BEST)
4. **fuzzy_filter** - Fuzzy weighted-mean
5. **dct_denoise** - DCT hard-thresholding
6. **wavelet_bayesshrink** - Wavelet with Bayesian shrinkage
7. **wavelet_visushrink** - Wavelet with universal threshold

**Metrics Collected**:
- **PSNR (Peak Signal-to-Noise Ratio)** in dB
  - Range: 21-31 dB (higher = better)
  - Optimal bilateral achieves **31.44 dB**
- **SSIM (Structural Similarity Index)** 0-1
  - Perceptual quality metric
  - Optimal bilateral achieves **0.783**
- **Runtime** in milliseconds per image
  - Real-time threshold: < 15 ms (66+ FPS)
  - Bilateral optimized: **12.3 ms** ✓

**Output Files**:
```
results/
├── benchmark_results.csv       # Per-image, per-method metrics
├── benchmark_summary.csv       # Averaged results (8 rows × 4 columns)
├── benchmark_chart.png         # PSNR/SSIM/runtime bar charts
└── sample_outputs/
    ├── comparison_*.png        # Side-by-side visual comparisons
    └── <method>_denoised.png   # Individual denoised outputs
```

**Example Results Table** (from README.md):
```
Method                      | PSNR (dB) | SSIM  | Runtime (ms)
────────────────────────────────────────────────────────────
Bilateral Optimized (BEST)  |   31.44   | 0.783 |   12.3
DCT Block-Thresholding      |   31.21   | 0.781 | 1310.0
Bilateral Baseline          |   30.60   | 0.713 |   11.4
Fuzzy Weighted-Mean         |   30.97   | 0.767 |  203.0
Wavelet VisuShrink          |   27.21   | 0.549 |   21.8
Wavelet BayesShrink         |   26.37   | 0.523 |   22.8
Noisy Input (Reference)     |   21.63   | 0.316 |    0.0
```

---

### **STAGE 7: Video Benchmark - FPS & Temporal Filtering**
**Job**: `video-benchmark`  
**Duration**: ~10-15 minutes (depends on video length)  
**Depends On**: `dependencies` + `generate-test-data`

**Purpose**: Evaluate continuous-video performance with temporal filtering

**What It Tests**:
- **Spatial-only denoising**: Frame-by-frame processing (no history)
- **Spatial + Temporal**: Motion-adaptive temporal blending
- **FPS Measurement**: Real-time capability assessment
- **Quality Consistency**: PSNR/SSIM across video frames

**Metrics**:
```
video_benchmark.csv columns:
├── method                 (Denoising algorithm)
├── spatial_fps            (Frames/sec, spatial-only)
├── temporal_fps           (Frames/sec, with temporal)
├── avg_psnr               (Quality over video)
├── avg_ssim               (Perceptual quality over video)
└── motion_score           (Temporal consistency metric)
```

**Video Pipeline** (`noise_reduction/video_pipeline.py`):
```python
TemporalDenoiser class:
  ├── apply(frame) → denoised_frame
  ├── motion_threshold: Detect camera/scene motion
  ├── alpha: Blending factor for temporal filtering
  └── update_history(): Maintain running average
```

**Expected Results**:
- Bilateral optimized: **66-71 FPS** (real-time at 480x320)
- Wavelet methods: **35-38 FPS** (smooth but slower)
- DCT method: **0.6 FPS** (too slow for real-time)

---

### **STAGE 8: CLI Scripts Demo - Real-World Usage**
**Job**: `demo-scripts`  
**Duration**: ~5-10 minutes  
**Depends On**: `dependencies` + `generate-test-data`

**Purpose**: Validate CLI entry points work correctly in production

**Demo 1: denoise_image.py** (all methods)
```bash
python scripts/denoise_image.py input.png output.png --method bilateral_optimized

# Tests all 5 methods:
✓ bilateral_baseline
✓ bilateral_optimized (with --downscale parameter)
✓ fuzzy_filter
✓ dct_denoise
✓ wavelet_denoise (BayesShrink + VisuShrink)
```

**CLI Features Tested**:
- Input/output path handling
- Method selection
- Optional downscaling (bilateral_optimized)
- Wavelet method selection (BayesShrink vs VisuShrink)
- `--enhance` flag (CLAHE brightening for low-light)

**Demo 2: denoise_video.py** (spatial + temporal)
```bash
python scripts/denoise_video.py input.mp4 output.mp4 \
    --method bilateral_optimized \
    --temporal \
    --temporal-alpha 0.35 \
    --max-frames 100

# Tests:
✓ Spatial-only denoising
✓ Spatial + temporal blending
✓ FPS calculation
✓ Frame counting
```

**Generated Outputs**:
```
results/demo/
├── bilateral_baseline.png
├── bilateral_optimized.png
├── fuzzy_filter.png
├── dct_denoise.png
├── wavelet_bayesshrink.png
├── wavelet_visushrink.png
├── video_bilateral_spatial.mp4
└── video_bilateral_temporal.mp4
```

---

### **STAGE 9: Final Report & Summary**
**Job**: `final-report`  
**Duration**: ~1 minute  
**Depends On**: ALL previous jobs

**Purpose**: Aggregate results and provide human-readable summary

**Report Contents**:
1. ✅ **Workflow Summary** - Status of all 8 stages
2. 📊 **Key Implementations Overview** - 7 denoising methods
3. 📈 **Expected Results Table** - Quality/performance benchmarks
4. 📁 **Artifacts Generated** - Output files and locations
5. 🔗 **Documentation References** - Links to README, papers, etc.

**Final Report Output**:
```
╔════════════════════════════════════════════════════════════╗
║  OpenCV Noise Reduction - Pipeline Execution Complete      ║
╚════════════════════════════════════════════════════════════╝

📊 WORKFLOW SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Stage 1: Environment Setup & Verification
✓ Stage 2: Dependencies Installation
✓ Stage 3: Unit Tests (Individual Modules) - 7 test files
✓ Stage 4: Complete Test Suite - Full coverage
✓ Stage 5: Synthetic Data Generation
✓ Stage 6: Image Benchmark - PSNR/SSIM/runtime
✓ Stage 7: Video Benchmark - FPS + quality
✓ Stage 8: CLI Scripts Demo - Real usage
✓ Stage 9: Final Report

[Detailed results for each stage...]
```

**Artifact Aggregation**:
- Downloads all artifacts from previous stages
- Organizes by stage (test-data, benchmarks, demo)
- Prepares for analysis and regression tracking

---

## 🔬 The 7 Key Implementations

### 1. **Bilateral Filter** (`noise_reduction/bilateral.py`)

#### `bilateral_baseline(frame, d=9, sigma_color=75, sigma_space=75)`
- Standard OpenCV bilateral filter
- Fixed parameters (baseline for comparison)
- Fast but not adaptive to noise level

#### `bilateral_optimized(frame, d=9, sigma_space=25, downscale=1)`
**3 Optimizations**:

1. **Noise-Adaptive Parameters**
   ```python
   # Estimate per-frame noise level from robust wavelet-MAD
   sigma_color = estimate_noise_sigma(frame)  
   # Scales smoothing strength to actual scene noise
   ```

2. **Luma/Chroma Split** (YCrCb Color Space)
   ```python
   # Filter luma (brightness) at full strength
   # Filter chroma (color) at reduced strength (faster)
   # Mirrors video codec practice (4:2:0 chroma subsampling)
   ```

3. **Multi-Resolution Filtering**
   ```python
   if downscale > 1:
       frame_small = downsample(frame, downscale)
       denoised_small = bilateral_filter(frame_small)
       result = upsample(denoised_small)  # 4x faster!
   ```

**Performance**: 
- **PSNR**: 31.44 dB (best)
- **SSIM**: 0.783 (best)
- **Runtime**: 12.3 ms (66-71 FPS)

---

### 2. **Fuzzy Filter** (`noise_reduction/fuzzy_filter.py`)

**Algorithm**:
- Fuzzy logic weighted-mean filtering
- Uses triangular membership functions on intensity differences
- Adaptive smoothing based on image content

**Implementation**:
```python
def fuzzy_filter(frame, d=5, membership_steepness=0.1):
    # For each pixel:
    #   1. Get neighborhood (5x5 window)
    #   2. Calculate fuzzy membership: how similar are neighbors?
    #   3. Compute weighted average with fuzzy weights
    #   4. Fuzzy IF-THEN rules for edge preservation
```

**Characteristics**:
- Preserves edges better than Gaussian
- Slower than bilateral (150+ ms overhead)
- Good SSIM (0.767) but not best

**Performance**:
- **PSNR**: 30.97 dB
- **SSIM**: 0.767
- **Runtime**: 203 ms (3.9 FPS) ❌ Not real-time

---

### 3. **DCT Denoise** (`noise_reduction/dct_denoise.py`)

**Algorithm**:
- Block-based DCT (Discrete Cosine Transform)
- Hard-thresholding in frequency domain
- Overlapping blocks with aggregation

**Implementation**:
```python
def dct_denoise(frame, block_size=8, k=4.0):
    # 1. Divide into 8×8 blocks (overlapping)
    # 2. Apply DCT to each block
    # 3. Hard-threshold coefficients (set small ones to 0)
    # 4. Inverse DCT
    # 5. Average overlapping regions
```

**Strengths**:
- Best PSNR (31.21 dB - nearly tied with bilateral!)
- Excellent SSIM (0.781)
- Good visual quality

**Weaknesses**:
- **Extremely slow**: 1310 ms per image! (0.6 FPS)
- Block artifacts visible on some images
- Impractical for real-time (but great for batch processing)

---

### 4. **Wavelet Denoise** (`noise_reduction/wavelet_denoise.py`)

**Two Shrinkage Methods**:

#### BayesShrink (Bayesian Shrinkage)
```python
def bayesshrink_shrink(coeffs, sigma):
    # Estimate noise variance in wavelet domain
    # Apply shrinkage: θ = E[X²] / (E[X²] + σ²)
    # Adaptive to local image statistics
```

#### VisuShrink (Universal Threshold)
```python
def visushrink_threshold(coeffs, size):
    # Universal threshold: λ = σ * √(2 * ln(N))
    # N = image size
    # Simpler, non-adaptive, often too strong
```

**Decomposition Levels**: Multi-scale wavelet transform
- Level 1: High-frequency detail (fine noise)
- Level 2: Medium frequencies
- Level 3+: Low-frequency structure

**Performance**:
- **PSNR**: 26-27 dB (lower than others)
- **SSIM**: 0.52-0.55 (lower perceptual quality)
- **Runtime**: 21-23 ms (35-38 FPS) ✓ Real-time but lower quality

---

### 5. **Low-Light Processing** (`noise_reduction/low_light.py`)

**Core Functions**:

#### `simulate_low_light(frame, exposure_factor, shot_noise_scale, read_noise_sigma)`
- Gamma darkening: `frame_dark = frame ** (1/gamma)`
- Poisson noise (shot noise): Proportional to light intensity
- Gaussian noise (read noise): Independent of intensity

#### `estimate_noise_sigma(frame)`
- Robust wavelet-MAD estimator
- Used by bilateral_optimized for adaptive parameters
- Resistant to edges/content variations

#### `enhance_low_light(frame)`
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Brightens dark images without over-saturation
- Used with `--enhance` flag in denoise_image.py

---

### 6. **Video Pipeline** (`noise_reduction/video_pipeline.py`)

**Class: TemporalDenoiser**
```python
class TemporalDenoiser:
    def __init__(self, alpha=0.35, motion_threshold=50.0):
        self.alpha = alpha  # Blending weight (0.35 = 35% temporal, 65% current frame)
        self.motion_threshold = motion_threshold  # Detect motion magnitude

    def apply(frame):
        # 1. Estimate motion (frame difference magnitude)
        # 2. If NO motion: blend with running average
        #    blended = alpha * avg + (1-alpha) * current
        # 3. If motion detected: use current frame (don't blend)
        #    Preserves temporal coherence without ghosting
        # 4. Update running average for next frame
```

**Function: `denoise_video(input_path, output_path, spatial_filter, use_temporal=False, ...)`**
- Frame-by-frame denoising with optional temporal blending
- Measures FPS and average runtime
- Returns statistics (frame_count, fps, avg_frame_ms)

**Video Statistics**:
```python
@dataclass
class VideoStats:
    frame_count: int
    fps: float
    avg_frame_ms: float
    total_time_sec: float
```

---

### 7. **Metrics & Measurement** (`noise_reduction/metrics.py`)

#### `psnr(image_true, image_noisy)`
```
PSNR = 10 * log10((2^bits - 1)² / MSE)
# Peak Signal-to-Noise Ratio
# Higher = better
# Typical: 20-40 dB
```

#### `ssim(image_true, image_noisy)`
```
SSIM = (2*μ_x*μ_y + C1)(2*σ_xy + C2) / 
       ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))
# Structural Similarity Index
# Perceptual quality (0-1)
# Better than MSE at matching human perception
```

#### `measure_runtime(func, *args, iterations=3)`
- Average execution time over 3 runs
- Excludes first run (warm-up)
- Returns milliseconds

---

## 📊 Workflow Dependencies Graph

```
┌─────────────────────┐
│    setup (1 min)    │
└──────────┬──────────┘
           │
    ┌──────▼──────┐
    │dependencies │
    │  (2-3 min)  │
    └──────┬──────┘
           │
    ┌──────┴───────────────────┬─────────────────┬──────────────────┐
    │                          │                 │                  │
    ▼                          ▼                 ▼                  ▼
┌─────────────┐      ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ unit-tests  │      │  test-suite      │  │generate-test-    │  │  image-benchmark │
│  (3-5 min)  │      │   (5-7 min)      │  │    data          │  │   (8-12 min)     │
└─────────────┘      │                  │  │  (2-3 min)       │  └──────────────────┘
                     └──────────────────┘  └──────────────────┘
                                                  │
                                    ┌─────────────┤
                                    │             │
                                    ▼             ▼
                            ┌──────────────┐  ┌──────────────┐
                            │video-        │  │demo-scripts  │
                            │benchmark     │  │(5-10 min)    │
                            │(10-15 min)   │  └──────────────┘
                            └──────────────┘
                                    │
                                    └────┬──────────┬────────────┐
                                         │          │            │
                                         ▼          ▼            ▼
                                    ┌───────────────────────────────────┐
                                    │    final-report (1 min)           │
                                    │ (ALL previous jobs complete)      │
                                    └───────────────────────────────────┘
                                              │
                                              ▼
                                    ✅ Pipeline Complete
```

**Execution Timeline** (Serial + Parallel):
- `setup`: 1 min
- `dependencies`: 2-3 min
- Parallel (while dependencies completes):
  - `unit-tests`: 3-5 min
  - `test-suite`: 5-7 min
  - `generate-test-data`: 2-3 min
  - `image-benchmark`: 8-12 min (after data)
  - `video-benchmark`: 10-15 min (after data)
  - `demo-scripts`: 5-10 min (after data)
- `final-report`: 1 min

**Total Pipeline Time**: ~30-45 minutes (parallel execution optimization)

---

## 🎯 Success Criteria & Exit Gates

### Stage Pass Criteria

| Stage | Success Criteria |
|-------|------------------|
| **Setup** | Python 3.11, project structure correct |
| **Dependencies** | All 8 packages importable, versions OK |
| **Unit Tests** | All 7 test modules pass (0 failures) |
| **Test Suite** | 100% test pass rate, no errors |
| **Data Gen** | All image/video files created |
| **Image Bench** | CSV reports generated, all methods complete |
| **Video Bench** | FPS > 0, quality metrics computed |
| **Demo Scripts** | All 5 methods produce output images |
| **Final Report** | Summary generated, artifacts aggregated |

### Quality Gates
- ❌ **Fail Fast**: If unit tests fail, image benchmark still runs but non-blocking
- ⚠️ **Warnings**: Slow methods (< 10 FPS) trigger warnings but don't fail
- ✅ **Success**: All artifacts present, report generated

---

## 📈 Performance Benchmarks (Expected Outputs)

### Image Benchmark Results
```
┌─────────────────────────┬──────────┬────────┬──────────┐
│ Method                  │ PSNR(dB) │ SSIM   │ Time(ms) │
├─────────────────────────┼──────────┼────────┼──────────┤
│ Bilateral Optimized ⭐  │   31.44  │ 0.783  │   12.3   │
│ DCT Denoise             │   31.21  │ 0.781  │ 1310.0   │
│ Bilateral Baseline      │   30.60  │ 0.713  │   11.4   │
│ Fuzzy Weighted-Mean     │   30.97  │ 0.767  │  203.0   │
│ Wavelet VisuShrink      │   27.21  │ 0.549  │   21.8   │
│ Wavelet BayesShrink     │   26.37  │ 0.523  │   22.8   │
│ Noisy Input (Baseline)  │   21.63  │ 0.316  │    0.0   │
└─────────────────────────┴──────────┴────────┴──────────┘

✅ VERDICT: Bilateral Optimized is BEST
   • Highest PSNR (31.44 dB)
   • Highest SSIM (0.783)
   • 2nd fastest (12.3 ms) - only 0.9 ms behind baseline
   • Real-time capable (66-71 FPS at 480x320)
```

### Video Benchmark Results
```
┌──────────────────────────────────────────────────────────┐
│ Method           │ Spatial FPS │ Temporal FPS │ Qual(dB) │
├──────────────────────────────────────────────────────────┤
│ Bilateral Opt ⭐ │    70-71    │    66-68     │   31.2   │
│ Bilateral Base   │    70-71    │    65-67     │   30.5   │
│ Wavelet (Both)   │   35-38     │    35-38     │  26-27   │
│ DCT (Very Slow)  │    0.6      │     0.6      │   31.0   │
│ Fuzzy           │     3.9      │     3.9      │   30.8   │
└──────────────────────────────────────────────────────────┘

✅ REAL-TIME CAPABLE: Bilateral Optimized achieves 66+ FPS
   even with temporal filtering (motion-adaptive blending)
```

---

## 🚀 Running the Workflow

### Local Testing (Before GitHub)
```bash
# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/

# Generate data
python scripts/generate_test_data.py

# Run benchmarks
python scripts/benchmark.py --images-only    # 1-2 min
python scripts/benchmark.py --video-only     # 5-10 min
python scripts/benchmark.py                  # Full run 10-15 min

# Try the CLI
python scripts/denoise_image.py data/noisy/astronaut_moderate.png output.png
python scripts/denoise_video.py input.mp4 output.mp4 --temporal
```

### GitHub Actions
```bash
# Trigger workflow
git push origin feature/my-feature

# Monitor in GitHub UI
# Settings → Actions → "OpenCV Noise Reduction - Full Pipeline"

# View logs
# Click job name → Expand stages → View output
```

---

## 📁 Artifact Retention Policy

| Artifact | Retention | Purpose |
|----------|-----------|---------|
| test-data | 1 day | Quick debugging, minimal disk |
| image-benchmark-results | 7 days | Quality/performance tracking |
| video-benchmark-results | 7 days | Video metrics trending |
| demo-results | 7 days | CLI validation samples |

---

## 🔍 Analysis & Insights

### Why Bilateral Optimized Wins
1. **Noise-adaptive parameters** automatically tune to frame content
2. **Luma/chroma split** leverages human vision (low-light flicker less on chroma)
3. **Multi-resolution option** provides speed/quality trade-off without sacrificing quality

### Why DCT is Impractical (Despite Best PSNR)
- 1310 ms per image = 0.6 FPS (unacceptable for real-time)
- Good for batch processing (videos shot days ago), not live streams

### Why Wavelets are Inconsistent
- No noise estimation → fixed threshold often too weak
- Good edges, but noticeable artifacts in flat regions
- Works well for moderate noise, struggles with severe noise

### Real-World Recommendation
For **low-light video** (the project's use case):
- ✅ **Use bilateral_optimized** with `downscale=1` (default, no speed loss)
- ⚙️ Enable `temporal=True` for 1-2 FPS boost and temporal stability
- 📱 Safe to deploy on Android (see ANDROID_PORTING.md)

---

## 📚 Related Documentation

- **[README.md](../README.md)** - Project overview and usage
- **[docs/LITERATURE_SURVEY.md](../docs/LITERATURE_SURVEY.md)** - Research methods & citations
- **[docs/BENCHMARK_REPORT.md](../docs/BENCHMARK_REPORT.md)** - Detailed results
- **[docs/ANDROID_PORTING.md](../docs/ANDROID_PORTING.md)** - Mobile deployment plan

---

## ✅ Workflow Quality Checklist

- ✅ All stages execute in optimal order (dependencies respected)
- ✅ Parallel execution where possible (7 unit tests in parallel)
- ✅ Artifact retention sized appropriately (1-7 days)
- ✅ Clear output messages for each stage
- ✅ Handles missing data gracefully (skips video if not generated)
- ✅ Matrix strategy for test coverage (7 modules × 1 job)
- ✅ Conditional execution for resilience (if: always())
- ✅ Human-readable final report
- ✅ Caching enabled for pip dependencies

---

**Generated**: 2026-09-13  
**Workflow Version**: 1.0  
**Python Target**: 3.11  
**Platform**: Ubuntu-latest (CI/CD)
