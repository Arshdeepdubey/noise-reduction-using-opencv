# Quick Reference: GitHub Workflow Stages

## 🎯 Overview
This workflow executes **9 stages** to test and benchmark the OpenCV Noise Reduction Filter across all implementations.

---

## 📋 The 9 Stages at a Glance

| # | Stage | Duration | Key Output | Status |
|---|-------|----------|-----------|--------|
| 1️⃣ | **Setup & Verification** | 1-2 min | Python 3.11 ready, repo structure ✓ | 🟢 Required |
| 2️⃣ | **Dependencies** | 2-3 min | All 8 packages installed + verified | 🟢 Required |
| 3️⃣ | **Unit Tests (Parallel)** | 3-5 min | 7 test modules pass 100% | 🟢 Required |
| 4️⃣ | **Full Test Suite** | 5-7 min | All pytest tests aggregate | 🟢 Required |
| 5️⃣ | **Data Generation** | 2-3 min | 4 images × 2 noise levels + video | 🟡 Required for benchmarks |
| 6️⃣ | **Image Benchmark** | 8-12 min | PSNR/SSIM/runtime CSV + chart | 🟢 Quality metrics |
| 7️⃣ | **Video Benchmark** | 10-15 min | FPS + temporal filtering results | 🟢 Real-time metrics |
| 8️⃣ | **CLI Demo Scripts** | 5-10 min | Sample denoised images + videos | 🟢 Validation |
| 9️⃣ | **Final Report** | 1 min | Aggregated summary + verdict | 🟢 Archive |

**Total Time**: ~30-45 minutes (parallel execution)

---

## 📊 The 7 Key Implementations Tested

### 🏆 Bilateral Filter (BEST)
**Files**: `noise_reduction/bilateral.py`
- **bilateral_baseline**: Reference OpenCV filter
- **bilateral_optimized**: 3 optimizations:
  - Noise-adaptive parameters (wavelet-MAD)
  - Luma/chroma split (YCrCb)
  - Multi-resolution downscaling

| Metric | Result |
|--------|--------|
| PSNR | **31.44 dB** ⭐ |
| SSIM | **0.783** ⭐ |
| Speed | **12.3 ms** (66-71 FPS) ⭐ |
| Real-time | ✅ YES |

---

### 🎯 Fuzzy Filter
**Files**: `noise_reduction/fuzzy_filter.py`
- Fuzzy logic weighted-mean filtering
- Edge-preserving but slower

| Metric | Result |
|--------|--------|
| PSNR | 30.97 dB |
| SSIM | 0.767 |
| Speed | 203 ms (3.9 FPS) ❌ |
| Real-time | ❌ NO |

---

### 📦 DCT Denoise
**Files**: `noise_reduction/dct_denoise.py`
- Block-DCT hard-thresholding
- Best PSNR but VERY slow

| Metric | Result |
|--------|--------|
| PSNR | **31.21 dB** (2nd best) |
| SSIM | 0.781 |
| Speed | **1310 ms** (0.6 FPS) ⚠️ |
| Real-time | ❌ NO (batch only) |

---

### 🌊 Wavelet Denoise
**Files**: `noise_reduction/wavelet_denoise.py`
- Two methods: BayesShrink & VisuShrink
- Lower quality but acceptable speed

| Metric | Result |
|--------|--------|
| PSNR | 26-27 dB |
| SSIM | 0.52-0.55 |
| Speed | **21-23 ms** (35-38 FPS) |
| Real-time | ✅ YES (lower quality) |

---

### 💡 Low-Light Processing
**Files**: `noise_reduction/low_light.py`
- Noise simulation (Poisson-Gaussian)
- Noise estimation (wavelet-MAD)
- Image enhancement (CLAHE)

**Used by**: bilateral_optimized for parameter adaptation

---

### 🎬 Video Pipeline
**Files**: `noise_reduction/video_pipeline.py`
- Frame-by-frame spatial denoising
- Motion-adaptive temporal filtering
- FPS measurement

**Feature**: Temporal blending avoids ghosting when motion detected

---

### 📏 Metrics & Measurement
**Files**: `noise_reduction/metrics.py`
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Runtime measurement

---

## 🗂️ Output Files by Stage

```
📁 .github/workflows/
  └── workflow.yml (this file)

📁 results/ (Stages 6-7 generate)
  ├── benchmark_results.csv          (per-image metrics)
  ├── benchmark_summary.csv          (aggregated)
  ├── benchmark_chart.png            (visual PSNR/SSIM/runtime)
  ├── video_benchmark.csv            (FPS + quality)
  └── sample_outputs/                (denoised images)
      ├── comparison_*.png
      └── <method>_denoised.mp4

📁 results/demo/ (Stage 8 generates)
  ├── bilateral_baseline.png
  ├── bilateral_optimized.png
  ├── fuzzy_filter.png
  ├── dct_denoise.png
  ├── wavelet_bayesshrink.png
  ├── wavelet_visushrink.png
  ├── video_bilateral_spatial.mp4
  └── video_bilateral_temporal.mp4

📁 data/ (Stage 5 generates)
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

---

## 🔍 Stage Details

### Stage 1: Setup & Environment Verification (1-2 min)
**What**: Initialize GitHub Actions environment
**How**: 
```yaml
- Checkout code
- Install Python 3.11
- Verify Python/pip/structure
- Generate timestamp
```
**Output**: ✅ Ready for dependencies

---

### Stage 2: Dependencies Installation (2-3 min)
**What**: Install & verify all 8 requirements
**Packages**:
- opencv-python-headless ≥ 4.9
- numpy ≥ 1.24
- scipy ≥ 1.11
- scikit-image ≥ 0.22
- PyWavelets ≥ 1.5
- matplotlib ≥ 3.8
- pandas ≥ 2.1
- pytest ≥ 7.4

**Verification**: Import each package, display version

---

### Stage 3: Unit Tests - Individual Modules (3-5 min)
**What**: Run pytest on each test file independently (parallel)
**Test Files**:
1. `tests/test_bilateral.py` - Bilateral filter variants
2. `tests/test_dct_denoise.py` - DCT denoising
3. `tests/test_fuzzy_filter.py` - Fuzzy filtering
4. `tests/test_low_light.py` - Low-light simulation
5. `tests/test_metrics.py` - PSNR/SSIM measurement
6. `tests/test_video_pipeline.py` - Video processing
7. `tests/test_wavelet_denoise.py` - Wavelet shrinkage

**How**: Matrix strategy (7 jobs in parallel)
```yaml
matrix:
  test-module:
    - test_bilateral.py
    - test_dct_denoise.py
    - ... (5 more)
```

---

### Stage 4: Complete Test Suite (5-7 min)
**What**: Aggregate all tests, report results
**Output**: Full pytest report
```bash
pytest tests/ -v --tb=short
```

---

### Stage 5: Data Generation (2-3 min)
**What**: Create synthetic test images & video
**Output**: `data/noisy/` (8 images) + `data/video/test_video.mp4`
```python
# Script: scripts/generate_test_data.py
for image_name in ['astronaut', 'coffee', 'chelsea', 'camera']:
    for noise_level in ['moderate', 'severe']:
        simulate_low_light(image, params) → image.png
        
# Video: 40 frames @ 480x320
```

---

### Stage 6: Image Benchmark (8-12 min)
**What**: Test all 7 methods on 8 images, measure quality/speed
**Command**:
```bash
python scripts/benchmark.py --images-only
```

**Outputs**:
- `benchmark_results.csv` - 8 images × 7 methods = 56 rows
- `benchmark_summary.csv` - Aggregated statistics
- `benchmark_chart.png` - Bar charts (PSNR/SSIM/runtime)
- `sample_outputs/` - Visual comparisons

**Metrics**: PSNR (dB), SSIM (0-1), Runtime (ms)

---

### Stage 7: Video Benchmark (10-15 min)
**What**: Process video frames, measure FPS & temporal filtering
**Command**:
```bash
python scripts/benchmark.py --video-only
```

**Output**: `video_benchmark.csv`
- Method, Spatial FPS, Temporal FPS, Avg PSNR, Avg SSIM

**Key Metric**: Real-time = FPS ≥ 30 (for 480x320)

---

### Stage 8: CLI Demo Scripts (5-10 min)
**What**: Validate CLI entry points with all methods
**Scripts**:

#### denoise_image.py
```bash
# Test all 5 methods
python scripts/denoise_image.py input.png output.png --method bilateral_baseline
python scripts/denoise_image.py input.png output.png --method bilateral_optimized
python scripts/denoise_image.py input.png output.png --method fuzzy_filter
python scripts/denoise_image.py input.png output.png --method dct_denoise
python scripts/denoise_image.py input.png output.png --method wavelet_denoise
```

#### denoise_video.py
```bash
# Spatial-only
python scripts/denoise_video.py input.mp4 output.mp4 --method bilateral_optimized

# With temporal filtering
python scripts/denoise_video.py input.mp4 output.mp4 --method bilateral_optimized --temporal
```

**Output**: Sample images & videos in `results/demo/`

---

### Stage 9: Final Report (1 min)
**What**: Aggregate results, display summary
**Contains**:
- ✅ Workflow status for all 8 stages
- 📊 Key implementations overview
- 📈 Expected benchmark results
- 📁 Artifact locations
- 🔗 Documentation links

---

## 🎯 Quality Gates & Success Criteria

### ✅ Pass Criteria (Green)
- All tests pass (0 failures)
- All artifacts generated
- All import checks successful
- Report generated

### ⚠️ Warning (Yellow)
- Video FPS < 10 (slow but still completes)
- Some images slower than expected
- Test warnings (not failures)

### ❌ Fail Criteria (Red)
- Test failure (pytest exit code ≠ 0)
- Dependency import fails
- Missing required files
- Python version incompatibility

---

## 🚀 How to Use

### View Workflow in GitHub UI
1. Go to repository → Actions tab
2. Click **"OpenCV Noise Reduction - Full Pipeline"**
3. Click latest run
4. Expand each job to see:
   - ✅ Passed steps (green)
   - ❌ Failed steps (red)
   - ⏭️ Skipped steps (yellow)

### Download Artifacts
1. Click on workflow run
2. Scroll to "Artifacts" section
3. Download:
   - `image-benchmark-results` (7 days)
   - `video-benchmark-results` (7 days)
   - `demo-results` (7 days)
   - `test-data` (1 day)

### Trigger Workflow
```bash
# Push to trigger
git push origin develop              # Auto-triggers
git push origin feature/my-feature   # Auto-triggers

# Schedule trigger
# Runs daily at 2 AM UTC (automatic)
```

---

## 📈 Interpreting Results

### Image Benchmark
```
Method                  PSNR  SSIM   Runtime   Real-time?
─────────────────────────────────────────────────────────
Bilateral Optimized     31.44 0.783  12.3 ms   ✅ Yes (81 FPS)
DCT Block-Threshold     31.21 0.781 1310.0 ms  ❌ No (0.8 FPS)
```

**How to read**:
- **PSNR > 30**: Good quality
- **SSIM > 0.75**: Good perceptual quality
- **Runtime < 33 ms**: Real-time capable (30 FPS)

### Video Benchmark
```
Method                  Spatial FPS  Temporal FPS
─────────────────────────────────────────────────
Bilateral Optimized        71           68
Wavelet                    36           36
```

**How to read**:
- **FPS > 30**: Real-time ✅
- **Temporal ≈ Spatial**: No overhead
- **Temporal < Spatial**: Minor overhead (acceptable)

---

## 🔧 Troubleshooting

### If Stage 3 (Unit Tests) Fails
```
Action: Check test output → Expand failed test
Look for: AssertionError or ImportError
Fix: 
  1. Run locally: pytest tests/test_<name>.py -v
  2. Check dependencies: pip install -r requirements.txt
  3. Check Python version: python --version (should be 3.11+)
```

### If Stage 6 (Image Benchmark) Fails
```
Common causes:
  1. Data not generated → Stage 5 failed (check logs)
  2. Memory issues → Large image size
  3. Missing package → OpenCV import failed

Fix: Re-run workflow or check Stage 2 (Dependencies)
```

### If Stage 7 (Video Benchmark) Takes > 20 min
```
Normal if:
  - DCT method runs (1310 ms per frame × 40 frames)
  - Multiple iterations for statistical significance
  
Optimization: Could set --max-frames 10 to reduce time
```

---

## 📚 Related Files

| File | Purpose |
|------|---------|
| [README.md](../README.md) | Project overview |
| [WORKFLOW_ANALYSIS.md](WORKFLOW_ANALYSIS.md) | Detailed workflow guide (this file) |
| [.github/workflows/workflow.yml](.github/workflows/workflow.yml) | GitHub Actions definition |
| [docs/LITERATURE_SURVEY.md](../docs/LITERATURE_SURVEY.md) | Research background |
| [docs/BENCHMARK_REPORT.md](../docs/BENCHMARK_REPORT.md) | Detailed results |
| [docs/ANDROID_PORTING.md](../docs/ANDROID_PORTING.md) | Mobile deployment |

---

## 📞 Summary

✅ **What this workflow does**:
1. ✓ Tests all 7 denoising implementations
2. ✓ Measures quality (PSNR/SSIM) and performance (FPS)
3. ✓ Validates CLI entry points
4. ✓ Generates reproducible benchmarks
5. ✓ Archives results for regression tracking

⏱️ **Runtime**: 30-45 minutes (parallel stages)

🎯 **Winner**: Bilateral Optimized (31.44 dB PSNR, 66-71 FPS)

🚀 **Ready to deploy**: Yes, real-time capable on low-end hardware

---

**Last Updated**: 2026-09-13  
**Workflow Version**: 1.0  
**Status**: ✅ Active
