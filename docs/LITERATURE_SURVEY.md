# Literature Survey — Image-Processing-Based Noise Reduction Filters

This survey supports the project brief's first deliverable: *"Literature
survey & document several image processing based optimized noise
reduction filter algorithms... Identify one method which gives better
results over other methods."* It covers the four families named in the
brief (bilateral, fuzzy, DCT, wavelet), the three references supplied
in the original slide deck, and additional low-light/video-specific
work found during this project.

## 1. Noise sources relevant to low-light video

Digital sensor noise in a low-light / high-ISO capture is dominated by
two roughly independent components (Healey & Kondepudy, 1994):

- **Shot noise** — Poisson-distributed, from the discreteness of photon
  arrivals; its *relative* magnitude grows as scene illumination (and
  therefore photon count) drops, which is exactly what happens in a
  dim scene.
- **Read/amplifier noise** — approximately signal-independent additive
  Gaussian noise from the sensor's analog front end, whose *relative*
  contribution to the final image also grows in low light because the
  signal itself is small.

A general survey of noise types and classical filters (linear/Gaussian,
adaptive/Wiener, median) is given in Yadav & Yadav's *"Image Noise
Reduction and Filtering Techniques"* — one of the three references
supplied in the project brief. Its conclusion (median filtering being
most robust to impulse-type noise, at the cost of fine detail) informed
the choice not to rely on simple order-statistic filters alone for this
project's noise model, since low-light noise is not purely impulsive.
[Image Noise Reduction and Filtering Techniques (IJSR)](https://www.ijsr.net/archive/v6i3/25031706.pdf)

## 2. Bilateral filtering (chosen family)

**Tomasi & Manduchi (1998)**, *"Bilateral filtering for gray and color
images,"* introduced the edge-preserving bilateral filter: a weighted
average where the weight of each neighboring pixel combines spatial
closeness and range (intensity) similarity. This directly targets the
project's requirement to smooth noise *without* destroying edges/detail
in a low-light video frame, which is why it was selected as the primary
method for this implementation (see `docs/BENCHMARK_REPORT.md` for the
head-to-head numbers that confirmed this choice).

Its main limitation is cost — a naive implementation scales with
`image_size x kernel_area`, and its fixed parameters do not adapt to a
scene's actual noise level. Two lines of published work address this:

- **Paris & Durand (2006)**, *"A fast approximation of the bilateral
  filter using a signal processing approach,"* show the bilateral
  filter can be approximated in a lower-dimensional space and computed
  at reduced resolution with controlled error — the basis for this
  project's *downscale-then-upsample* optimization in
  `bilateral_optimized()`.
- **Yang, Tan & Ahuja (2009)**, *"Real-time O(1) bilateral filtering,"*
  achieve filtering cost independent of kernel size via a small number
  of pre-filtered "Principal Bilateral Filtered Image Components"
  interpolated per pixel, reporting ~10x speedups with negligible
  accuracy loss versus a brute-force filter.
  [Real-Time O(1) Bilateral Filtering (PDF)](https://vision.ai.illinois.edu/html-files-to-import/publications/yang_cvpr09.pdf)

Parameter-tuning work such as *"Optimization of bilateral filter
parameters using a whale optimization algorithm"* confirms that
hand-fixed `sigma_color`/`sigma_space` values generalize poorly across
scenes/noise levels — motivating this project's own noise-adaptive
`sigma_color` (Section 4 of `BENCHMARK_REPORT.md`), a lighter-weight
alternative to a full metaheuristic search that is cheap enough to run
every video frame.
[Optimization of bilateral filter parameters (Taylor & Francis)](https://www.tandfonline.com/doi/full/10.1080/27684830.2022.2140863)

Low-light/video-specific bilateral and related edge-aware filters:

- *"Tracking-Based Denoising: A Trilateral Filter-Based Denoiser for
  Real-World Surveillance Video in Extreme Low-Light Conditions"*
  (Sensors, 2025) extends the bilateral idea with a third,
  motion/tracking-aware kernel specifically for extreme low-light
  video — the same problem this project targets, at a more elaborate
  operating point than time allowed for here; noted as future work in
  `ANDROID_PORTING.md`. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12431109/) · [DOI](https://doi.org/10.3390/s25175567)
- Noise-adaptive spatio-temporal filters for low-light-level images
  informed this project's motion-adaptive temporal EMA filter in
  `noise_reduction/video_pipeline.py`.
  [Noise-adaptive spatio-temporal filter (ResearchGate)](https://www.researchgate.net/publication/3181175_Noise-adaptive_spatio-temporal_filter_for_real-time_noise_removal_in_low_light_level_images)
- Recent deep-learning raw-to-raw denoisers targeting real-time,
  extreme low-light UHD video (CVPR 2026) illustrate where the field is
  heading beyond classical filtering, and are discussed as a longer-term
  option in `ANDROID_PORTING.md`.
  [Efficient Real-Time Raw-to-Raw Denoising (CVPR 2026, PDF)](https://openaccess.thecvf.com/content/CVPR2026/papers/Pochimireddy_Efficient_Real-Time_Raw-to-Raw_Denoising_for_Extreme_Low-Light_Ultra_HD_Video_CVPR_2026_paper.pdf)

## 3. Fuzzy filters

Fuzzy denoising filters (e.g., Farbiz & Menhaj 1998; Kwan 2003; the
GOA/fuzzy-gradient family) replace a hard statistical range kernel with
a fuzzy membership function expressing similarity/edge-strength in
`[0, 1]`, without assuming a specific noise distribution. A recent
comprehensive survey (*"A comprehensive survey on impulse and Gaussian
denoising filters,"* Mafi et al., also published as *"Survey on mixed
impulse and Gaussian denoising filters,"* IET Image Processing, 2020)
benchmarks this family against classical filters for *mixed*
impulse+Gaussian noise — closer to a real compressed/low-light sensor
stream than a pure-Gaussian assumption.
[Survey on mixed impulse and Gaussian denoising filters (IET)](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-ipr.2018.6335) · [comprehensive survey (ScienceDirect, author manuscript)](https://www.sciencedirect.com/science/article/am/pii/S0165168418303979)

`noise_reduction/fuzzy_filter.py` implements a representative
gradient-and-similarity fuzzy weighted-mean filter in this spirit.
**Implementation lesson learned** (documented in code comments and
`BENCHMARK_REPORT.md`): computing the edge/gradient membership directly
on a noisy frame is a known pitfall — sensor noise itself produces
large Sobel responses everywhere, which fools a naive fuzzy-edge term
into treating noise as "edges" and refusing to smooth them. The
implementation here computes that membership from a lightly
median-smoothed proxy of the frame instead, which is consistent with
how published fuzzy filters pre-condition their gradient/derivative
terms.

## 4. DCT-based denoising

**Yaroslavsky's DCT-domain denoising** and its clean, reproducible
formalization by **Guoshen Yu & Guillermo Sapiro** — *"DCT image
denoising: a simple and effective image denoising algorithm"* (IPOL,
2011) — hard-threshold the 2D DCT coefficients of small (e.g. 8x8)
overlapping blocks and average the overlapping reconstructions. This
exploits the same energy-compaction property JPEG relies on: natural
image content concentrates in a few low-frequency coefficients while
additive Gaussian noise spreads uniformly, so thresholding at
`~3 sigma` removes most noise-only coefficients.
[DCT image denoising (IPOL article + reference code)](https://www.ipol.im/pub/art/2011/ys-dct/article.pdf)

`noise_reduction/dct_denoise.py` implements this algorithm directly
(overlapping 8x8 blocks, `k * sigma` hard threshold, per-pixel
aggregation by averaging). It is the highest-*quality* single-image
method measured in this project's benchmark (see
`BENCHMARK_REPORT.md`) but, run in pure Python/OpenCV, over 100x too
slow for continuous video at the tested resolution — a finding directly
relevant to the brief's "check if the method works well for a
continuous video stream" requirement.

BM3D (**Dabov, Foi, Katkovnik & Egiazarian, 2007**, referenced directly
in the project brief) is the natural extension of DCT denoising: instead
of denoising each block independently, it *groups* similar 2D patches
from across the image into 3D stacks and jointly (collaboratively)
filters and thresholds them in a 3D transform domain before aggregating
overlapping estimates. This consistently outperforms single-block DCT
thresholding in the literature, at higher computational cost again —
noted here as the natural "if more accuracy is needed than real-time
constraints allow" upgrade path, but out of scope for the real-time
low-light video target of this project.
[BM3D project page (Tampere University)](https://webpages.tuni.fi/foi/GCF-BM3D/)

## 5. Wavelet-shrinkage denoising

Wavelet denoising decomposes the image into multi-resolution
sub-bands and shrinks small coefficients (assumed noise-dominated)
toward zero:

- **Donoho & Johnstone (1994)** — *VisuShrink*: a single, universal
  threshold `sigma * sqrt(2 log N)` for the whole image. Simple and
  robust but tends to over-smooth, since one threshold is applied
  uniformly regardless of local sub-band statistics.
- **Chang, Yu & Vetterli (2000)** — *BayesShrink*: a Bayesian,
  sub-band-adaptive threshold, generally preserving more detail than
  VisuShrink on natural images.

The unr.edu survey supplied in the project brief, *"Image Denoising
Techniques"*, reaches the same conclusion after reviewing this family:
adaptive shrinkage rules (BayesShrink, SUREShrink) consistently
outperform the non-adaptive VisuShrink baseline, and notes that most
published wavelet methods assume a known noise variance and a Gaussian
noise model — an assumption this project's benchmark stress-tests
directly, since the simulated low-light noise here is a *signal-dependent*
Poisson-Gaussian mixture, not pure Gaussian.
[Image Denoising Techniques (UNR, PDF)](https://www.cse.unr.edu/~fredh/papers/conf/034-asoidt/paper.pdf)

`noise_reduction/wavelet_denoise.py` wraps `skimage.restoration.denoise_wavelet`
and exposes both rules. Counter-intuitively, this project's own
benchmark found **VisuShrink outperformed BayesShrink** on the
synthetic low-light test set (see `BENCHMARK_REPORT.md` §2) — a useful,
concrete illustration of the unr.edu survey's caveat that these rules'
relative performance is sensitive to how well the true (here,
signal-dependent) noise matches each rule's Gaussian assumption.

## 6. Comparative summary

| Method | Adapts to edges? | Handles mixed/signal-dependent noise? | Relative cost | Real-time video (this project's measurements) |
|---|---|---|---|---|
| Bilateral (optimized, this project) | Yes (range kernel) | Yes, via per-frame noise-adaptive `sigma_color` | Low | **69 fps** spatial-only, 480x320 |
| Bilateral (baseline/fixed params) | Yes | No (fixed params) | Low | 71 fps |
| DCT block-thresholding | Partially (block-local) | Assumes additive Gaussian sigma | High | 0.6 fps -- not viable |
| Fuzzy weighted-mean | Yes (fuzzy edge term) | Yes, no explicit noise model needed | High | 3.9 fps -- not viable |
| Wavelet (BayesShrink/VisuShrink) | Partially (multiscale) | Assumes additive Gaussian sigma | Medium | 30-38 fps |
| BM3D (referenced, not implemented) | Yes (patch grouping) | Yes (best-in-class quality) | Very high | Not evaluated (out of scope) |

**Conclusion**: the optimized bilateral filter was selected as the
primary method for this project's low-light video use case. It
matched or exceeded every other method's average PSNR/SSIM on the
synthetic benchmark while being one of the two fastest methods tested,
comfortably real-time in pure Python/OpenCV even before any C++/NDK
port — see `docs/BENCHMARK_REPORT.md` for the full numbers and
`docs/ANDROID_PORTING.md` for the deployment plan.
