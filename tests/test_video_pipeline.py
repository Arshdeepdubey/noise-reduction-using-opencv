import cv2
import numpy as np

from noise_reduction.bilateral import bilateral_optimized
from noise_reduction.video_pipeline import TemporalDenoiser, denoise_video, iter_frames


def _write_tiny_video(path, num_frames=6, size=(48, 32)):
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    rng = np.random.default_rng(0)
    for i in range(num_frames):
        frame = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return num_frames


def test_iter_frames_yields_expected_count(tmp_path):
    video_path = tmp_path / "in.mp4"
    n = _write_tiny_video(video_path, num_frames=5)
    frames = list(iter_frames(str(video_path)))
    assert len(frames) == n
    assert frames[0].shape == (32, 48, 3)


def test_iter_frames_respects_max_frames(tmp_path):
    video_path = tmp_path / "in.mp4"
    _write_tiny_video(video_path, num_frames=8)
    frames = list(iter_frames(str(video_path), max_frames=3))
    assert len(frames) == 3


def test_denoise_video_produces_matching_frame_count(tmp_path):
    in_path = tmp_path / "in.mp4"
    out_path = tmp_path / "out.mp4"
    n = _write_tiny_video(in_path, num_frames=6)

    stats = denoise_video(str(in_path), str(out_path), lambda f: bilateral_optimized(f), use_temporal=False)

    assert stats.frame_count == n
    assert stats.fps > 0
    out_frames = list(iter_frames(str(out_path)))
    assert len(out_frames) == n


def test_denoise_video_with_temporal_filter_runs(tmp_path):
    in_path = tmp_path / "in.mp4"
    out_path = tmp_path / "out.mp4"
    _write_tiny_video(in_path, num_frames=6)

    stats = denoise_video(str(in_path), str(out_path), lambda f: f, use_temporal=True, temporal_alpha=0.5)
    assert stats.frame_count == 6


def test_temporal_denoiser_blends_static_content():
    td = TemporalDenoiser(alpha=0.5, motion_threshold=50.0)
    frame = np.full((10, 10, 3), 100, dtype=np.uint8)
    first = td.apply(frame)
    np.testing.assert_array_equal(first, frame)  # first frame: no history yet

    # A slightly different but "static" (small-diff) second frame should
    # be blended toward the running average, not passed through raw.
    second = np.full((10, 10, 3), 110, dtype=np.uint8)
    blended = td.apply(second)
    assert blended.mean() < 110  # pulled toward the previous (100) frame


def test_temporal_denoiser_passes_through_motion():
    td = TemporalDenoiser(alpha=0.5, motion_threshold=5.0)
    frame = np.full((10, 10, 3), 50, dtype=np.uint8)
    td.apply(frame)

    moved = np.full((10, 10, 3), 220, dtype=np.uint8)  # large jump = "motion"
    out = td.apply(moved)
    np.testing.assert_array_equal(out, moved)
