"""Unit tests for Task C4: held-prefix release cut (`clean_track`) and guarded gravity scale (`choose_scale`)."""
from __future__ import annotations

import pytest

from foresight import scale, track


def test_clean_track_cuts_long_held_prefix_when_released_one_frame_before_end() -> None:
    """Reproducer for ball-and-block-fall_right / _left: 30 stationary held frames + 1 late drop step."""
    size_px = 298.0  # radius_px = 149.0 -> still threshold = 11.92 px/frame
    pts = [{"t_s": round(1.9333 + i / 30.0, 4), "x": 786.0, "y": 484.0} for i in range(30)]
    pts.append({"t_s": 2.9333, "x": 778.0, "y": 492.0})  # step ~8.94 px (still <= 11.92)
    pts.append({"t_s": 2.9667, "x": 778.0, "y": 532.0})  # step 40.0 px (> 11.92 -> moving_after == 1)

    cleaned, info = track.clean_track(pts, size_px=size_px)
    assert len(cleaned) == 3
    assert info["dropped_before_release"] == 29
    assert cleaned[-1]["y"] == 532.0  # final falling point is preserved, not pruned as an outlier

    sc = scale.choose_scale(size_px=size_px, size_cm=6.7, track=cleaned, motion_type="fall")
    assert sc["source"] == "size"
    assert sc["px_per_m"] == pytest.approx(4447.76, abs=0.05)
    assert "too few free-fall points" in sc["reason"]


def test_choose_scale_rejects_shallow_ramp_mislabeled_as_fall() -> None:
    """Reproducer for marble-run-x_center: 6 noisy shallow-ramp points mislabeled as motion_type='fall'."""
    marble_track = [
        {"t_s": 2.7667, "x": 1724.0, "y": 172.0},
        {"t_s": 2.8000, "x": 1724.0, "y": 176.0},
        {"t_s": 2.8333, "x": 1724.0, "y": 180.0},
        {"t_s": 2.8667, "x": 1722.0, "y": 174.0},
        {"t_s": 2.9000, "x": 1718.0, "y": 190.0},
        {"t_s": 2.9333, "x": 1724.0, "y": 192.0},
    ]
    sc = scale.choose_scale(size_px=64.8, size_cm=3.0, track=marble_track, motion_type="fall")
    assert sc["source"] == "size"
    assert sc["px_per_m"] == pytest.approx(2160.0)
    assert sc["gravity_check"] is not None
    assert "rejected_reason" in sc["gravity_check"]
    assert sc["reason"] == sc["gravity_check"]["rejected_reason"]


def test_choose_scale_accepts_clean_free_fall_within_2x_of_size() -> None:
    """Clean free-fall parabola (8 pts, 0.233 s) at true 3000 px/m vs 2000 px/m size prior (1.50x) adopts gravity."""
    true_ppm = 3000.0
    g_px = true_ppm * scale.G
    pts = [
        {"t_s": round(2.5 + i / 30.0, 6), "x": 1000.0, "y": 200.0 + 120.0 * (i / 30.0) + 0.5 * g_px * (i / 30.0) ** 2}
        for i in range(8)
    ]
    # size_px=200, size_cm=10 -> ppm_size = 2000 px/m; true_ppm / ppm_size = 1.50x (within [0.5, 2.0], >30% diff)
    sc = scale.choose_scale(size_px=200.0, size_cm=10.0, track=pts, motion_type="fall")
    assert sc["source"] == "gravity"
    assert sc["px_per_m"] == pytest.approx(true_ppm, rel=1e-3)
    assert "clean free fall" in sc["reason"]


def test_choose_scale_rejects_clean_parabola_beyond_2x_of_size() -> None:
    """Even a noiseless parabola is rejected if it implies >2.0x divergence from the object's size scale."""
    implied_ppm = 5000.0
    g_px = implied_ppm * scale.G
    pts = [
        {"t_s": round(2.5 + i / 30.0, 6), "x": 1000.0, "y": 200.0 + 0.5 * g_px * (i / 30.0) ** 2}
        for i in range(8)
    ]
    # ppm_size = 2000 px/m -> ratio = 2.50x (> 2.0x)
    sc = scale.choose_scale(size_px=200.0, size_cm=10.0, track=pts, motion_type="fall")
    assert sc["source"] == "size"
    assert sc["px_per_m"] == pytest.approx(2000.0)
    assert "outside [0.50x, 2.00x]" in sc["reason"]


def test_choose_scale_rejects_bounce_reversal_track() -> None:
    """A bounce track (falls then rebounds upward) is not single-arc free fall and keeps the size scale."""
    ys = [300.0, 360.0, 440.0, 540.0, 660.0, 560.0, 480.0, 420.0]
    pts = [{"t_s": round(2.5 + i / 30.0, 4), "x": 900.0 + 20 * i, "y": y} for i, y in enumerate(ys)]
    sc = scale.choose_scale(size_px=150.0, size_cm=6.7, track=pts, motion_type="bounce")
    assert sc["source"] == "size"
    assert "bounce reversal" in sc["reason"]
