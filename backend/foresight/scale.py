"""Pixels-per-metre from the object's estimated real size, cross-checked with gravity on free-fall segments."""
from __future__ import annotations

import numpy as np

G = 9.81


def px_per_m_from_size(size_px: float, size_cm: float) -> float:
    return float(size_px) / (float(size_cm) / 100.0)


def gravity_check(track: list[dict], px_per_m: float) -> dict | None:
    """Fit y(t) = a + b t + 0.5 g_px t^2 on a track (orig px, y down). Needs >= 5 points over >= 0.15 s."""
    if len(track) < 5:
        return None
    t = np.array([p["t_s"] for p in track])
    y = np.array([p["y"] for p in track])
    if t.max() - t.min() < 0.15:
        return None
    A = np.stack([np.ones_like(t), t - t[0], 0.5 * (t - t[0]) ** 2], axis=1)
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    g_px = float(coef[2])
    rmse = float(np.sqrt(np.mean((A @ coef - y) ** 2)))
    return {"g_px_s2": round(g_px, 1), "g_est_m_s2": round(g_px / px_per_m, 2), "rmse_px": round(rmse, 2),
            "n_points": len(track), "span_s": round(float(t.max() - t.min()), 3)}


def choose_scale(size_px: float, size_cm: float, track: list[dict], motion_type: str) -> dict:
    ppm = px_per_m_from_size(size_px, size_cm)
    out = {"px_per_m": round(ppm, 2), "source": "size", "gravity_check": None}
    if motion_type in ("fall", "bounce"):
        gc = gravity_check(track, ppm)
        out["gravity_check"] = gc
        # Trust gravity only with enough clean free-fall points and a clear disagreement with the size prior.
        if gc and gc["n_points"] >= 6 and gc["g_px_s2"] > 0 and abs(gc["g_est_m_s2"] / G - 1) > 0.3:
            out["px_per_m"] = round(gc["g_px_s2"] / G, 2)
            out["source"] = "gravity"
        elif gc and gc["g_px_s2"] > 0:
            out["source"] = "size+gravity"
    return out
