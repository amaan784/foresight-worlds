"""Pixels-per-metre from the object's estimated real size, cross-checked with gravity on clean free-fall segments."""
from __future__ import annotations

import numpy as np

G = 9.81
MIN_GRAVITY_RATIO = 0.5
MAX_GRAVITY_RATIO = 2.0


def px_per_m_from_size(size_px: float, size_cm: float) -> float:
    return float(size_px) / (float(size_cm) / 100.0)


def gravity_check(track: list[dict], px_per_m: float, size_px: float | None = None) -> dict | None:
    """Fit y(t) = a + b t + 0.5 g_px t^2 on a track (orig px, y down). Needs >= 5 points over >= 0.15 s."""
    if len(track) < 5:
        return None
    t = np.array([p["t_s"] for p in track], dtype=float)
    y = np.array([p["y"] for p in track], dtype=float)
    span_s = float(t.max() - t.min())
    if span_s < 0.15:
        return None
    dt = t - t[0]
    A = np.stack([np.ones_like(dt), dt, 0.5 * dt ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    g_px = float(coef[2])
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    drop_px = float(y[-1] - y[0])
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((yhat - y) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else 0.0

    dy = np.diff(y)
    step_tol = max(4.0, 0.05 * float(size_px or 0.0))
    down_idx = np.where(dy > step_tol)[0]
    up_idx = np.where(dy < -step_tol)[0]
    has_bounce = bool(len(down_idx) and len(up_idx) and int(up_idx[-1]) > int(down_idx[0]))

    return {
        "g_px_s2": round(g_px, 1),
        "g_est_m_s2": round(g_px / px_per_m, 2),
        "ppm_gravity": round(g_px / G, 2) if g_px > 0 else None,
        "rmse_px": round(rmse, 2),
        "r2": round(r2, 4),
        "drop_px": round(drop_px, 1),
        "has_bounce": has_bounce,
        "n_points": len(track),
        "span_s": round(span_s, 3),
    }


def _reject(out: dict, gc: dict, reason: str) -> dict:
    gc["rejected_reason"] = reason
    out["reason"] = reason
    return out


def choose_scale(size_px: float, size_cm: float, track: list[dict], motion_type: str) -> dict:
    ppm_size = px_per_m_from_size(size_px, size_cm)
    out = {
        "px_per_m": round(ppm_size, 2),
        "px_per_m_size": round(ppm_size, 2),
        "source": "size",
        "reason": "from object size prior",
        "gravity_check": None,
    }
    if motion_type not in ("fall", "bounce"):
        out["reason"] = f"motion '{motion_type}' is not free fall; using size scale"
        return out

    gc = gravity_check(track, ppm_size, size_px=size_px)
    out["gravity_check"] = gc
    if gc is None:
        out["reason"] = f"too few free-fall points ({len(track)} pts; need >= 5 over >= 0.15 s); keeping size scale"
        return out

    if gc["n_points"] < 6:
        return _reject(out, gc, f"need >= 6 free-fall points (got {gc['n_points']}); keeping size scale")
    if gc["has_bounce"]:
        return _reject(out, gc, "track contains a bounce reversal; keeping size scale")
    if gc["g_px_s2"] <= 0:
        return _reject(out, gc, f"fitted downward acceleration ({gc['g_px_s2']} px/s^2) is not positive; keeping size scale")

    min_drop_px = max(20.0, 0.5 * float(size_px))
    if gc["drop_px"] < min_drop_px:
        return _reject(
            out,
            gc,
            f"net downward drop ({gc['drop_px']} px) < {min_drop_px:.1f} px required for free fall; keeping size scale",
        )

    max_rmse_size = max(4.0, 0.08 * float(size_px))
    max_rmse_drop = 0.05 * gc["drop_px"]
    if gc["rmse_px"] > max_rmse_size or gc["rmse_px"] > max_rmse_drop or gc["r2"] < 0.98:
        return _reject(
            out,
            gc,
            f"parabola residual too large (rmse={gc['rmse_px']} px, R^2={gc['r2']:.3f}); keeping size scale",
        )

    ppm_grav = gc["g_px_s2"] / G
    ratio = ppm_grav / ppm_size
    if not (MIN_GRAVITY_RATIO <= ratio <= MAX_GRAVITY_RATIO):
        return _reject(
            out,
            gc,
            f"gravity scale ({ppm_grav:.1f} px/m) is {ratio:.2f}x size scale ({ppm_size:.1f} px/m), outside [0.50x, 2.00x]; keeping size scale",
        )

    if abs(gc["g_est_m_s2"] / G - 1.0) > 0.3:
        out["px_per_m"] = round(ppm_grav, 2)
        out["source"] = "gravity"
        out["reason"] = (
            f"clean free fall ({gc['n_points']} pts, drop={gc['drop_px']} px, rmse={gc['rmse_px']} px, "
            f"R^2={gc['r2']:.3f}) within {ratio:.2f}x of size scale"
        )
    else:
        out["source"] = "size+gravity"
        out["reason"] = (
            f"size and gravity agree within 30% (g_est={gc['g_est_m_s2']} m/s^2, ratio={ratio:.2f}x); keeping size scale"
        )
    return out
