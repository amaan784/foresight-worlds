"""OpenCV CSRT tracking with appearance gating (CSRT rarely reports loss on its own).

Phase 0 (phase0/REPORT.md): CSRT held 12/12 visible frames, median 20 px vs. an independent
background-subtraction reference at 960x540; Robotics-ER 2 was 39 px, so CSRT is primary.
"""
from __future__ import annotations

import cv2
import numpy as np

APPEARANCE_GATE = 45.0  # mean abs BGR diff (0-255) vs. the init patch; above = lost


def clip_box(box, w: int, h: int) -> list[int]:
    x, y, bw, bh = [int(v) for v in box]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def _patch(frame: np.ndarray, box) -> np.ndarray | None:
    x, y, bw, bh = clip_box(box, frame.shape[1], frame.shape[0])
    p = frame[y:y + bh, x:x + bw]
    return cv2.resize(p, (32, 32)).astype(np.float32) if p.size else None


def norm_to_box(b: list[int], w: int, h: int) -> tuple[int, int, int, int]:
    """[ymin,xmin,ymax,xmax] normalized 0-1000 -> (x, y, w, h) pixels."""
    y0, x0, y1, x1 = b
    return (int(x0 / 1000 * w), int(y0 / 1000 * h), max(2, int((x1 - x0) / 1000 * w)), max(2, int((y1 - y0) / 1000 * h)))


def csrt_run(frames: list[np.ndarray], init_idx: int, init_box, direction: int) -> list[dict]:
    h, w = frames[0].shape[:2]
    init_box = tuple(clip_box(init_box, w, h))
    tr = cv2.TrackerCSRT_create()
    tr.init(frames[init_idx], init_box)
    ref = _patch(frames[init_idx], init_box)
    out = [{"frame": init_idx, "ok": True, "box": list(init_box), "app": 0.0}]
    misses, idx = 0, init_idx + direction
    while 0 <= idx < len(frames):
        ok, box = tr.update(frames[idx])
        box = clip_box(box, w, h)
        p = _patch(frames[idx], box)
        app = float(np.abs(p - ref).mean()) if p is not None else 999.0
        ok = bool(ok) and app < APPEARANCE_GATE
        out.append({"frame": idx, "ok": ok, "box": box, "app": round(app, 1)})
        misses = 0 if ok else misses + 1
        if misses >= 3:
            break
        idx += direction
    return out


def merge(runs: list[list[dict]], n: int) -> list[dict]:
    best: dict[int, dict] = {}
    for run in runs:
        for r in run:
            cur = best.get(r["frame"])
            if cur is None or (r["ok"] and (not cur["ok"] or r["app"] < cur["app"])):
                best[r["frame"]] = r
    return [best[i] for i in range(n) if i in best]


def track_object(frames: list[np.ndarray], bbox_first: list[int] | None, bbox_last: list[int] | None) -> list[dict]:
    """Track one object (normalized bboxes from perception) forward from frame 0 and/or backward from the last frame."""
    h, w = frames[0].shape[:2]
    runs = []
    if bbox_last and bbox_last != [0, 0, 0, 0]:
        runs.append(csrt_run(frames, len(frames) - 1, norm_to_box(bbox_last, w, h), -1))
    if bbox_first and bbox_first != [0, 0, 0, 0]:
        runs.append(csrt_run(frames, 0, norm_to_box(bbox_first, w, h), +1))
    return merge(runs, len(frames)) if runs else []


def _loo_miss(pts: list[dict], i: int) -> float:
    """Distance (px) between point i and the low-order polynomial through all the other points."""
    rest = [p for j, p in enumerate(pts) if j != i]
    t = np.array([p["t_s"] for p in rest])
    deg = 2 if len(rest) >= 4 else 1
    miss = 0.0
    for k in ("x", "y"):
        c = np.polyfit(t - t[0], np.array([p[k] for p in rest]), deg)
        miss += (np.polyval(c, pts[i]["t_s"] - t[0]) - pts[i][k]) ** 2
    return float(np.sqrt(miss))


def clean_track(pts: list[dict], size_px: float, min_keep: int = 3) -> tuple[list[dict], dict]:
    """Keep only the points that belong to the object's final free motion (orig px, real timestamps).

    1. Release: a held or stuck prefix is not free motion. Keep from the last still point before the final run of
       motion (that point is where the object was let go).
    2. Ends: a first or last point the smooth path through the other points misses by more than 0.2 object sizes is
       a tracker error (the object half inside the pipe it rolls out of, partly out of frame), not motion.
    """
    info: dict = {"n_in": len(pts)}
    if len(pts) <= min_keep:
        return pts, info | {"n_out": len(pts)}
    still = max(4.0, 0.04 * size_px)  # px per 1/30 s
    rate = [np.hypot(b["x"] - a["x"], b["y"] - a["y"]) / max(1e-3, (b["t_s"] - a["t_s"]) * 30) for a, b in zip(pts, pts[1:])]
    last_still = max((k for k, v in enumerate(rate) if v <= still), default=-1)
    moving_after = len(rate) - 1 - last_still
    if last_still >= 0 and moving_after >= 2:
        info["dropped_before_release"] = last_still + 1
        pts = pts[last_still + 1:]
    dropped = []
    while len(pts) > min_keep:
        ends = [0, 1, len(pts) - 1] if len(pts) > 4 else [0, len(pts) - 1]
        miss, i = max((_loo_miss(pts, i), i) for i in ends)
        if miss <= 0.2 * size_px:
            break
        dropped.append({"t_s": pts[i]["t_s"], "miss_px": round(miss, 1)})
        pts = pts[:i] + pts[i + 1:]
    if dropped:
        info["dropped_outliers"] = dropped
    return pts, info | {"n_out": len(pts)}


def moving_segment(track: list[dict], min_step_px: float = 1.5) -> list[dict]:
    """Ok frames from the last stationary frame before motion onward (drops the long static 'held' prefix,
    keeping one anchor frame so the release position is known)."""
    ok = [r for r in track if r["ok"]]
    if len(ok) < 2:
        return ok
    c = np.array([[r["box"][0] + r["box"][2] / 2, r["box"][1] + r["box"][3] / 2] for r in ok])
    steps = np.linalg.norm(np.diff(c, axis=0), axis=1)
    moving = np.where(steps > min_step_px)[0]
    if len(moving) == 0:
        return ok[-1:]
    start = max(0, moving[0] - 1)
    return ok[start:]
