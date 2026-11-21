"""Sprite-on-background trajectory renderer producing H.264 MP4s via phase0/vidio.py.

Exports:
    render_trajectory(scene, trajectory, out_path, out_size=None) -> Path
Reused by the constant-velocity baseline in backend/foresight/ so only dynamics differ.
"""
from __future__ import annotations

import importlib.util
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_VIDIO_MOD = None

OnEvent = Callable[[str, dict[str, Any]], None]


def _vidio():
    global _VIDIO_MOD
    if _VIDIO_MOD is None:
        vidio_path = ROOT / "phase0" / "vidio.py"
        spec = importlib.util.spec_from_file_location("foresight_phase0_vidio", vidio_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load vidio helper at {vidio_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _VIDIO_MOD = mod
    return _VIDIO_MOD


def _load_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Background image not found or unreadable: {path}")
    return img


def _load_sprite_premul(
    path: str | Path,
    scale_x: float,
    scale_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load an RGBA/BGRA sprite, resize to output scale, and pad for lossless rotation.

    Returns (bgr_premul_f32, alpha_f32) centered in a square canvas of size `ceil(hypot(w, h))`.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Sprite image not found or unreadable: {path}")
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGRA)
    elif raw.shape[2] == 3:
        alpha = np.full((*raw.shape[:2], 1), 255, dtype=raw.dtype)
        raw = np.concatenate([raw, alpha], axis=2)

    sh, sw = raw.shape[:2]
    tw = max(2, int(round(sw * scale_x)))
    th = max(2, int(round(sh * scale_y)))

    # Premultiply before resizing so transparent borders never bleed dark fringes
    a = raw[:, :, 3:4].astype(np.float32) / 255.0
    bgr_pre = raw[:, :, :3].astype(np.float32) * a
    packed = np.concatenate([bgr_pre, a], axis=2)

    if tw != sw or th != sh:
        interp = cv2.INTER_AREA if (tw < sw or th < sh) else cv2.INTER_LINEAR
        packed = cv2.resize(packed, (tw, th), interpolation=interp)

    # Pad into an odd-sided square big enough to contain any 2D rotation without clipping
    diag = int(math.ceil(math.hypot(tw, th))) + 2
    if diag % 2 == 0:
        diag += 1
    canvas = np.zeros((diag, diag, 4), dtype=np.float32)
    ox = (diag - tw) // 2
    oy = (diag - th) // 2
    canvas[oy:oy + th, ox:ox + tw] = packed
    return np.ascontiguousarray(canvas[:, :, :3]), np.ascontiguousarray(canvas[:, :, 3:4])


def _composite_sprite(
    frame_bgr: np.ndarray,
    bgr_pre: np.ndarray,
    alpha: np.ndarray,
    cx: float,
    cy: float,
    angle_rad: float,
) -> None:
    """Alpha-composite a premultiplied sprite onto `frame_bgr` in-place at `(cx, cy)`."""
    h_img, w_img = frame_bgr.shape[:2]
    sh, sw = bgr_pre.shape[:2]
    half_w = sw // 2
    half_h = sh // 2

    ix = int(round(cx))
    iy = int(round(cy))
    dx0 = ix - half_w
    dy0 = iy - half_h
    dx1 = dx0 + sw
    dy1 = dy0 + sh

    # Skip completely off-screen sprites
    if dx1 <= 0 or dy1 <= 0 or dx0 >= w_img or dy0 >= h_img:
        return

    if abs(angle_rad) > 1e-4:
        angle_deg = math.degrees(angle_rad)
        rot_m = cv2.getRotationMatrix2D((sw / 2.0, sh / 2.0), angle_deg, 1.0)
        stacked = np.concatenate([bgr_pre, alpha], axis=2)
        rot = cv2.warpAffine(
            stacked,
            rot_m,
            (sw, sh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0, 0.0, 0.0, 0.0),
        )
        src_bgr = rot[:, :, :3]
        src_a = rot[:, :, 3:4]
    else:
        src_bgr = bgr_pre
        src_a = alpha

    # Clip destination and source ROIs to image boundaries
    cx0 = max(0, dx0)
    cy0 = max(0, dy0)
    cx1 = min(w_img, dx1)
    cy1 = min(h_img, dy1)

    sx0 = cx0 - dx0
    sy0 = cy0 - dy0
    sx1 = sx0 + (cx1 - cx0)
    sy1 = sy0 + (cy1 - cy0)

    roi = frame_bgr[cy0:cy1, cx0:cx1].astype(np.float32)
    sb = src_bgr[sy0:sy1, sx0:sx1]
    sa = src_a[sy0:sy1, sx0:sx1]
    blended = sb + roi * (1.0 - sa)
    frame_bgr[cy0:cy1, cx0:cx1] = np.clip(blended, 0.0, 255.0).astype(np.uint8)


def render_trajectory(
    scene: dict[str, Any],
    trajectory: list[dict[str, Any]],
    out_path: str | Path,
    out_size: tuple[int, int] | None = None,
    *,
    on_event: OnEvent | None = None,
) -> Path:
    """Render `trajectory` onto `scene["background_png"]` using per-object sprites -> MP4.

    Verifies output duration against `len(trajectory) / fps_out` (±0.001 s) with ffprobe.
    """
    if not trajectory:
        raise ValueError("trajectory must contain at least one frame")

    out_path = Path(out_path)
    fps_out = float(scene.get("fps_out", 24))
    orig_w, orig_h = int(scene["orig_size"][0]), int(scene["orig_size"][1])

    bg = _load_bgr(scene["background_png"])
    bg_h, bg_w = bg.shape[:2]

    if out_size is not None:
        tw, th = int(out_size[0]), int(out_size[1])
    else:
        tw, th = orig_w, orig_h
        max_w_env = os.environ.get("FORESIGHT_RENDER_MAX_W")
        if max_w_env:
            max_w = int(max_w_env)
            if tw > max_w:
                th = int(round(th * (max_w / float(tw))))
                tw = max_w
    # H.264 yuv420p requires even width and height
    tw = max(2, tw - (tw % 2))
    th = max(2, th - (th % 2))

    if (bg_w, bg_h) != (tw, th):
        interp = cv2.INTER_AREA if (tw < bg_w or th < bg_h) else cv2.INTER_LINEAR
        bg = cv2.resize(bg, (tw, th), interpolation=interp)

    scale_x = tw / float(orig_w)
    scale_y = th / float(orig_h)

    # Pre-load and scale sprites once
    sprites: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    obj_order: list[str] = []
    for obj in scene.get("objects", []):
        oid = obj["id"]
        obj_order.append(oid)
        sprite_path = obj.get("sprite_png")
        if sprite_path:
            sprites[oid] = _load_sprite_premul(sprite_path, scale_x, scale_y)

    # Stream one frame at a time into ffmpeg stdin (avoids holding 120 full-res frames in RAM)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{tw}x{th}", "-r", str(fps_out), "-i", "-",
        "-frames:v", str(len(trajectory)), "-c:v", "libx264", "-preset", "veryfast", "-threads", "2",
        "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
        "-r", str(fps_out), str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    frame = np.empty_like(bg)
    try:
        for step in trajectory:
            np.copyto(frame, bg)
            objs_state = step.get("objects", {})
            for oid in obj_order:
                st = objs_state.get(oid)
                if st is None or oid not in sprites:
                    continue
                if not st.get("visible", True):
                    continue
                bgr_pre, alpha = sprites[oid]
                _composite_sprite(
                    frame,
                    bgr_pre,
                    alpha,
                    cx=float(st["x"]) * scale_x,
                    cy=float(st["y"]) * scale_y,
                    angle_rad=float(st.get("angle", 0.0)),
                )
            proc.stdin.write(np.ascontiguousarray(frame[:th, :tw]).tobytes())
    finally:
        proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {out_path}")

    vidio = _vidio()
    written = out_path
    info = vidio.probe(written)

    expected_dur = len(trajectory) / fps_out
    if abs(info["duration"] - expected_dur) > 0.0015:
        raise RuntimeError(
            f"Rendered MP4 duration {info['duration']:.6f}s != expected {expected_dur:.6f}s "
            f"(frames={info['frames']}, fps={info['fps']})"
        )
    if info["frames"] != len(trajectory):
        raise RuntimeError(
            f"Rendered MP4 frame count {info['frames']} != expected {len(trajectory)}"
        )

    if on_event is not None:
        work_dir = scene.get("work_dir", "")
        job_name = Path(work_dir).name if work_dir else out_path.parent.name
        on_event("render", {
            "url": f"/outputs/{job_name}/{out_path.name}",
            "frames": info["frames"],
            "fps": info["fps"],
            "duration_s": round(info["duration"], 3),
        })
    return written
