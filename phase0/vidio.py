"""Small video I/O helpers (OpenCV read, ffmpeg H.264 write with exact frame counts)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def read_frames(path: str | Path, max_frames: int | None = None) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok or (max_frames is not None and len(frames) >= max_frames):
            break
        frames.append(f)
    cap.release()
    return frames, fps


def write_mp4(path: str | Path, frames: list[np.ndarray], fps: float) -> Path:
    """Write BGR frames to H.264 MP4 (yuv420p, browser-playable) at exactly len(frames) frames."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    w2, h2 = w - w % 2, h - h % 2
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w2}x{h2}", "-r", str(fps), "-i", "-",
        "-frames:v", str(len(frames)), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-r", str(fps), str(path),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f[:h2, :w2]).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")
    return path


def probe(path: str | Path) -> dict:
    """Duration (format), fps, frame count via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    d = json.loads(out)
    s = d["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {
        "duration": float(d["format"]["duration"]),
        "fps": float(num) / float(den),
        "frames": int(s["nb_read_frames"]),
        "width": int(s["width"]),
        "height": int(s["height"]),
    }
