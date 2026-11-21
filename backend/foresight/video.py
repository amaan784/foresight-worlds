"""Video I/O for the backend: streaming reads at processing size, exact-frame-count H.264 writes."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def read_video(path: str | Path, proc_w: int | None = None, keep_full: set[int] | None = None,
               max_frames: int | None = None) -> dict:
    """Stream a video. Returns {fps, orig_size, n, proc (list, resized to proc_w if given), full {idx: frame}}."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    proc, full, i = [], {}, 0
    keep_full = keep_full or set()
    W = H = None
    while True:
        ok, f = cap.read()
        if not ok or (max_frames is not None and i >= max_frames):
            break
        H, W = f.shape[:2]
        if i in keep_full or -1 in keep_full:
            full[i] = f
        if proc_w and W != proc_w:
            f = cv2.resize(f, (proc_w, int(round(H * proc_w / W))), interpolation=cv2.INTER_AREA)
        proc.append(f)
        i += 1
    cap.release()
    if -1 in keep_full and i:
        full[i - 1] = full.get(i - 1)
    return {"fps": fps, "orig_size": [W, H], "n": i, "proc": proc, "full": full}


def last_frame(path: str | Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 3))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frame = f
    cap.release()
    return frame


def write_mp4(path: str | Path, frames: list[np.ndarray], fps: float) -> Path:
    """BGR frames -> H.264 yuv420p MP4 with exactly len(frames) frames (120 @ 24 -> 5.000 s)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    w2, h2 = w - w % 2, h - h % 2
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w2}x{h2}",
           "-r", str(fps), "-i", "-", "-frames:v", str(len(frames)), "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-crf", "20", "-r", str(fps), "-movflags", "+faststart", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f[:h2, :w2]).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")
    return path


def transcode(src: str | Path, dst: str | Path, fps: float, start_s: float = 0.0, dur_s: float | None = None,
              max_w: int | None = None) -> Path:
    """Constant-frame-rate H.264 copy of [start_s, start_s+dur_s) (phone uploads are VFR webm/mp4/mov)."""
    vf = [f"fps={fps}"]
    if max_w:
        vf.append(f"scale='min({max_w},iw)':-2")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start_s:.3f}", "-i", str(src)]
    if dur_s is not None:
        cmd += ["-t", f"{dur_s:.3f}"]
    cmd += ["-vf", ",".join(vf), "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-crf", "18", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)
    return Path(dst)


def probe(path: str | Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"duration": float(d["format"]["duration"]), "fps": float(num) / float(den),
            "frames": int(s["nb_read_frames"]), "width": int(s["width"]), "height": int(s["height"])}
