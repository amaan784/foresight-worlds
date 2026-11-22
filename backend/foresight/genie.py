"""Genie 4 world model on the stage: last conditioning frame + a scene prompt -> 5 s of streamed video.

Same call as the eval baseline (eval/run_genie.py): a live interaction on the v1beta API, a world_setup event with
the prompt and a reference image, then video frames streamed back. The first 5 s (fps * 5 frames) are kept, padded
with the last frame if the stream ends early (the padding is reported).
"""
from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path

import cv2
import numpy as np

from . import config, gemini
from .video import read_video, write_mp4

GENERIC_PROMPT = "A static locked-off camera films a tabletop physics scene in real time with realistic physics."
STREAM_TIMEOUT_S = 90.0


async def _stream(ref_jpeg_b64: str, prompt: str, n_frames: int) -> list[np.ndarray]:
    c = gemini.client_for(config.GENIE)
    inter = await c.aio.interactions.create(model=config.GENIE, background=True)
    frames: list[np.ndarray] = []
    t0 = time.time()
    async with c.aio.interactions.connect(inter.id) as conn:
        await conn.send({"event_type": "world_setup", "world_prompt": {"text": prompt},
                         "reference_image": {"data": ref_jpeg_b64, "mime_type": "image/jpeg"}})
        async for ev in conn.receive():
            d = getattr(ev, "delta", None)
            if d is not None and getattr(d, "type", "") == "video" and getattr(d, "data", None):
                raw = base64.b64decode(d.data) if isinstance(d.data, str) else d.data
                f = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if f is not None:
                    frames.append(f)
            if len(frames) >= n_frames or time.time() - t0 > STREAM_TIMEOUT_S:
                await conn.cancel()
                break
    return frames


def continue_from_frame(cond_video: Path, out_5s: Path, prompt: str = GENERIC_PROMPT, fps: int = config.BENCH_FPS,
                        tag: str = "") -> dict:
    """Stream 5 s from Genie 4 starting at the last frame of `cond_video`; write an MP4 at `fps`."""
    cond = read_video(Path(cond_video), keep_full={-1}, keep_proc=False)
    ref = cv2.resize(cond["full"][cond["n"] - 1], (1280, 720), interpolation=cv2.INTER_AREA)
    b64 = base64.b64encode(cv2.imencode(".jpg", ref, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()).decode()
    n = int(round(config.PRED_SECONDS * fps))
    t0 = time.time()
    rec: dict = {"model": config.GENIE}
    try:
        frames = asyncio.run(_stream(b64, prompt, n))
        if not frames:
            raise RuntimeError("Genie streamed no video frames")
        padded = max(0, n - len(frames))
        frames = (frames + [frames[-1]] * padded)[:n]
        write_mp4(Path(out_5s), frames, fps)
        rec |= {"latency_s": round(time.time() - t0, 1), "frames_received": n - padded, "frames_padded": padded}
        return rec
    except Exception as e:  # noqa: BLE001
        rec |= {"latency_s": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {e}"[:400]}
        raise
    finally:
        gemini._log({"kind": "interactions.connect", "model": config.GENIE, "tag": tag, "latency_s": rec["latency_s"],
                     "request": {"world_prompt": prompt, "reference_image": "<last conditioning frame, 1280x720 jpeg>"},
                     "response": rec})
