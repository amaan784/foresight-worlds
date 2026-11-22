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


async def _stream(model: str, ref_jpeg_b64: str, prompt: str, n_frames: int) -> list[np.ndarray]:
    c = gemini.client_for(model)
    inter = await c.aio.interactions.create(model=model, background=True)
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


def _stream_to_mp4(*, model: str, world_prompt: str, reference_image: dict[str, str], out_path: str, fps: int,
                   n_frames: int) -> dict:
    frames = asyncio.run(_stream(model, reference_image["data"], world_prompt, n_frames))
    if not frames:
        raise RuntimeError("Genie streamed no video frames")
    padded = max(0, n_frames - len(frames))
    frames = (frames + [frames[-1]] * padded)[:n_frames]
    write_mp4(Path(out_path), frames, fps)
    return {"frames_received": n_frames - padded, "frames_padded": padded}


def continue_from_frame(cond_video: Path, out_5s: Path, prompt: str = GENERIC_PROMPT, fps: int = config.BENCH_FPS,
                        tag: str = "") -> dict:
    """Stream 5 s from Genie 4 starting at the last frame of `cond_video`; write an MP4 at `fps`."""
    cond = read_video(Path(cond_video), keep_full={-1}, keep_proc=False)
    ref = cv2.resize(cond["full"][cond["n"] - 1], (1280, 720), interpolation=cv2.INTER_AREA)
    b64 = base64.b64encode(cv2.imencode(".jpg", ref, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()).decode()
    n = int(round(config.PRED_SECONDS * fps))
    t0 = time.time()
    resp = gemini.call(
        _stream_to_mp4,
        kind="interactions.connect",
        model=config.GENIE,
        tag=tag,
        world_prompt=prompt,
        reference_image={"data": b64, "mime_type": "image/jpeg"},
        out_path=str(out_5s),
        fps=fps,
        n_frames=n,
    )
    return {"model": config.GENIE, "latency_s": round(time.time() - t0, 1), **resp}
