"""Baseline: gemini-omni-1.2-flash continues the 3 s clip (video extension). Proven in phase0/REPORT.md (0c)."""
from __future__ import annotations

import base64
import time
from pathlib import Path

import cv2

from . import config, gemini
from .video import probe, transcode

GENERIC_PROMPT = "Static locked-off camera, real-time, realistic physics. Continue the scene."


def continue_clip(cond_video: Path, out_5s: Path, prompt: str = GENERIC_PROMPT, fps: int = config.BENCH_FPS,
                  tag: str = "", max_w: int = 1280) -> dict:
    """Upload the conditioning clip (<= max_w wide), ask Omni to extend it, save the 5 s right after the input."""
    out_5s = Path(out_5s)
    work = out_5s.parent
    pc = probe(cond_video)
    cond_s = pc["duration"]
    small = transcode(cond_video, work / "omni_input.mp4", pc["fps"], max_w=max_w)
    f = gemini.upload_video(small)
    t0 = time.time()
    resp = gemini.call(
        gemini.client_for(config.OMNI).interactions.create, kind="interactions.create", model=config.OMNI, tag=tag,
        input=[{"type": "video", "uri": f.uri, "mime_type": f.mime_type}, {"type": "text", "text": prompt}],
        generation_config={"video_config": {"task": "extend"}},
    )
    latency = round(time.time() - t0, 1)
    vid = resp.output_video
    if vid is None or not getattr(vid, "data", None):
        raise RuntimeError(f"Omni returned no video (text: {str(resp.output_text)[:200]})")
    raw = work / "omni_raw.mp4"
    raw.write_bytes(base64.b64decode(vid.data) if isinstance(vid.data, str) else vid.data)
    pr = probe(raw)
    dur = pr["duration"]
    start = cond_s if dur >= cond_s + 4.0 else 0.0  # Omni returns input + continuation (Phase 0: 3 s + 10 s)
    transcode(raw, out_5s, fps, start_s=start, dur_s=5.0)
    po = probe(out_5s)
    padded = max(0, int(round(5.0 * fps)) - po["frames"])
    return {"latency_s": latency, "raw_duration_s": round(dur, 2), "start_offset_s": start, "frames_padded": padded,
            "size": [po["width"], po["height"]]}
