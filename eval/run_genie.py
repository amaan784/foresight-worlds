"""Genie 4 baseline: from the last conditioning frame and the official Physics-IQ prompt, stream video live.

The first 120 streamed frames are taken as the 5 s at 24 FPS (padded with the last frame if the stream is short,
and recorded). Same scoring as every other system:
  PYTHONPATH=backend:phase0 python eval/run_genie.py && PYTHONPATH=backend:phase0 python eval/run_variant.py score genie
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "phase0"), str(ROOT / "eval")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
import run_eval as E  # noqa: E402
from foresight import config  # noqa: E402
from foresight.gemini import _log, client_for  # noqa: E402
from vidio import read_frames, write_mp4  # noqa: E402

NAME, N_FRAMES = "genie", 120


async def one(scene: str, sem: asyncio.Semaphore) -> None:
    async with sem:
        bench = E.pq.benchmark_dir("gcs")
        out_dir = E.PREDS / NAME
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = E.CACHE / f"variant_{NAME}" / f"{scene}.json"
        meta.parent.mkdir(parents=True, exist_ok=True)
        cond, _ = read_frames(E.cond_path(bench, scene, E.FPS))
        ref = cv2.resize(cond[-1], (1280, 720))
        b64 = base64.b64encode(cv2.imencode(".jpg", ref, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()).decode()
        prompt = E.benchmark_prompt(scene).replace(" Continue the scene.", "")
        t0, frames = time.time(), []
        try:
            c = client_for(config.GENIE)
            inter = await c.aio.interactions.create(model=config.GENIE, background=True)
            async with c.aio.interactions.connect(inter.id) as conn:
                await conn.send({"event_type": "world_setup", "world_prompt": {"text": prompt},
                                 "reference_image": {"data": b64, "mime_type": "image/jpeg"}})
                async for ev in conn.receive():
                    d = getattr(ev, "delta", None)
                    if d is not None and getattr(d, "type", "") == "video" and getattr(d, "data", None):
                        raw = base64.b64decode(d.data) if isinstance(d.data, str) else d.data
                        f = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                        if f is not None:
                            frames.append(f)
                    if len(frames) >= N_FRAMES or time.time() - t0 > 90:
                        await conn.cancel()
                        break
            if not frames:
                raise RuntimeError("Genie streamed no video frames")
            padded = max(0, N_FRAMES - len(frames))
            frames = (frames + [frames[-1]] * padded)[:N_FRAMES]
            write_mp4(out_dir / E.pred_name(scene), frames, E.FPS)
            rec = {"ok": True, "latency_s": round(time.time() - t0, 2), "frames_received": N_FRAMES - padded, "frames_padded": padded}
        except Exception as e:  # noqa: BLE001
            E.static_prediction(bench, scene, E.VIEW, E.FPS, out_dir)
            rec = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400], "fallback": "static"}
        _log({"kind": "interactions.connect", "model": config.GENIE, "tag": f"eval:{scene}", "latency_s": round(time.time() - t0, 2),
              "request": {"world_prompt": prompt, "reference_image": "<last conditioning frame, 1280x720 jpeg>"}, "response": rec})
        meta.write_text(json.dumps(rec, indent=2))
        print(scene, json.dumps(rec)[:160], flush=True)


async def main() -> None:
    sem = asyncio.Semaphore(5)
    await asyncio.gather(*[one(s, sem) for s in E.SCENES])


if __name__ == "__main__":
    asyncio.run(main())
