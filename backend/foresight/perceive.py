"""Scene parsing with gemini-4-hydrogen: agentic video + first/last frames, native structured output."""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import List, Literal, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field

from . import config, gemini


class SceneObject(BaseModel):
    name: str = Field(description="short noun phrase, e.g. 'grey tennis ball'")
    shape: Literal["circle", "box"]
    est_real_size_cm: float = Field(description="real-world diameter (circle) or width (box) in cm")
    moving: bool = Field(description="true if this object moves at any time in the clip")
    visible_frame0: bool
    bbox_frame0: List[int] = Field(description="[ymin, xmin, ymax, xmax] normalized 0-1000 in the FIRST frame; [0,0,0,0] if not visible")
    bbox_last_frame: List[int] = Field(description="[ymin, xmin, ymax, xmax] normalized 0-1000 in the LAST frame; [0,0,0,0] if not visible")
    est_mass_g: Optional[float] = Field(default=None, description="estimated real mass in grams from everyday knowledge "
                                        "(e.g. coffee mug ~300, brick ~2000)")
    material: Literal["felt", "rubber", "hard plastic", "foam", "wood", "metal", "glass", "other"] = Field(
        default="other", description="what the object's outer surface is made of")


class SupportSurface(BaseModel):
    name: str
    kind: Literal["table", "floor", "ramp", "wall", "container", "other"]
    line: List[int] = Field(description="top edge the object rests/rolls on, as [y1, x1, y2, x2] normalized 0-1000 in the last frame")
    material: Literal["hard", "soft"] = Field(default="hard", description="soft: objects sink in and stop (bedding, cushions, foam, "
                                              "sand); hard: they bounce or roll (wood, stone, plastic, glass)")


class ScenePerception(BaseModel):
    objects: List[SceneObject]
    support_surfaces: List[SupportSurface]
    motion_type: Literal["roll", "slide", "fall", "bounce", "other"]
    motion_start_s: float = Field(description="time in seconds when the main object starts moving (or appears)")


PROMPT = """You are the perception stage of a physics simulator for robotics teams.
The video is a short side-view clip from a static camera. Image A is its first frame, image B its last frame.
List the rigid objects that move (or are about to move) and any large static objects they could hit.
Approximate each object as a circle or a box and estimate its real size in cm from everyday knowledge
(e.g. tennis ball ~6.7 cm). Give the support surfaces as line segments along their top edges.
Estimate each object's mass in grams and say what it is made of, and say whether each support surface is hard or
soft.
Classify the main object's motion. Bounding boxes and lines use [y, x] order normalized to 0-1000."""


def _all_required(schema: dict) -> dict:
    """Fields with defaults (so older cached readings still load) are still required from the model."""
    for node in [schema, *schema.get("$defs", {}).values()]:
        if "properties" in node:
            node["required"] = list(node["properties"])
            for prop in node["properties"].values():
                opts = [o for o in prop.pop("anyOf", []) if o.get("type") != "null"]
                if opts:
                    prop.update(opts[0])
                prop.pop("default", None)
    return schema


def _jpg_b64(img: np.ndarray, max_w: int = 1280) -> str:
    if img.shape[1] > max_w:
        img = cv2.resize(img, (max_w, int(img.shape[0] * max_w / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode()


def perceive(video_path: Path, first: np.ndarray, last: np.ndarray, tag: str = "",
             cache_file: Path | None = None) -> tuple[ScenePerception, dict]:
    """Returns (scene, meta). `video_path` should be a small (e.g. 960x540) copy of the conditioning clip."""
    if cache_file and cache_file.exists():
        d = json.loads(cache_file.read_text())
        return ScenePerception.model_validate(d["scene"]), {**d["meta"], "cached": True}
    f = gemini.upload_video(video_path)
    t0 = time.time()
    resp = gemini.call(
        gemini.client().interactions.create,
        kind="interactions.create", model=config.PERCEPTION, tag=tag, fallback=True,
        input=[
            {"type": "video", "uri": f.uri, "mime_type": f.mime_type, "processing": "agentic"},
            {"type": "text", "text": "Image A (first frame):"},
            {"type": "image", "data": _jpg_b64(first), "mime_type": "image/jpeg"},
            {"type": "text", "text": "Image B (last frame):"},
            {"type": "image", "data": _jpg_b64(last), "mime_type": "image/jpeg"},
            {"type": "text", "text": PROMPT},
        ],
        generation_config={"thinking_level": config.PERCEPTION_THINKING},
        response_format={"type": "text", "mime_type": "application/json", "schema": _all_required(ScenePerception.model_json_schema())},
    )
    scene = ScenePerception.model_validate_json(resp.output_text)
    meta = {"model": getattr(resp, "model", None) or config.PERCEPTION, "latency_s": round(time.time() - t0, 2), "cached": False}
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"scene": scene.model_dump(), "meta": meta}, indent=2))
    return scene, meta
