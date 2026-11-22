"""Optional Managed Agent batch runner using antigravity-preview-09-2026.

Provisions ONE remote Linux sandbox (`environment="remote"`), installs pymunk,
scipy, and opencv-python-headless once alongside our physics modules, and reuses
that single `environment_id` across a batch of Foresight scene JSONs.

Sends lightweight trajectory/scene JSON (never raw video) and routes every API
call through `foresight.gemini.call(...)` with `foresight.gemini.client_for(config.MANAGED_AGENT)`.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from foresight import config, gemini

from .fit import fit_scene

PKG_DIR = Path(__file__).resolve().parent
REMOTE_MODULES = ["layers.py", "world.py", "fit.py"]  # fit imports world and layers; world imports layers


class RemoteBodyParams(BaseModel):
    """Fitted y-up metre parameters for a single dynamic object."""

    x0: float
    y0: float
    vx0: float
    vy0: float
    friction: float
    elasticity: float
    damping: float


class RemoteFitOutput(BaseModel):
    """Structured output returned by the Managed Agent after running fit_scene."""

    clip_id: str = Field(description="Scene clip_id that was fitted")
    motion_model: str = Field(description="Motion model used by the fitter")
    fit_error_px: float = Field(description="RMSE over fitted track points in orig px")
    fit_params: Dict[str, RemoteBodyParams] = Field(
        description="Per-object fitted parameters keyed by object id"
    )
    notes: str = Field(default="", description="Short diagnostic note from the sandbox run")


def strip_scene_for_agent(scene: dict[str, Any]) -> dict[str, Any]:
    """Strip local image/video file paths, keeping only numeric geometry and tracks."""
    slim_objects = []
    for obj in scene.get("objects", []):
        slim_objects.append({
            "id": obj["id"],
            "name": obj.get("name"),
            "shape": obj.get("shape", "circle"),
            "radius_px": obj.get("radius_px"),
            "w_px": obj.get("w_px"),
            "h_px": obj.get("h_px"),
            "est_real_size_cm": obj.get("est_real_size_cm", 6.7),
            "est_mass_g": obj.get("est_mass_g"),
            "track": obj.get("track", []),
            "visible_at_end": obj.get("visible_at_end", True),
        })
    return {
        "clip_id": scene.get("clip_id", "clip"),
        "orig_size": scene["orig_size"],
        "fps_out": scene.get("fps_out", 24),
        "t_cond_end_s": scene.get("t_cond_end_s", 3.0),
        "pred_seconds": scene.get("pred_seconds", 5.0),
        "px_per_m": scene["px_per_m"],
        "px_per_m_source": scene.get("px_per_m_source", "size"),
        "gravity_m_s2": scene.get("gravity_m_s2", 9.81),
        "motion_type_hint": scene.get("motion_type_hint", "roll"),
        "objects": slim_objects,
        "static_objects": [{k: v for k, v in st.items() if k != "sprite_png"} for st in scene.get("static_objects", [])],
        "surfaces": scene.get("surfaces", []),
    }


def build_setup_prompt() -> str:
    """Build the one-time environment bootstrap prompt installing deps and physics code."""
    files = {name: (PKG_DIR / name).read_text() for name in REMOTE_MODULES}
    listing = ", ".join(f"`/workspace/physics/{n}`" for n in files)
    blocks = "\n\n".join(f"--- /workspace/physics/{n} ---\n```python\n{src}\n```" for n, src in files.items())
    return (
        "Set up this persistent Python workspace for Foresight physics fitting:\n"
        "1. Install packages once: `pip install -q pymunk scipy opencv-python-headless numpy`.\n"
        f"2. Create `/workspace/physics/__init__.py` (empty file) and {listing} with the exact source code below.\n"
        "3. Verify `PYTHONPATH=/workspace python -c 'import pymunk, scipy, cv2; from physics.fit import fit_scene; print(\"READY\")'`.\n\n"
        f"{blocks}\n"
    )


def build_fit_prompt(scene: dict[str, Any]) -> str:
    """Build a per-clip fit prompt referencing the pre-installed `/workspace/physics` package."""
    slim = strip_scene_for_agent(scene)
    scene_json = json.dumps(slim, indent=2)
    return (
        "Run the Foresight physics fitter already installed in `/workspace/physics` on the scene JSON below.\n"
        "Write the JSON to `/workspace/current_scene.json`, execute:\n"
        "```bash\n"
        "PYTHONPATH=/workspace python -c \"\n"
        "import json\n"
        "from physics.fit import fit_scene\n"
        "s = json.loads(open('/workspace/current_scene.json').read())\n"
        "r = fit_scene(s)\n"
        "print(json.dumps({'clip_id': s['clip_id'], 'motion_model': r.motion_model, "
        "'fit_error_px': r.fit_error_px, 'fit_params': r.fit_params_dict}))\n"
        "\"\n"
        "```\n"
        "Return the exact fitted values in the structured response.\n\n"
        f"```json\n{scene_json}\n```\n"
    )


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_remote_fit_output(text: str) -> RemoteFitOutput:
    """Parse RemoteFitOutput from raw JSON or Managed Agent markdown with fenced JSON."""
    raw = (text or "").strip()
    try:
        return RemoteFitOutput.model_validate_json(raw)
    except Exception:
        pass

    matches = _FENCED_JSON_RE.findall(raw)
    for candidate in reversed(matches):
        try:
            return RemoteFitOutput.model_validate_json(candidate)
        except Exception:
            continue

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        return RemoteFitOutput.model_validate_json(raw[start : end + 1])

    raise ValueError(f"Could not extract RemoteFitOutput JSON from Managed Agent response: {raw[:240]}")


class ManagedPhysicsBatchRunner:
    """Runs batch physics fits in antigravity-preview-09-2026 reusing one environment_id."""

    def __init__(self, environment_id: str | None = None) -> None:
        self.environment_id: str | None = environment_id
        self.setup_interaction_id: str | None = None

    def _create_agent_interaction(self, *, tag: str, **kwargs: Any) -> Any:
        client = gemini.client_for(config.MANAGED_AGENT)
        return gemini.call(
            lambda model, **kw: client.interactions.create(agent=model, **kw),
            kind="interactions.create.agent",
            model=config.MANAGED_AGENT,
            tag=tag,
            **kwargs,
        )

    def ensure_environment(self) -> str:
        """Provision the remote sandbox and install pymunk/scipy/opencv once."""
        if self.environment_id:
            return self.environment_id

        resp = self._create_agent_interaction(
            tag="agent_runner:setup",
            environment="remote",
            input=build_setup_prompt(),
        )
        env_id = getattr(resp, "environment_id", None)
        if not env_id:
            raise RuntimeError(
                f"Managed Agent setup did not return environment_id (interaction id={getattr(resp, 'id', None)})"
            )
        self.environment_id = str(env_id)
        self.setup_interaction_id = getattr(resp, "id", None)
        return self.environment_id

    def fit_scene_remote(self, scene: dict[str, Any]) -> dict[str, Any]:
        """Run `fit_scene` for one scene JSON inside the reused Managed Agent sandbox."""
        env_id = self.ensure_environment()
        clip_id = str(scene.get("clip_id", "clip"))

        resp = self._create_agent_interaction(
            tag=f"agent_runner:fit:{clip_id}",
            environment=env_id,
            input=build_fit_prompt(scene),
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": RemoteFitOutput.model_json_schema(),
            },
        )
        parsed = parse_remote_fit_output(resp.output_text)
        return {
            "clip_id": parsed.clip_id,
            "motion_model": parsed.motion_model,
            "fit_error_px": parsed.fit_error_px,
            "fit_params": {oid: bp.model_dump() for oid, bp in parsed.fit_params.items()},
            "notes": parsed.notes,
            "environment_id": env_id,
            "interaction_id": getattr(resp, "id", None),
        }

    def run_batch(
        self,
        scenes: list[dict[str, Any]],
        *,
        pace_s: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Fit a batch of scenes sequentially in one reused sandbox environment."""
        if not scenes:
            return []
        self.ensure_environment()
        results: list[dict[str, Any]] = []
        for idx, scene in enumerate(scenes):
            if idx > 0 and pace_s > 0:
                time.sleep(pace_s)
            results.append(self.fit_scene_remote(scene))
        return results


def fit_batch_local(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Local deterministic fallback when eval runs without the Managed Agent."""
    out: list[dict[str, Any]] = []
    for scene in scenes:
        res = fit_scene(scene)
        out.append({
            "clip_id": str(scene.get("clip_id", "clip")),
            "motion_model": res.motion_model,
            "fit_error_px": res.fit_error_px,
            "fit_params": res.fit_params_dict,
            "environment_id": None,
        })
    return out
