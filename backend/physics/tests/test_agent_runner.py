"""Unit tests for backend/physics/agent_runner.py payload construction and SDK validation."""
from __future__ import annotations

import json
from pathlib import Path

from google.genai._gaos.types.interactions import CreateAgentInteraction

from foresight import config
from physics.agent_runner import (
    RemoteFitOutput,
    build_fit_prompt,
    build_setup_prompt,
    fit_batch_local,
    parse_remote_fit_output,
    strip_scene_for_agent,
)


def test_strip_scene_removes_media_paths_and_validates_agent_request() -> None:
    scene_path = Path(__file__).resolve().parents[3] / "phase0" / "out" / "scene_ball-rolls-off" / "scene.json"
    scene = json.loads(scene_path.read_text())

    slim = strip_scene_for_agent(scene)
    assert "background_png" not in slim
    assert "keyframes" not in slim
    assert "sprite_png" not in slim["objects"][0]
    assert len(slim["objects"][0]["track"]) == 14
    assert slim["px_per_m"] == scene["px_per_m"]

    setup_req = CreateAgentInteraction(
        agent=config.MANAGED_AGENT,
        environment="remote",
        input=build_setup_prompt(),
    )
    assert setup_req.agent == config.MANAGED_AGENT
    assert setup_req.environment == "remote"

    fit_req = CreateAgentInteraction(
        agent=config.MANAGED_AGENT,
        environment="env_test123",
        input=build_fit_prompt(scene),
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": RemoteFitOutput.model_json_schema(),
        },
    )
    assert fit_req.agent == config.MANAGED_AGENT
    assert fit_req.environment == "env_test123"


def test_fit_batch_local_on_real_ball_rolls_off_scene() -> None:
    scene_path = Path(__file__).resolve().parents[3] / "phase0" / "out" / "scene_ball-rolls-off" / "scene.json"
    scene = json.loads(scene_path.read_text())

    results = fit_batch_local([scene])
    assert len(results) == 1
    r = results[0]
    assert r["clip_id"] == scene["clip_id"]
    assert r["motion_model"] == "roll"
    # At 4K (3840x2160), ball radius is 121 px; fit error should be well under 15 px
    assert r["fit_error_px"] < 15.0
    p = r["fit_params"]["obj0"]
    assert 1.0 < p["vx0"] < 1.6
    assert 0.55 <= p["elasticity"] <= 0.75
    assert 0.40 <= p["friction"] <= 0.60


def test_parse_remote_fit_output_handles_fenced_markdown() -> None:
    md = (
        "The scene JSON was fitted using `physics.fit.fit_scene`.\n\n"
        "```json\n"
        '{"clip_id": "ball-rolls-off", "motion_model": "roll", "fit_error_px": 7.884, '
        '"fit_params": {"obj0": {"x0": 0.118, "y0": 0.110, "vx0": 1.265, "vy0": -0.018, '
        '"friction": 0.5, "elasticity": 0.65, "damping": 0.424}}}\n'
        "```"
    )
    parsed = parse_remote_fit_output(md)
    assert parsed.clip_id == "ball-rolls-off"
    assert parsed.fit_error_px == 7.884
    assert parsed.fit_params["obj0"].vx0 == 1.265
