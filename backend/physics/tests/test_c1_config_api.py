"""Offline tests for Task C1: public model config defaults, client routing, Genie logging wrapper, and API info."""
from __future__ import annotations

import numpy as np

from foresight import api, config, gemini, genie


def test_config_public_defaults_and_fallbacks() -> None:
    assert config.PERCEPTION == "gemini-3.8-flash"
    assert config.FALLBACK_MODEL == "gemini-3.6-flash"
    assert config.VERIFY == "gemini-3.8-flash"
    assert config.VERIFY_MODE in ("structured", "decisions")
    assert config.REPAIR_MODE in ("local", "llm")
    assert config.REPAIR == "gemini-3.8-flash"
    assert config.CARBON == config.REPAIR
    assert config.FALLBACK["gemini-3.8-flash"] == "gemini-3.6-flash"
    assert config.FALLBACK["gemini-3.6-flash"] == "gemini-3.8-flash"
    assert config.FALLBACK["gemini-4-argon"] == "gemini-3.8-flash"
    assert config.FALLBACK["gemini-4-hydrogen"] == "gemini-4-neon"


def test_client_for_routes_v1_vs_v1beta() -> None:
    assert gemini.client_for("gemini-3.8-flash") is gemini.client()
    assert gemini.client_for("gemini-3.6-flash") is gemini.client()
    assert gemini.client_for("gemini-4-argon") is gemini.client()
    assert gemini.client_for("gemini-omni-1.1-flash") is gemini.beta_client()
    assert gemini.client_for("gemini-omni-1.2-flash") is gemini.beta_client()
    assert gemini.client_for("genie-4") is gemini.beta_client()
    assert gemini.client_for("gemini-decision-1") is gemini.beta_client()
    assert gemini.client_for("gemini-3.1-pro-preview") is gemini.beta_client()
    assert gemini.client_for("antigravity-preview-09-2026") is gemini.beta_client()


def test_api_info_reflects_config_toggles_and_models() -> None:
    inf = api.info()
    assert inf["physics_available"] is True
    assert inf["omni_enabled"] is config.OMNI_ENABLED
    assert inf["genie_enabled"] is config.GENIE_ENABLED
    assert inf["verify_mode"] == config.VERIFY_MODE
    assert inf["repair_mode"] == config.REPAIR_MODE
    assert inf["models"]["perception"] == config.PERCEPTION
    assert inf["models"]["fallback"] == config.FALLBACK_MODEL
    expected_repair = "local-multistart" if config.REPAIR_MODE == "local" else config.REPAIR
    assert inf["models"]["repair"] == expected_repair
    assert inf["models"]["baseline_video"] == config.OMNI
    assert inf["models"]["world_model"] == config.GENIE


def test_genie_stream_to_mp4_uses_write_mp4_and_pads(tmp_path, monkeypatch) -> None:
    dummy_frame = np.zeros((36, 64, 3), dtype=np.uint8)

    async def _fake_stream(model: str, ref_b64: str, prompt: str, n_frames: int):
        assert model == "genie-4"
        return [dummy_frame, dummy_frame]

    monkeypatch.setattr(genie, "_stream", _fake_stream)
    out_mp4 = tmp_path / "genie_test.mp4"
    res = genie._stream_to_mp4(
        model="genie-4",
        world_prompt="test prompt",
        reference_image={"data": "abc", "mime_type": "image/jpeg"},
        out_path=str(out_mp4),
        fps=24,
        n_frames=5,
    )
    assert out_mp4.exists()
    assert res == {"frames_received": 2, "frames_padded": 3}
