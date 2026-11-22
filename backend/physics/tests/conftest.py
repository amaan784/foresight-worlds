"""Shared offline CI safety fixtures: strip API keys and block accidental live Gemini calls."""
from __future__ import annotations

from typing import Any

import pytest

from foresight import gemini

ORIGINAL_GEMINI_CALL = gemini.call


def _forbid_live_gemini_call(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError(
        "Unit tests must never invoke live foresight.gemini.call(...) or require secrets; "
        "monkeypatch foresight.gemini.call (or caller) explicitly."
    )


@pytest.fixture(autouse=True)
def _enforce_offline_zero_secret_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure every unit test passes with empty API keys and makes zero live Gemini calls."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setattr(gemini, "_client", None)
    monkeypatch.setattr(gemini, "_beta_client", None)
    monkeypatch.setattr(gemini, "call", _forbid_live_gemini_call)
