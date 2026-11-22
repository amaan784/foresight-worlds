"""Foresight: watch 3 s of real video, simulate the physics, predict the next 5 s."""
from __future__ import annotations

from .audit import audit_continuation, audit_video

__all__ = [
    "audit_continuation",
    "audit_video",
]
