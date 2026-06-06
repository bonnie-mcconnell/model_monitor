"""Inference layer: batch prediction, shadow mode, and request handling.

Wraps the underlying model and routes predictions through monitoring
before returning results to callers.
"""

from __future__ import annotations
