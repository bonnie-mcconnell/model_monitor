"""FastAPI application, routing, and request/response schemas.

The server is thin: it validates input, delegates to storage and
inference layers, and returns typed responses. No business logic lives here.
"""

from __future__ import annotations
