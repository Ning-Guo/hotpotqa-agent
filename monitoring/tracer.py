"""
monitoring/tracer.py

Langfuse request-trace wrapper. Gracefully disabled when:
  - LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set, OR
  - the `langfuse` package is not installed.

Usage in api.py:
    init_langfuse()                        # called once at startup
    trace = RequestTrace(question, req_id)
    span  = trace.span("classify", {"question": q})
    span.end({"qtype": "bridge"})
    trace.end({"answer": "..."}, status="OK")
"""
from __future__ import annotations

import os
from typing import Any

_langfuse = None  # module-level singleton


def init_langfuse() -> None:
    global _langfuse
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not pk or not sk:
        print("[tracer] Langfuse disabled — LANGFUSE_PUBLIC_KEY / SECRET_KEY not set")
        return
    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            public_key=pk,
            secret_key=sk,
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        print("[tracer] Langfuse enabled")
    except ImportError:
        print("[tracer] Langfuse disabled — `langfuse` package not installed")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RequestTrace:
    """
    Wraps one Langfuse trace for a single /ask request.
    All methods are no-ops when Langfuse is not configured.
    """

    def __init__(self, question: str, request_id: str, tenant_id: str = "default"):
        self._trace = None
        if _langfuse:
            self._trace = _langfuse.trace(
                name="hotpotqa-agent",
                input={"question": question},
                session_id=request_id,
                metadata={"tenant_id": tenant_id},
            )

    def span(self, name: str, input_data: dict[str, Any] | None = None) -> "_Span":
        if self._trace:
            return _Span(self._trace.span(name=name, input=input_data or {}))
        return _NoopSpan()

    def end(self, output: dict[str, Any], status: str = "OK") -> None:
        if self._trace:
            self._trace.update(output=output, status_message=status)
            _langfuse.flush()


class _Span:
    """Thin wrapper around a real Langfuse span."""

    def __init__(self, span):
        self._span = span

    def end(self, output: Any = None) -> None:
        self._span.end(output=output)


class _NoopSpan:
    """Silent no-op used when Langfuse is disabled."""

    def end(self, output: Any = None) -> None:
        pass
