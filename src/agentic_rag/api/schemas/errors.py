from pydantic import BaseModel

from agentic_rag.core.models import FailureMode


class ErrorResponse(BaseModel):
    """Uniform error body for every domain-error response.

    Never includes a stack trace or raw exception text — only the stable
    `code`, a human-readable `message`, and a `trace_id` the caller can hand
    back for support/debugging (spec §29: don't leak internals to consumers).
    """

    code: FailureMode
    message: str
    trace_id: str | None = None
