"""Domain error hierarchy.

Every error the system can surface to a caller is a subclass of `AgenticRAGError`
and carries a stable `code` matching the failure-mode enum in `core/models.py`
(spec §35). Routes translate these into HTTP responses without ever leaking raw
stack traces (see api/dependencies/error_handling.py, added in Phase 1's API wiring).

Never catch-and-swallow one of these into a generic success response — a caller
must always be able to distinguish "no answer because no evidence" from
"answer, possibly wrong."
"""

from __future__ import annotations

from agentic_rag.core.models import FailureMode


class AgenticRAGError(Exception):
    """Base class for all domain errors. Always carries a machine-readable code."""

    code: FailureMode = FailureMode.MODEL_ERROR

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NoKnowledgeError(AgenticRAGError):
    code = FailureMode.NO_KNOWLEDGE


class InsufficientEvidenceError(AgenticRAGError):
    code = FailureMode.INSUFFICIENT_EVIDENCE


class ConflictingEvidenceError(AgenticRAGError):
    code = FailureMode.CONFLICTING_EVIDENCE


class RetrievalError(AgenticRAGError):
    code = FailureMode.RETRIEVAL_ERROR


class ModelProviderError(AgenticRAGError):
    code = FailureMode.MODEL_ERROR


class QueryTimeoutError(AgenticRAGError):
    code = FailureMode.TIMEOUT


class BudgetExceededError(AgenticRAGError):
    code = FailureMode.BUDGET_EXCEEDED


class InvalidDocumentError(AgenticRAGError):
    code = FailureMode.INVALID_DOCUMENT


class UnsupportedFileTypeError(AgenticRAGError):
    code = FailureMode.UNSUPPORTED_FILE_TYPE


class PromptInjectionDetectedError(AgenticRAGError):
    code = FailureMode.PROMPT_INJECTION_DETECTED


class RateLimitExceededError(AgenticRAGError):
    code = FailureMode.RATE_LIMITED


class UnauthorizedError(AgenticRAGError):
    code = FailureMode.UNAUTHORIZED
