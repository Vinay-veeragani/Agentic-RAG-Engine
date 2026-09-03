"""Deterministic, schema-introspecting mock LLM provider.

Rather than calling a real model, `MockLLMProvider.complete_structured`
inspects the *actual pydantic schema type* it was asked for and the prompt
text, then fills each field with a deterministic, heuristically-derived
value: keyword rules for fields whose name signals intent (`query_type`,
`decompose`, `subqueries`, `sufficient`, ...), generic rules by Python type
otherwise (bool/int/float/str/Enum/list/nested BaseModel).

This is not a language model — it cannot handle a schema or a prompt it has
no heuristic for gracefully beyond generic type-based defaults. It exists so
every agent built on `LLMProvider.complete_structured` (query analysis,
planning, expansion, decomposition, evidence assessment, and later citation/
synthesis agents) has a zero-dependency, fully offline path end to end, the
same role `MockEmbeddingProvider` and `MockReranker` play for their layers.
"""

from __future__ import annotations

import re
import types
import typing
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel

from agentic_rag.core.models import QueryType, RetrievalStrategy

T = TypeVar("T", bound=BaseModel)

_COMPARISON_WORDS = ("compare", "versus", " vs ", "difference between", "which is better")
_TEMPORAL_WORDS = ("changed", "trend", "over time", "since", "compared to last")
_ANALYTICAL_WORDS = ("why", "how does", "how did", "explain", "analy", "cause of", "reason")
_SUMMARY_WORDS = ("summar", "overview of", "give me an overview")
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


@dataclass(slots=True, frozen=True)
class _Context:
    """Everything a field-filling rule might need, extracted once per
    `complete_structured` call rather than re-parsed per field."""

    query: str
    evidence: str


def _classify_query_type(query: str) -> QueryType:
    q = query.lower().strip()
    if not q or len(q.split()) <= 1:
        return QueryType.AMBIGUOUS
    if any(w in q for w in _COMPARISON_WORDS):
        return QueryType.COMPARISON
    if len(_YEAR_PATTERN.findall(q)) >= 2 or any(w in q for w in _TEMPORAL_WORDS):
        return QueryType.TEMPORAL
    if any(w in q for w in _SUMMARY_WORDS):
        return QueryType.SUMMARIZATION
    if " and " in q and ("?" in q or q.count(" and ") >= 2):
        return QueryType.MULTI_HOP
    if any(w in q for w in _ANALYTICAL_WORDS):
        return QueryType.ANALYTICAL
    return QueryType.SIMPLE_FACTUAL


def _should_decompose(query: str) -> bool:
    query_type = _classify_query_type(query)
    return query_type in (QueryType.COMPARISON, QueryType.MULTI_HOP) or query.lower().count(
        " and "
    ) >= 2


def _should_expand(query: str) -> bool:
    query_type = _classify_query_type(query)
    if query_type in (QueryType.AMBIGUOUS, QueryType.ANALYTICAL, QueryType.SUMMARIZATION):
        return True
    return len(query.split()) <= 6


def _derive_expanded_queries(query: str) -> list[str]:
    base = query.strip().rstrip("?")
    if not base:
        return [query] if query else ["query"]
    variants = [base, f"{base} details", f"information about {base}"]
    seen: list[str] = []
    for v in variants:
        if v not in seen:
            seen.append(v)
    return seen[:5]


def _derive_subqueries(query: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\band\b|,", query, flags=re.IGNORECASE) if p.strip()]
    return parts[:8] if len(parts) > 1 else [query.strip() or "query"]


def _is_evidence_sufficient(ctx: _Context) -> bool:
    """Heuristic stand-in for real evidence judgment: some minimal lexical
    overlap between the query and the evidence text. Not semantically
    meaningful — same caveat as every other mock in this codebase — but
    lets tests exercise "insufficient evidence -> refine -> retry" without a
    real model, by simply retrieving evidence that doesn't share query terms."""
    if not ctx.evidence.strip():
        return False
    query_terms = {w for w in ctx.query.lower().split() if len(w) > 2}
    evidence_terms = set(ctx.evidence.lower().split())
    if not query_terms:
        return True
    overlap = len(query_terms & evidence_terms)
    return overlap >= max(1, len(query_terms) // 3)


def _missing_information(ctx: _Context) -> list[str]:
    return [] if _is_evidence_sufficient(ctx) else [f"more detail about: {ctx.query[:60]}"]


_QUERY_LINE = re.compile(r"^Query:\s*(.*)$", re.MULTILINE)
_EVIDENCE_BLOCK = re.compile(r"^Evidence:\s*\n(.*)", re.MULTILINE | re.DOTALL)


def _build_context(user_prompt: str) -> _Context:
    """Every prompt in this codebase starts a line with "Query: <text>", and
    evidence-bearing prompts (evidence assessment) follow with an
    "Evidence:\\n..." block. Rules must operate on just these extracted
    pieces — not the raw prompt blob, which may also contain e.g. a
    classification JSON dump — or unrelated prompt content could
    accidentally trip a keyword/pattern rule."""
    query_match = _QUERY_LINE.search(user_prompt)
    evidence_match = _EVIDENCE_BLOCK.search(user_prompt)
    return _Context(
        query=query_match.group(1).strip() if query_match else user_prompt,
        evidence=evidence_match.group(1).strip() if evidence_match else "",
    )


_FIELD_NAME_RULES: dict[str, Any] = {
    "query_type": lambda ctx: _classify_query_type(ctx.query),
    "strategy": lambda ctx: RetrievalStrategy.HYBRID,
    "expand_query": lambda ctx: _should_expand(ctx.query),
    "requires_expansion": lambda ctx: _should_expand(ctx.query),
    "decompose": lambda ctx: _should_decompose(ctx.query),
    "requires_decomposition": lambda ctx: _should_decompose(ctx.query),
    "is_ambiguous": lambda ctx: _classify_query_type(ctx.query) == QueryType.AMBIGUOUS,
    "is_answerable": lambda ctx: bool(ctx.query.strip()),
    "reasoning": lambda ctx: f"Mock heuristic classification for query: {ctx.query[:80]!r}",
    "expanded_queries": lambda ctx: _derive_expanded_queries(ctx.query),
    "subqueries": lambda ctx: _derive_subqueries(ctx.query),
    "max_iterations": lambda ctx: 3,
    "top_k": lambda ctx: 10,
    "candidate_pool_size": lambda ctx: 30,
    "sufficient": _is_evidence_sufficient,
    "missing_information": _missing_information,
}

# Nested model types that should get their own (mostly-empty) default rather
# than being deep-filled field-by-field — e.g. an all-optional filter object
# whose sensible mock default is simply "no filters".
_DEFAULT_INSTANCE_TYPES = {"MetadataFilter"}


class MockLLMProvider:
    @property
    def model_name(self) -> str:
        return "mock"

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        return f"[mock completion for prompt of length {len(user_prompt)}]"

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        temperature: float = 0.0,
    ) -> T:
        ctx = _build_context(user_prompt)
        data = {
            name: _fill_field(name, field.annotation, ctx)
            for name, field in schema.model_fields.items()
        }
        return schema.model_validate(data)


def _fill_field(name: str, annotation: Any, ctx: _Context) -> Any:
    if name in _FIELD_NAME_RULES:
        return _FIELD_NAME_RULES[name](ctx)
    return _fill_by_type(annotation, ctx)


def _fill_by_type(annotation: Any, ctx: _Context) -> Any:
    origin = get_origin(annotation)

    if origin in (Union, types.UnionType):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        return _fill_by_type(non_none[0], ctx) if non_none else None

    if origin in (list, typing.List):  # noqa: UP006 - comparing against typing.List itself
        (item_type,) = get_args(annotation) or (str,)
        if isinstance(item_type, type) and issubclass(item_type, BaseModel):
            return [_fill_model(item_type, ctx)]
        return _derive_subqueries(ctx.query) if item_type is str else []

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        members = list(annotation)
        return members[0] if members else None

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _fill_model(annotation, ctx)

    if annotation is bool:
        return False
    if annotation is int:
        return 1
    if annotation is float:
        return 0.5
    if annotation is str:
        return f"mock value for: {ctx.query[:60]}"
    return None


def _fill_model(model_type: type[BaseModel], ctx: _Context) -> BaseModel:
    if model_type.__name__ in _DEFAULT_INSTANCE_TYPES:
        return model_type()
    data = {
        name: _fill_field(name, field.annotation, ctx)
        for name, field in model_type.model_fields.items()
    }
    return model_type.model_validate(data)
