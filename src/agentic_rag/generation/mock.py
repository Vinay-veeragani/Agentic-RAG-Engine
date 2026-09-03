"""Deterministic, schema-introspecting mock LLM provider.

Rather than calling a real model, `MockLLMProvider.complete_structured`
inspects the *actual pydantic schema type* it was asked for and the query
text in `user_prompt`, then fills each field with a deterministic,
heuristically-derived value: keyword rules for fields whose name signals
intent (`query_type`, `decompose`, `subqueries`, ...), generic rules by
Python type otherwise (bool/int/float/str/Enum/list/nested BaseModel).

This is not a language model — it cannot handle a schema or a query it has
no heuristic for gracefully beyond generic type-based defaults. It exists so
every agent built on `LLMProvider.complete_structured` (query analysis,
planning, expansion, decomposition, and later evidence/citation agents) has
a zero-dependency, fully offline path end to end, the same role
`MockEmbeddingProvider` and `MockReranker` play for their layers.
"""

from __future__ import annotations

import re
import types
import typing
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


_FIELD_NAME_RULES: dict[str, Any] = {
    "query_type": lambda q: _classify_query_type(q),
    "strategy": lambda q: RetrievalStrategy.HYBRID,
    "expand_query": _should_expand,
    "requires_expansion": _should_expand,
    "decompose": _should_decompose,
    "requires_decomposition": _should_decompose,
    "is_ambiguous": lambda q: _classify_query_type(q) == QueryType.AMBIGUOUS,
    "is_answerable": lambda q: bool(q.strip()),
    "reasoning": lambda q: f"Mock heuristic classification for query: {q[:80]!r}",
    "expanded_queries": _derive_expanded_queries,
    "subqueries": _derive_subqueries,
    "max_iterations": lambda q: 3,
    "top_k": lambda q: 10,
    "candidate_pool_size": lambda q: 30,
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
        query = _extract_query(user_prompt)
        data = {
            name: _fill_field(name, field.annotation, query)
            for name, field in schema.model_fields.items()
        }
        return schema.model_validate(data)


_QUERY_LINE = re.compile(r"^Query:\s*(.*)$", re.MULTILINE)


def _extract_query(user_prompt: str) -> str:
    """Every prompt in this codebase starts a line with "Query: <text>".
    Heuristics must operate on just that text — not on whatever else a
    prompt appends after it (e.g. RetrievalPlanner also appends a
    "Classification: {...}" JSON blob) — or unrelated prompt content could
    accidentally trip a keyword/pattern rule."""
    match = _QUERY_LINE.search(user_prompt)
    return match.group(1).strip() if match else user_prompt


def _fill_field(name: str, annotation: Any, query: str) -> Any:
    if name in _FIELD_NAME_RULES:
        return _FIELD_NAME_RULES[name](query)
    return _fill_by_type(annotation, query)


def _fill_by_type(annotation: Any, query: str) -> Any:
    origin = get_origin(annotation)

    if origin in (Union, types.UnionType):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        return _fill_by_type(non_none[0], query) if non_none else None

    if origin in (list, typing.List):  # noqa: UP006 - comparing against typing.List itself
        (item_type,) = get_args(annotation) or (str,)
        if isinstance(item_type, type) and issubclass(item_type, BaseModel):
            return [_fill_model(item_type, query)]
        return _derive_subqueries(query) if item_type is str else []

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        members = list(annotation)
        return members[0] if members else None

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _fill_model(annotation, query)

    if annotation is bool:
        return False
    if annotation is int:
        return 1
    if annotation is float:
        return 0.5
    if annotation is str:
        return f"mock value for: {query[:60]}"
    return None


def _fill_model(model_type: type[BaseModel], query: str) -> BaseModel:
    if model_type.__name__ in _DEFAULT_INSTANCE_TYPES:
        return model_type()
    data = {
        name: _fill_field(name, field.annotation, query)
        for name, field in model_type.model_fields.items()
    }
    return model_type.model_validate(data)
