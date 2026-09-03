"""Evidence evaluation (spec §15), contradiction detection (spec §18),
temporal awareness (spec §19), and source authority (spec §20).

Split deliberately by how much genuine reasoning each judgment needs
(engineering principle #1/#2):

- Sufficiency/relevance/coverage/directness: an LLM call
  (`EvidenceAgent._assess`) — whether evidence actually answers a query in
  the way it was asked is a language-understanding task.
- Contradiction detection: fully deterministic regex-based numeric-claim
  matching (`_detect_contradictions`). This does not attempt general
  semantic contradiction detection (e.g. two sources disagreeing in prose
  with no shared number) — that would need real LLM reasoning and is a
  documented gap. What it catches — two different sources reporting a
  different percentage/number for the same named metric — matches this
  spec section's own worked example ("Revenue declined 4%.") and is exactly
  the kind of claim deterministic extraction handles reliably.
- Source authority: a pure lookup against a *configured* order (spec §20:
  "Do not hardcode that hierarchy as universally correct"), never an LLM
  guess at which source is more authoritative.
- Temporal spread: regex year extraction — informational, not a hard block.

Never invents a contradiction resolution: `Contradiction.resolution` is set
only when the configured authority order actually distinguishes the two
sources; otherwise it stays `None` and the conflict is surfaced as-is
(spec §18: "If the system cannot resolve the conflict, tell the user
explicitly").
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from agentic_rag.generation.llm import LLMProvider
from agentic_rag.retrieval.base import RetrievedCandidate

_SYSTEM_PROMPT = (
    "You judge whether retrieved evidence sufficiently answers a query. "
    "Evidence that confirms WHAT happened but not WHY (when the query asks "
    "why) is NOT sufficient. Evidence that is only tangentially related is "
    "NOT sufficient. If insufficient, list the specific missing information "
    "needed, not a restatement of the query. Also rate relevance (does the "
    "evidence relate to the query topic), coverage (does it address all "
    "parts of the query), and directness (does it state the answer plainly "
    "vs. requiring inference), each from 0.0 to 1.0."
)

DEFAULT_SOURCE_AUTHORITY_ORDER = [
    "annual report",
    "investor presentation",
    "press release",
    "secondary source",
]

_METRIC_PATTERN = re.compile(
    r"(revenue|profit|margin|growth|decline|earnings|sales|income)"
    r"\D{0,20}?(-?\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r"\b((?:19|20)\d{2})\b")
_NUMERIC_EPSILON = 1e-9


class EvidenceAssessment(BaseModel):
    sufficient: bool
    reason: str
    missing_information: list[str] = Field(default_factory=list)
    relevance: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    directness: float = Field(ge=0.0, le=1.0)


@dataclass(slots=True)
class Contradiction:
    claim_a: str
    claim_b: str
    document_a: str
    document_b: str
    chunk_id_a: uuid.UUID
    chunk_id_b: uuid.UUID
    resolution: str | None = None


@dataclass(slots=True)
class EvidenceEvaluation:
    assessment: EvidenceAssessment
    contradictions: list[Contradiction] = field(default_factory=list)
    years_referenced: list[int] = field(default_factory=list)
    spans_multiple_periods: bool = False


class EvidenceAgent:
    def __init__(
        self,
        llm: LLMProvider,
        *,
        source_authority_order: list[str] | None = None,
    ) -> None:
        self._llm = llm
        self._authority_order = [
            s.strip().lower() for s in (source_authority_order or DEFAULT_SOURCE_AUTHORITY_ORDER)
        ]

    async def evaluate(
        self, query: str, candidates: list[RetrievedCandidate]
    ) -> EvidenceEvaluation:
        assessment = await self._assess(query, candidates)
        years = sorted({y for c in candidates for y in _extract_years(c.content)})
        return EvidenceEvaluation(
            assessment=assessment,
            contradictions=self._detect_contradictions(candidates),
            years_referenced=years,
            spans_multiple_periods=len(years) > 1,
        )

    async def _assess(
        self, query: str, candidates: list[RetrievedCandidate]
    ) -> EvidenceAssessment:
        if not candidates:
            return EvidenceAssessment(
                sufficient=False,
                reason="No evidence was retrieved for this query.",
                missing_information=[query],
                relevance=0.0,
                coverage=0.0,
                directness=0.0,
            )

        evidence_text = "\n\n".join(f"[{i + 1}] {c.content}" for i, c in enumerate(candidates))
        return await self._llm.complete_structured(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=f"Query: {query}\n\nEvidence:\n{evidence_text}",
            schema=EvidenceAssessment,
        )

    def _detect_contradictions(
        self, candidates: list[RetrievedCandidate]
    ) -> list[Contradiction]:
        mentions: dict[str, list[tuple[float, RetrievedCandidate]]] = defaultdict(list)
        for candidate in candidates:
            for keyword, value in _METRIC_PATTERN.findall(candidate.content):
                mentions[keyword.lower()].append((float(value), candidate))

        contradictions: list[Contradiction] = []
        for keyword, entries in mentions.items():
            for i in range(len(entries)):
                value_a, cand_a = entries[i]
                for j in range(i + 1, len(entries)):
                    value_b, cand_b = entries[j]
                    if cand_a.document_id == cand_b.document_id:
                        continue
                    if abs(value_a - value_b) < _NUMERIC_EPSILON:
                        continue
                    contradictions.append(
                        Contradiction(
                            claim_a=f"{keyword} {value_a}%",
                            claim_b=f"{keyword} {value_b}%",
                            document_a=cand_a.document_filename,
                            document_b=cand_b.document_filename,
                            chunk_id_a=cand_a.chunk_id,
                            chunk_id_b=cand_b.chunk_id,
                            resolution=self._resolve_via_authority(cand_a, cand_b),
                        )
                    )
        return contradictions

    def _resolve_via_authority(
        self, a: RetrievedCandidate, b: RetrievedCandidate
    ) -> str | None:
        rank_a = self._authority_rank(a.document_source)
        rank_b = self._authority_rank(b.document_source)
        if rank_a == rank_b:
            return None
        preferred = a if rank_a < rank_b else b
        return (
            f"Prefers {preferred.document_filename!r} per the configured source "
            "authority order — a provenance-based preference, not a claim that "
            "its content is factually correct."
        )

    def _authority_rank(self, source: str | None) -> int:
        if source is None:
            return len(self._authority_order)
        normalized = source.strip().lower()
        if normalized not in self._authority_order:
            return len(self._authority_order)
        return self._authority_order.index(normalized)


def _extract_years(text: str) -> set[int]:
    return {int(y) for y in _YEAR_PATTERN.findall(text)}
