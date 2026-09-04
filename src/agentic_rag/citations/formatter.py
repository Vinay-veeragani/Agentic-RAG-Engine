"""Deterministic citation display formatting (example:
"[1] Annual Report 2025, page 42, Revenue Recognition")."""

from __future__ import annotations

from agentic_rag.citations.resolver import Citation


def format_citation(citation: Citation, index: int) -> str:
    label = citation.source or citation.document_filename
    parts = [label]
    if citation.page is not None:
        parts.append(f"page {citation.page}")
    if citation.section:
        parts.append(citation.section)
    return f"[{index}] {', '.join(parts)}"
