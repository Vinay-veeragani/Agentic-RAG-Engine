"""Deterministic prompt-injection heuristics for retrieved chunk content.

Retrieved documents are untrusted data (top-level engineering principle):
nothing in a chunk's text may be treated as an instruction to the model, no
matter how it's phrased. This module doesn't attempt semantic detection —
that would need LLM reasoning over content the LLM shouldn't already trust
to reason about honestly — it's a bounded, deterministic regex sweep for
the common textual patterns real injection attempts use, in the same spirit
as `evidence/contradiction.py`'s regex-based contradiction detector.

A match doesn't prove malicious intent, but is a conservative enough signal
that the caller (see `agents/verifier.py`) excludes that chunk's content
from the synthesis/citation prompt entirely rather than silently including
it and risking an instruction embedded in retrieved text influencing the
model's behavior or being echoed into the answer.
"""

from __future__ import annotations

import re

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_instructions",
        re.compile(r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above)\s+instructions", re.I),
    ),
    (
        "disregard_instructions",
        re.compile(r"disregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above)", re.I),
    ),
    ("new_instructions", re.compile(r"\b(new|updated)\s+instructions\s*:", re.I)),
    (
        "system_prompt_probe",
        re.compile(r"\b(reveal|print|show|output)\s+(your\s+|the\s+)?system\s+prompt\b", re.I),
    ),
    ("role_override", re.compile(r"\byou\s+are\s+now\b", re.I)),
    ("act_as", re.compile(r"\bact\s+as\s+(?:an?|the)\b", re.I)),
    ("developer_mode", re.compile(r"\b(developer|jailbreak|dan)\s+mode\b", re.I)),
    (
        "instruction_delimiter_injection",
        re.compile(r"(<\|?(system|im_start|im_end)\|?>)", re.I),
    ),
    (
        "refusal_injection",
        re.compile(
            r"\b(do\s+not|don'?t|never|refuse\s+to)\s+(answer|respond(\s+to)?)\s+"
            r"(the\s+|this\s+|that\s+|user'?s?\s+)*(question|query|request)\b",
            re.I,
        ),
    ),
    (
        "exfiltration",
        re.compile(
            r"\b(send|email|forward|post|upload|transmit)\s+(this|the|all|any)\s+"
            r"(information|data|content|details|conversation)\b",
            re.I,
        ),
    ),
)


def detect_injection_patterns(text: str) -> list[str]:
    """Returns the names of every heuristic pattern that matched `text` —
    empty if none did. Never raises; a pure, side-effect-free function."""
    return [name for name, pattern in _PATTERNS if pattern.search(text)]
