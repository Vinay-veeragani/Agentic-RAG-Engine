"""Token counting shared by every chunker.

Originally used tiktoken's `cl100k_base` encoding, but that requires
downloading its BPE merge table from a Microsoft blob-storage host on first
use — a real network dependency for something called constantly in the
ingestion hot path, and one that turned out to be unreachable from this
machine's Python (its DNS resolution fails for that specific host even
though general internet access, including PyPI, works fine).

Instead this is a small rule-based, fully offline, dependency-free
tokenizer: text is split into whitespace-vs-non-whitespace runs, and each
run is one "token". It is exactly reversible (`"".join(encode(text)) ==
text`), deterministic, and requires no model file or network call — at the
cost of not matching any real LLM's actual subword tokenizer exactly. Chunk
budgeting only needs a consistent, reproducible count across strategies and
providers, not per-model exactness; this is documented as an approximation,
not silently assumed to be precise.
"""

from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"\S+|\s+")


def encode(text: str) -> list[str]:
    """Splits `text` into token strings such that `"".join(encode(text)) ==
    text` — used by the chunkers to slice/reconstruct token windows."""
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text)


def decode(tokens: list[str]) -> str:
    return "".join(tokens)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(encode(text))
