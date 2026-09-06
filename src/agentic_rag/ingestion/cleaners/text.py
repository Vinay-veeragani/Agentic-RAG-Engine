"""Deterministic text cleanup applied uniformly after parsing, regardless of
source format. Kept intentionally minimal — this is normalization, not
content rewriting."""

from __future__ import annotations

import re
import unicodedata

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")

# Tell-tale markers of one specific, common double-encoding bug: a source
# PDF/document whose real UTF-8 bytes (e.g. E2 80 99 for the ’ apostrophe)
# got decoded one byte at a time as cp1252 instead, producing "â€™" instead
# of "’". Found running the PDF parser against a real (non-synthetic) lease
# document, where every smart quote/dash/ellipsis came out this way. "â"
# followed immediately by one of cp1252's C1-control-range characters
# (0x80-0x9F, which is where UTF-8's continuation bytes for this range land)
# is specific enough to real mojibake that it essentially never occurs in
# genuine text otherwise.
_MOJIBAKE_MARKER = re.compile(
    "â[€‚ƒ„…†‡ˆ‰Š‹Œ‘’“”•–—˜™š›œ]"
)


def _repair_mojibake(text: str) -> str:
    """Repairs UTF-8-decoded-as-cp1252 mojibake by round-tripping through
    cp1252 bytes — but only when the marker pattern is present *and* doing
    so actually reduces it, never blindly. Ordinary text (including
    non-English text or text that merely contains "â") is returned
    unchanged: the round-trip either fails outright (most real text isn't
    cp1252-encodable) or doesn't reduce the marker count, in which case the
    original is kept."""
    if not _MOJIBAKE_MARKER.search(text):
        return text
    try:
        repaired = text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    if len(_MOJIBAKE_MARKER.findall(repaired)) < len(_MOJIBAKE_MARKER.findall(text)):
        return repaired
    return text


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _repair_mojibake(text)
    text = _CONTROL_CHARS.sub("", text)
    text = _TRAILING_SPACE.sub("\n", text)
    text = _MULTI_BLANK_LINES.sub("\n\n", text)
    return text.strip()
