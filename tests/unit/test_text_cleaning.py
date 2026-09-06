from agentic_rag.ingestion.cleaners.text import clean_text


def test_clean_text_repairs_utf8_as_cp1252_mojibake() -> None:
    """Found running the PDF parser against a real (non-synthetic) lease
    document: every smart apostrophe/quote came out as this specific
    double-encoding artifact — its real UTF-8 bytes got decoded one byte
    at a time as cp1252 somewhere in the extraction pipeline."""
    mojibake = "Landlordâ€™s consent"
    assert clean_text(mojibake) == "Landlord’s consent"


def test_clean_text_repairs_en_dash_mojibake() -> None:
    mojibake = "2018â€“2024 term"
    assert clean_text(mojibake) == "2018–2024 term"


def test_clean_text_leaves_ordinary_text_untouched() -> None:
    text = "Revenue declined 4% due to weaker enterprise demand."
    assert clean_text(text) == text


def test_clean_text_leaves_genuine_non_english_text_untouched() -> None:
    # Must not be "repaired" into nonsense just because it contains
    # non-ASCII characters unrelated to the mojibake pattern.
    text = "El alquiler debe pagarse antes del día primero de cada mes."
    assert clean_text(text) == text


def test_clean_text_leaves_a_lone_a_with_circumflex_untouched() -> None:
    # "â" alone (not followed by a cp1252 C1-control-range character) is
    # not the mojibake pattern and must be left alone.
    text = "The château requires â  careful maintenance."
    assert clean_text(text) == text
