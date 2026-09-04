from agentic_rag.core.config import Settings
from agentic_rag.security.auth import auth_required, extract_api_key, is_valid_api_key


def test_auth_disabled_when_no_keys_configured() -> None:
    settings = Settings(api_keys=[])
    assert auth_required(settings) is False


def test_auth_required_when_keys_configured() -> None:
    settings = Settings(api_keys=["secret-key"])
    assert auth_required(settings) is True


def test_extract_api_key_prefers_bearer_header() -> None:
    key = extract_api_key(authorization="Bearer abc123", x_api_key="other")
    assert key == "abc123"


def test_extract_api_key_falls_back_to_x_api_key_header() -> None:
    key = extract_api_key(authorization=None, x_api_key="abc123")
    assert key == "abc123"


def test_extract_api_key_returns_none_when_absent() -> None:
    assert extract_api_key(authorization=None, x_api_key=None) is None


def test_extract_api_key_ignores_non_bearer_authorization() -> None:
    assert extract_api_key(authorization="Basic abc123", x_api_key=None) is None


def test_is_valid_api_key_accepts_configured_key() -> None:
    settings = Settings(api_keys=["secret-key"])
    assert is_valid_api_key(settings, "secret-key") is True


def test_is_valid_api_key_rejects_unknown_key() -> None:
    settings = Settings(api_keys=["secret-key"])
    assert is_valid_api_key(settings, "wrong-key") is False
    assert is_valid_api_key(settings, None) is False


def test_is_valid_api_key_checks_every_configured_key() -> None:
    """Uses hmac.compare_digest per candidate (constant-time), not `in` —
    still must accept any of several configured keys, not just the first."""
    settings = Settings(api_keys=["key-one", "key-two", "key-three"])
    assert is_valid_api_key(settings, "key-one") is True
    assert is_valid_api_key(settings, "key-two") is True
    assert is_valid_api_key(settings, "key-three") is True
    assert is_valid_api_key(settings, "key-four") is False
