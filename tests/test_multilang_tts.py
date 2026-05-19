"""Test multi-language TTS resolver — không gọi gTTS thật (network)."""
from core.tts import SUPPORTED_TTS_LANGS, _resolve_lang


def test_supported_langs_include_vietnamese():
    assert "vi" in SUPPORTED_TTS_LANGS


def test_resolve_lang_uses_input_when_supported():
    assert _resolve_lang("en") == "en"
    assert _resolve_lang("ja") == "ja"


def test_resolve_lang_normalizes_zh_to_zh_cn():
    """zh → zh-CN (gTTS code)."""
    assert _resolve_lang("zh") == "zh-CN"


def test_resolve_lang_fallback_when_empty():
    assert _resolve_lang("") == "vi"  # fallback default = config.TTS_LANG = vi
    assert _resolve_lang(None) == "vi"


def test_resolve_lang_fallback_when_unsupported():
    assert _resolve_lang("xx-INVALID") == "vi"


def test_resolve_lang_strips_whitespace():
    assert _resolve_lang("  en  ") == "en"


def test_resolve_lang_explicit_fallback_arg():
    assert _resolve_lang("", fallback="ja") == "ja"
    assert _resolve_lang(None, fallback="en") == "en"
