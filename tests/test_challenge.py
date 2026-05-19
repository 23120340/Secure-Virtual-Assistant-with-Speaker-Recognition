"""Test challenge-response module: phrase generation + fuzzy match.

ASR tiếng Việt có WER cao trên audio mic không chuyên → fuzzy match phải
nhận audio user đọc hơi sai chính tả nhưng vẫn đúng nội dung.
"""
from core.challenge import (
    gen_phrase, make_token, phrase_match, _DIGITS_VI, _COLORS_VI,
)


def test_gen_phrase_length():
    p = gen_phrase(4)
    assert len(p.split()) == 4
    p3 = gen_phrase(3)
    assert len(p3.split()) == 3


def test_gen_phrase_uses_pool():
    """Mỗi từ trong phrase phải từ pool đã định."""
    pool = set(_DIGITS_VI + _COLORS_VI)
    for _ in range(20):
        for w in gen_phrase(4).split():
            assert w in pool


def test_gen_phrase_random():
    """50 phrase sinh ra phải gần như không trùng (entropy đủ lớn)."""
    samples = {gen_phrase(4) for _ in range(50)}
    # Với 1296 possible 4-grams, va chạm là cực hiếm; cho ngưỡng 45/50 unique.
    assert len(samples) >= 45


def test_make_token_unique_and_urlsafe():
    tokens = {make_token() for _ in range(100)}
    assert len(tokens) == 100
    for t in tokens:
        # urlsafe → chỉ chứa A-Z a-z 0-9 _-
        assert all(c.isalnum() or c in "_-" for c in t)


def test_phrase_match_exact():
    ok, ratio = phrase_match("ba bốn năm sáu", "ba bốn năm sáu")
    assert ok is True
    assert ratio == 0.0


def test_phrase_match_one_word_off_pass():
    """1/4 từ sai (= 0.25 ratio) — dưới tolerance 0.34 → pass."""
    ok, ratio = phrase_match("ba bốn năm sáu", "ba bốn năm bảy")
    assert ok is True
    assert ratio == 0.25


def test_phrase_match_two_words_off_fail():
    """2/4 từ sai (= 0.5 ratio) — vượt tolerance → fail."""
    ok, ratio = phrase_match("ba bốn năm sáu", "ba bốn bảy tám")
    assert ok is False


def test_phrase_match_case_and_punctuation_insensitive():
    ok, _ = phrase_match("Ba Bốn Năm Sáu", "ba, bốn. năm! sáu?")
    assert ok is True


def test_phrase_match_one_filler_word_pass():
    """1 filler word đầu = 1 insertion = 1/4 = 0.25 → pass."""
    ok, _ = phrase_match("ba bốn năm sáu", "à ba bốn năm sáu")
    assert ok is True


def test_phrase_match_long_prefix_filler_fail():
    """Filler 2 từ ("Xác nhận") = 2 insertions = 0.5 ratio → fail.
    Đây là design choice: user phải đọc đúng phrase, không được thêm dài dòng.
    """
    ok, _ = phrase_match("ba bốn năm sáu", "Xác nhận ba bốn năm sáu")
    assert ok is False


def test_phrase_match_empty_hypothesis_fail():
    ok, _ = phrase_match("ba bốn năm sáu", "")
    assert ok is False


def test_phrase_match_empty_expected_fail():
    """Edge case: expected rỗng → không thể match (avoid divide-by-zero)."""
    ok, ratio = phrase_match("", "ba bốn")
    assert ok is False
    assert ratio == 1.0
