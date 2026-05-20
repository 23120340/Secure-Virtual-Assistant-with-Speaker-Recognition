from core import asr_corrections
from core.asr_corrections import _Rule


def teardown_function():
    asr_corrections._set_rules_for_test([])


def test_apply_literal_correction_from_voice_log():
    asr_corrections._set_rules_for_test([
        _Rule("đã phía chú", "đọc ghi chú"),
    ])

    assert asr_corrections.apply_corrections("Đã phía chú.") == "đọc ghi chú."


def test_apply_regex_correction_from_voice_log():
    asr_corrections._set_rules_for_test([
        _Rule(r"\bnào,?\s*ghi chú\b", "đọc ghi chú", regex=True),
    ])

    assert asr_corrections.apply_corrections("Nào, ghi chú.") == "đọc ghi chú."


def test_apply_email_corrections_from_voice_log():
    asr_corrections._set_rules_for_test([
        _Rule("bị email", "gửi email"),
        _Rule("đi mail", "gửi mail"),
        _Rule("giới mail", "gửi mail"),
        _Rule("mèo trương nhi", "gửi mail cho nhi"),
        _Rule("chín miếu", "gửi mail"),
        _Rule("unk email", "gửi email"),
    ])

    assert asr_corrections.apply_corrections("bị email.") == "gửi email."
    assert asr_corrections.apply_corrections("đi mail.") == "gửi mail."
    assert asr_corrections.apply_corrections("giới mail cho nhi.") == "gửi mail cho nhi."
    assert asr_corrections.apply_corrections("mèo trương nhi.") == "gửi mail cho nhi."
    assert asr_corrections.apply_corrections("chín miếu.") == "gửi mail."
    assert asr_corrections.apply_corrections("unk email.") == "gửi email."


def test_load_user_vocab_ignores_blank_and_comments(tmp_path, monkeypatch):
    vocab = tmp_path / "asr_user_vocab.txt"
    vocab.write_text(
        "# comment\n\nđọc ghi chú\nkiểm tra số dư\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(asr_corrections, "_VOCAB_PATH", vocab)

    assert asr_corrections.load_user_vocab() == ["đọc ghi chú", "kiểm tra số dư"]
