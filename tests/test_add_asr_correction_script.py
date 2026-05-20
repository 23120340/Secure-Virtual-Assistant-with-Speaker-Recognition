import json

from scripts.add_asr_correction import add_or_update_rule


def test_add_asr_correction_creates_file(tmp_path):
    path = tmp_path / "asr_corrections.json"

    result = add_or_update_rule(
        wrong="đã phía chú",
        right="đọc ghi chú",
        path=path,
    )

    assert result == "added"
    rules = json.loads(path.read_text(encoding="utf-8"))
    assert rules == [{
        "pattern": "đã phía chú",
        "replacement": "đọc ghi chú",
        "regex": False,
        "ignore_case": True,
    }]


def test_add_asr_correction_updates_existing_rule(tmp_path):
    path = tmp_path / "asr_corrections.json"
    path.write_text(json.dumps([{
        "pattern": "ghi chu",
        "replacement": "ghi chú",
        "regex": False,
        "ignore_case": True,
    }], ensure_ascii=False), encoding="utf-8")

    result = add_or_update_rule(
        wrong="ghi chu",
        right="đọc ghi chú",
        yes=True,
        path=path,
    )

    assert result == "updated"
    rules = json.loads(path.read_text(encoding="utf-8"))
    assert len(rules) == 1
    assert rules[0]["replacement"] == "đọc ghi chú"


def test_add_asr_correction_accepts_regex(tmp_path):
    path = tmp_path / "asr_corrections.json"

    result = add_or_update_rule(
        wrong=r"\bnào,?\s*ghi chú\b",
        right="đọc ghi chú",
        regex=True,
        path=path,
    )

    assert result == "added"
    rules = json.loads(path.read_text(encoding="utf-8"))
    assert rules[0]["regex"] is True
