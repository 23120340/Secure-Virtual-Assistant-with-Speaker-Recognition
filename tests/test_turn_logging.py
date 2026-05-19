import csv
import json

from core.turn_logging import build_nlu_candidate, build_turn_event, log_turn
from scripts.export_nlu_dataset import export_csv


def _sample_turn():
    return {
        "transcript": "gửi email cho user@example.com số 0909123456",
        "intent": "send_email",
        "auth_level": "important",
        "entities": {
            "recipient_email": "user@example.com",
            "subject": "hello",
            "body": "secret mail body",
            "token": "abc",
        },
        "identified_user_id": "u_123",
        "identified_user_name": "Lan",
        "sid_score": 0.91,
        "sv_required": True,
        "sv_passed": True,
        "sv_score": 0.87,
        "response": "ok",
        "blocked": False,
    }


def test_turn_event_redacts_sensitive_fields():
    event = build_turn_event(_sample_turn(), source="test")

    assert event["schema_version"] == "secva.turn.v1"
    assert event["transcript"] == "gửi email cho <email> số <phone>"
    assert event["identified_user_id_hash"]
    assert event["identified_user_id_hash"] != "u_123"
    assert event["entities"]["recipient_email"] == "<email>"
    assert event["entities"]["body"] == "<redacted_text>"
    assert event["entities"]["token"] == "<redacted>"


def test_log_turn_writes_turn_and_nlu_jsonl(tmp_path):
    turn_path = tmp_path / "turns.jsonl"
    nlu_path = tmp_path / "nlu.jsonl"

    event = log_turn(_sample_turn(), source="test",
                     turn_path=turn_path, nlu_path=nlu_path)

    turn_rows = [json.loads(line) for line in turn_path.read_text("utf-8").splitlines()]
    nlu_rows = [json.loads(line) for line in nlu_path.read_text("utf-8").splitlines()]
    assert turn_rows == [event]
    assert nlu_rows == [build_nlu_candidate(event)]
    assert nlu_rows[0]["label_status"] == "unlabeled"


def test_export_nlu_dataset_csv(tmp_path):
    nlu_path = tmp_path / "nlu.jsonl"
    csv_path = tmp_path / "nlu.csv"
    event = build_turn_event(_sample_turn(), source="test")
    candidate = build_nlu_candidate(event)
    nlu_path.write_text(json.dumps(candidate, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    count = export_csv(nlu_path, csv_path)

    assert count == 1
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["utterance"] == "gửi email cho <email> số <phone>"
    assert rows[0]["predicted_intent"] == "send_email"

