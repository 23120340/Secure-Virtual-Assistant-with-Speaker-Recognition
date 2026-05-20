"""Regression tests for small source-level issues tracked in PROJECT_REVIEW_AND_TODO."""
from types import SimpleNamespace

import numpy as np

from core import handlers
from core.database import UserDB
from core.intents import INTENTS, AuthLevel
from core.nlu import GeminiChat, GeminiNLU, RuleBasedNLU


def test_explanatory_weather_question_stays_general_question():
    result = RuleBasedNLU().parse("Vì sao trời mưa")

    assert result == {
        "intent": "general_question",
        "entities": {"query": "Vì sao trời mưa"},
    }


def test_gemini_tool_call_for_explanatory_question_is_guarded():
    class DummyModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(function_calls=[SimpleNamespace(name="get_weather", args={})])

    class DummyTypes:
        GenerateContentConfig = staticmethod(lambda **kwargs: kwargs)
        ToolConfig = staticmethod(lambda **kwargs: kwargs)
        FunctionCallingConfig = staticmethod(lambda **kwargs: kwargs)

    nlu = object.__new__(GeminiNLU)
    nlu._client = SimpleNamespace(models=DummyModels())
    nlu._model = "dummy"
    nlu._types = DummyTypes
    nlu._tools = []
    nlu._system = "system"

    result = nlu.parse("Tại sao trời nắng")

    assert result == {
        "intent": "general_question",
        "entities": {"query": "Tại sao trời nắng"},
    }


def _add_user(db: UserDB, user_id: str, name: str):
    db.add_user(user_id, name, np.zeros(192, dtype=np.float32), preferences={})
    return db.get_user(user_id)


def test_find_user_by_name_substring_escapes_like_wildcards(tmp_db):
    _add_user(tmp_db, "u1", "Alice")
    _add_user(tmp_db, "u2", "Bob")
    _add_user(tmp_db, "u3", "Carol_100")

    assert tmp_db.find_user_by_name_substring("%") is None
    assert tmp_db.find_user_by_name_substring("alice%") is None
    assert tmp_db.find_user_by_name_substring("bob_") is None
    assert tmp_db.find_user_by_name_substring("carol_100")["user_id"] == "u3"


def test_open_files_returns_handler_result_with_count(tmp_path, monkeypatch, tmp_db):
    from core import config

    monkeypatch.setattr(config, "USER_FILES_DIR", tmp_path)
    user = _add_user(tmp_db, "u1", "Alice")
    user_dir = tmp_path / "u1"
    user_dir.mkdir()
    (user_dir / "a.txt").write_text("a", encoding="utf-8")
    (user_dir / "b.txt").write_text("b", encoding="utf-8")

    result = handlers.handle_open_files({}, user)

    assert isinstance(result, handlers.HandlerResult)
    assert result.action_type == "show_files"
    assert result.action_data["file_count"] == 2
    assert "2 file" in result.text


def test_play_music_guest_default_matches_web_default():
    assert "v-pop" in handlers.handle_play_music({}, None)


def test_email_flow_declared_internal_and_hidden_from_gemini_tools():
    assert INTENTS["email_flow"]["level"] == AuthLevel.IMPORTANT
    assert INTENTS["email_flow"]["internal"] is True

    built = []
    nlu = object.__new__(GeminiNLU)
    nlu._build_one_decl = lambda name, spec, hints, types: built.append(name) or name
    tools = nlu._build_tools(SimpleNamespace(Tool=lambda function_declarations: function_declarations))

    assert tools == [built]
    assert "email_flow" not in built
    assert "unknown" not in built


def test_gemini_chat_handles_empty_text_response():
    class DummyModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text=None)

    chat = object.__new__(GeminiChat)
    chat._client = SimpleNamespace(models=DummyModels())
    chat._model = "dummy"
    chat._types = SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs)

    assert "không trả nội dung" in chat.answer("hello")


def test_gemini_chat_uses_offline_answer_on_api_failure():
    class DummyModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota")

    chat = object.__new__(GeminiChat)
    chat._client = SimpleNamespace(models=DummyModels())
    chat._model = "dummy"
    chat._types = SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs)

    assert chat.answer("thủ đô Việt Nam là gì?") == "Thủ đô Việt Nam là Hà Nội."


def test_gemini_chat_reports_key_or_quota_errors():
    class DummyModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("API key was reported as leaked")

    chat = object.__new__(GeminiChat)
    chat._client = SimpleNamespace(models=DummyModels())
    chat._model = "dummy"
    chat._types = SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs)

    assert "GEMINI_API_KEY" in chat.answer("câu hỏi lạ")


def test_gemini_nlu_no_function_call_becomes_general_question():
    class DummyModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(function_calls=[])

    class DummyTypes:
        GenerateContentConfig = staticmethod(lambda **kwargs: kwargs)
        ToolConfig = staticmethod(lambda **kwargs: kwargs)
        FunctionCallingConfig = staticmethod(lambda **kwargs: kwargs)

    nlu = object.__new__(GeminiNLU)
    nlu._client = SimpleNamespace(models=DummyModels())
    nlu._model = "dummy"
    nlu._types = DummyTypes
    nlu._tools = []
    nlu._system = "system"

    result = nlu.parse("vì sao trời mưa")

    assert result == {
        "intent": "general_question",
        "entities": {"query": "vì sao trời mưa"},
    }
