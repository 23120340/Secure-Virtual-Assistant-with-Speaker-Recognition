"""NLU: text → {intent, entities}.

Chiến lược:
  1. Nếu có GEMINI_API_KEY → dùng Gemini với structured output (JSON).
  2. Nếu không → rule-based fallback (pattern match keywords) — dev offline.

Đầu ra chuẩn:
    {"intent": "<intent_name>", "entities": {"key": "value", ...}}
"""
import json
import re
from typing import Dict, Any

from . import config
from .intents import INTENTS


# ==========================================================================
# Build system prompt cho LLM từ INTENTS (để khi sửa intents.py thì NLU
# tự động cập nhật theo, không cần sửa prompt thủ công)
# ==========================================================================
def _build_system_prompt() -> str:
    lines = [
        "Bạn là module hiểu ý định (NLU) cho trợ lý ảo tiếng Việt.",
        "Phân tích câu nói của người dùng và trả về JSON với 2 field:",
        '  "intent": tên intent (1 trong các intent dưới đây)',
        '  "entities": object chứa các thực thể trích xuất được',
        "",
        "Danh sách intent:",
    ]
    for name, spec in INTENTS.items():
        ents = ", ".join(spec["entities"]) if spec["entities"] else "không có"
        ex = " | ".join(spec["examples"][:3]) if spec["examples"] else ""
        lines.append(f'  - "{name}": {spec["desc"]}')
        lines.append(f"      entities: {ents}")
        if ex:
            lines.append(f"      ví dụ: {ex}")
    lines.extend([
        "",
        "Luật:",
        "- CHỈ trả về JSON thuần, KHÔNG markdown, KHÔNG giải thích.",
        "- Nếu không khớp intent nào, trả về intent='unknown'.",
        "- entities dùng key đúng như đã liệt kê. Nếu thiếu thông tin thì bỏ qua key đó.",
        "- entity values là string ngắn gọn, không có dấu câu thừa.",
    ])
    return "\n".join(lines)


# ==========================================================================
# Gemini-based NLU
# ==========================================================================
class GeminiNLU:
    def __init__(self, api_key: str = config.GEMINI_API_KEY,
                 model_name: str = config.GEMINI_MODEL):
        from google import genai
        from google.genai import types
        self._client = genai.Client(api_key=api_key)
        self._model  = model_name
        self._system = _build_system_prompt()
        self._types  = types

    def parse(self, text: str) -> Dict[str, Any]:
        try:
            resp = self._client.models.generate_content(
                model=self._model,
                contents=text,
                config=self._types.GenerateContentConfig(
                    system_instruction=self._system,
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            data = json.loads(resp.text)
        except Exception as e:
            print(f"  [NLU error: {e}] → fallback rule-based")
            return RuleBasedNLU().parse(text)

        intent = data.get("intent", "unknown")
        if intent not in INTENTS:
            intent = "unknown"
        return {
            "intent": intent,
            "entities": data.get("entities", {}) or {},
        }


# ==========================================================================
# Rule-based fallback (khi không có API key, hoặc dev offline)
# ==========================================================================
class RuleBasedNLU:
    """Pattern matching đơn giản. Match keyword trong câu input."""

    KEYWORD_MAP = [
        # (intent, keyword list — match nếu có 1 trong các keyword)
        ("get_time", ["mấy giờ", "giờ rồi", "giờ hiện tại"]),
        ("get_weather", ["thời tiết", "trời", "mưa", "nắng"]),
        ("tell_joke", ["chuyện cười", "kể cười", "câu cười"]),
        ("read_notes", ["đọc ghi chú", "mở nhật ký", "đọc nhật ký"]),
        ("send_email", ["gửi email", "gửi mail", "soạn mail"]),
        ("check_balance", ["số dư", "bao nhiêu tiền", "tài khoản"]),
        ("delete_data", ["xóa", "xoá"]),
        ("open_files", ["mở file", "xem file", "file của tôi", "danh sách file"]),
        ("greet", ["xin chào", "chào bạn", "hello", "hi"]),
        ("play_music", ["mở nhạc", "phát nhạc", "bật nhạc", "nghe nhạc"]),
        ("show_schedule", ["lịch", "việc gì", "nhắc việc"]),
    ]

    def parse(self, text: str) -> Dict[str, Any]:
        t = text.lower().strip()
        for intent, kws in self.KEYWORD_MAP:
            if any(kw in t for kw in kws):
                return {"intent": intent, "entities": self._extract(intent, t)}
        return {"intent": "general_question", "entities": {"query": text}}

    def _extract(self, intent: str, text: str) -> dict:
        if intent == "get_weather":
            # Regex bắt địa danh sau "ở" hoặc "tại"
            m = re.search(r"(?:ở|tại)\s+([\w\s]+)", text)
            if m:
                return {"location": m.group(1).strip()}
        if intent == "play_music":
            for genre in ["rock", "pop", "ballad", "edm", "jazz", "rap"]:
                if genre in text:
                    return {"genre": genre}
        return {}


# ==========================================================================
# Factory
# ==========================================================================
_nlu_instance = None


def get_nlu():
    global _nlu_instance
    if _nlu_instance is None:
        if config.GEMINI_API_KEY:
            print("  Loading Gemini NLU...")
            _nlu_instance = GeminiNLU()
        else:
            print("  GEMINI_API_KEY chưa set → dùng rule-based NLU")
            _nlu_instance = RuleBasedNLU()
    return _nlu_instance
