"""NLU: text → {intent, entities}.

Chiến lược:
  1. Nếu có GEMINI_API_KEY → dùng Gemini với native function-calling. Mỗi
     intent là một tool; LLM tự chọn tool đúng và bind tham số → schema
     ràng buộc chặt, ít hallucinate entities lạ hơn so với JSON-output prompt.
  2. Nếu không → rule-based fallback (pattern match keywords) — dev offline.

Đầu ra chuẩn (giữ nguyên cho mọi backend):
    {"intent": "<intent_name>", "entities": {"key": "value", ...}}
"""
import logging
import re
from typing import Dict, Any

from . import config
from .intents import INTENTS

_log = logging.getLogger("secva.nlu")


# ==========================================================================
# Gemini function-calling NLU
# ==========================================================================
class GeminiNLU:
    """NLU dùng Gemini function-calling.

    Mỗi intent (trừ `unknown`) được dựng thành 1 FunctionDeclaration với
    schema parameters khớp `entities` của intent đó. Description chứa cả
    examples (mô tả phân bố câu input) lẫn counter_examples (biên).

    Khi câu không khớp tool nào → Gemini không gọi function → fallback unknown.
    """

    # Class-level fallback singleton — không tạo RuleBasedNLU mới mỗi lần Gemini fail.
    _fallback: "RuleBasedNLU | None" = None

    def __init__(self, api_key: str = config.GEMINI_API_KEY,
                 model_name: str = config.GEMINI_MODEL):
        from google import genai
        from google.genai import types
        self._client = genai.Client(api_key=api_key)
        self._model  = model_name
        self._types  = types
        self._tools  = self._build_tools(types)
        self._system = (
            "Bạn là module hiểu ý định (NLU) cho trợ lý ảo tiếng Việt. "
            "Đọc câu nói của người dùng và gọi đúng 1 function tương ứng với "
            "ý định người dùng muốn thực hiện. "
            "Trích xuất tham số (entities) từ câu nói, điền vào đúng schema "
            "của function. Không bịa thông tin không có trong câu. "
            "Nếu câu không khớp với bất kỳ function nào, hãy không gọi function "
            "(sẽ được xử lý như 'unknown')."
        )

    # ----- Build tool declarations từ INTENTS -----
    def _build_tools(self, types):
        # Entity → description ngắn giúp Gemini bind giá trị đúng kiểu.
        entity_hints = {
            "location":        "tên thành phố / địa danh (vd: Hà Nội, Sài Gòn)",
            "query":           "nội dung câu hỏi gốc của người dùng",
            "recipient":       "tên người nhận email (vd: anh Tuấn, sếp, mẹ)",
            "recipient_email": "địa chỉ email người nhận (chứa ký tự @)",
            "subject":         "tiêu đề email",
            "body":            "nội dung email",
            "content":         "nội dung chính (alias của body)",
            "target":          "đối tượng cần xóa (ghi chú, lịch, dữ liệu, ...)",
            "filename":        "tên file (nếu user nói rõ)",
            "genre":           "thể loại nhạc (rock, pop, ballad, edm, ...)",
            "date":            "ngày tham chiếu (hôm nay, ngày mai, tuần này, ...)",
        }

        decls = []
        for name, spec in INTENTS.items():
            if name == "unknown" or spec.get("internal"):
                continue  # unknown = không gọi function nào
            decl = self._build_one_decl(name, spec, entity_hints, types)
            decls.append(decl)
        return [types.Tool(function_declarations=decls)]

    def _build_one_decl(self, name: str, spec: dict,
                        entity_hints: dict, types):
        # Description chứa cả examples + counter_examples → cho LLM nhìn được
        # phân phối in-class và biên out-of-class trong cùng schema docstring.
        desc = [spec["desc"]]
        examples = spec.get("examples") or []
        if examples:
            desc.append("Ví dụ KHỚP function này: "
                        + " | ".join(examples[:8]))
        counters = spec.get("counter_examples") or []
        if counters:
            desc.append("KHÔNG dùng cho: " + " | ".join(counters))
        description = "\n".join(desc)

        # Parameters: chỉ tạo OBJECT schema nếu có entities (Gemini cho phép
        # function không có params, nhưng phải truyền None hoặc bỏ qua).
        entities = spec.get("entities") or []
        parameters = None
        if entities:
            properties = {
                ent: types.Schema(
                    type=types.Type.STRING,
                    description=entity_hints.get(ent, ent),
                )
                for ent in entities
            }
            parameters = types.Schema(
                type=types.Type.OBJECT,
                properties=properties,
                # KHÔNG required — nếu user nói thiếu thông tin (vd: "thời tiết
                # thế nào" không có location), handler tự xử lý/hỏi lại.
            )

        return types.FunctionDeclaration(
            name=name,
            description=description,
            parameters=parameters,
        )

    # ----- Parse -----
    def parse(self, text: str) -> Dict[str, Any]:
        try:
            resp = self._client.models.generate_content(
                model=self._model,
                contents=text,
                config=self._types.GenerateContentConfig(
                    system_instruction=self._system,
                    tools=self._tools,
                    tool_config=self._types.ToolConfig(
                        function_calling_config=self._types.FunctionCallingConfig(
                            # AUTO: model tự quyết có gọi function không.
                            # Câu không khớp tool nào → trả text → unknown.
                            mode="AUTO",
                        ),
                    ),
                    temperature=0.1,
                ),
            )
            fcs = resp.function_calls or []
        except Exception as e:
            _log.warning("NLU error: %s → fallback rule-based", e)
            if GeminiNLU._fallback is None:
                GeminiNLU._fallback = RuleBasedNLU()
            return GeminiNLU._fallback.parse(text)

        if not fcs:
            return {"intent": "unknown", "entities": {}}

        fc = fcs[0]  # single-intent semantics — bỏ qua multi-tool composition
        intent = fc.name if fc.name in INTENTS else "unknown"
        if intent == "unknown":
            return {"intent": "unknown", "entities": {}}

        # Sanitize entities: chỉ giữ key đã khai báo (defense-in-depth dù
        # Gemini đã bind theo schema, vẫn check để tránh prompt injection
        # đẩy key lạ vào args).
        allowed = set(INTENTS[intent].get("entities", []) or [])
        raw_args = dict(fc.args) if fc.args else {}
        entities = {k: str(v) for k, v in raw_args.items() if k in allowed and v is not None}
        return {"intent": intent, "entities": entities}


# ==========================================================================
# Gemini-based Answer Generation — dùng cho general_question handler
# ==========================================================================
class GeminiChat:
    """Trả lời câu hỏi tổng quát bằng Gemini, ngắn gọn, tiếng Việt tự nhiên."""

    _SYSTEM = (
        "Bạn là trợ lý ảo thông minh, giao tiếp bằng tiếng Việt tự nhiên. "
        "Trả lời ngắn gọn, rõ ràng (tối đa 3–4 câu). "
        "Không dùng markdown, không gạch đầu dòng, không giải thích dài dòng. "
        "Nếu không biết câu trả lời, nói thật một cách lịch sự."
    )

    def __init__(self, api_key: str = config.GEMINI_API_KEY,
                 model_name: str = config.GEMINI_MODEL):
        from google import genai
        from google.genai import types
        self._client = genai.Client(api_key=api_key)
        self._model  = model_name
        self._types  = types

    def answer(self, question: str, user_name: str = "") -> str:
        prompt = f"{user_name} hỏi: {question}" if user_name else question
        try:
            resp = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    system_instruction=self._SYSTEM,
                    temperature=0.7,
                ),
            )
            answer = (resp.text or "").strip()
            if not answer:
                return "Xin lỗi, mình chưa tra được thông tin lúc này. Thử lại sau nhé."
            return answer
        except Exception:
            return "Xin lỗi, mình chưa tra được thông tin lúc này. Thử lại sau nhé."


_chat_instance: "GeminiChat | None" = None


def get_chat() -> "GeminiChat | None":
    """Singleton GeminiChat — trả về None nếu chưa có API key."""
    global _chat_instance
    if _chat_instance is None and config.GEMINI_API_KEY:
        _chat_instance = GeminiChat()
    return _chat_instance


# ==========================================================================
# Rule-based fallback (khi không có API key, hoặc dev offline)
# ==========================================================================
class RuleBasedNLU:
    """Pattern matching đơn giản. Match keyword trong câu input."""

    # First-match-wins: cụ thể đặt trước generic.
    # "xóa" raw quá broad — "xóa file" phải match open_files, không phải delete_data.
    # add_* (IMPORTANT bidirectional) đặt TRƯỚC read_notes/show_schedule để
    # "thêm ghi chú X" không nhầm thành "đọc ghi chú".
    KEYWORD_MAP = [
        ("get_time", ["mấy giờ", "giờ rồi", "giờ hiện tại"]),
        ("get_weather", ["thời tiết", "mưa", "nắng"]),
        ("tell_joke", ["chuyện cười", "kể cười", "câu cười"]),
        ("add_contact", ["thêm liên hệ", "lưu contact", "thêm contact",
                         "tạo contact", "thêm vào danh bạ"]),
        ("add_note", ["thêm ghi chú", "tạo ghi chú", "lưu ghi chú", "tạo note",
                      "thêm note", "ghi chú mới", "note lại", "viết ghi chú",
                      "ghi vào note", "thêm vào ghi chú"]),
        ("add_schedule", ["thêm lịch", "đặt lịch", "lên lịch", "ghi vào lịch",
                          "thêm cuộc hẹn", "thêm vào lịch", "tạo lịch", "đặt nhắc",
                          "thêm task vào lịch"]),
        ("read_notes", ["đọc ghi chú", "mở nhật ký", "đọc nhật ký", "ghi chú của tôi"]),
        ("send_email", ["gửi email", "gửi mail", "soạn mail", "viết mail", "viết email"]),
        ("check_balance", ["số dư", "bao nhiêu tiền", "kiểm tra tài khoản"]),
        # open_files đặt TRƯỚC delete_data: "xóa file" phải match open_files (giả định
        # user muốn vào panel files để xóa, không phải xóa preferences/notes).
        ("open_files", ["mở file", "xem file", "file của tôi", "danh sách file",
                        "xóa file", "xoá file"]),
        ("delete_data", ["xóa ghi chú", "xoá ghi chú", "xóa lịch", "xoá lịch",
                         "xóa dữ liệu", "xoá dữ liệu", "xóa thông tin", "xoá thông tin",
                         "xóa tất cả", "xoá tất cả"]),
        ("greet", ["xin chào", "chào bạn", "hello", "hi"]),
        ("play_music", ["mở nhạc", "phát nhạc", "bật nhạc", "nghe nhạc"]),
        ("show_schedule", ["lịch hôm nay", "lịch của tôi", "việc gì", "nhắc việc"]),
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
            for genre in ["rock", "pop", "ballad", "edm", "jazz", "rap", "v-pop", "vpop"]:
                if genre in text:
                    return {"genre": genre}
        if intent == "send_email":
            # "gửi email cho anh Tuấn" / "soạn mail tới lan@gmail.com"
            m = re.search(r"(?:cho|tới|đến|gửi)\s+([^.,!?\n]+?)(?:\s+về|\s+với|\s+nội dung|$)",
                          text)
            if m:
                recipient = m.group(1).strip()
                if "@" in recipient:
                    return {"recipient_email": recipient}
                return {"recipient": recipient}
        if intent == "delete_data":
            for target in ("ghi chú", "ghi chu", "lịch", "lich", "tất cả", "tat ca"):
                if target in text:
                    return {"target": target}
        if intent == "add_note":
            # "thêm ghi chú <content>" — lấy phần sau keyword trigger làm content.
            for kw in ("thêm ghi chú", "tạo ghi chú", "lưu ghi chú", "tạo note",
                       "thêm note", "ghi chú mới", "note lại", "viết ghi chú",
                       "ghi vào note", "thêm vào ghi chú"):
                idx = text.find(kw)
                if idx >= 0:
                    content = text[idx + len(kw):].strip(" :,.-")
                    if content:
                        return {"content": content}
            return {}
        if intent == "add_schedule":
            # Tách time pattern (hh giờ / hh:mm / sáng/chiều/tối) + content.
            time_m = re.search(
                r"(\d{1,2}\s*(?:giờ|h|:)\s*\d{0,2}\s*(?:sáng|chiều|tối)?|"
                r"sáng mai|chiều nay|tối nay|ngày mai|hôm nay)",
                text, flags=re.IGNORECASE,
            )
            time_str = time_m.group(0).strip() if time_m else ""
            content = text
            for kw in ("thêm lịch", "đặt lịch", "lên lịch", "ghi vào lịch",
                       "thêm cuộc hẹn", "thêm vào lịch", "tạo lịch", "đặt nhắc",
                       "thêm task vào lịch"):
                idx = content.find(kw)
                if idx >= 0:
                    content = content[idx + len(kw):]
                    break
            if time_str:
                content = content.replace(time_str, "", 1)
            content = content.strip(" :,.-")
            out: dict = {}
            if content:
                out["content"] = content
            if time_str:
                out["time"] = time_str
            return out
        if intent == "add_contact":
            email_m = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
            email = email_m.group(0) if email_m else ""
            # Lấy name: token alphabetic giữa "liên hệ"/"contact" và "email"/"mail"
            name_m = re.search(
                r"(?:liên\s+hệ|contact|danh\s+bạ|liên\s+lạc)\s+(?:mới\s+)?"
                r"(?:tên\s+)?([\w\sÀ-ỹ]+?)(?:\s+(?:email|mail|số\s+mail|với\s+mail))",
                text, flags=re.IGNORECASE,
            )
            name = name_m.group(1).strip() if name_m else ""
            out = {}
            if name:  out["name"] = name
            if email: out["email"] = email
            return out
        return {}


# ==========================================================================
# Factory
# ==========================================================================
_nlu_instance = None


def get_nlu():
    global _nlu_instance
    if _nlu_instance is None:
        if config.GEMINI_API_KEY:
            _log.info("Loading Gemini NLU...")
            _nlu_instance = GeminiNLU()
        else:
            _log.info("GEMINI_API_KEY chưa set → dùng rule-based NLU")
            _nlu_instance = RuleBasedNLU()
    return _nlu_instance


# ==========================================================================
# Conditional ASR correction: chỉ gọi Gemini sửa transcript khi NLU không
# hiểu được câu gốc. Giảm latency/cost so với eager-correct mỗi turn.
# ==========================================================================
def parse_with_correction(text: str, nlu=None) -> tuple:
    """Parse text. Nếu intent='unknown' và có Gemini API key → thử sửa
    transcript rồi parse lại 1 lần.

    Returns (final_text, nlu_result). final_text có thể đã được sửa
    so với input để caller log/hiển thị lên UI.

    Tốn tối đa 1 extra Gemini call CHỈ khi câu gốc không khớp intent — câu
    bình thường (>90% case) không tốn gì thêm.
    """
    if nlu is None:
        nlu = get_nlu()

    result = nlu.parse(text)
    if result["intent"] != "unknown" or not config.GEMINI_API_KEY:
        return text, result

    # Lazy import để tránh vòng tròn nlu ↔ asr lúc module load.
    from .asr import correct_transcript
    corrected = correct_transcript(text, force=True)
    if corrected == text or not corrected:
        return text, result

    retry = nlu.parse(corrected)
    if retry["intent"] == "unknown":
        # Sửa rồi vẫn không hiểu → trả về kết quả gốc, đỡ confuse user
        # về việc transcript bị thay đổi mà chẳng giúp gì.
        return text, result
    _log.info("NLU fallback corrected %r → %r (intent=%s)",
              text, corrected, retry["intent"])
    return corrected, retry
