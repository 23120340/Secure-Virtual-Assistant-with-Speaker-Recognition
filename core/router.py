"""Intent router: orchestrate toàn bộ logic gating + dispatch.

Flow chính:
    audio + identified_user → NLU → check auth_level →
        ├─ NORMAL    → handler chạy ngay, user có thể None (guest)
        ├─ IMPORTANT → SV check (pass mới handler chạy)
        └─ PERSONAL  → handler chạy với user info để cá nhân hóa
                       (nếu guest thì handler tự xử lý)
"""
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote
import numpy as np

from . import handlers
from .intents import INTENTS, AuthLevel
from .database import SpeakerManager


@dataclass
class TurnResult:
    """Kết quả 1 lượt tương tác — debug + log + báo cáo."""
    transcript: str
    intent: str
    auth_level: str
    entities: dict
    identified_user_id: Optional[str]
    identified_user_name: str
    sid_score: float
    sv_required: bool
    sv_passed: Optional[bool]    # None nếu không cần SV
    sv_score: Optional[float]
    response: str
    blocked: bool                # True nếu SV fail
    action_url: Optional[str] = None  # URL browser tự mở (vd: YouTube search)


class Router:
    def __init__(self, speaker_manager: SpeakerManager):
        self.spk = speaker_manager

    def handle_turn(self, audio: np.ndarray, transcript: str,
                    nlu_result: dict, extra_context: Optional[dict] = None) -> TurnResult:
        """Xử lý 1 turn hoàn chỉnh.

        Args:
            audio: raw audio đã trim VAD (dùng cho SID + SV)
            transcript: text từ ASR
            nlu_result: {intent, entities} từ NLU
            extra_context: kwargs forwarded to handlers (db, ...)
        """
        intent = nlu_result["intent"]
        entities = nlu_result["entities"]
        spec = INTENTS.get(intent, INTENTS["unknown"])
        level = spec["level"]

        # ----- Bước 1: SID (luôn chạy để biết ai đang nói) -----
        uid, name, sid_score = self.spk.identify(audio)
        user = self.spk.db.get_user(uid) if uid else None

        # ----- Bước 2: gate theo auth level -----
        sv_required = (level == AuthLevel.IMPORTANT)
        sv_passed = None
        sv_score = None
        blocked = False
        response = ""

        if sv_required:
            # Important intent: phải verify
            if uid is None:
                response = ("Đây là tác vụ quan trọng. Mình không nhận ra "
                            "giọng bạn nên không thể thực hiện. Vui lòng đăng ký trước.")
                blocked = True
            else:
                sv_passed, sv_score = self.spk.verify(audio, uid)
                if not sv_passed:
                    response = (f"Xác thực thất bại (score={sv_score:.2f}). "
                                "Mình không thể thực hiện tác vụ này.")
                    blocked = True

        # ----- Bước 3: dispatch handler nếu chưa bị block -----
        action_url = None
        if not blocked:
            handler = handlers.HANDLERS.get(intent, handlers.handle_unknown)
            ctx = extra_context or {}
            response = handler(entities, user, **ctx)

            # action_url không dùng nữa — xử lý ở app.py qua action_type/action_data

        return TurnResult(
            transcript=transcript,
            intent=intent,
            auth_level=level.value,
            entities=entities,
            identified_user_id=uid,
            identified_user_name=name,
            sid_score=sid_score,
            sv_required=sv_required,
            sv_passed=sv_passed,
            sv_score=sv_score,
            response=response,
            blocked=blocked,
            action_url=action_url,
        )
