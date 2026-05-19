"""Intent router: orchestrate toàn bộ logic gating + dispatch.

Flow chính:
    audio + identified_user → NLU → check auth_level →
        ├─ NORMAL    → handler chạy ngay, user có thể None (guest)
        ├─ IMPORTANT → SV check (pass mới handler chạy)
        └─ PERSONAL  → handler chạy với user info để cá nhân hóa
                       (nếu guest thì handler tự xử lý)
"""
import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from . import config
from . import handlers
from .audit import audit as _audit
from .intents import INTENTS, AuthLevel
from .database import SpeakerManager
from .challenge import gen_phrase, make_token

_log = logging.getLogger(__name__)


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
    # Optional action signal từ handler (vd: send_email cần OAuth)
    action_type: Optional[str] = None
    action_data: Optional[dict] = None


class Router:
    def __init__(self, speaker_manager: SpeakerManager):
        self.spk = speaker_manager

    def complete_challenge(self, audio: np.ndarray, transcript_2nd: str,
                           state: dict, extra_context: Optional[dict] = None
                           ) -> "TurnResult":
        """Hoàn tất 1 challenge-response.

        Args:
            audio: audio thứ 2 (user đọc lại phrase). Đã VAD-trim ở route.
            transcript_2nd: ASR của audio thứ 2.
            state: pending state lưu từ turn đầu (chứa phrase, user_id, intent...).
            extra_context: forward cho handler (db, ...).

        Trả về TurnResult — block nếu phrase hoặc SV fail; dispatch handler
        nếu cả 2 pass.
        """
        from .challenge import phrase_match

        intent   = state["intent"]
        entities = state.get("entities") or {}
        uid      = state["user_id"]
        name     = state["user_name"]
        phrase   = state["phrase"]
        spec     = INTENTS.get(intent, INTENTS["unknown"])
        level    = spec["level"]
        user     = self.spk.db.get_user(uid) if uid else None

        # 1) Phrase match
        ok_phrase, ratio = phrase_match(phrase, transcript_2nd)
        if not ok_phrase:
            _audit("auth.challenge_check", intent=intent, user_id=uid,
                   outcome="fail_phrase", phrase_err_ratio=round(ratio, 3))
            return TurnResult(
                transcript=transcript_2nd, intent=intent,
                auth_level=level.value, entities=entities,
                identified_user_id=uid, identified_user_name=name,
                sid_score=state.get("sid_score") or 0.0,
                sv_required=True, sv_passed=False, sv_score=None,
                response=f"Câu xác nhận không khớp. Tác vụ bị từ chối.",
                blocked=True,
            )

        # 2) Re-verify SV trên audio mới
        sv_passed, sv_score = self.spk.verify(audio, uid)
        if not sv_passed:
            _audit("auth.challenge_check", intent=intent, user_id=uid,
                   outcome="fail_sv", sv_score=float(sv_score))
            return TurnResult(
                transcript=transcript_2nd, intent=intent,
                auth_level=level.value, entities=entities,
                identified_user_id=uid, identified_user_name=name,
                sid_score=state.get("sid_score") or 0.0,
                sv_required=True, sv_passed=False, sv_score=sv_score,
                response=(f"Giọng nói lần 2 không khớp với hồ sơ "
                          f"(score={sv_score:.2f}). Tác vụ bị từ chối."),
                blocked=True,
            )

        # 3) Pass cả 2 → dispatch handler đã pending
        _audit("auth.challenge_check", intent=intent, user_id=uid,
               outcome="success", sv_score=float(sv_score),
               phrase_err_ratio=round(ratio, 3))

        # Incremental re-enroll trên audio_2 (sau khi cả phrase + SV đều pass)
        # — chỉ kick khi challenge enabled (handle_turn path đã skip ở case này).
        if (config.SPEAKER_INCREMENTAL_REENROLL
                and config.CHALLENGE_RESPONSE_ENABLED):
            try:
                ok, _ = self.spk.incremental_update_centroid(
                    audio, uid, alpha=config.SPEAKER_INCREMENTAL_ALPHA)
                if ok:
                    _audit("speaker.incremental_update",
                           user_id=uid, alpha=config.SPEAKER_INCREMENTAL_ALPHA,
                           trigger="challenge_pass")
            except Exception:
                _log.exception("incremental update centroid failed for %s", uid)

        action_type = None
        action_data = None
        response = ""
        try:
            raw = handlers.HANDLERS.get(intent, handlers.handle_unknown)(
                entities, user, **(extra_context or {}))
        except Exception:
            _log.exception("Handler %s failed", intent)
            response = "Đã có lỗi khi xử lý lệnh này. Bạn thử lại sau nhé."
            blocked = True
        else:
            blocked = False
            if isinstance(raw, handlers.HandlerResult):
                response, action_type, action_data = raw.text, raw.action_type, raw.action_data
            else:
                response = raw

        return TurnResult(
            transcript=transcript_2nd, intent=intent,
            auth_level=level.value, entities=entities,
            identified_user_id=uid, identified_user_name=name,
            sid_score=state.get("sid_score") or 0.0,
            sv_required=True, sv_passed=True, sv_score=sv_score,
            response=response, blocked=blocked,
            action_type=action_type, action_data=action_data,
        )

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

        # ----- Bước 1: SID với margin check -----
        # Margin = Top1 − Top2. Margin nhỏ → 2 user có giọng tương tự → ambiguous.
        # Với IMPORTANT, ambiguous → block để chống mạo danh nhẹ (kẻ giả đủ tốt vượt
        # threshold nhưng vẫn cạnh tranh với user thật khác).
        uid, name, sid_score, margin = self.spk.identify_with_margin(audio)
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
                if name == "Ambiguous":
                    response = ("Mình nghe giọng bạn giống vài người trong hệ thống. "
                                "Không thể chắc là ai để thực hiện tác vụ quan trọng này.")
                else:
                    response = ("Đây là tác vụ quan trọng. Mình không nhận ra "
                                "giọng bạn nên không thể thực hiện. Vui lòng đăng ký trước.")
                blocked = True
                _audit("auth.voice_verify", intent=intent, outcome="block",
                       reason="sid_unidentified", sid_score=float(sid_score),
                       sid_margin=float(margin),
                       identified_name=name)
            else:
                sv_passed, sv_score = self.spk.verify(audio, uid)
                if not sv_passed:
                    response = (f"Xác thực thất bại (score={sv_score:.2f}). "
                                "Mình không thể thực hiện tác vụ này.")
                    blocked = True
                    _audit("auth.voice_verify", intent=intent, user_id=uid,
                           outcome="fail", sv_score=float(sv_score),
                           sid_score=float(sid_score), sid_margin=float(margin))
                else:
                    _audit("auth.voice_verify", intent=intent, user_id=uid,
                           outcome="success", sv_score=float(sv_score),
                           sid_score=float(sid_score), sid_margin=float(margin))
                    # ── Incremental re-enroll: chỉ apply khi KHÔNG challenge.
                    # Khi challenge enabled, chờ tới complete_challenge để update
                    # vì audio_2 là pass cuối cùng — tránh poisoning từ audio_1
                    # khi attacker có recording (challenge sẽ chặn ở turn 2).
                    if (config.SPEAKER_INCREMENTAL_REENROLL
                            and not config.CHALLENGE_RESPONSE_ENABLED):
                        try:
                            ok, _ = self.spk.incremental_update_centroid(
                                audio, uid, alpha=config.SPEAKER_INCREMENTAL_ALPHA)
                            if ok:
                                _audit("speaker.incremental_update",
                                       user_id=uid, alpha=config.SPEAKER_INCREMENTAL_ALPHA,
                                       trigger="sv_pass")
                        except Exception:
                            _log.exception("incremental update centroid failed for %s", uid)
                    # ── Challenge-response: opt-in via env flag.
                    # Khi bật, KHÔNG dispatch handler — trả signal challenge.
                    # Caller (web/app.py) lưu state vào session, prompt user
                    # đọc phrase, gọi endpoint /api/assistant/challenge-response.
                    if config.CHALLENGE_RESPONSE_ENABLED:
                        phrase = gen_phrase(config.CHALLENGE_PHRASE_LEN)
                        token  = make_token()
                        challenge_payload = {
                            "phrase":     phrase,
                            "token":      token,
                            # Caller cần state để dispatch sau khi pass:
                            "intent":     intent,
                            "entities":   entities,
                            "user_id":    uid,
                            "user_name":  name,
                            "sid_score":  float(sid_score),
                            "sv_score":   float(sv_score),
                        }
                        _audit("auth.challenge_issued", intent=intent,
                               user_id=uid, token_prefix=token[:8])
                        return TurnResult(
                            transcript=transcript,
                            intent=intent,
                            auth_level=level.value,
                            entities=entities,
                            identified_user_id=uid,
                            identified_user_name=name,
                            sid_score=sid_score,
                            sv_required=True,
                            sv_passed=True,
                            sv_score=sv_score,
                            response=f"Hãy đọc lại để xác nhận: \"{phrase}\"",
                            blocked=False,   # chưa block — đang chờ challenge
                            action_type="challenge_required",
                            action_data=challenge_payload,
                        )

        # ----- Bước 3: dispatch handler nếu chưa bị block -----
        action_type = None
        action_data = None
        if not blocked:
            handler = handlers.HANDLERS.get(intent, handlers.handle_unknown)
            ctx = extra_context or {}
            try:
                raw = handler(entities, user, **ctx)
            except Exception:
                _log.exception("Handler %s failed (entities=%s)", intent, entities)
                response = "Đã có lỗi khi xử lý lệnh này. Bạn thử lại sau nhé."
                blocked = True
            else:
                # Handler có thể trả str (backward compat) hoặc HandlerResult
                if isinstance(raw, handlers.HandlerResult):
                    response    = raw.text
                    action_type = raw.action_type
                    action_data = raw.action_data
                else:
                    response = raw

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
            action_type=action_type,
            action_data=action_data,
        )
