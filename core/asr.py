"""ASR module dùng faster-whisper."""
import logging
import re
import threading
from functools import lru_cache
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel

from . import config
from . import audio_io
from .intents import INTENTS

_log = logging.getLogger("secva.asr")


def _build_initial_prompt(max_chars: int = 600,
                          examples_per_intent: int = 2) -> str:
    """Build prompt bias cho Whisper từ INTENTS + user vocab.

    Whisper dùng initial_prompt như văn bản ngữ cảnh trước audio → các từ trong
    prompt được ưu tiên decode. Giúp giảm homophone errors cho từ-khóa-lệnh
    (vd: "số dư" thay vì "sờ dư", "ghi chú" thay vì "ghi chu").

    Giới hạn cứng: Whisper prompt limit ~244 tokens (~700 chars tiếng Việt).
    Ưu tiên: user vocab (file `data/asr_user_vocab.txt`) > INTENTS examples.
    User vocab xuất hiện trước → Whisper cân nhắc kỹ hơn.
    """
    from .asr_corrections import load_user_vocab
    user_vocab = load_user_vocab()

    phrases: list[str] = []
    seen: set = set()
    # User vocab đầu danh sách (highest priority).
    for p in user_vocab:
        if p not in seen:
            seen.add(p)
            phrases.append(p)
    # INTENTS examples sau (general command vocabulary).
    for spec in INTENTS.values():
        for ex in (spec.get("examples", []) or [])[:examples_per_intent]:
            ex = ex.strip()
            if ex and ex not in seen:
                seen.add(ex)
                phrases.append(ex)

    prefix = "Trợ lý ảo tiếng Việt nhận lệnh người dùng. Câu thường gặp: "
    body = ", ".join(phrases) + "."
    full = prefix + body
    if len(full) <= max_chars:
        return full
    # Truncate giữ trọn câu cuối: cắt tới dấu phẩy gần nhất trước max_chars
    cut = full[:max_chars].rsplit(", ", 1)[0] + "."
    return cut


# Whisper được pre-train trên YouTube → khi audio im lặng / quá ngắn / nhiễu,
# model "hallucinate" bằng các phrase outro hay gặp ở cuối video. Pattern này
# xuất hiện trong `data/turn_log.jsonl` khi user click không giữ đủ lâu.
# Toàn bộ pattern phải match (whole-text), không chỉ chứa — tránh false positive
# khi user thật sự nói "xin chào".
_WHISPER_HALLUCINATION_PATTERNS = [
    # Standalone "Hẹn gặp lại!" — bare outro phrase, không kèm object.
    re.compile(r"^hẹn gặp lại\s*[!.]?$", re.IGNORECASE),
    re.compile(r"^hẹn gặp lại( với)? (mọi người|các bạn|quý vị)\.?$", re.IGNORECASE),
    re.compile(r"^xin chào (và )?(tạm biệt|hẹn gặp lại).*$", re.IGNORECASE),
    re.compile(r"^cảm ơn (các bạn|mọi người) đã (xem|theo dõi).*$", re.IGNORECASE),
    re.compile(r"^(hẹn gặp lại|xin chào).*(video|tập|tuần|lần) (sau|tới|tiếp theo).*$", re.IGNORECASE),
    re.compile(r"^(đừng quên|nhớ) (subscribe|like|đăng ký|bấm chuông).*$", re.IGNORECASE),
    re.compile(r"^chúc.*(buổi|ngày|tuần).*(vui|tốt lành|hạnh phúc)\.?$", re.IGNORECASE),
    re.compile(r"^.*(theo dõi|đăng ký kênh).*$", re.IGNORECASE),
    re.compile(r"^xin chào tất cả các bạn\.?$", re.IGNORECASE),
    # Ký tự đơn lẻ kiểu "11h", "5h", "2 giờ" — Whisper hay sinh ra khi
    # audio quá ngắn / không có speech rõ. Threshold: tổng ≤ 4 char.
    re.compile(r"^\d{1,2}\s*(h|giờ|phút|m|s)?\.?$", re.IGNORECASE),
]


def _is_whisper_youtube_hallucination(text: str) -> bool:
    """Match text với pattern outro YouTube thường gặp."""
    t = text.strip()
    if not t:
        return False
    return any(rx.match(t) for rx in _WHISPER_HALLUCINATION_PATTERNS)


def _is_hallucinated(text: str, n_samples: int,
                     sample_rate: int = 16000) -> bool:
    """Phát hiện Whisper hallucination: lặp câu, quá dài so với audio,
    hoặc match phrase outro YouTube thường gặp.

    Whisper dễ sinh ra văn bản lặp lại (repetition loop) khi audio ngắn/im lặng.
    VD: 'Xin chào tất cả các bạn...' nhân lên 9 lần mặc dù chỉ nói 2 chữ.
    """
    if not text:
        return False

    duration_s = n_samples / sample_rate

    # YouTube outro hallucination — bắt sớm trước các check khác.
    if _is_whisper_youtube_hallucination(text):
        return True

    # Quá dài so với thời gian audio. Trước đây ngưỡng 8 ký tự/s gây false positive
    # với tiếng Việt có dấu (mỗi từ ~5 char). Câu "ngoài đường mưa rất to" 22 char
    # trong 2.5s → 8.8 char/s nhưng vẫn hợp lệ. Đặt 15 để chỉ bắt hallucination thực.
    if len(text) > max(60, duration_s * 15):
        return True

    # Phát hiện câu lặp lại
    clauses = [c.strip() for c in re.split(r'[.!?。\n]', text)
               if len(c.strip()) > 8]
    if len(clauses) >= 3:
        unique_ratio = len(set(clauses)) / len(clauses)
        if unique_ratio < 0.5:
            return True

    return False


class ASR:
    """Wrapper cho faster-whisper, cache model singleton."""

    def __init__(self,
                 model_size: str = config.WHISPER_MODEL,
                 device: str = config.WHISPER_DEVICE,
                 compute_type: str = config.WHISPER_COMPUTE):
        _log.info("Loading Whisper '%s' trên %s/%s...", model_size, device, compute_type)
        self.model = WhisperModel(model_size, device=device,
                                  compute_type=compute_type)
        self.language = config.ASR_LANGUAGE
        self.initial_prompt = _build_initial_prompt()
        # Whisper inference không thread-safe → lock cho threaded=True
        self._lock = threading.Lock()

    def transcribe(self, audio: np.ndarray,
                   sample_rate: int = config.SAMPLE_RATE) -> str:
        """Audio float32 mono → text. faster-whisper accept ndarray trực tiếp
        nếu sample rate = 16k."""
        if sample_rate != 16000:
            raise ValueError("Whisper expect 16kHz")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            if np.abs(audio).max() > 1.5:  # nhiều khả năng là int16 đặt vào float
                audio = audio / 32768.0

        with self._lock:
            segments, info = self.model.transcribe(
                audio,
                language=self.language,
                # task="transcribe" EXPLICIT — pin về transcribe mode, không
                # cho model fallback sang "translate" (translate sang English)
                # khi audio chứa từ-phonetic-giống-English. Đây là cause user-
                # report "Mấy giờ rồi" → "May Roy" / "May your roll": Whisper-
                # small với task default + short audio đôi khi cross sang
                # English path. Pin task → ép decoder dùng vocab tiếng Việt.
                task="transcribe",
                beam_size=5,             # giảm 10 → 5: ~2x nhanh hơn, WER tương đương vi
                best_of=1,
                temperature=0.0,            # deterministic, tránh hallucination
                vad_filter=False,           # đã VAD trước rồi
                condition_on_previous_text=False,
                # GHI CHÚ: từng siết 0.5 / -0.8 để filter outro hallucination,
                # nhưng tone tiếng Việt làm log_prob thấp tự nhiên → false-negative
                # toàn câu thật. Quay về ngưỡng cũ; outro hallucination giờ catch
                # bằng `_WHISPER_HALLUCINATION_PATTERNS` ở post-process (text-level).
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=1.8,  # thấp hơn = bắt lặp câu sớm hơn
                initial_prompt=self.initial_prompt,  # bias vocab command-domain
            )
            # Debug: log info nếu detect language ≠ "vi" (chỉ ra prompt/audio fail).
            detected = getattr(info, "language", None)
            if detected and detected != self.language:
                _log.warning("Whisper detect lang=%r (prob=%.2f) thay vì %r — "
                             "kết quả có thể bị English-ize.", detected,
                             getattr(info, "language_probability", 0.0),
                             self.language)

            # Lọc từng segment: bỏ qua segment có khả năng cao là không có speech.
            # Ngưỡng nới (no_speech 0.7, log_prob -1.2) — giữ recall tiếng Việt.
            # Hallucination outro vẫn bị catch ở text-level qua _is_hallucinated.
            parts = []
            for seg in segments:
                if getattr(seg, "no_speech_prob", 0) > 0.7:
                    continue
                if getattr(seg, "avg_logprob", 0) < -1.2:
                    continue
                parts.append(seg.text.strip())

        text = " ".join(parts).strip()

        # Post-process: phát hiện hallucination tổng thể
        if _is_hallucinated(text, len(audio)):
            return ""

        # Layer 1 correction — áp dụng quy tắc sửa cố định từ
        # `data/asr_corrections.json` (zero-cost, mỗi turn).
        # Layer 3 (Gemini) gọi tách riêng qua `correct_transcript()` ở caller.
        from .asr_corrections import apply_corrections
        text = apply_corrections(text)

        return text

    def transcribe_file(self, wav_path: Path) -> str:
        audio = audio_io.load_wav(wav_path)
        return self.transcribe(audio)


@lru_cache(maxsize=256)
def _correct_cached(text: str) -> str:
    """Gọi Gemini sửa lỗi cho 1 transcript. Cache để 2 lần gọi cùng text chỉ tốn 1 API call."""
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        system = (
            "Bạn là module sửa lỗi nhận dạng giọng nói tiếng Việt.\n"
            "Nhiệm vụ: nhận văn bản thô từ speech-to-text, trả về câu đã sửa.\n"
            "Quy tắc:\n"
            "- Câu đúng → trả về nguyên văn, không thay đổi gì.\n"
            "- Lỗi âm thanh hoặc chính tả nhỏ → sửa lỗi.\n"
            "- Nhiều từ nghe giống nhau → chọn nghĩa hợp lý nhất theo ngữ cảnh.\n"
            "- Giữ nguyên ý định của người nói mọi lúc.\n"
            "- KHÔNG thêm thông tin mới ngoài những gì có trong câu gốc.\n"
            "- Câu mơ hồ → chọn cách diễn giải có xác suất cao nhất.\n"
            "- Ưu tiên cụm từ thông dụng và ngôn ngữ tự nhiên.\n"
            "CHỈ trả về câu đã sửa, không giải thích gì thêm."
        )
        resp = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.1,
            ),
        )
        corrected = (resp.text or "").strip()
        return corrected if corrected else text
    except Exception as e:
        _log.warning("correct_transcript error: %s → giữ nguyên", e)
        return text


def correct_transcript(text: str, force: bool = False) -> str:
    """Sửa lỗi ASR bằng Gemini.

    - Mặc định opt-in qua config.ASR_CORRECT_ENABLED (eager mode, mỗi turn).
    - force=True bỏ qua flag — dùng cho fallback khi NLU không hiểu, chỉ tốn
      Gemini call khi thật cần (xem nlu.parse_with_correction).
    - Cache LRU theo text → 2 lần gọi cùng câu chỉ tốn 1 API call.
    - Skip khi text quá ngắn (< 3 ký tự) hoặc thiếu API key.
    """
    if (not text
            or not config.GEMINI_API_KEY
            or len(text.strip()) < 3):
        return text
    if not force and not config.ASR_CORRECT_ENABLED:
        return text
    return _correct_cached(text)


# ==========================================================================
# PhoWhisper backend (transformers pipeline) — stub cho session 2
# ==========================================================================
class PhoWhisperASR:
    """ASR backend dùng PhoWhisper qua transformers pipeline.

    PhoWhisper (VinAI) fine-tune Whisper trên 844h audio tiếng Việt → WER trên
    benchmark tiếng Việt thấp hơn nhiều so với multilingual Whisper cùng size.
    Trade-off: dùng transformers (eager mode) → chậm hơn faster-whisper int8
    nếu cùng size. CPU: nên dùng PhoWhisper-base; GPU: small/medium tốt.

    Dùng song song với class ASR (faster-whisper) — get_asr() dispatch theo
    config.ASR_BACKEND.
    """

    def __init__(self, model_name: str = config.PHOWHISPER_MODEL,
                 device: str = ""):
        from transformers import pipeline
        import torch as _torch

        if not device:
            device = "cuda" if _torch.cuda.is_available() else "cpu"

        _log.info("Loading PhoWhisper '%s' on %s...", model_name, device)
        # device là str ('cuda'/'cpu') hoặc int (GPU index); pipeline accept cả 2.
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            device=device,
        )
        self.language = config.ASR_LANGUAGE
        # Whisper prompt token limit 224 (hard) — vượt sẽ generate garbage.
        # faster-whisper auto-truncate, nhưng transformers thì không → ta cap
        # chars input ở 200 (~80-90 tokens Vietnamese) cho an toàn rộng rãi.
        self.initial_prompt = _build_initial_prompt(max_chars=200,
                                                     examples_per_intent=1)
        # Lazy: chỉ build prompt_ids khi cần. Một số tokenizer Whisper lỗi với
        # prompt quá dài → wrap try/except để không crash transcribe.
        self._prompt_ids = None
        # transformers Whisper model.generate() không thread-safe → lock như ASR.
        self._lock = threading.Lock()

    def _get_prompt_ids(self):
        """Lazy build prompt_ids để bias vocab — fail-safe.

        Whisper hard-limit 224 prompt tokens. Vượt → model generate garbage.
        Tự disable prompt_ids nếu vượt 200 thay vì để output sai.
        """
        if self._prompt_ids is False:
            return None     # đã thử và fail trước đó → skip
        if self._prompt_ids is not None:
            return self._prompt_ids
        try:
            tok = self.pipe.tokenizer
            ids = tok.get_prompt_ids(self.initial_prompt, return_tensors="pt")
            if ids.shape[0] > 200:
                _log.warning("PhoWhisper prompt %d tokens > 200 → disable bias",
                             ids.shape[0])
                self._prompt_ids = False
                return None
            self._prompt_ids = ids
            return ids
        except Exception as e:
            _log.warning("PhoWhisper prompt_ids build failed: %s — bỏ qua bias", e)
            self._prompt_ids = False
            return None

    def transcribe(self, audio: np.ndarray,
                   sample_rate: int = config.SAMPLE_RATE) -> str:
        """Audio float32 mono 16kHz → text. API tương đương ASR.transcribe."""
        if sample_rate != 16000:
            raise ValueError("PhoWhisper expect 16kHz")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            if np.abs(audio).max() > 1.5:
                audio = audio / 32768.0

        # Chỉ truyền language + task. KHÔNG truyền num_beams/do_sample/temperature
        # vì xung đột với generation_config có sẵn trong PhoWhisper checkpoint
        # → transformers throw deprecation warning + empty output.
        #
        # KHÔNG truyền prompt_ids: trong transformers 5.x, kết hợp
        # language+task+prompt_ids khiến Whisper "echo" lại nội dung prompt
        # thay vì transcribe audio (xác nhận bằng test: hai audio khác hẳn ra
        # cùng 1 string trùng phrase trong prompt). PhoWhisper đã fine-tune trên
        # 844h tiếng Việt → không cần prompt bias.
        generate_kwargs = {
            "language": self.language,
            "task": "transcribe",
        }

        with self._lock:
            try:
                result = self.pipe(
                    {"raw": audio, "sampling_rate": sample_rate},
                    generate_kwargs=generate_kwargs,
                )
            except Exception as e:
                _log.warning("PhoWhisper transcribe failed: %s", e)
                return ""

        text = (result.get("text") or "").strip() if isinstance(result, dict) else ""
        _log.info("PhoWhisper raw output (%.2fs audio): %r", len(audio)/sample_rate, text)

        # PhoWhisper không expose no_speech_prob/avg_logprob qua pipeline → chỉ
        # dùng được length-based hallucination check.
        if _is_hallucinated(text, len(audio), sample_rate):
            _log.info("PhoWhisper output filtered as hallucination: %r", text)
            return ""
        return text

    def transcribe_file(self, wav_path):
        audio = audio_io.load_wav(wav_path)
        return self.transcribe(audio)


# ==========================================================================
# Backend factory
# ==========================================================================
# Singleton — tránh load model nhiều lần (Whisper-base CPU mất ~2s).
_asr_instance = None


def get_asr():
    """Trả về ASR backend hiện tại theo config.ASR_BACKEND.

    Return type là Protocol có method `transcribe(audio, sample_rate) -> str`.
    Callers chỉ cần dùng method này — không phụ thuộc class cụ thể.
    """
    global _asr_instance
    if _asr_instance is not None:
        return _asr_instance

    backend = config.ASR_BACKEND
    if backend in ("faster-whisper", "whisper", ""):
        _asr_instance = ASR()
    elif backend == "phowhisper":
        _asr_instance = PhoWhisperASR()
    else:
        raise ValueError(
            f"ASR_BACKEND={backend!r} không hỗ trợ. "
            "Chọn 'faster-whisper' hoặc 'phowhisper'."
        )
    return _asr_instance
