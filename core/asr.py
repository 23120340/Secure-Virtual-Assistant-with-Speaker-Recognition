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

_log = logging.getLogger("secva.asr")


def _is_hallucinated(text: str, n_samples: int,
                     sample_rate: int = 16000) -> bool:
    """Phát hiện Whisper hallucination: lặp câu hoặc quá dài so với audio.

    Whisper dễ sinh ra văn bản lặp lại (repetition loop) khi audio ngắn/im lặng.
    VD: 'Xin chào tất cả các bạn...' nhân lên 9 lần mặc dù chỉ nói 2 chữ.
    """
    if not text:
        return False

    duration_s = n_samples / sample_rate

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
            segments, _info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=5,             # giảm 10 → 5: ~2x nhanh hơn, WER tương đương vi
                best_of=1,
                temperature=0.0,            # deterministic, tránh hallucination
                vad_filter=False,           # đã VAD trước rồi
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,    # lọc segment xác suất thấp
                compression_ratio_threshold=1.8,  # thấp hơn = bắt lặp câu sớm hơn
            )

            # Lọc từng segment: bỏ qua segment có khả năng cao là không có speech
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


def correct_transcript(text: str) -> str:
    """Sửa lỗi ASR bằng Gemini. Opt-in qua config.ASR_CORRECT_ENABLED.

    - Tốn 500-2000ms / turn → ngoài thực tế nên tắt mặc định trên CPU.
    - Cache LRU theo text để các turn lặp lại không bắn API mới.
    - Skip khi text quá ngắn (< 3 ký tự).
    """
    if (not config.ASR_CORRECT_ENABLED
            or not text
            or not config.GEMINI_API_KEY
            or len(text.strip()) < 3):
        return text
    return _correct_cached(text)


# Singleton
_asr_instance = None


def get_asr() -> ASR:
    global _asr_instance
    if _asr_instance is None:
        _asr_instance = ASR()
    return _asr_instance
