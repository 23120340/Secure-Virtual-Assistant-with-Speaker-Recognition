"""TTS module với gTTS — đơn giản, online, hỗ trợ tiếng Việt."""
import io
import logging
import os
import tempfile
from pathlib import Path

_log = logging.getLogger("secva.tts")

# Thêm ffmpeg vào PATH TRƯỚC khi import pydub (pydub check PATH lúc import)
_FFMPEG_BIN_HINTS = [
    os.getenv("FFMPEG_BIN", ""),
    r"C:\Users\Acer\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin",
]
for _bin_dir in _FFMPEG_BIN_HINTS:
    if _bin_dir and Path(_bin_dir).exists():
        os.environ["PATH"] = _bin_dir + os.pathsep + os.environ.get("PATH", "")
        break

import numpy as np
import sounddevice as sd
import soundfile as sf
from gtts import gTTS
from pydub import AudioSegment

from . import config

# Set explicit path sau khi import để pydub dùng đúng binary
for _bin_dir in _FFMPEG_BIN_HINTS:
    if _bin_dir and Path(_bin_dir).exists():
        AudioSegment.converter = str(Path(_bin_dir) / "ffmpeg.exe")
        AudioSegment.ffprobe   = str(Path(_bin_dir) / "ffprobe.exe")
        break


# Supported gTTS language codes (subset chính). Multi-language switch.
SUPPORTED_TTS_LANGS = {
    "vi": "Tiếng Việt",
    "en": "English",
    "ja": "日本語",
    "ko": "한국어",
    "zh-CN": "中文",
    "fr": "Français",
    "de": "Deutsch",
}


def _resolve_lang(lang: str | None, fallback: str = config.TTS_LANG) -> str:
    """Normalize lang code, fallback nếu None/empty/unsupported."""
    if not lang:
        return fallback
    code = lang.strip()
    # Cho phép "zh" → "zh-CN", các code phổ biến.
    if code == "zh":
        code = "zh-CN"
    if code in SUPPORTED_TTS_LANGS:
        return code
    return fallback


class TTS:
    def __init__(self, lang: str = config.TTS_LANG):
        self.lang = lang

    def synthesize(self, text: str, save_path: Path = None,
                   lang: str | None = None) -> np.ndarray:
        """Text → audio float32 mono ở 16kHz. Lưu wav nếu save_path != None.

        `lang` override per-call (multi-language theo user preferences); None →
        dùng `self.lang` (mặc định Vietnamese).
        """
        effective_lang = _resolve_lang(lang, self.lang)
        # gTTS tạo MP3 → decode bằng pydub → resample về 16kHz mono
        mp3_buf = io.BytesIO()
        gTTS(text=text, lang=effective_lang).write_to_fp(mp3_buf)
        mp3_buf.seek(0)

        seg = AudioSegment.from_file(mp3_buf, format="mp3")
        seg = seg.set_frame_rate(config.SAMPLE_RATE).set_channels(1)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        # Convert int16 range → [-1, 1]
        if seg.sample_width == 2:
            samples /= 32768.0

        if save_path is not None:
            sf.write(str(save_path), samples, config.SAMPLE_RATE)
        return samples

    def synthesize_to_mp3_bytes(self, text: str, lang: str | None = None) -> bytes:
        """Text → MP3 bytes (cho browser <audio> tag stream về client).

        `lang` override theo user preferences; None → mặc định self.lang.
        """
        effective_lang = _resolve_lang(lang, self.lang)
        mp3_buf = io.BytesIO()
        gTTS(text=text, lang=effective_lang).write_to_fp(mp3_buf)
        return mp3_buf.getvalue()

    def speak(self, text: str):
        """Synthesize và play qua loa. Chỉ chạy khi có sounddevice (CLI mode)."""
        if not text.strip():
            return
        try:
            import sounddevice as _sd
        except (ImportError, OSError):
            _log.info("[TTS skipped — no audio device] %s", text)
            return
        _log.info("🔊 Speaking: %s", text)
        try:
            audio = self.synthesize(text)
            _sd.play(audio, config.SAMPLE_RATE)
            _sd.wait()
        except Exception as e:
            _log.warning("TTS error: %s — fall back to text-only", e)


_tts_instance = None


def get_tts() -> TTS:
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTS()
    return _tts_instance
