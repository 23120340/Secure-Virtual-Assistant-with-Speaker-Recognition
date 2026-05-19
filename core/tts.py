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
        # Offline fallback (pyttsx3) — lazy-init: nếu không cài, set None → skip.
        self._offline_engine = None
        self._offline_engine_tried = False

    # ----- Offline fallback (pyttsx3) ---------------------------------------
    def _ensure_offline_engine(self):
        """Lazy init pyttsx3 engine. Trả về engine hoặc None nếu không có lib.

        pyttsx3 dùng SAPI5 trên Windows, NSSpeechSynthesizer trên macOS,
        espeak trên Linux — không phụ thuộc mạng. Chất lượng kém hơn gTTS
        nhưng đủ cho fallback khi offline / mất mạng / quota gTTS.
        """
        if self._offline_engine_tried:
            return self._offline_engine
        self._offline_engine_tried = True
        try:
            import pyttsx3  # type: ignore
        except ImportError:
            _log.info("pyttsx3 chưa cài — offline TTS fallback disabled. "
                      "Cài bằng: pip install pyttsx3")
            return None
        try:
            eng = pyttsx3.init()
            # 150-160 wpm dễ nghe; default ~200 hơi nhanh cho tiếng Việt.
            eng.setProperty("rate", 160)
            self._offline_engine = eng
            _log.info("pyttsx3 offline TTS đã sẵn sàng (fallback khi gTTS fail).")
            return eng
        except Exception as e:
            _log.warning("pyttsx3 init failed (%s) — offline TTS fallback disabled", e)
            return None

    def _synthesize_offline_to_wav_bytes(self, text: str) -> bytes:
        """Synthesize bằng pyttsx3 → đọc file WAV → return bytes.

        pyttsx3 không có in-memory API ổn định trên Windows; cách an toàn nhất
        là save_to_file → đọc lại WAV.
        """
        engine = self._ensure_offline_engine()
        if engine is None:
            return b""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_path = tf.name
        try:
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            with open(tmp_path, "rb") as f:
                return f.read()
        except Exception as e:
            _log.warning("pyttsx3 synth failed: %s", e)
            return b""
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _wav_bytes_to_mp3_bytes(self, wav_bytes: bytes) -> bytes:
        """Convert WAV (pyttsx3 output) → MP3 (browser stream)."""
        if not wav_bytes:
            return b""
        seg = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        seg = seg.set_frame_rate(config.SAMPLE_RATE).set_channels(1)
        out = io.BytesIO()
        seg.export(out, format="mp3", bitrate="64k")
        return out.getvalue()

    # ----- Online primary (gTTS) -------------------------------------------
    def synthesize(self, text: str, save_path: Path = None,
                   lang: str | None = None) -> np.ndarray:
        """Text → audio float32 mono ở 16kHz. Lưu wav nếu save_path != None.

        `lang` override per-call (multi-language theo user preferences); None →
        dùng `self.lang` (mặc định Vietnamese).

        gTTS fail (mất mạng/quota) → fallback pyttsx3 nếu có. Nếu cả 2 fail,
        raise — caller phải xử lý (route TTS trả 204).
        """
        effective_lang = _resolve_lang(lang, self.lang)
        try:
            mp3_buf = io.BytesIO()
            gTTS(text=text, lang=effective_lang).write_to_fp(mp3_buf)
            mp3_buf.seek(0)
            seg = AudioSegment.from_file(mp3_buf, format="mp3")
        except Exception as e:
            # Fallback pyttsx3 (offline) — chỉ tiếng Anh tốt; tiếng Việt brittle
            # với engine mặc định nhưng vẫn hơn câm.
            _log.info("gTTS failed (%s) → thử pyttsx3 offline", e)
            wav_bytes = self._synthesize_offline_to_wav_bytes(text)
            if not wav_bytes:
                raise
            seg = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")

        seg = seg.set_frame_rate(config.SAMPLE_RATE).set_channels(1)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        if seg.sample_width == 2:
            samples /= 32768.0

        if save_path is not None:
            sf.write(str(save_path), samples, config.SAMPLE_RATE)
        return samples

    def synthesize_to_mp3_bytes(self, text: str, lang: str | None = None) -> bytes:
        """Text → MP3 bytes (cho browser <audio> tag stream về client).

        Primary: gTTS. Fallback: pyttsx3 → WAV → MP3 (qua pydub).
        Trả `b""` nếu cả 2 fail (caller route /api/tts đã handle b"" thành 204).

        `lang` override theo user preferences; None → mặc định self.lang.
        """
        effective_lang = _resolve_lang(lang, self.lang)
        try:
            mp3_buf = io.BytesIO()
            gTTS(text=text, lang=effective_lang).write_to_fp(mp3_buf)
            return mp3_buf.getvalue()
        except Exception as e:
            _log.info("gTTS mp3 failed (%s) → thử pyttsx3 offline", e)
            wav_bytes = self._synthesize_offline_to_wav_bytes(text)
            return self._wav_bytes_to_mp3_bytes(wav_bytes)

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
