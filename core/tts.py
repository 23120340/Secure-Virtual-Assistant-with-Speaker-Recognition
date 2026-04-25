"""TTS module với gTTS — đơn giản, online, hỗ trợ tiếng Việt."""
import io
import tempfile
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
from gtts import gTTS
from pydub import AudioSegment

from . import config


class TTS:
    def __init__(self, lang: str = config.TTS_LANG):
        self.lang = lang

    def synthesize(self, text: str, save_path: Path = None) -> np.ndarray:
        """Text → audio float32 mono ở 16kHz. Lưu wav nếu save_path != None."""
        # gTTS tạo MP3 → decode bằng pydub → resample về 16kHz mono
        mp3_buf = io.BytesIO()
        gTTS(text=text, lang=self.lang).write_to_fp(mp3_buf)
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

    def speak(self, text: str):
        """Synthesize và play qua loa."""
        if not text.strip():
            return
        print(f"  🔊 Speaking: {text}")
        try:
            audio = self.synthesize(text)
            sd.play(audio, config.SAMPLE_RATE)
            sd.wait()
        except Exception as e:
            print(f"  [TTS error: {e}] Falling back to text-only")


_tts_instance = None


def get_tts() -> TTS:
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTS()
    return _tts_instance
