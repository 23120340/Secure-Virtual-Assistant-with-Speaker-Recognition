"""ASR module dùng faster-whisper."""
from pathlib import Path
import tempfile
import numpy as np
from faster_whisper import WhisperModel

from . import config
from . import audio_io


class ASR:
    """Wrapper cho faster-whisper, cache model singleton."""

    def __init__(self,
                 model_size: str = config.WHISPER_MODEL,
                 device: str = config.WHISPER_DEVICE,
                 compute_type: str = config.WHISPER_COMPUTE):
        print(f"  Loading Whisper '{model_size}' trên {device}/{compute_type}...")
        self.model = WhisperModel(model_size, device=device,
                                  compute_type=compute_type)
        self.language = config.ASR_LANGUAGE

    def transcribe(self, audio: np.ndarray,
                   sample_rate: int = config.SAMPLE_RATE) -> str:
        """Audio float32 mono → text. faster-whisper accept ndarray trực tiếp
        nếu sample rate = 16k."""
        if sample_rate != 16000:
            raise ValueError("Whisper expect 16kHz")

        segments, info = self.model.transcribe(
            audio, language=self.language,
            beam_size=5, vad_filter=False,  # ta đã VAD trước đó
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text

    def transcribe_file(self, wav_path: Path) -> str:
        audio = audio_io.load_wav(wav_path)
        return self.transcribe(audio)


# Singleton
_asr_instance = None


def get_asr() -> ASR:
    global _asr_instance
    if _asr_instance is None:
        _asr_instance = ASR()
    return _asr_instance
