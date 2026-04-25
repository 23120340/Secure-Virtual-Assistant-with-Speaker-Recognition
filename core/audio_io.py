"""Audio capture, file I/O, và VAD-based trimming."""
import io
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

from . import config

# sounddevice chỉ cần khi chạy CLI có mic. Flask server thường không có audio device.
try:
    import sounddevice as sd
    _HAS_SD = True
except (ImportError, OSError):
    sd = None
    _HAS_SD = False


# ==========================================================================
# Recording
# ==========================================================================
def record(duration: float, sample_rate: int = config.SAMPLE_RATE) -> np.ndarray:
    """Record từ default microphone, trả về float32 mono [N]."""
    if not _HAS_SD:
        raise RuntimeError("sounddevice không khả dụng. Dùng web UI hoặc --audio_files.")
    print(f"  🎙  Recording {duration:.1f}s...")
    audio = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze()


def decode_browser_audio(blob: bytes,
                         target_sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Decode audio từ browser (WebM/Opus, OGG, WAV...) → float32 mono [N] @ 16kHz."""
    from pydub import AudioSegment
    seg = AudioSegment.from_file(io.BytesIO(blob))
    seg = seg.set_frame_rate(target_sr).set_channels(1)
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    if seg.sample_width == 2:
        samples /= 32768.0
    elif seg.sample_width == 4:
        samples /= 2147483648.0
    return samples


def save_wav(audio: np.ndarray, path: Path,
             sample_rate: int = config.SAMPLE_RATE):
    sf.write(str(path), audio, sample_rate)


def load_wav(path: Path, sample_rate: int = config.SAMPLE_RATE) -> np.ndarray:
    """Load wav, resample về sample_rate, mono float32."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        # Resample đơn giản bằng torchaudio
        import torchaudio.functional as F
        audio = F.resample(torch.from_numpy(audio), sr, sample_rate).numpy()
    return audio


# ==========================================================================
# VAD (Silero) — cắt bỏ khoảng lặng đầu/cuối
# ==========================================================================
class SileroVAD:
    """Wrapper cho silero-vad. Lazy load để tránh chậm khi import."""
    _model = None
    _utils = None

    @classmethod
    def _load(cls):
        if cls._model is None:
            print("  Loading Silero VAD...")
            cls._model, cls._utils = torch.hub.load(
                "snakers4/silero-vad", "silero_vad", trust_repo=True
            )
        return cls._model, cls._utils

    @classmethod
    def trim(cls, audio: np.ndarray,
             sample_rate: int = config.SAMPLE_RATE,
             min_speech_ms: int = 250) -> np.ndarray:
        """Giữ lại các đoạn có speech, ghép lại. Trả về audio đã trim."""
        model, utils = cls._load()
        get_speech_timestamps = utils[0]

        wav_t = torch.from_numpy(audio).float()
        ts = get_speech_timestamps(
            wav_t, model,
            sampling_rate=sample_rate,
            min_speech_duration_ms=min_speech_ms,
        )
        if not ts:
            return audio  # không detect được → trả về nguyên
        chunks = [wav_t[t["start"]:t["end"]] for t in ts]
        return torch.cat(chunks).numpy()


# ==========================================================================
# Convenience: record + trim + save
# ==========================================================================
def record_and_trim(duration: float, save_path: Path = None) -> np.ndarray:
    audio = record(duration)
    audio = SileroVAD.trim(audio)
    if save_path is not None:
        save_wav(audio, save_path)
    return audio
