"""Audio capture, file I/O, và VAD-based trimming."""
import io
import logging
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

from . import config

_log = logging.getLogger("secva.audio")

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
    _log.info("🎙  Recording %.1fs...", duration)
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
    if seg.sample_width == 1:
        # 8-bit PCM unsigned [0, 255] → normalize về [-1, 1]
        samples = (samples - 128.0) / 128.0
    elif seg.sample_width == 2:
        samples /= 32768.0
    elif seg.sample_width == 4:
        samples /= 2147483648.0
    else:
        # Sample width lạ → normalize bằng max abs để tránh clipping
        max_val = float(np.abs(samples).max() or 1.0)
        samples /= max_val
    return samples


def save_wav(audio: np.ndarray, path: Path,
             sample_rate: int = config.SAMPLE_RATE):
    """Lưu audio sang WAV. Khi `config.ENCRYPT_BIOMETRIC_WAV=true`, ghi
    `<path>.enc` (Fernet ciphertext) thay vì plaintext WAV — bảo vệ mẫu giọng
    người dùng ở filesystem (DB leak / backup leak không lộ audio gốc).

    Cả `path.wav` và `path.wav.enc` đều được caller truyền dưới dạng `path`
    với extension .wav. Wrapper thêm `.enc` khi encrypt mode bật.
    """
    if not config.ENCRYPT_BIOMETRIC_WAV:
        sf.write(str(path), audio, sample_rate)
        return
    # Encrypt mode: encode → bytes → fernet → ghi .enc
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    from .database import _encrypt
    ciphertext = _encrypt(buf.read().decode("latin-1"))  # binary → latin-1 → str
    enc_path = Path(str(path) + ".enc")
    enc_path.write_text(ciphertext, encoding="ascii")
    # Xoá plaintext nếu trùng tên — tránh inconsistency.
    plain_path = Path(path)
    if plain_path.exists() and plain_path != enc_path:
        try:
            plain_path.unlink()
        except OSError:
            pass


def load_wav(path: Path, sample_rate: int = config.SAMPLE_RATE) -> np.ndarray:
    """Load wav, resample về sample_rate, mono float32.

    Tự detect encrypted (.enc) — ưu tiên file `.enc` nếu tồn tại,
    fallback `.wav` plaintext (backward compat với data cũ).
    """
    plain_path = Path(path)
    enc_path = Path(str(path) + ".enc")
    if enc_path.exists():
        from .database import _decrypt
        ciphertext = enc_path.read_text(encoding="ascii")
        wav_bytes = _decrypt(ciphertext).encode("latin-1")
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    elif plain_path.exists():
        audio, sr = sf.read(str(plain_path), dtype="float32")
    else:
        raise FileNotFoundError(f"Không tìm thấy {plain_path} (cũng đã thử {enc_path})")
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
import threading


class SileroVAD:
    """Wrapper cho silero-vad. Lazy load để tránh chậm khi import.

    Dùng package PyPI `silero-vad` thay cho `torch.hub.load(snakers4/silero-vad)`
    — torch.hub tải Python source + execute → supply-chain RCE nếu repo bị compromise.
    Package PyPI đã có trong requirements.txt.
    """
    _model = None
    _lock = threading.Lock()

    @classmethod
    def _load(cls):
        if cls._model is None:
            with cls._lock:
                if cls._model is None:  # double-checked locking
                    _log.info("Loading Silero VAD (from silero_vad pip package)...")
                    from silero_vad import load_silero_vad
                    cls._model = load_silero_vad()
        return cls._model

    @classmethod
    def trim(cls, audio: np.ndarray,
             sample_rate: int = config.SAMPLE_RATE,
             min_speech_ms: int = 250,
             return_speech_detected: bool = False):
        """Giữ lại các đoạn có speech, ghép lại. Trả về audio đã trim.

        Nếu return_speech_detected=True → trả tuple (audio, has_speech: bool).
        has_speech=False khi VAD không tìm thấy đoạn nào → caller có thể reject
        thay vì gọi encoder/ASR trên silence."""
        from silero_vad import get_speech_timestamps
        model = cls._load()

        wav_t = torch.from_numpy(audio).float()
        ts = get_speech_timestamps(
            wav_t, model,
            sampling_rate=sample_rate,
            min_speech_duration_ms=min_speech_ms,
        )
        if not ts:
            if return_speech_detected:
                return audio, False
            return audio  # không detect được → trả về nguyên
        chunks = [wav_t[t["start"]:t["end"]] for t in ts]
        result = torch.cat(chunks).numpy()
        if return_speech_detected:
            return result, True
        return result


# ==========================================================================
# Audio preprocessing cho speaker encoder
# ==========================================================================
# noisereduce là optional dep — không bắt buộc cài để chạy hệ thống.
_NOISEREDUCE = None
_NOISEREDUCE_TRIED = False


def _try_noisereduce():
    """Lazy import noisereduce. Cache None nếu không có package."""
    global _NOISEREDUCE, _NOISEREDUCE_TRIED
    if not _NOISEREDUCE_TRIED:
        _NOISEREDUCE_TRIED = True
        try:
            import noisereduce  # type: ignore
            _NOISEREDUCE = noisereduce
            _log.info("noisereduce available → denoise có thể bật")
        except ImportError:
            _log.info("noisereduce chưa cài (pip install noisereduce) → denoise no-op")
    return _NOISEREDUCE


def preprocess_audio(audio: np.ndarray,
                     sample_rate: int = config.SAMPLE_RATE,
                     denoise: bool | None = None,
                     normalize: bool | None = None) -> np.ndarray:
    """Pre-process audio trước khi đưa vào speaker encoder.

    - DC-removal (mean centering): bỏ DC bias của micro
    - Denoise (optional): spectral gating qua noisereduce (nếu cài + flag bật)
    - RMS-normalize (optional): đưa loudness về RMS ~0.1 → giảm ảnh hưởng âm
      lượng tới embedding

    Cả 2 flag (denoise, normalize) khi None sẽ đọc từ config — bật/tắt qua env.
    Quan trọng: nếu đổi preprocessing sau khi enroll, hãy re-enroll user để
    centroid khớp domain mới (audio đã preprocess vs chưa).
    """
    if denoise is None:
        denoise = config.SPEAKER_DENOISE
    if normalize is None:
        normalize = config.SPEAKER_PREPROCESS

    if audio.size == 0:
        return audio

    out = audio.astype(np.float32, copy=False)

    # Luôn DC-remove khi preprocess (rất rẻ, không gây side effect)
    if normalize or denoise:
        out = out - out.mean()

    if denoise:
        nr = _try_noisereduce()
        if nr is not None:
            try:
                # stationary=True dùng cho noise nền liên tục (ổn định hơn).
                # prop_decrease=0.8 để không quá aggressive (giữ formant tiếng nói).
                out = nr.reduce_noise(
                    y=out, sr=sample_rate,
                    stationary=True, prop_decrease=0.8,
                ).astype(np.float32)
            except Exception as e:
                _log.warning("noisereduce failed: %s — bỏ qua denoise", e)

    if normalize:
        # RMS normalize → target 0.1 (~ -20 dBFS) — speech điển hình
        rms = float(np.sqrt(np.mean(out ** 2)) + 1e-9)
        if rms > 1e-5:    # tránh khuếch đại silence/noise nhỏ
            out = out * (0.1 / rms)
            # Soft clip để chặn peak > 1.0 (gây distortion)
            np.clip(out, -1.0, 1.0, out=out)

    return out


# ==========================================================================
# Convenience: record + trim + save
# ==========================================================================
def record_and_trim(duration: float, save_path: Path = None) -> np.ndarray:
    audio = record(duration)
    audio = SileroVAD.trim(audio)
    if save_path is not None:
        save_wav(audio, save_path)
    return audio
