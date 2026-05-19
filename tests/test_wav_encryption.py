"""Test WAV encryption at-rest: save → load roundtrip + ciphertext không là plaintext."""
import numpy as np

from core import audio_io


def test_save_load_plaintext_roundtrip(tmp_path, monkeypatch):
    """ENCRYPT_BIOMETRIC_WAV=False → ghi WAV bình thường."""
    from core import config
    monkeypatch.setattr(config, "ENCRYPT_BIOMETRIC_WAV", False)
    audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.05
    p = tmp_path / "plain.wav"
    audio_io.save_wav(audio, p)
    assert p.exists()
    assert not (tmp_path / "plain.wav.enc").exists()
    loaded = audio_io.load_wav(p)
    # PCM 16-bit roundtrip có quantization noise — tolerance lỏng.
    np.testing.assert_allclose(loaded, audio, atol=1e-3)


def test_save_load_encrypted_roundtrip(tmp_path, monkeypatch):
    """ENCRYPT_BIOMETRIC_WAV=True → ghi .enc thay vì .wav."""
    from core import config, database as _db
    monkeypatch.setattr(config, "ENCRYPT_BIOMETRIC_WAV", True)
    _db._FERNET = None  # reset cache, force re-derive with current FLASK_SECRET
    audio = np.random.RandomState(1).randn(16000).astype(np.float32) * 0.05
    p = tmp_path / "secret.wav"
    audio_io.save_wav(audio, p)
    # KHÔNG có file plaintext
    assert not p.exists()
    # CÓ file .enc
    enc = tmp_path / "secret.wav.enc"
    assert enc.exists()
    # Ciphertext không chứa pattern WAV header
    ciphertext = enc.read_text(encoding="ascii")
    assert ciphertext.startswith("gAAAA")
    assert "RIFF" not in ciphertext  # WAV magic bytes
    # Load lại → khớp audio gốc
    loaded = audio_io.load_wav(p)
    np.testing.assert_allclose(loaded, audio, atol=1e-3)
    _db._FERNET = None


def test_load_prefers_enc_over_plain(tmp_path, monkeypatch):
    """Nếu cả .wav và .wav.enc tồn tại → ưu tiên .enc (vì plaintext stale)."""
    from core import config, database as _db
    monkeypatch.setattr(config, "ENCRYPT_BIOMETRIC_WAV", True)
    _db._FERNET = None
    audio_enc = np.ones(16000, dtype=np.float32) * 0.5
    audio_plain = np.zeros(16000, dtype=np.float32)
    p = tmp_path / "both.wav"
    # Ghi enc trước
    audio_io.save_wav(audio_enc, p)
    # Ghi plain riêng để test priority — bypass save_wav.
    import soundfile as sf
    sf.write(str(p), audio_plain, 16000)
    # save_wav (encrypt mode) đã xoá plain — nhưng vừa rồi đè vào → file plain tồn tại.
    # Load phải trả về encrypted audio (= 0.5), không phải plain (= 0).
    loaded = audio_io.load_wav(p)
    assert abs(loaded.mean() - 0.5) < 0.01
    _db._FERNET = None


def test_load_falls_back_to_plain_when_no_enc(tmp_path, monkeypatch):
    """Backward compat: file cũ chỉ có .wav (chưa migrate) → vẫn load được."""
    from core import config
    monkeypatch.setattr(config, "ENCRYPT_BIOMETRIC_WAV", True)
    audio = np.random.RandomState(2).randn(16000).astype(np.float32) * 0.2
    p = tmp_path / "legacy.wav"
    # Ghi plaintext bypass save_wav.
    import soundfile as sf
    sf.write(str(p), audio, 16000)
    loaded = audio_io.load_wav(p)
    np.testing.assert_allclose(loaded, audio, atol=1e-3)
