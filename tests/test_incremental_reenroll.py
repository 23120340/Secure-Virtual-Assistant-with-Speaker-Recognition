"""Test incremental re-enroll: cập nhật centroid theo audio mới sau khi SV pass.

Mục tiêu chống speaker drift — alpha nhỏ giữ centroid stable, alpha=1 thì
centroid biến thành audio mới hoàn toàn (extreme case).
"""
import numpy as np

from core.database import SpeakerManager


class _FakeEncoder:
    backend_id = "ecapa"
    sv_threshold = 0.45
    sid_min_threshold = 0.35

    def encode_centroid(self, audios):
        # Return mean of fake "embeddings" — đủ cho test, không cần model thật.
        return np.ones(192, dtype=np.float32) / np.sqrt(192)

    def _hash_to_vec(self, audio):
        h = abs(hash(audio.tobytes())) % (10**8)
        rng = np.random.default_rng(h)
        v = rng.normal(size=192).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-9
        return v

    def encode(self, audio):
        return self._hash_to_vec(audio)

    def encode_multiwindow(self, audio):
        """SpeakerManager._prepare uses encode_multiwindow when
        SPEAKER_MULTIWINDOW=True (default). Fake bằng encode đơn lẻ."""
        return self._hash_to_vec(audio)


def _setup_user_with_centroid(tmp_db, monkeypatch):
    """Tạo user với fake centroid, monkeypatch SpeakerManager.encoder."""
    from core import speaker_encoder
    mgr = SpeakerManager(tmp_db)
    # Override encoder bằng fake
    mgr.encoder = _FakeEncoder()
    # Insert fake user + centroid (unit vector)
    centroid = np.ones(192, dtype=np.float32) / np.sqrt(192)
    tmp_db.add_user("u1", "Test", centroid)
    mgr._cache = {"u1": ("Test", centroid)}
    return mgr


def test_incremental_update_returns_false_for_unknown_user(tmp_db, monkeypatch):
    mgr = _setup_user_with_centroid(tmp_db, monkeypatch)
    fake_audio = np.zeros(16000, dtype=np.float32)
    ok, _ = mgr.incremental_update_centroid(fake_audio, "nonexistent")
    assert ok is False


def test_incremental_update_with_alpha_zero_keeps_centroid(tmp_db, monkeypatch):
    """alpha=0 → centroid không đổi."""
    mgr = _setup_user_with_centroid(tmp_db, monkeypatch)
    original = mgr._cache["u1"][1].copy()
    fake_audio = np.ones(16000, dtype=np.float32)
    ok, _ = mgr.incremental_update_centroid(fake_audio, "u1", alpha=0.0)
    assert ok
    # Centroid sau update vẫn bằng original (alpha=0 → giữ nguyên + re-normalize).
    new = mgr._cache["u1"][1]
    np.testing.assert_allclose(new, original, atol=1e-6)


def test_incremental_update_with_alpha_one_replaces_centroid(tmp_db, monkeypatch):
    """alpha=1 → centroid hoàn toàn là embedding mới (re-normalized)."""
    mgr = _setup_user_with_centroid(tmp_db, monkeypatch)
    fake_audio = np.ones(16000, dtype=np.float32)
    new_emb = mgr._prepare(fake_audio)
    mgr.incremental_update_centroid(fake_audio, "u1", alpha=1.0)
    new_centroid = mgr._cache["u1"][1]
    np.testing.assert_allclose(new_centroid, new_emb, atol=1e-6)


def test_incremental_update_preserves_unit_norm(tmp_db, monkeypatch):
    """Sau mọi update, centroid phải unit-norm (cosine = dot product)."""
    mgr = _setup_user_with_centroid(tmp_db, monkeypatch)
    fake_audio = np.ones(16000, dtype=np.float32)
    mgr.incremental_update_centroid(fake_audio, "u1", alpha=0.3)
    norm = float(np.linalg.norm(mgr._cache["u1"][1]))
    assert abs(norm - 1.0) < 1e-5


def test_incremental_update_persisted_to_db(tmp_db, monkeypatch):
    """Centroid mới phải được persist vào DB, không chỉ cache."""
    mgr = _setup_user_with_centroid(tmp_db, monkeypatch)
    fake_audio = np.ones(16000, dtype=np.float32)
    mgr.incremental_update_centroid(fake_audio, "u1", alpha=0.5)
    new_centroid = mgr._cache["u1"][1]

    # Reload từ DB không qua cache
    cache_fresh = tmp_db.load_all_embeddings(expected_dim=192, backend_id="ecapa")
    persisted = cache_fresh["u1"][1]
    np.testing.assert_allclose(persisted, new_centroid, atol=1e-6)


def test_alpha_clamped_to_valid_range(tmp_db, monkeypatch):
    """alpha out-of-range không crash — clamp [0, 1]."""
    mgr = _setup_user_with_centroid(tmp_db, monkeypatch)
    fake_audio = np.ones(16000, dtype=np.float32)
    # alpha < 0 → clamp 0 → centroid unchanged
    original = mgr._cache["u1"][1].copy()
    mgr.incremental_update_centroid(fake_audio, "u1", alpha=-0.5)
    np.testing.assert_allclose(mgr._cache["u1"][1], original, atol=1e-6)
    # alpha > 1 → clamp 1 → centroid = audio embedding
    mgr.incremental_update_centroid(fake_audio, "u1", alpha=10.0)
    np.testing.assert_allclose(mgr._cache["u1"][1],
                               mgr._prepare(fake_audio), atol=1e-6)
