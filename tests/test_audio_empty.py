import io

import pytest

from core import audio_io


def test_decode_browser_audio_tiny_blob_is_empty_audio():
    with pytest.raises(audio_io.EmptyAudio):
        audio_io.decode_browser_audio(b"")

    with pytest.raises(audio_io.EmptyAudio):
        audio_io.decode_browser_audio(b"\x1a\x45\xdf\xa3")


def test_decode_browser_audio_invalid_webm_header_is_empty_audio():
    blob = b"\x1a\x45\xdf\xa3" + (b"\x00" * 1024)
    with pytest.raises(audio_io.EmptyAudio):
        audio_io.decode_browser_audio(blob)


def test_decode_browser_audio_valid_wav_still_decodes():
    import numpy as np
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.zeros(1600, dtype=np.float32), 16000,
             format="WAV", subtype="PCM_16")
    audio = audio_io.decode_browser_audio(buf.getvalue())
    assert audio.dtype == np.float32
    assert audio.size == 1600
