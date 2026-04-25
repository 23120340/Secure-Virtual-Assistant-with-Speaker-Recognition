"""Smoke test toàn bộ pipeline với file audio sẵn (không cần mic).

Hữu ích khi:
  - Dev trên Kaggle/Colab không có mic
  - Reproducibility test cho báo cáo
  - Calibrate threshold

Cách dùng:
    python cli/test_pipeline.py --audio sample.wav
    python cli/test_pipeline.py --audio sample.wav --claimed_user minh
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import audio_io
from core.asr import get_asr
from core.nlu import get_nlu
from core.database import UserDB, SpeakerManager
from core.router import Router


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True, help="File wav input")
    p.add_argument("--skip_asr", action="store_true",
                   help="Bỏ qua ASR, dùng --transcript thay thế")
    p.add_argument("--transcript", default=None,
                   help="Text override (dùng để test NLU/router không qua ASR)")
    args = p.parse_args()

    # Load
    audio = audio_io.load_wav(Path(args.audio))
    audio = audio_io.SileroVAD.trim(audio)
    print(f"Audio: {audio.size/16000:.2f}s sau VAD trim")

    # ASR
    if args.skip_asr:
        transcript = args.transcript or ""
        print(f"Transcript (override): '{transcript}'")
    else:
        asr = get_asr()
        transcript = asr.transcribe(audio)
        print(f"Transcript: '{transcript}'")

    # NLU
    nlu = get_nlu()
    nlu_result = nlu.parse(transcript)
    print(f"NLU: {nlu_result}")

    # Router
    db = UserDB()
    spk_mgr = SpeakerManager(db)
    router = Router(spk_mgr)
    result = router.handle_turn(audio, transcript, nlu_result)

    print("\n=== Router Output ===")
    print(f"  intent     : {result.intent}  [{result.auth_level}]")
    print(f"  speaker    : {result.identified_user_name} "
          f"(score={result.sid_score:.3f})")
    if result.sv_required:
        print(f"  SV         : {'PASS' if result.sv_passed else 'FAIL'} "
              f"(score={result.sv_score:.3f})")
    print(f"  blocked    : {result.blocked}")
    print(f"  response   : {result.response}")


if __name__ == "__main__":
    main()
