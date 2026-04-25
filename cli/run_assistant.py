"""REPL chạy virtual assistant end-to-end qua mic.

Cách dùng:
    python cli/run_assistant.py             # mic mode
    python cli/run_assistant.py --no-tts    # không phát loa, chỉ in text
    python cli/run_assistant.py --text-mode # nhập text thay vì nói (debug NLU)

Workflow mỗi turn:
    1. Press Enter → record 5s từ mic
    2. VAD trim → ASR → speaker encoder song song
    3. NLU phân tích intent
    4. Router gate (SV/SID) + dispatch handler
    5. TTS đọc response

Bấm Ctrl+C hoặc gõ 'quit' để thoát.
"""
import sys
import argparse
import json
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import audio_io, config
from core.asr import get_asr
from core.tts import get_tts
from core.nlu import get_nlu
from core.database import UserDB, SpeakerManager
from core.router import Router


def print_turn(result):
    """In log đẹp cho debugging + báo cáo."""
    print("\n" + "─" * 60)
    print(f"  Transcript:  '{result.transcript}'")
    print(f"  Intent:      {result.intent}  [{result.auth_level}]")
    print(f"  Entities:    {result.entities}")
    print(f"  Speaker:     {result.identified_user_name} "
          f"(uid={result.identified_user_id}, sid_score={result.sid_score:.3f})")
    if result.sv_required:
        status = "PASS" if result.sv_passed else "FAIL"
        print(f"  SV check:    {status} (score={result.sv_score:.3f})")
    if result.blocked:
        print(f"  ⛔ BLOCKED")
    print(f"  Response:    {result.response}")
    print("─" * 60)


def run_mic_mode(args):
    print("Đang load các module (lần đầu sẽ chậm)...")
    asr = get_asr()
    tts = get_tts() if not args.no_tts else None
    nlu = get_nlu()
    db = UserDB()
    spk_mgr = SpeakerManager(db)
    router = Router(spk_mgr)

    users = db.list_users()
    print(f"\n✅ Sẵn sàng. Database có {len(users)} users: "
          f"{[u['name'] for u in users] or '(chưa ai)'}")
    print(f"   ASR: Whisper {config.WHISPER_MODEL}")
    print(f"   NLU: {nlu.__class__.__name__}")
    print(f"   Encoder: {spk_mgr.encoder.mode}")
    print("\nBấm Enter để nói (hoặc gõ 'quit' để thoát).\n")

    log = []
    while True:
        try:
            cmd = input("> ").strip().lower()
            if cmd in ("quit", "q", "exit"):
                break

            audio = audio_io.record_and_trim(config.RECORD_DURATION)
            if audio.size < config.SAMPLE_RATE // 2:
                print("  Không nghe rõ. Nói lại nhé.")
                continue

            transcript = asr.transcribe(audio)
            if not transcript:
                print("  ASR không nhận diện được nội dung.")
                continue

            nlu_result = nlu.parse(transcript)
            result = router.handle_turn(audio, transcript, nlu_result)
            print_turn(result)

            if tts:
                tts.speak(result.response)

            log.append(asdict(result))

        except KeyboardInterrupt:
            break

    # Lưu log để báo cáo
    if log and args.log_file:
        with open(args.log_file, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu {len(log)} turn log vào {args.log_file}")


def run_text_mode(args):
    """Debug NLU + handler không qua audio. Hữu ích để test logic intent."""
    print("Text mode (chỉ test NLU + handler, không qua ASR/SV/SID)")
    nlu = get_nlu()
    db = UserDB()
    users = db.list_users()
    print(f"Users in DB: {[u['name'] for u in users]}\n")

    while True:
        try:
            text = input("text> ").strip()
            if not text or text.lower() in ("quit", "q"):
                break
            result = nlu.parse(text)
            print(f"  → intent={result['intent']}, entities={result['entities']}")
        except KeyboardInterrupt:
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-tts", action="store_true",
                   help="Không phát loa, chỉ in text")
    p.add_argument("--text-mode", action="store_true",
                   help="Test NLU bằng text, bỏ qua audio pipeline")
    p.add_argument("--log-file", default="data/turn_log.json",
                   help="File JSON lưu log từng turn (cho báo cáo)")
    args = p.parse_args()

    if args.text_mode:
        run_text_mode(args)
    else:
        run_mic_mode(args)


if __name__ == "__main__":
    main()
