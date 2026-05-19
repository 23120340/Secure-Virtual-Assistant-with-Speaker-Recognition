"""Re-encode audio đã lưu của user và lưu embedding cho backend hiện tại.

Bối cảnh: đổi SPEAKER_BACKEND (vd: ecapa → wavlm) → DB filter theo backend_id
nên embedding cũ không xài được cho backend mới. Script này đọc audio enroll
gốc trong `data/enroll_audio/<user>/` và encode lại với encoder của backend
đang chạy, lưu vào DB như 1 embedding mới (không động vào embedding cũ).

Cách dùng:
    # Trước đó đã set SPEAKER_BACKEND=wavlm trong .env hoặc inline:
    SPEAKER_BACKEND=wavlm python scripts/reenroll_backend.py             # tất cả user
    SPEAKER_BACKEND=wavlm python scripts/reenroll_backend.py --user minh # chỉ 1 user
    python scripts/reenroll_backend.py --dry-run                          # chỉ list
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import audio_io, config
from core.database import UserDB, SpeakerManager


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user", default=None,
                   help="user_id cụ thể, mặc định re-enroll tất cả")
    p.add_argument("--dry-run", action="store_true",
                   help="Chỉ list user/audio cần xử lý, không encode")
    args = p.parse_args()

    db = UserDB()
    mgr = SpeakerManager(db)
    backend = mgr.encoder.backend_id
    print(f"Current SPEAKER_BACKEND = {backend} (dim={mgr.encoder.embedding_dim})")
    print(f"Encoder type: {type(mgr.encoder).__name__}")

    all_users = db.list_users()
    already_enrolled = set(db.users_with_backend(backend))
    target_users = ([u for u in all_users if u["user_id"] == args.user]
                    if args.user else all_users)
    if not target_users:
        print(f"Không có user nào match {args.user!r}")
        return

    print(f"\n{len(target_users)} user trong DB, "
          f"{len(already_enrolled)} đã có embedding cho '{backend}':")
    for u in target_users:
        uid = u["user_id"]
        audio_dir = config.ENROLL_AUDIO_DIR / uid
        wavs = sorted(audio_dir.glob("*.wav")) if audio_dir.exists() else []
        status = "✓ đã có" if uid in already_enrolled else "✗ thiếu"
        print(f"  {status}  {uid:15s} ({u['name']:20s}) — {len(wavs)} audio file")

    if args.dry_run:
        print("\n--dry-run → không enroll.")
        return

    todo = [u for u in target_users
            if not args.user or u["user_id"] == args.user]
    if not todo:
        print("\nKhông có user nào cần xử lý.")
        return

    print(f"\nBắt đầu re-encode cho {len(todo)} user...")
    n_ok, n_skip, n_err = 0, 0, 0
    for u in todo:
        uid = u["user_id"]
        audio_dir = config.ENROLL_AUDIO_DIR / uid
        if not audio_dir.exists():
            print(f"  [{uid}] SKIP — không có thư mục {audio_dir}")
            n_skip += 1
            continue
        wavs = sorted(audio_dir.glob("*.wav"))
        if len(wavs) < 2:
            print(f"  [{uid}] SKIP — chỉ có {len(wavs)} audio, cần ≥ 2")
            n_skip += 1
            continue
        try:
            audios = []
            for wav in wavs:
                a = audio_io.load_wav(wav)
                a = audio_io.SileroVAD.trim(a)
                if a.size < config.SAMPLE_RATE:
                    continue
                audios.append(a)
            if len(audios) < 2:
                print(f"  [{uid}] SKIP — sau VAD chỉ còn {len(audios)} audio đủ length")
                n_skip += 1
                continue

            centroid = mgr.enroll_additional_backend(uid, audios)
            print(f"  [{uid}] OK — {len(audios)} samples → "
                  f"centroid {centroid.shape}, norm={(centroid**2).sum()**0.5:.4f}")
            n_ok += 1
        except Exception as e:
            print(f"  [{uid}] ERROR — {e}")
            n_err += 1

    print(f"\nDone. OK={n_ok}, SKIP={n_skip}, ERROR={n_err}")


if __name__ == "__main__":
    main()
