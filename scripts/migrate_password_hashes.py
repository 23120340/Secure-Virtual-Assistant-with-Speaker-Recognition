"""Migration script cho password_hash.

Mục đích: Sau khi đổi từ SHA-256 unsalted → PBKDF2-HMAC-SHA256 + salt (xem
core/database.py), các hash cũ trong DB không còn xác thực được nữa.

Script này quét tất cả user có password_hash:
  - legacy_sha256: 64 ký tự hex (SHA-256), KHÔNG có dấu "$"
  - empty       : "" — backdoor cũ "no password = always pass"

→ Reset password tạm thời, in ra console để admin báo user đổi sau khi đăng nhập.

Cách chạy:
    python scripts/migrate_password_hashes.py
    # hoặc dry-run trước:
    python scripts/migrate_password_hashes.py --dry-run

Lưu ý: hash gốc SHA-256 không thể decode → không thể giữ password cũ.
"""
import argparse
import secrets
import sqlite3
import sys
from pathlib import Path

# Thêm project root vào path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.database import _hash_pw  # noqa: E402  (PBKDF2 mới)
from core import config              # noqa: E402


def is_legacy_sha256(stored: str) -> bool:
    """Hash cũ: 64 ký tự hex, không có '$' (định dạng PBKDF2 mới là 'salt$hash')."""
    if not stored or "$" in stored:
        return False
    if len(stored) != 64:
        return False
    try:
        int(stored, 16)
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Migrate password hashes to PBKDF2")
    parser.add_argument("--dry-run", action="store_true",
                        help="In ra danh sách user sẽ bị reset, không ghi DB")
    parser.add_argument("--db", default=str(config.DB_PATH),
                        help=f"Path tới DB (default {config.DB_PATH})")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB không tồn tại: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT user_id, password_hash FROM users"
    ).fetchall()

    to_reset = []
    already_ok = 0
    for uid, ph in rows:
        if ph and "$" in ph:
            already_ok += 1
            continue
        # legacy SHA-256 hoặc empty — đều cần reset
        reason = "empty(backdoor)" if not ph else "legacy_sha256"
        to_reset.append((uid, reason))

    print(f"Tổng số user: {len(rows)}")
    print(f"  Hash PBKDF2 (đã OK): {already_ok}")
    print(f"  Cần reset:           {len(to_reset)}")
    print()

    if not to_reset:
        print("Không có gì cần làm. Migration đã xong.")
        return

    print("─" * 60)
    print(f"{'USER_ID':<24} {'REASON':<18} {'TEMP_PASSWORD'}")
    print("─" * 60)

    new_hashes = []
    for uid, reason in to_reset:
        # Random 12-char URL-safe password — báo user đổi ngay sau khi login
        temp_pw = secrets.token_urlsafe(9)
        new_hash = _hash_pw(temp_pw)
        new_hashes.append((uid, new_hash, temp_pw, reason))
        print(f"{uid:<24} {reason:<18} {temp_pw}")

    print("─" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Không có thay đổi nào trên DB.")
        return

    print("\nĐang ghi vào DB...")
    for uid, new_hash, _temp, _reason in new_hashes:
        conn.execute("UPDATE users SET password_hash=? WHERE user_id=?",
                     (new_hash, uid))
    conn.commit()
    print(f"Đã reset {len(new_hashes)} password.")
    print("\n*** QUAN TRỌNG: lưu lại danh sách temp password ở trên, "
          "báo cho từng user đổi mật khẩu ngay sau khi đăng nhập lại. ***")


if __name__ == "__main__":
    main()
