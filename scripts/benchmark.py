"""Benchmark end-to-end ASR + speaker backends trên audio enroll đã có.

Tính:
  - WER (Word Error Rate) per ASR backend, dùng ENROLL_PROMPTS làm ground truth.
    Audio không match prompt (vd: user đọc lệch) sẽ ra WER cao cho CẢ 2 backend
    → so sánh tương đối vẫn fair.
  - SID accuracy: %% audio mà target user có top-1 score.
  - SV at config threshold: TPR (true positive rate, đúng nhận target) +
    FPR (false positive rate, sai nhận impostor).

Cách dùng:
    # Test 1 combo (mặc định: faster-whisper + ecapa, theo .env)
    python scripts/benchmark.py

    # Test combo cụ thể:
    ASR_BACKEND=phowhisper SPEAKER_BACKEND=wavlm python scripts/benchmark.py

    # Test tất cả 4 combos (chạy 1 process — load nhiều model nên chậm):
    python scripts/benchmark.py --all
"""
import sys
import argparse
import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import audio_io, config


# Ground truth prompt thứ tự theo cli/enroll_user.py:ENROLL_PROMPTS
ENROLL_PROMPTS = [
    "Trợ lý ơi, hôm nay thời tiết thế nào?",
    "Phát một bản nhạc tôi yêu thích đi.",
    "Cho tôi xem lịch làm việc hôm nay.",
    "Đọc cho tôi ghi chú cuối tuần.",
    "Một hai ba bốn năm sáu bảy tám chín mười.",
]


# ==========================================================================
# WER: word-level Levenshtein / |ref|
# ==========================================================================
def _normalize_vi(text: str) -> str:
    """Lowercase, remove punctuation (giữ dấu thanh tiếng Việt)."""
    text = text.lower()
    # Bỏ punctuation Unicode (category P*), giữ chữ + space
    text = "".join(c for c in text
                   if not unicodedata.category(c).startswith("P"))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _edit_distance(a: list, b: list) -> int:
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j-1], dp[j])
            prev = cur
    return dp[m]


def wer(ref: str, hyp: str) -> float:
    r = _normalize_vi(ref).split()
    h = _normalize_vi(hyp).split()
    if not r:
        return 1.0 if h else 0.0
    return _edit_distance(r, h) / len(r)


# ==========================================================================
# Build test set: liệt kê audio + ground truth
# ==========================================================================
def collect_test_set():
    """Yield (user_id, audio_idx, wav_path, expected_text)."""
    for user_dir in sorted(config.ENROLL_AUDIO_DIR.iterdir()):
        if not user_dir.is_dir():
            continue
        for wav in sorted(user_dir.glob("sample_*.wav")):
            # sample_N.wav → index N → prompt (N-1) % 5
            m = re.search(r"sample_(\d+)", wav.stem)
            if not m:
                continue
            idx = int(m.group(1))
            expected = ENROLL_PROMPTS[(idx - 1) % len(ENROLL_PROMPTS)]
            yield user_dir.name, idx, wav, expected


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.ROOT.resolve()))
    except ValueError:
        return str(path)


def asr_rows_to_samples(rows):
    return [
        {
            "user_id": user,
            "sample_idx": idx,
            "expected_text": exp,
            "hypothesis": hyp,
            "wer": round(float(w), 4),
            "latency_s": round(float(dt), 4),
        }
        for user, idx, exp, hyp, w, dt in rows
    ]


# ==========================================================================
# ASR benchmark
# ==========================================================================
def bench_asr(asr_backend: str):
    """Return list of (user, idx, expected, hypothesis, wer_value, dt)."""
    # Reset singleton để swap backend
    import core.asr as asr_mod
    asr_mod._asr_instance = None
    config.ASR_BACKEND = asr_backend
    from core.asr import get_asr

    print(f"\n--- ASR: {asr_backend} ---")
    asr = get_asr()
    rows = []
    for user, idx, wav, expected in collect_test_set():
        audio = audio_io.load_wav(wav)
        audio = audio_io.SileroVAD.trim(audio)
        if audio.size < config.SAMPLE_RATE:
            continue
        t0 = time.time()
        hyp = asr.transcribe(audio)
        dt = time.time() - t0
        w = wer(expected, hyp)
        rows.append((user, idx, expected, hyp, w, dt))
    return rows


# ==========================================================================
# Speaker benchmark (SID + SV @ config threshold)
# ==========================================================================
def bench_speaker(speaker_backend: str):
    """Return metrics dict for SID + SV using configured threshold."""
    # Reset singleton
    import core.speaker_encoder as se_mod
    se_mod._encoder_instance = None
    config.SPEAKER_BACKEND = speaker_backend
    from core.database import UserDB, SpeakerManager
    import core.speaker_encoder as se

    print(f"\n--- Speaker: {speaker_backend} ---")
    mgr = SpeakerManager(UserDB())
    mgr._refresh_cache()
    if not mgr._cache:
        print(f"  ⚠ Không có embedding cho backend '{speaker_backend}' "
              "— bỏ qua. Chạy `python scripts/reenroll_backend.py` trước.")
        return None

    sv_thr = mgr.encoder.sv_threshold
    sid_correct, sid_total = 0, 0
    tp = fn = fp = tn = 0
    sample_dists = []  # per-sample SID/SV rows
    n_users = len(mgr._cache)

    for user, idx, wav, expected in collect_test_set():
        if user not in mgr._cache:
            continue
        audio = audio_io.load_wav(wav)
        audio = audio_io.SileroVAD.trim(audio)
        if audio.size < config.SAMPLE_RATE:
            continue

        emb = mgr._prepare(audio)
        scores = {uid: se.cosine(emb, ref)
                  for uid, (_, ref) in mgr._cache.items()}
        top_uid = max(scores, key=scores.get)
        sid_total += 1
        if top_uid == user:
            sid_correct += 1
        sample_dists.append({
            "true_user_id": user,
            "sample_idx": idx,
            "wav_path": _rel(wav),
            "top_user_id": top_uid,
            "top_score": round(float(scores[top_uid]), 6),
            "target_score": round(float(scores[user]), 6),
            "sid_correct": top_uid == user,
            "sv_threshold": round(float(sv_thr), 6),
            "sv_target_passed": scores[user] >= sv_thr,
        })

        for claimed_uid, sc in scores.items():
            is_target = (claimed_uid == user)
            pred_match = (sc >= sv_thr)
            if is_target and pred_match: tp += 1
            elif is_target and not pred_match: fn += 1
            elif not is_target and pred_match: fp += 1
            else: tn += 1

    return {
        "backend":   speaker_backend,
        "n_users":   n_users,
        "n_audio":   sid_total,
        "sid_acc":   sid_correct / sid_total if sid_total else 0,
        "sv_thr":    sv_thr,
        "tpr":       tp / (tp + fn) if (tp + fn) else 0,
        "fpr":       fp / (fp + tn) if (fp + tn) else 0,
        "n_pos":     tp + fn,
        "n_neg":     fp + tn,
        "samples":   sample_dists,
    }


# ==========================================================================
# Main: print pretty report
# ==========================================================================
def print_asr(label, rows):
    if not rows:
        print(f"  no rows"); return
    wers = [r[4] for r in rows]
    dts  = [r[5] for r in rows]
    perfect = sum(1 for w in wers if w == 0)
    print(f"\n[{label}] {len(rows)} samples")
    print(f"  WER  avg={np.mean(wers):.3f}  median={np.median(wers):.3f}  "
          f"min={min(wers):.3f}  max={max(wers):.3f}")
    print(f"  Perfect (WER=0): {perfect}/{len(rows)}")
    print(f"  Latency: avg={np.mean(dts):.2f}s  min={min(dts):.2f}s  max={max(dts):.2f}s")
    # Per-sample
    print(f"\n  Per-sample (user/idx, WER, hyp[:60]):")
    for user, idx, exp, hyp, w, _ in rows:
        marker = "✓" if w == 0 else "✗"
        print(f"    {marker} {user[:8]:8s}/{idx} W={w:.2f}  exp={exp[:30]:30s} | hyp={hyp[:35]}")


def print_speaker(label, m):
    if m is None:
        print(f"  no data"); return
    print(f"\n[{label}] users={m['n_users']}  audio={m['n_audio']}")
    print(f"  SID accuracy: {m['sid_acc']:.3f}  ({int(m['sid_acc']*m['n_audio'])}/{m['n_audio']} correct)")
    print(f"  SV @ thr={m['sv_thr']}: TPR={m['tpr']:.3f}  FPR={m['fpr']:.3f}  "
          f"(n_pos={m['n_pos']}, n_neg={m['n_neg']})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true",
                   help="Chạy tất cả 4 combinations (2 ASR × 2 Speaker)")
    p.add_argument("--asr-only", action="store_true",
                   help="Chỉ benchmark ASR")
    p.add_argument("--speaker-only", action="store_true",
                   help="Chỉ benchmark Speaker")
    p.add_argument("--out", default="",
                   help="Path JSON để dump kết quả (ví dụ: docs/results/benchmark.json)")
    args = p.parse_args()

    if args.all:
        asrs = ["faster-whisper", "phowhisper"]
        spks = ["ecapa", "wavlm"]
    else:
        asrs = [config.ASR_BACKEND]
        spks = [config.SPEAKER_BACKEND]

    print("=" * 70)
    print(f"Benchmark — ASR: {asrs}, Speaker: {spks}")
    print("=" * 70)

    summary: dict = {
        "schema_version": "secva.benchmark.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "asr_backends": asrs,
            "speaker_backends": spks,
            "sample_rate": config.SAMPLE_RATE,
            "enroll_audio_dir": _rel(config.ENROLL_AUDIO_DIR),
        },
        "asr": {},
        "speaker": {},
    }

    if not args.speaker_only:
        for asr in asrs:
            t0 = time.time()
            rows = bench_asr(asr)
            print_asr(asr, rows)
            dt = time.time() - t0
            print(f"  Total time: {dt:.1f}s")
            wers = [r[4] for r in rows]
            dts  = [r[5] for r in rows]
            summary["asr"][asr] = {
                "n_samples":  len(rows),
                "wer_mean":   float(np.mean(wers)) if wers else None,
                "wer_median": float(np.median(wers)) if wers else None,
                "latency_mean_s": float(np.mean(dts)) if dts else None,
                "total_time_s": round(dt, 2),
                "samples": asr_rows_to_samples(rows),
            }

    if not args.asr_only:
        for spk in spks:
            t0 = time.time()
            m = bench_speaker(spk)
            print_speaker(spk, m)
            print(f"  Total time: {time.time()-t0:.1f}s")
            if m:
                summary["speaker"][spk] = {
                    "n_users":  m["n_users"],
                    "n_audio":  m["n_audio"],
                    "sid_acc":  round(m["sid_acc"], 4),
                    "sv_threshold": m["sv_thr"],
                    "tpr": round(m["tpr"], 4),
                    "fpr": round(m["fpr"], 4),
                    "n_pos": m["n_pos"],
                    "n_neg": m["n_neg"],
                    "samples": m["samples"],
                }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nĐã lưu JSON tổng hợp: {out_path}")


if __name__ == "__main__":
    main()
