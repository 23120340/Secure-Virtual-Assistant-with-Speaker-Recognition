"""Sinh `docs/results/threshold_calibration.md` từ JSON output của benchmark.py.

Mục đích: thay vì tính tay từ raw JSON, script này tự:
  1. Tóm tắt config + dataset (n_users, n_audio per backend).
  2. Báo cáo SID accuracy + confusion matrix.
  3. Báo cáo SV @ threshold mặc định (TPR/FPR/EER nếu sweep).
  4. Sweep threshold từ 0.05 → 0.95 step 0.025, tìm EER và threshold-tại-EER.
  5. So sánh threshold mặc định trong config với threshold tại EER → recommendation.
  6. ASR WER table.
  7. Methodology caveats.

Cách dùng:
    python scripts/build_calibration_report.py docs/results/benchmark_ecapa.json
    # → docs/results/threshold_calibration.md (overwrite)

    # Multiple file (so sánh backend):
    python scripts/build_calibration_report.py \
        docs/results/benchmark_ecapa.json \
        docs/results/benchmark_wavlm.json \
        --out docs/results/threshold_calibration.md
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ==========================================================================
# Threshold sweep — tái cấu trúc từ raw per-sample scores
# ==========================================================================
def sweep_threshold(samples: list[dict], all_user_ids: list[str],
                    lo: float = 0.05, hi: float = 0.95, step: float = 0.025):
    """Trả về list (threshold, tpr, fpr, |fpr-fnr|) cho từng giá trị threshold.

    Mỗi sample có top_score + target_score; nhưng để có TPR/FPR đầy đủ ta cần
    re-compute scores cho TẤT CẢ pairs (audio_i × claimed_user_j). JSON
    benchmark.v1 chỉ lưu top_score + target_score per audio → đủ để biết
    positive trial (target_score), nhưng negative trial chỉ có top_score
    (impostor có score cao nhất).

    Vì vậy script này dùng:
      - Positive trials: target_score (genuine, n=n_audio)
      - Negative trials: top_score khi top != true_user (impostor mạnh nhất, n≤n_audio)

    Đây là "max impostor" convention — strict hơn random pair, phản ánh đúng
    threat model (attacker chọn user gần giọng họ nhất).
    """
    pos_scores = []  # target_score (genuine)
    neg_scores = []  # top impostor score khi đoán sai

    for s in samples:
        pos_scores.append(s["target_score"])
        # Impostor case: top != true_user → top_score là score impostor mạnh nhất
        if s["top_user_id"] != s["true_user_id"]:
            neg_scores.append(s["top_score"])

    if not pos_scores:
        return [], None

    out = []
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    t = lo
    while t <= hi + 1e-9:
        tp = sum(1 for x in pos_scores if x >= t)
        fn = n_pos - tp
        fp = sum(1 for x in neg_scores if x >= t)
        tn = n_neg - fp

        tpr = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        fnr = fn / n_pos if n_pos else 0.0
        out.append({
            "threshold": round(t, 4),
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "eer_gap": round(abs(fpr - fnr), 4),
        })
        t += step

    # EER point: argmin |fpr - fnr|
    eer_row = min(out, key=lambda r: r["eer_gap"])
    eer = (eer_row["fpr"] + eer_row["fnr"]) / 2
    return out, {
        "threshold": eer_row["threshold"],
        "eer": round(eer, 4),
        "n_pos_trials": n_pos,
        "n_neg_trials": n_neg,
    }


# ==========================================================================
# Markdown builders
# ==========================================================================
def build_header(reports: list[dict]) -> str:
    first = reports[0]
    backends_spk = sorted({b for r in reports for b in r["speaker"]})
    backends_asr = sorted({b for r in reports for b in r["asr"]})
    return (
        "# Threshold Calibration Report\n\n"
        f"- **Sinh tự động bởi**: `scripts/build_calibration_report.py`\n"
        f"- **Nguồn dữ liệu**: audio enroll thật trong `data/enroll_audio/`\n"
        f"- **Generated at (nguồn JSON đầu tiên)**: `{first.get('generated_at', '?')}`\n"
        f"- **Speaker backends báo cáo**: {', '.join(backends_spk) if backends_spk else '(không có)'}\n"
        f"- **ASR backends báo cáo**: {', '.join(backends_asr) if backends_asr else '(không có)'}\n\n"
    )


def build_speaker_section(reports: list[dict]) -> str:
    parts = ["## 1. Speaker Verification — Calibration\n\n"]

    # Bảng tổng hợp ngắn
    parts.append("### 1.1 Tổng hợp ở threshold mặc định (trong `core/config.py`)\n\n")
    parts.append("| Backend | n_users | n_audio | SID acc | SV thr | TPR | FPR | n_pos | n_neg |\n")
    parts.append("|---------|--------:|--------:|--------:|-------:|----:|----:|------:|------:|\n")
    for r in reports:
        for backend, m in r["speaker"].items():
            parts.append(
                f"| `{backend}` | {m['n_users']} | {m['n_audio']} | "
                f"{m['sid_acc']:.3f} | {m['sv_threshold']:.3f} | "
                f"{m['tpr']:.3f} | {m['fpr']:.3f} | {m['n_pos']} | {m['n_neg']} |\n"
            )
    parts.append("\n")

    # Threshold sweep từng backend
    parts.append("### 1.2 Threshold sweep (tìm EER + recommendation)\n\n")
    for r in reports:
        for backend, m in r["speaker"].items():
            samples = m.get("samples", [])
            if not samples:
                continue
            all_uids = sorted({s["true_user_id"] for s in samples}
                              | {s["top_user_id"] for s in samples})
            sweep, eer_info = sweep_threshold(samples, all_uids)
            cfg_thr = m["sv_threshold"]

            parts.append(f"#### Backend `{backend}`\n\n")
            if eer_info is None:
                parts.append("_Không có sample → bỏ qua._\n\n")
                continue

            parts.append(
                f"- **EER**: {eer_info['eer']*100:.2f}%\n"
                f"- **Threshold tại EER**: **{eer_info['threshold']}**\n"
                f"- **Threshold đang dùng (`config.py`)**: {cfg_thr}\n"
                f"- **n_pos_trials** (genuine): {eer_info['n_pos_trials']}\n"
                f"- **n_neg_trials** (max-impostor): {eer_info['n_neg_trials']}\n\n"
            )

            # Recommendation
            if abs(cfg_thr - eer_info["threshold"]) < 0.05:
                rec = (f"✅ **OK**: threshold mặc định ({cfg_thr}) gần threshold "
                       f"EER ({eer_info['threshold']}), không cần đổi.")
            elif cfg_thr < eer_info["threshold"]:
                rec = (f"⚠️ **Threshold đang LỎNG** so với EER: hiện {cfg_thr} < EER {eer_info['threshold']}. "
                       f"Cân nhắc nâng lên {eer_info['threshold']} để giảm false-accept "
                       "(an toàn hơn cho IMPORTANT task).")
            else:
                rec = (f"⚠️ **Threshold đang CHẶT** hơn EER: hiện {cfg_thr} > EER {eer_info['threshold']}. "
                       f"User thật có thể bị reject. Cân nhắc hạ xuống {eer_info['threshold']} "
                       "nếu false-reject rate cao trong demo.")
            parts.append(rec + "\n\n")

            # Sweep table (rút gọn từng step 0.05)
            parts.append("**Sweep table** (TPR/FPR theo threshold, step 0.05):\n\n")
            parts.append("| Threshold | TPR | FPR | FNR | |FPR-FNR| |\n")
            parts.append("|----------:|----:|----:|----:|---------:|\n")
            for row in sweep:
                # show every 2nd row (step 0.025 → 0.05)
                if int(round(row["threshold"] * 1000)) % 50 != 0:
                    continue
                marker = " ← EER" if row["threshold"] == eer_info["threshold"] else ""
                marker = " ← config" if abs(row["threshold"] - cfg_thr) < 0.013 else marker
                parts.append(
                    f"| {row['threshold']:.3f} | {row['tpr']:.3f} | "
                    f"{row['fpr']:.3f} | {row['fnr']:.3f} | "
                    f"{row['eer_gap']:.3f} |{marker}\n"
                )
            parts.append("\n")

    # SID confusion matrix
    parts.append("### 1.3 SID Confusion Matrix\n\n")
    for r in reports:
        for backend, m in r["speaker"].items():
            samples = m.get("samples", [])
            if not samples:
                continue
            users = sorted({s["true_user_id"] for s in samples})
            conf = defaultdict(Counter)
            for s in samples:
                conf[s["true_user_id"]][s["top_user_id"]] += 1

            parts.append(f"#### Backend `{backend}`\n\n")
            parts.append("| True \\ Pred |" + "|".join(f" `{u}` " for u in users) + "|\n")
            parts.append("|---|" + "|".join("---:" for _ in users) + "|\n")
            for tu in users:
                row = "| `" + tu + "` |"
                for pu in users:
                    row += f" {conf[tu].get(pu, 0)} |"
                parts.append(row + "\n")
            parts.append("\n")

    return "".join(parts)


def build_asr_section(reports: list[dict]) -> str:
    parts = ["## 2. ASR — Word Error Rate (WER)\n\n"]
    parts.append("Ground truth: 5 prompts từ enrollment (`ENROLL_PROMPTS` trong `scripts/benchmark.py`). "
                 "WER tính theo word-level Levenshtein, normalize lowercase + bỏ punctuation.\n\n")
    parts.append("| Backend | n_samples | WER mean | WER median | Latency mean (s) |\n")
    parts.append("|---------|----------:|---------:|-----------:|-----------------:|\n")
    for r in reports:
        for backend, m in r["asr"].items():
            parts.append(
                f"| `{backend}` | {m['n_samples']} | "
                f"{(m['wer_mean'] or 0):.3f} | "
                f"{(m['wer_median'] or 0):.3f} | "
                f"{(m['latency_mean_s'] or 0):.2f} |\n"
            )
    parts.append("\n")
    return "".join(parts)


def build_caveats(reports: list[dict]) -> str:
    n_users = max((m["n_users"] for r in reports for m in r["speaker"].values()), default=0)
    n_audio = max((m["n_audio"] for r in reports for m in r["speaker"].values()), default=0)
    return (
        "## 3. Methodology caveats (PHẢI ghi vào báo cáo)\n\n"
        f"- **Dataset rất nhỏ**: chỉ {n_users} user enrolled, {n_audio} audio total. "
        "EER/TPR/FPR ở đây là **indicative**, KHÔNG so sánh trực tiếp với benchmark "
        "VoxCeleb hay NIST SRE.\n"
        "- **Same-session bias**: cả 5 mẫu/user thu liên tiếp cùng 1 session, cùng mic, "
        "cùng noise floor — score genuine sẽ optimistic so với enrollment-vs-future-session.\n"
        "- **Impostor convention**: dùng \"max-impostor per audio\" (chọn user gần giọng "
        "nhất trong số non-target). Conservative cho FPR — phản ánh threat model attacker chọn "
        "user gần giọng họ nhất.\n"
        "- **Ground truth ASR cho user có `sample_0..4`** (thay vì `sample_1..5`): mapping prompt "
        "lệch 1 → WER user đó có thể cao bất thường, không phản ánh ASR quality.\n"
        "- **Threshold recommendation chỉ là gợi ý** từ EER trên dataset hiện tại; "
        "production cần audio cross-session + augmentation noise/reverb để robust.\n"
    )


def build_full_report(reports: list[dict]) -> str:
    sections = [
        build_header(reports),
        build_speaker_section(reports),
        build_asr_section(reports),
        build_caveats(reports),
    ]
    return "".join(sections)


# ==========================================================================
# Main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+",
                    help="Một hoặc nhiều JSON file output của scripts/benchmark.py")
    ap.add_argument("--out", default="docs/results/threshold_calibration.md",
                    help="Path Markdown output (default: docs/results/threshold_calibration.md)")
    args = ap.parse_args()

    reports = []
    for p in args.inputs:
        path = Path(p)
        if not path.exists():
            print(f"✗ Không tìm thấy: {p}", file=sys.stderr)
            sys.exit(1)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("schema_version") != "secva.benchmark.v1":
            print(f"⚠ {p} không phải schema secva.benchmark.v1 — bỏ qua")
            continue
        reports.append(data)

    if not reports:
        print("✗ Không có report hợp lệ", file=sys.stderr)
        sys.exit(1)

    md = build_full_report(reports)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"✓ Đã sinh: {out_path}")
    print(f"  ({len(reports)} report, {sum(len(r['speaker']) for r in reports)} speaker backend)")


if __name__ == "__main__":
    main()