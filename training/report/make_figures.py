"""Sinh các biểu đồ cho báo cáo training từ artifact trong training/results/.

Chạy bằng Python có matplotlib (vd C:\\Python314\\python.exe):
    python training/report/make_figures.py
Output PNG -> training/report/figs/
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGS = HERE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

VOX = RESULTS / "voxceleb_indian"
VIV = RESULTS / "vivos"

# Bảng màu nhất quán
C_TRAIN, C_VAL = "#2563eb", "#dc2626"
C_VOX, C_VIV = "#6366f1", "#10b981"


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def curves():
    """2x2: hàng trên = loss, hàng dưới = accuracy; cột = VoxCeleb / VIVOS."""
    vox_log, viv_log = load(VOX / "training_log.json"), load(VIV / "training_log.json")
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))

    for col, (log, title) in enumerate([(vox_log, "VoxCeleb Indian (24 spk)"),
                                        (viv_log, "VIVOS (65 spk)")]):
        ep = [r["epoch"] for r in log]
        # Loss
        ax[0, col].plot(ep, [r["train_loss"] for r in log], "-o", color=C_TRAIN,
                        ms=4, label="Train loss")
        ax[0, col].plot(ep, [r["val_loss"] for r in log], "-s", color=C_VAL,
                        ms=4, label="Val loss")
        ax[0, col].set_title(title, fontweight="bold")
        ax[0, col].set_xlabel("Epoch"); ax[0, col].set_ylabel("Loss")
        ax[0, col].grid(True, alpha=0.3); ax[0, col].legend()
        # Accuracy
        ax[1, col].plot(ep, [r["train_acc"] * 100 for r in log], "-o", color=C_TRAIN,
                        ms=4, label="Train acc")
        ax[1, col].plot(ep, [r["val_acc"] * 100 for r in log], "-s", color=C_VAL,
                        ms=4, label="Val acc")
        ax[1, col].set_xlabel("Epoch"); ax[1, col].set_ylabel("Accuracy (%)")
        ax[1, col].set_ylim(40, 101)
        ax[1, col].grid(True, alpha=0.3); ax[1, col].legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(FIGS / "curves.png", dpi=150)
    plt.close(fig)
    print("  curves.png")


def sid_bar():
    vox, viv = load(VOX / "sid_results.json"), load(VIV / "sid_results.json")
    labels = ["Top-1 accuracy", "Top-5 accuracy", "Micro-F1 (= Top-1)"]
    vox_v = [vox["top1_accuracy"], vox["top5_accuracy"], vox["top1_accuracy"]]
    viv_v = [viv["top1_accuracy"], viv["top5_accuracy"], viv["top1_accuracy"]]
    x = range(len(labels)); w = 0.36
    fig, axx = plt.subplots(figsize=(8, 4.5))
    b1 = axx.bar([i - w / 2 for i in x], vox_v, w, label="VoxCeleb Indian", color=C_VOX)
    b2 = axx.bar([i + w / 2 for i in x], viv_v, w, label="VIVOS", color=C_VIV)
    for b in list(b1) + list(b2):
        axx.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                 f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    axx.set_ylim(80, 102); axx.set_ylabel("(%)")
    axx.set_title("Speaker Identification — so sánh 2 model", fontweight="bold")
    axx.set_xticks(list(x)); axx.set_xticklabels(labels)
    axx.legend(); axx.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "sid_compare.png", dpi=150); plt.close(fig)
    print("  sid_compare.png")


def sv_bar():
    vox, viv = load(VOX / "sv_results.json"), load(VIV / "sv_results.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.2))
    # EER
    eer = [vox["eer_percent"], viv["eer_percent"]]
    b = a1.bar(["VoxCeleb\nIndian", "VIVOS"], eer, color=[C_VOX, C_VIV], width=0.5)
    for r in b:
        a1.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.05,
                f"{r.get_height():.2f}%", ha="center", va="bottom")
    a1.set_title("Equal Error Rate (thấp = tốt)", fontweight="bold")
    a1.set_ylabel("EER (%)"); a1.set_ylim(0, max(eer) * 1.3); a1.grid(True, axis="y", alpha=0.3)
    # minDCF
    dcf = [vox["min_dcf"], viv["min_dcf"]]
    b = a2.bar(["VoxCeleb\nIndian", "VIVOS"], dcf, color=[C_VOX, C_VIV], width=0.5)
    for r in b:
        a2.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.004,
                f"{r.get_height():.4f}", ha="center", va="bottom")
    a2.set_title("minDCF (thấp = tốt)", fontweight="bold")
    a2.set_ylabel("minDCF"); a2.set_ylim(0, max(dcf) * 1.3); a2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "sv_compare.png", dpi=150); plt.close(fig)
    print("  sv_compare.png")


def val_overlay():
    """Overlay val-accuracy của 2 dataset trên cùng trục → so sánh tốc độ hội tụ."""
    vox_log, viv_log = load(VOX / "training_log.json"), load(VIV / "training_log.json")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot([r["epoch"] for r in vox_log], [r["val_acc"] * 100 for r in vox_log],
            "-o", color=C_VOX, ms=4, label="VoxCeleb Indian (24 spk)")
    ax.plot([r["epoch"] for r in viv_log], [r["val_acc"] * 100 for r in viv_log],
            "-s", color=C_VIV, ms=4, label="VIVOS (65 spk)")
    ax.axhline(95, ls="--", color="gray", alpha=0.6, lw=1)
    ax.text(1.2, 95.4, "95%", color="gray", fontsize=8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Tốc độ hội tụ: VIVOS đạt >97% sau 5 epoch", fontweight="bold")
    ax.set_ylim(45, 101); ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIGS / "val_overlay.png", dpi=150); plt.close(fig)
    print("  val_overlay.png")


if __name__ == "__main__":
    print("Sinh figures:")
    curves()
    sid_bar()
    sv_bar()
    val_overlay()
    print(f"Xong -> {FIGS}")
