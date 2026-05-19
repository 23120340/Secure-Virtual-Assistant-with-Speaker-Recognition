# Training Results (placeholder)

Folder này chứa output thực nghiệm của `train_ecapa.py`, `evaluate_sid.py`, `evaluate_sv.py`.

Xem `docs/results/README.md` cho checklist đầy đủ.

## Bắt buộc commit trước khi nộp

- `training_log.json` (auto-generate bởi `train_ecapa.py --save_dir <this_folder>`)
- `spk2idx.json` (auto-generate cùng `train_ecapa.py`)
- `sid_results.json` (`evaluate_sid.py --out training/results/sid_results.json`)
- `sv_results.json` (`evaluate_sv.py --out training/results/sv_results.json`)

## Khuyến nghị bổ sung

- `roc_curve.png`, `confusion_matrix.png`, `loss_curves.png` — plot bằng matplotlib từ JSON ở trên.
- `hparams.json` — argparse args + commit SHA + torch version (cần thêm 1 dòng vào `train_ecapa.py`).
