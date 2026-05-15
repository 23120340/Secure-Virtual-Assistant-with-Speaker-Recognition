"""Speaker encoder: wrap ECAPA-TDNN từ Tuần 1, hoặc fallback pretrained SpeechBrain.

Output: 192-dim L2-normalized embedding để cosine = dot product.
"""
import logging
import threading
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from . import config

_log = logging.getLogger("secva.encoder")


class SpeakerEncoder:
    """Singleton encoder. Tự động chọn model: own checkpoint hoặc pretrained."""

    def __init__(self, ckpt_path: str = config.SPEAKER_CKPT):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # PyTorch model không thread-safe → lock quanh forward pass.
        # Cho phép Flask threaded=True mà không corrupt internal state.
        self._lock = threading.Lock()
        ckpt_path = Path(ckpt_path)

        if ckpt_path.exists():
            _log.info("Loading own ECAPA checkpoint: %s", ckpt_path)
            self._load_own(ckpt_path)
            self.mode = "own"
            self.sv_threshold       = config.SV_THRESHOLD_OWN
            self.sid_min_threshold  = config.SID_MIN_THRESHOLD_OWN
        else:
            _log.info("Checkpoint không tồn tại tại %s", ckpt_path)
            _log.info("→ Fallback sang pretrained SpeechBrain ECAPA (VoxCeleb2)")
            self._load_pretrained()
            self.mode = "pretrained"
            self.sv_threshold       = config.SV_THRESHOLD_PRETRAINED
            self.sid_min_threshold  = config.SID_MIN_THRESHOLD_PRETRAINED
        _log.info("→ SV threshold = %s, SID min = %s (%s mode)",
                  self.sv_threshold, self.sid_min_threshold, self.mode)

    def _load_own(self, ckpt_path: Path):
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        self.model = ECAPA_TDNN(input_size=80, lin_neurons=192,
                                channels=[512, 512, 512, 512, 1536]).to(self.device).eval()
        # weights_only=True chặn arbitrary pickle execution (CVE-class)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state)

        # Mel extractor giống lúc train
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE, n_fft=512, win_length=400,
            hop_length=160, n_mels=80,
        ).to(self.device)

    def _load_pretrained(self):
        # SpeechBrain có sẵn pretrained model trên VoxCeleb2
        from speechbrain.inference.speaker import EncoderClassifier
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(Path(config.DATA_DIR) / "pretrained_ecapa"),
            run_opts={"device": self.device},
        )

    @torch.no_grad()
    def encode(self, audio: np.ndarray) -> np.ndarray:
        """Audio float32 mono 16kHz [N] → 192-d L2-normalized embedding.

        Reject audio < 0.5s thay vì zero-pad: zero-pad không phải speech,
        embedding sẽ lệch hẳn và pollute centroid/identify. Caller phải
        VAD + length check trước khi gọi.
        """
        if audio.size < config.SAMPLE_RATE // 2:
            raise ValueError(
                f"Audio quá ngắn ({audio.size / config.SAMPLE_RATE:.2f}s, cần ≥ 0.5s). "
                "Caller phải VAD-trim và check length trước."
            )

        with self._lock:
            wav = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            if self.mode == "own":
                mel = self.mel(wav)
                mel = torch.log(mel + 1e-6)
                mel = mel - mel.mean(dim=-1, keepdim=True)
                emb = self.model(mel.transpose(1, 2)).squeeze(1)
            else:
                emb = self.encoder.encode_batch(wav).squeeze(1)
            emb = F.normalize(emb, dim=1)
            return emb.cpu().numpy()[0]

    def encode_centroid(self, audios: list,
                        min_pairwise: float = 0.6,
                        min_remaining: int = 3) -> np.ndarray:
        """Trung bình nhiều embedding rồi L2-normalize lại → centroid của user.

        Quality filter: drop sample có pairwise similarity trung bình < min_pairwise
        so với các sample khác — outlier (noise, sai user). Yêu cầu còn ≥ min_remaining
        sample sau khi lọc.
        """
        embs = np.stack([self.encode(a) for a in audios])
        n = len(embs)

        # Cosine pairwise (đã L2-normalized → dot product)
        sim_matrix = embs @ embs.T
        # Trung bình similarity của mỗi sample với các sample khác
        # (trừ chính nó: diagonal = 1.0, sum trừ 1 / (n-1))
        if n > 1:
            avg_sim = (sim_matrix.sum(axis=1) - 1.0) / (n - 1)
            keep_mask = avg_sim >= min_pairwise
            if keep_mask.sum() < min_remaining:
                raise ValueError(
                    f"Chất lượng audio không đồng nhất "
                    f"(chỉ {int(keep_mask.sum())}/{n} mẫu đạt ngưỡng {min_pairwise}). "
                    "Hãy thu lại trong môi trường yên tĩnh, nói rõ hơn."
                )
            kept = embs[keep_mask]
        else:
            kept = embs

        centroid = kept.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        return centroid


_encoder_instance = None


def get_encoder() -> SpeakerEncoder:
    global _encoder_instance
    if _encoder_instance is None:
        _encoder_instance = SpeakerEncoder()
    return _encoder_instance


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity giữa 2 vector đã L2-normalized → dot product."""
    return float(np.dot(a, b))
