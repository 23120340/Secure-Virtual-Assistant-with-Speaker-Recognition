"""Speaker encoder: wrap ECAPA-TDNN từ Tuần 1, hoặc fallback pretrained SpeechBrain.

Output: 192-dim L2-normalized embedding để cosine = dot product.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from . import config


class SpeakerEncoder:
    """Singleton encoder. Tự động chọn model: own checkpoint hoặc pretrained."""

    def __init__(self, ckpt_path: str = config.SPEAKER_CKPT):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_path = Path(ckpt_path)

        if ckpt_path.exists():
            print(f"  Loading own ECAPA checkpoint: {ckpt_path}")
            self._load_own(ckpt_path)
            self.mode = "own"
        else:
            print(f"  Checkpoint không tồn tại tại {ckpt_path}")
            print("  → Fallback sang pretrained SpeechBrain ECAPA (VoxCeleb2)")
            self._load_pretrained()
            self.mode = "pretrained"

    def _load_own(self, ckpt_path: Path):
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        self.model = ECAPA_TDNN(input_size=80, lin_neurons=192,
                                channels=[512, 512, 512, 512, 1536]).to(self.device).eval()
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])

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
        """Audio float32 mono 16kHz [N] → 192-d L2-normalized embedding."""
        if audio.size < config.SAMPLE_RATE // 2:  # < 0.5s thì pad
            pad = config.SAMPLE_RATE // 2 - audio.size
            audio = np.concatenate([audio, np.zeros(pad, dtype=np.float32)])

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

    def encode_centroid(self, audios: list) -> np.ndarray:
        """Trung bình nhiều embedding rồi L2-normalize lại → centroid của user."""
        embs = np.stack([self.encode(a) for a in audios])
        centroid = embs.mean(axis=0)
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
