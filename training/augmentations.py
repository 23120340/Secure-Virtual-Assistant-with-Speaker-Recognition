"""Training-time augmentation utilities for ECAPA-TDNN.

The transforms here are optional and run only during training. They do not
affect runtime speaker verification in the web/CLI assistant.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def _find_audio_files(root: str | Path | None) -> list[Path]:
    if not root:
        return []
    root = Path(root)
    if not root.exists():
        return []
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def _load_mono(path: Path, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.mean(dim=0)


def _fit_length(wav: torch.Tensor, num_samples: int) -> torch.Tensor:
    if wav.numel() >= num_samples:
        start = random.randint(0, wav.numel() - num_samples)
        return wav[start:start + num_samples]
    return F.pad(wav, (0, num_samples - wav.numel()))


class WaveformAugmenter:
    """Apply speed perturbation, MUSAN noise, and RIR reverb to a waveform."""

    def __init__(
        self,
        *,
        sample_rate: int,
        num_samples: int,
        musan_root: str | Path | None = None,
        rir_root: str | Path | None = None,
        speed_rates: Iterable[float] = (0.9, 1.0, 1.1),
        speed_prob: float = 1.0,
        noise_prob: float = 0.5,
        rir_prob: float = 0.3,
        snr_db: tuple[float, float] = (5.0, 20.0),
    ) -> None:
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.musan_files = _find_audio_files(musan_root)
        self.rir_files = _find_audio_files(rir_root)
        self.speed_rates = tuple(speed_rates)
        self.speed_prob = speed_prob
        self.noise_prob = noise_prob
        self.rir_prob = rir_prob
        self.snr_db = snr_db

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if self.speed_rates and random.random() < self.speed_prob:
            wav = self._speed_perturb(wav)
        if self.rir_files and random.random() < self.rir_prob:
            wav = self._apply_rir(wav)
        if self.musan_files and random.random() < self.noise_prob:
            wav = self._mix_noise(wav)
        return _fit_length(wav, self.num_samples).clamp(-1.0, 1.0)

    def _speed_perturb(self, wav: torch.Tensor) -> torch.Tensor:
        rate = random.choice(self.speed_rates)
        if abs(rate - 1.0) < 1e-6:
            return wav
        virtual_sr = max(1, int(self.sample_rate * rate))
        return torchaudio.functional.resample(wav, virtual_sr, self.sample_rate)

    def _mix_noise(self, wav: torch.Tensor) -> torch.Tensor:
        noise = _load_mono(random.choice(self.musan_files), self.sample_rate)
        noise = _fit_length(noise, wav.numel()).to(wav.dtype)

        clean_power = wav.pow(2).mean().clamp_min(1e-8)
        noise_power = noise.pow(2).mean().clamp_min(1e-8)
        snr = random.uniform(*self.snr_db)
        scale = torch.sqrt(clean_power / (noise_power * (10 ** (snr / 10))))
        return wav + scale * noise

    def _apply_rir(self, wav: torch.Tensor) -> torch.Tensor:
        rir = _load_mono(random.choice(self.rir_files), self.sample_rate).to(wav.dtype)
        rir = rir / rir.abs().max().clamp_min(1e-8)
        reverbed = F.conv1d(
            wav.view(1, 1, -1),
            rir.flip(0).view(1, 1, -1),
            padding=rir.numel() - 1,
        ).view(-1)
        return reverbed[: wav.numel()]


class SpecAugment(nn.Module):
    """Time/frequency masking on log-mel features shaped [B, T, F]."""

    def __init__(
        self,
        *,
        freq_masks: int = 2,
        time_masks: int = 2,
        max_freq_width: int = 8,
        max_time_width: int = 40,
    ) -> None:
        super().__init__()
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.max_freq_width = max_freq_width
        self.max_time_width = max_time_width

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mel
        out = mel.clone()
        batch, frames, bins = out.shape

        for b in range(batch):
            for _ in range(self.freq_masks):
                width = random.randint(0, min(self.max_freq_width, bins))
                start = random.randint(0, bins - width) if width else 0
                out[b, :, start:start + width] = 0.0
            for _ in range(self.time_masks):
                width = random.randint(0, min(self.max_time_width, frames))
                start = random.randint(0, frames - width) if width else 0
                out[b, start:start + width, :] = 0.0
        return out
