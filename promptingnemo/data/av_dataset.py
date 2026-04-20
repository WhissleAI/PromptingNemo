"""
Audio-Visual dataset with SNR-based noise mixing for noisy speech recognition.

Loads clean audio, mixes with noise audio extracted from video files at a
configurable SNR ratio, and provides pre-extracted CLIP ViT-L/14 visual features.

Reference:
  "Visual-Aware Speech Recognition for Noisy Scenarios"
  Darur & Singla, EMNLP 2025
  https://aclanthology.org/2025.emnlp-main.845/
"""

import io
import json
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer

log = logging.getLogger(__name__)


def _load_noise_audio_from_video(video_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Extract audio waveform from a video file using pydub."""
    from pydub import AudioSegment as PydubSegment

    audio_seg = PydubSegment.from_file(video_path)
    audio_seg = audio_seg.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)
    samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
    # Normalize to [-1, 1]
    samples = samples / (2**15)
    return samples


def _mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_ratio: float) -> np.ndarray:
    """Mix clean and noise signals at the given SNR ratio.

    snr_ratio is the fraction of clean signal in the mix:
      mixed = snr_ratio * clean + (1 - snr_ratio) * noise
    Values closer to 1.0 mean cleaner audio, closer to 0.0 mean noisier.
    """
    # Repeat or truncate noise to match clean length
    if len(noise) == 0:
        return clean
    if len(noise) < len(clean):
        repeats = (len(clean) // len(noise)) + 1
        noise = np.tile(noise, repeats)
    noise = noise[: len(clean)]

    mixed = snr_ratio * clean + (1.0 - snr_ratio) * noise
    return mixed


class AVToBPEDataset(Dataset):
    """Audio-Visual dataset for CTC training with noise mixing.

    Each manifest entry should have:
      - audio_filepath: path to clean audio file
      - video_filepath: path to video file (noise source; audio is extracted)
      - feature_file: path to pre-extracted CLIP visual features (.npy)
      - text: transcript text (may include <N\\d+> noise label)
      - duration: audio duration in seconds

    Args:
        manifest_filepath: Path to JSONL manifest file.
        tokenizer: NeMo BPE tokenizer instance.
        sample_rate: Audio sample rate (default 16000).
        override_snr_ratio: Fixed SNR ratio (float), "rand" for uniform [0.3, 0.6],
            or None to skip noise mixing.
        get_zero_vid_feats: If True, return zero tensors instead of real video features
            (for ablation studies).
        video_feat_dim: Dimension of CLIP feature vectors (default 768 for ViT-L/14).
        video_frame_rate: Frame rate of pre-extracted video features (default 1.0 fps).
        max_duration: Maximum audio duration in seconds (samples longer are skipped).
        min_duration: Minimum audio duration in seconds (samples shorter are skipped).
    """

    def __init__(
        self,
        manifest_filepath: str,
        tokenizer,
        sample_rate: int = 16000,
        override_snr_ratio: Optional[Union[float, str]] = None,
        get_zero_vid_feats: bool = False,
        video_feat_dim: int = 768,
        video_frame_rate: float = 1.0,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
    ):
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.override_snr_ratio = override_snr_ratio
        self.get_zero_vid_feats = get_zero_vid_feats
        self.video_feat_dim = video_feat_dim
        self.video_frame_rate = video_frame_rate

        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate)

        self.samples = []
        with open(manifest_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                duration = entry.get("duration", 0.0)
                if max_duration is not None and duration > max_duration:
                    continue
                if min_duration is not None and duration < min_duration:
                    continue
                self.samples.append(entry)

        log.info(
            "AVToBPEDataset: loaded %d samples from %s (snr=%s, zero_vid=%s)",
            len(self.samples),
            manifest_filepath,
            override_snr_ratio,
            get_zero_vid_feats,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _get_snr_ratio(self) -> Optional[float]:
        if self.override_snr_ratio is None:
            return None
        if isinstance(self.override_snr_ratio, str) and self.override_snr_ratio == "rand":
            return random.uniform(0.3, 0.6)
        return float(self.override_snr_ratio)

    def __getitem__(self, index: int):
        entry = self.samples[index]

        # Load clean audio
        audio_path = entry["audio_filepath"]
        audio = self.featurizer.process(audio_path)
        audio_len = len(audio)

        # Optionally mix with noise from video
        snr = self._get_snr_ratio()
        if snr is not None and "video_filepath" in entry:
            video_path = entry["video_filepath"]
            try:
                noise = _load_noise_audio_from_video(video_path, self.sample_rate)
                audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
                mixed = _mix_at_snr(audio_np, noise, snr)
                audio = torch.tensor(mixed, dtype=torch.float32)
            except Exception as e:
                log.warning("Failed to extract noise from %s: %s", video_path, e)

        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio, dtype=torch.float32)

        # Load video features
        if self.get_zero_vid_feats:
            # For ablation: use a single zero frame
            video_feats = torch.zeros(1, self.video_feat_dim, dtype=torch.float32)
        else:
            feature_file = entry.get("feature_file")
            if feature_file and Path(feature_file).exists():
                feats = np.load(feature_file)
                video_feats = torch.tensor(feats, dtype=torch.float32)
                if video_feats.ndim == 1:
                    video_feats = video_feats.unsqueeze(0)
            else:
                video_feats = torch.zeros(1, self.video_feat_dim, dtype=torch.float32)

        # Tokenize transcript
        text = entry.get("text", "")
        tokens = self.tokenizer.text_to_ids(text)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens_len = len(tokens)

        return audio, torch.tensor(audio_len, dtype=torch.long), video_feats, tokens, torch.tensor(tokens_len, dtype=torch.long)


def av_collate_fn(batch, pad_id: int = 0):
    """Collate function for AVToBPEDataset.

    Pads audio waveforms, video feature sequences, and token sequences
    to their respective max lengths within the batch.

    Returns:
        (audio, audio_len, video_feats, tokens, tokens_len)
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        empty_f = torch.empty(0, 0, dtype=torch.float32)
        empty_l = torch.tensor([], dtype=torch.long)
        return empty_f, empty_l, empty_f, empty_l.clone(), empty_l.clone()

    audios, audio_lens, video_feats_list, tokens_list, tokens_lens = zip(*batch)

    # Pad audio
    max_audio_len = max(a.shape[0] for a in audios)
    padded_audios = []
    for a in audios:
        padded = torch.zeros(max_audio_len, dtype=a.dtype)
        padded[: a.shape[0]] = a
        padded_audios.append(padded)
    audio_batch = torch.stack(padded_audios)
    audio_len_batch = torch.stack(list(audio_lens))

    # Pad video features: (B, max_frames, feat_dim)
    max_frames = max(vf.shape[0] for vf in video_feats_list)
    feat_dim = video_feats_list[0].shape[-1]
    padded_video = []
    for vf in video_feats_list:
        padded = torch.zeros(max_frames, feat_dim, dtype=vf.dtype)
        padded[: vf.shape[0]] = vf
        padded_video.append(padded)
    video_batch = torch.stack(padded_video)

    # Pad tokens
    tokens_len_batch = torch.stack(list(tokens_lens))
    tokens_batch = nn.utils.rnn.pad_sequence(list(tokens_list), batch_first=True, padding_value=pad_id)

    return audio_batch, audio_len_batch, video_batch, tokens_batch, tokens_len_batch
