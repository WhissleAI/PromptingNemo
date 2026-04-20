"""
Audio-Visual WER metric for noisy speech recognition.

Computes labelled WER (including noise tags), unlabelled WER (tags stripped),
and noise label classification accuracy. Noise tags follow the <N\d+> pattern
from the VANS dataset (Visual-Aware Noisy Speech).

Reference:
  "Visual-Aware Speech Recognition for Noisy Scenarios"
  Darur & Singla, EMNLP 2025
  https://aclanthology.org/2025.emnlp-main.845/
"""

import re
from typing import Dict, List, Optional, Tuple

import torch
from torchmetrics import Metric

NOISE_TAG_PATTERN = re.compile(r"<N\d+>")


def separate_labels_from_text(text: str) -> Tuple[str, Optional[str]]:
    """Strip noise label tags from transcript text.

    Returns (clean_text, noise_label) where noise_label is the first <N\\d+>
    tag found, or None if no tag is present.
    """
    match = NOISE_TAG_PATTERN.search(text)
    noise_label = match.group(0) if match else None
    clean_text = NOISE_TAG_PATTERN.sub("", text).strip()
    # Collapse multiple spaces left behind after tag removal
    clean_text = re.sub(r"\s+", " ", clean_text)
    return clean_text, noise_label


def _word_errors(hypothesis: str, reference: str) -> Tuple[int, int]:
    """Compute (edit_distance_in_words, num_reference_words)."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    r = len(ref_words)
    h = len(hyp_words)

    # Dynamic programming for word-level Levenshtein distance
    d = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1):
        d[i][0] = i
    for j in range(h + 1):
        d[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[r][h], r


class AVWordErrorRate(Metric):
    """Audio-Visual Word Error Rate metric.

    Tracks three quantities across batches:
      - Labelled WER: WER on full transcripts including <N\\d+> noise tags
      - Unlabelled WER: WER on transcripts with noise tags stripped
      - Noise label accuracy: fraction of samples where predicted noise tag
        matches the reference noise tag
    """

    full_state_update: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("labelled_errors", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("labelled_words", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("unlabelled_errors", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("unlabelled_words", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("label_correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("label_total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, hypotheses: List[str], references: List[str]) -> None:
        """Update metric state with a batch of decoded hypotheses and references.

        Args:
            hypotheses: List of predicted transcript strings (may contain <N\\d+> tags).
            references: List of ground-truth transcript strings (may contain <N\\d+> tags).
        """
        for hyp, ref in zip(hypotheses, references):
            # Labelled WER (full text including noise tags)
            errors, words = _word_errors(hyp, ref)
            self.labelled_errors += errors
            self.labelled_words += words

            # Unlabelled WER (noise tags stripped)
            clean_hyp, hyp_label = separate_labels_from_text(hyp)
            clean_ref, ref_label = separate_labels_from_text(ref)
            errors_u, words_u = _word_errors(clean_hyp, clean_ref)
            self.unlabelled_errors += errors_u
            self.unlabelled_words += words_u

            # Noise label accuracy (only counted when reference has a tag)
            if ref_label is not None:
                self.label_total += 1
                if hyp_label == ref_label:
                    self.label_correct += 1

    def compute(self) -> Dict[str, float]:
        """Compute aggregated metrics.

        Returns dict with keys: labelled_wer, unlabelled_wer, noise_label_accuracy
        """
        labelled_wer = (
            float(self.labelled_errors) / max(float(self.labelled_words), 1.0)
        )
        unlabelled_wer = (
            float(self.unlabelled_errors) / max(float(self.unlabelled_words), 1.0)
        )
        noise_acc = (
            float(self.label_correct) / max(float(self.label_total), 1.0)
            if self.label_total > 0
            else 0.0
        )
        return {
            "labelled_wer": labelled_wer,
            "unlabelled_wer": unlabelled_wer,
            "noise_label_accuracy": noise_acc,
        }
