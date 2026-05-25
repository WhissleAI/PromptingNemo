"""Dual-head trailing tag classifier for ASR training.

Removes trailing sentence-level tags (AGE, GENDER, EMOTION, INTENT, DIALECT)
from CTC targets and predicts them via a separate cross-entropy classification
head on the *pooled* encoder output — one prediction per utterance, not per
frame — so that tag-classifier gradients cannot interfere with the temporal
structure the CTC head relies on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Set, Tuple

TRAILING_TAG_PREFIXES = ('AGE_', 'GENDER_', 'EMOTION_', 'INTENT_', 'DIALECT_')


def build_all_special_token_ids(vocabulary, prefixes):
    """Build a set of ALL special token IDs from vocabulary given prefix list.

    Args:
        vocabulary: list of vocab tokens
        prefixes: list of prefix strings (e.g. ['ENTITY_', 'INTENT_', 'AGE_'])

    Returns:
        set of vocab indices for all tokens matching any prefix
    """
    prefixes = tuple(p.upper() for p in prefixes)
    special_ids = set()
    for idx, token in enumerate(vocabulary):
        clean = token.lstrip('▁')
        for prefix in prefixes:
            if clean.startswith(prefix):
                special_ids.add(idx)
                break
        if clean == 'END':
            special_ids.add(idx)
    return special_ids


def build_sp_boundary_ids(vocabulary):
    """Find sentencepiece boundary token IDs ('▁' alone) in the vocabulary."""
    return {idx for idx, token in enumerate(vocabulary) if token == '▁'}


def strip_all_special_from_targets(transcript, transcript_len, all_special_ids, sp_boundary_ids=None):
    """Remove ALL special tokens (inline + trailing) from CTC targets for WER.

    Unlike strip_trailing_tags_and_get_labels which only walks backward from the
    end, this removes special tokens anywhere in the sequence and compacts what
    remains. Use this for WER computation so inline ENTITY_/END tokens don't
    inflate the error count.

    Keeps all sentencepiece boundary tokens ('▁') intact — NeMo's WER metric
    normalizes whitespace, so extra boundaries are harmless, but removing them
    concatenates adjacent words and inflates WER.

    Args:
        transcript: [B, max_len] token IDs
        transcript_len: [B] valid lengths
        all_special_ids: set of vocab IDs to remove
        sp_boundary_ids: unused, kept for call-site compatibility

    Returns:
        clean_transcript: [B, max_len] compacted
        clean_transcript_len: [B] new lengths
    """
    batch_size, max_len = transcript.shape
    clean = torch.zeros_like(transcript)
    clean_len = torch.zeros_like(transcript_len)

    for i in range(batch_size):
        seq_len = int(transcript_len[i].item())
        tokens = transcript[i, :seq_len].tolist()

        kept = [tid for tid in tokens if tid not in all_special_ids]

        clean_len[i] = len(kept)
        for j, tid in enumerate(kept):
            clean[i, j] = tid

    return clean, clean_len


def build_trailing_tag_maps(vocabulary, categories=None):
    """Build mappings from vocabulary for trailing tag classification.

    Args:
        vocabulary: list of vocab tokens
        categories: optional list of category names (e.g. ['AGE', 'GENDER']).
                    If None, auto-detect from vocabulary.

    Returns:
        trailing_tag_ids: set of vocab indices that are trailing tags
        category_to_id: dict of category_name -> {vocab_id: class_index}
        category_sizes: dict of category_name -> num_classes (including NONE at 0)
    """
    if categories:
        active_prefixes = tuple(f"{cat}_" for cat in categories)
    else:
        active_prefixes = TRAILING_TAG_PREFIXES

    category_members = defaultdict(list)
    for idx, token in enumerate(vocabulary):
        clean = token.lstrip('▁')
        for prefix in active_prefixes:
            if clean.startswith(prefix):
                cat_name = prefix.rstrip('_')
                category_members[cat_name].append((idx, clean))
                break

    trailing_tag_ids = set()
    category_to_id = {}
    category_sizes = {}

    for cat_name, members in sorted(category_members.items()):
        cat_map = {}
        for class_idx, (vocab_id, _) in enumerate(sorted(members, key=lambda x: x[1])):
            cat_map[vocab_id] = class_idx + 1  # 0 = NONE
            trailing_tag_ids.add(vocab_id)
        category_to_id[cat_name] = cat_map
        category_sizes[cat_name] = len(members) + 1
    return trailing_tag_ids, category_to_id, category_sizes


class TrailingTagClassifier(nn.Module):
    """Multi-head classifier for sentence-level tags with attention pooling.

    Uses self-attention to pool encoder frames (learns which frames matter),
    then a shared 2-layer MLP before per-category classification heads.
    """

    def __init__(self, encoder_dim, category_sizes, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.category_names = sorted(category_sizes.keys())

        self.attention = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.feature_extractor = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleDict({
            cat: nn.Linear(hidden_dim, n_classes)
            for cat, n_classes in sorted(category_sizes.items())
        })

    def forward(self, encoder_output, encoded_len):
        """
        Args:
            encoder_output: [B, T, D] encoder output (time-major)
            encoded_len: [B] valid frame counts
        Returns:
            dict of category_name -> [B, num_classes] logits
        """
        attn_scores = self.attention(encoder_output).squeeze(-1)
        mask = torch.arange(encoder_output.size(1), device=encoder_output.device).unsqueeze(0) < encoded_len.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
        pooled = (encoder_output * attn_weights).sum(dim=1)

        features = self.feature_extractor(pooled)
        return {cat: head(features) for cat, head in self.heads.items()}


def strip_trailing_tags_and_get_labels(transcript, transcript_len, trailing_tag_ids,
                                       category_to_id, category_names,
                                       all_special_ids=None,
                                       sp_boundary_ids=None):
    """Strip trailing tag tokens from CTC targets and extract classification labels.

    Uses all_special_ids (if provided) to walk past ALL trailing special tokens
    (INTENT_, ENTITY_, etc.), not just active category tokens. Labels are only
    extracted for tokens in trailing_tag_ids.

    sp_boundary_ids: set of sentencepiece boundary token IDs (e.g. '▁') that
    appear between special tokens in aggregate tokenizers and should also be
    skipped during backward walk.

    Returns:
        clean_transcript: [B, max_len] with trailing tags removed
        clean_transcript_len: [B] new lengths
        tag_labels: [B, num_categories] classification labels (0=NONE)
    """
    skip_ids = all_special_ids if all_special_ids is not None else trailing_tag_ids
    if sp_boundary_ids:
        skip_ids = skip_ids | sp_boundary_ids

    batch_size = transcript.size(0)
    num_cats = len(category_names)
    clean_transcript = transcript.clone()
    clean_transcript_len = transcript_len.clone()
    tag_labels = torch.zeros(batch_size, num_cats, dtype=torch.long, device=transcript.device)

    for i in range(batch_size):
        seq_len = int(transcript_len[i].item())
        end_pos = seq_len
        while end_pos > 0:
            token_id = int(transcript[i, end_pos - 1].item())
            if token_id in skip_ids:
                if token_id in trailing_tag_ids:
                    for cat_idx, cat_name in enumerate(category_names):
                        if token_id in category_to_id.get(cat_name, {}):
                            tag_labels[i, cat_idx] = category_to_id[cat_name][token_id]
                            break
                end_pos -= 1
            else:
                break
        clean_transcript_len[i] = end_pos
        if end_pos < seq_len:
            clean_transcript[i, end_pos:seq_len] = 0

    return clean_transcript, clean_transcript_len, tag_labels


def masked_mean_pool(encoder_output, encoded_len, input_format='BDT'):
    """Mean-pool encoder output over valid (non-padded) time steps.

    Args:
        encoder_output: encoder output tensor
        encoded_len: [B] valid lengths
        input_format: 'BDT' for NeMo convention [B, D, T], 'BTD' for [B, T, D]

    Returns:
        [B, D] pooled representation
    """
    if input_format == 'BDT':
        encoder_output = encoder_output.transpose(1, 2)

    B, T, D = encoder_output.shape
    time_mask = torch.arange(T, device=encoder_output.device).unsqueeze(0) < encoded_len.unsqueeze(1)
    time_mask = time_mask.unsqueeze(-1).float()
    pooled = (encoder_output * time_mask).sum(dim=1) / time_mask.sum(dim=1).clamp(min=1)
    return pooled


def compute_tag_classification_loss(tag_logits, tag_labels, class_weights=None):
    """Compute cross-entropy loss for trailing tags from pooled predictions.

    Args:
        tag_logits: dict of category_name -> [B, num_classes] logits
        tag_labels: [B, num_categories] ground truth labels
        class_weights: optional dict of category_name -> [num_classes] weight tensor

    Returns:
        scalar loss averaged across categories
    """
    if not tag_logits:
        return torch.tensor(0.0, requires_grad=True)

    total_loss = torch.tensor(0.0, device=next(iter(tag_logits.values())).device)
    cat_names = sorted(tag_logits.keys())

    for cat_idx, cat_name in enumerate(cat_names):
        logits = tag_logits[cat_name]
        labels = tag_labels[:, cat_idx]
        w = class_weights.get(cat_name) if class_weights else None
        if w is not None:
            w = w.to(logits.device)
        loss = F.cross_entropy(logits, labels, weight=w)
        total_loss = total_loss + loss

    return total_loss / max(len(cat_names), 1)
