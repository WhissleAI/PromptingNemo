"""Decoder manipulation utilities: scanning, extending, slimming, and rescaling."""

import json
import logging
import re
from typing import Dict, List

import torch
import torch.nn as nn
from omegaconf import open_dict
from nemo.utils import logging as nemo_logging
from nemo.collections.asr.losses.ctc import CTCLoss

from promptingnemo.tokenizer.dedup_aggregate import DedupAggregateTokenizer
from promptingnemo.tokenizer.aggregate import _family_name_for_lang


def scan_manifest_for_new_tokens(manifest_path: str, current_vocab: set) -> List[str]:
    """Scan training manifest for special tokens not in the current vocabulary."""
    tag_pattern = re.compile(r'^[A-Z][A-Z0-9_]*_[A-Z0-9_<>+.]*$|^END$')
    found = set()
    with open(manifest_path, encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            for word in entry.get('text', '').split():
                if tag_pattern.match(word) and word not in current_vocab:
                    found.add(word)
    return sorted(found)


def extend_decoder_for_new_tokens(model, new_tokens: List[str]):
    """Extend decoder output layer to accommodate new vocabulary tokens.

    CTC blank is at the last index -- new rows are inserted before it so the
    blank stays at position len(new_vocab).
    """
    n_new = len(new_tokens)
    if n_new == 0:
        return

    decoder_layer = model.decoder.decoder_layers[0]
    old_weight = decoder_layer.weight.data  # [old_num_classes, hidden, 1]
    old_bias = decoder_layer.bias.data      # [old_num_classes]
    hidden_dim = old_weight.shape[1]
    old_num_classes = old_weight.shape[0]

    blank_weight = old_weight[-1:, :, :]
    blank_bias = old_bias[-1:]
    vocab_weight = old_weight[:-1, :, :]
    vocab_bias = old_bias[:-1]

    new_weight = torch.randn(n_new, hidden_dim, 1, device=old_weight.device, dtype=old_weight.dtype) * 0.02
    new_bias = torch.zeros(n_new, device=old_bias.device, dtype=old_bias.dtype)

    extended_weight = torch.cat([vocab_weight, new_weight, blank_weight], dim=0)
    extended_bias = torch.cat([vocab_bias, new_bias, blank_bias], dim=0)

    decoder_layer.weight = nn.Parameter(extended_weight)
    decoder_layer.bias = nn.Parameter(extended_bias)

    new_vocab = list(model.decoder.vocabulary) + ['\u2581' + t for t in new_tokens]
    new_num_classes = len(new_vocab) + 1
    with open_dict(model.cfg):
        model.cfg.decoder.vocabulary = new_vocab
        model.cfg.decoder.num_classes = len(new_vocab)

    model.decoder._num_classes = new_num_classes
    model.decoder._ConvASRDecoder__vocabulary = new_vocab

    nemo_logging.info(
        f"Extended decoder: {old_num_classes} -> {new_num_classes} outputs "
        f"(+{n_new} tokens, blank repositioned to idx {new_num_classes - 1}). "
        f"New tokens: {new_tokens}"
    )


def slim_decoder_for_training(model, training_families):
    """Remove non-target language transcription tokens from the decoder.

    Keeps all tokens from target language SP tokenizers (which include shared
    special tokens like ENTITY_*, INTENT_*, EMOTION_*, GENDER_*, AGE_*
    baked into each SP model). Only removes transcription tokens unique to
    other language families. All kept tokens reuse pretrained decoder weights.
    """
    if not training_families:
        return

    training_families_upper = {f.upper() for f in training_families}

    if not hasattr(model.tokenizer, 'tokenizers_dict'):
        nemo_logging.warning("Model tokenizer has no tokenizers_dict -- cannot slim decoder")
        return

    all_langs = list(model.tokenizer.tokenizers_dict.keys())
    target_tokenizers = {
        lang: model.tokenizer.tokenizers_dict[lang]
        for lang in all_langs
        if lang.upper() in training_families_upper
    }

    if not target_tokenizers:
        nemo_logging.warning(
            "No tokenizers matched training families %s (available: %s). Skipping slim.",
            training_families, all_langs
        )
        return

    if len(target_tokenizers) == len(all_langs):
        nemo_logging.info("All language families selected -- no decoder slimming needed.")
        return

    pretrained_vocab = list(model.decoder.vocabulary)
    pretrained_token_to_idx = {t: i for i, t in enumerate(pretrained_vocab)}
    decoder_layer = model.decoder.decoder_layers[0]
    old_weight = decoder_layer.weight.data
    old_bias = decoder_layer.bias.data
    hidden_dim = old_weight.shape[1]

    slim_tokenizer = DedupAggregateTokenizer(target_tokenizers)
    slim_vocab = list(slim_tokenizer.vocabulary)

    n_slim = len(slim_vocab) + 1
    new_weight = torch.zeros(n_slim, hidden_dim, 1, device=old_weight.device, dtype=old_weight.dtype)
    new_bias = torch.zeros(n_slim, device=old_bias.device, dtype=old_bias.dtype)

    copied = 0
    missing_tokens = []
    for new_idx, token in enumerate(slim_vocab):
        old_idx = pretrained_token_to_idx.get(token)
        if old_idx is not None:
            new_weight[new_idx] = old_weight[old_idx]
            new_bias[new_idx] = old_bias[old_idx]
            copied += 1
        else:
            new_weight[new_idx, :, 0] = torch.randn(
                hidden_dim, device=old_weight.device, dtype=old_weight.dtype
            ) * 0.02
            missing_tokens.append(token)

    new_weight[-1] = old_weight[-1]
    new_bias[-1] = old_bias[-1]

    decoder_layer.weight = nn.Parameter(new_weight)
    decoder_layer.bias = nn.Parameter(new_bias)

    with open_dict(model.cfg):
        model.cfg.decoder.vocabulary = slim_vocab
        model.cfg.decoder.num_classes = len(slim_vocab)
    model.decoder._num_classes = n_slim
    model.decoder._ConvASRDecoder__vocabulary = slim_vocab

    model.tokenizer = slim_tokenizer

    if hasattr(model, 'loss'):
        old_cfg = getattr(model.loss, 'config', {})
        model.loss = CTCLoss(
            num_classes=n_slim,
            zero_infinity=old_cfg.get('zero_infinity', True),
            reduction=old_cfg.get('reduction', 'mean_batch'),
        )

    removed = len(pretrained_vocab) - len(slim_vocab)
    nemo_logging.info(
        f"Slim decoder: {len(pretrained_vocab)+1} -> {n_slim} outputs "
        f"(removed {removed} non-target tokens, copied {copied} pretrained weights). "
        f"Target families: {sorted(training_families_upper)}"
    )
    if missing_tokens:
        nemo_logging.warning(
            f"{len(missing_tokens)} slim vocab tokens not found in pretrained decoder "
            f"(random init): {missing_tokens[:10]}"
        )


def scale_down_tag_decoder_weights(model, scale_factor=0.01):
    """Scale down decoder weights for sentence-level meta-tag tokens.

    Tags like AGE_*, GENDER_*, EMOTION_*, INTENT_*, DIALECT_* should only
    fire at end-of-utterance positions. Their pretrained weights cause them
    to dominate at every frame after slim decoder removes competing tokens.
    Re-initializing to small random values forces them to learn proper
    positional firing from the CTC loss signal.
    """
    SENTENCE_TAG_PREFIXES = (
        'AGE_', 'GENDER_', 'GER_', 'GGENDER_', 'GENSION_',
        'EMOTION_', 'INTENT_', 'DIALECT_',
    )

    vocab = list(model.decoder.vocabulary)
    decoder_layer = model.decoder.decoder_layers[0]
    hidden_dim = decoder_layer.weight.shape[1]

    scaled_count = 0
    scaled_tokens = []
    for idx, token in enumerate(vocab):
        clean = token.lstrip('\u2581')
        if any(clean.startswith(prefix) for prefix in SENTENCE_TAG_PREFIXES):
            decoder_layer.weight.data[idx] = torch.randn_like(
                decoder_layer.weight.data[idx]
            ) * scale_factor
            decoder_layer.bias.data[idx] = 0.0
            scaled_count += 1
            if len(scaled_tokens) < 20:
                scaled_tokens.append(clean)

    nemo_logging.info(
        f"Re-initialized decoder weights for {scaled_count} sentence-level tag tokens "
        f"(scale={scale_factor}) to prevent mid-utterance spam. "
        f"Sample: {scaled_tokens}"
    )
