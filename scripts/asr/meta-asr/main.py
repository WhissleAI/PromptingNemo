import argparse
import os
import logging
import subprocess
import sys
import tempfile

# Add the project root to the Python path to allow running the script directly
# and to ensure NeMo can find the custom model as a module.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# nemo_repo_root = os.path.abspath(os.path.join(project_root, '..', 'NeMo-W'))
# if os.path.isdir(nemo_repo_root) and nemo_repo_root not in sys.path:
#     sys.path.insert(0, nemo_repo_root)

from pathlib import Path
import yaml
import torch
from omegaconf import OmegaConf, open_dict
import lightning.pytorch as pl
from nemo.utils.callbacks import NeMoModelCheckpoint
from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.utils import model_utils
from nemo.core import adapter_mixins
import sentencepiece as spm
from nemo.utils import exp_manager
import re
import os
import json
import logging
import math
import torch.nn as nn
import torch.nn.functional as F
import nemo
from nemo.collections.asr.data import audio_to_text
try:
    from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
except ImportError:
    DALIOutputs = tuple()
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.metrics.wer import word_error_rate_detail
# from nemo.collections.asr.parts.utils.transcribe_utils import transcribe_partial_audio
from nemo.collections.asr.data.audio_to_text import AudioToBPEDataset, _speech_collate_fn
from nemo.utils import logging as nemo_logging
from omegaconf import OmegaConf, open_dict
from nemo.collections.asr.parts.preprocessing.perturb import WhiteNoisePerturbation, ShiftPerturbation
from nemo.collections.asr.parts.preprocessing.features import AudioAugmentor
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.losses.ctc import CTCLoss
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import torch.distributed as dist
from nemo.core.classes.mixins import AccessMixin
from lightning.pytorch.loggers import TensorBoardLogger
# No longer need the manifest processor import
# from nemo.collections.asr.data.audio_to_text import ASRManifestProcessor as ManifestProcessor
from collections import Counter, defaultdict
from typing import Dict, IO, List, Set, Union

from scripts.asr.meta_asr.validate_data import (
    determine_output_path,
    validate_manifest as run_validate_manifest,
)

try:
    from nemo.collections.common.tokenizers import aggregate_tokenizer as nemo_agg
    from nemo.collections.common import tokenizers as nemo_tokenizers_pkg
except ImportError:
    nemo_agg = None
    nemo_tokenizers_pkg = None


LANG_FAMILIES: Dict[str, List[str]] = {}
LANG_TO_FAMILY: Dict[str, str] = {}


def set_language_families(language_families) -> None:
    """Configure language family mapping from config."""

    if not language_families:
        raise ValueError("language_families mapping is required but was empty")

    normalized: Dict[str, List[str]] = {}
    mapping: Dict[str, str] = {}

    def _register_family(family_name, languages) -> None:
        if not family_name:
            return
        family_key = str(family_name).upper()
        normalized_langs: List[str] = []
        if languages is None:
            languages = [family_key]
        elif isinstance(languages, (str, bytes)):
            languages = [languages]
        elif not isinstance(languages, (list, tuple, set)):
            raise ValueError(f"Expected iterable of languages for family '{family_name}', got {type(languages)}")

        for lang in languages:
            if not lang:
                continue
            lang_upper = str(lang).upper()
            normalized_langs.append(lang_upper)
            mapping[lang_upper] = family_key

        if not normalized_langs:
            normalized_langs.append(family_key)
            mapping[family_key] = family_key

        normalized[family_key] = normalized_langs

    if isinstance(language_families, dict):
        for family, langs in language_families.items():
            _register_family(family, langs)
    elif isinstance(language_families, (list, tuple, set)):
        for entry in language_families:
            if isinstance(entry, dict):
                for family, langs in entry.items():
                    _register_family(family, langs)
            else:
                _register_family(entry, entry)
    else:
        raise ValueError(f"Unsupported language_families type: {type(language_families)}")

    if not mapping:
        raise ValueError("language_families must include at least one language entry")

    LANG_FAMILIES.clear()
    LANG_FAMILIES.update(normalized)

    LANG_TO_FAMILY.clear()
    LANG_TO_FAMILY.update(mapping)


def resolve_model_path(cfg, path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = Path(cfg.model.model_root).expanduser().resolve() / path
    return path


def store_tokenizer_langs(cfg, mapping: Dict[str, Dict[str, str]]) -> None:
    path_str = cfg.model.get('tokenizer_langs_path')
    if not path_str:
        path_str = 'tokenizer_langs.yaml'
        with open_dict(cfg):
            cfg.model.tokenizer_langs_path = path_str
    path = resolve_model_path(cfg, path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(mapping, f, sort_keys=True, allow_unicode=True)
    with open_dict(cfg):
        cfg.model.tokenizer_langs = {}


def store_shared_special_tokens(cfg, tokens: List[str]) -> None:
    path_str = cfg.model.get('shared_special_tokens_path')
    if not path_str:
        path_str = 'shared_special_tokens.yaml'
        with open_dict(cfg):
            cfg.model.shared_special_tokens_path = path_str
    path = resolve_model_path(cfg, path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(tokens, f, sort_keys=False, allow_unicode=True)
    with open_dict(cfg):
        cfg.model.shared_special_tokens = []


def store_aggregate_vocabulary(cfg, vocab: List[str]) -> Path:
    path_str = cfg.model.get('aggregate_vocabulary_path')
    if not path_str:
        path_str = 'aggregate_vocab.txt'
        with open_dict(cfg):
            cfg.model.aggregate_vocabulary_path = path_str
    path = resolve_model_path(cfg, path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for token in vocab:
            f.write(f"{token}\n")
    with open_dict(cfg):
        cfg.model.aggregate_vocabulary = []
    return path


def load_tokenizer_langs(cfg) -> Dict[str, Dict[str, str]]:
    tokenizer_langs_cfg = cfg.model.get('tokenizer_langs')
    if tokenizer_langs_cfg:
        return OmegaConf.to_container(tokenizer_langs_cfg, resolve=True)
    path_str = cfg.model.get('tokenizer_langs_path')
    if not path_str:
        return {}
    path = resolve_model_path(cfg, path_str)
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data


def load_shared_special_tokens(cfg) -> List[str]:
    tokens_cfg = cfg.model.get('shared_special_tokens')
    if tokens_cfg:
        return list(tokens_cfg)
    path_str = cfg.model.get('shared_special_tokens_path')
    if not path_str:
        return []
    path = resolve_model_path(cfg, path_str)
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or []
    return list(data)


def load_aggregate_vocabulary(cfg) -> List[str]:
    vocab_cfg = cfg.model.get('aggregate_vocabulary')
    if vocab_cfg:
        return list(vocab_cfg)
    path_str = cfg.model.get('aggregate_vocabulary_path')
    if not path_str:
        return []
    path = resolve_model_path(cfg, path_str)
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

if nemo_agg is not None:
    TokenizerSpecBase = nemo_agg.TokenizerSpec
    DummyTokenizer = nemo_agg.DummyTokenizer

    class DedupAggregateTokenizer(TokenizerSpecBase):
        def __init__(self, tokenizers: Dict):
            self.tokenizers_dict = tokenizers
            self.vocabulary: List[str] = []
            self._token_to_global: Dict[str, int] = {}
            self.lang_local_to_global: Dict[str, Dict[int, int]] = {}
            self.global_to_lang_local: Dict[str, Dict[int, int]] = {}
            self.lang_candidates_by_global: Dict[int, Set[str]] = {}

            for lang, tokenizer in self.tokenizers_dict.items():
                lang_map: Dict[int, int] = {}
                vocab_list = list(tokenizer.vocab)
                for local_id, token in enumerate(vocab_list):
                    global_id = self._token_to_global.get(token)
                    if global_id is None:
                        global_id = len(self.vocabulary)
                        self.vocabulary.append(token)
                        self._token_to_global[token] = global_id
                        self.lang_candidates_by_global[global_id] = set()
                    self.lang_candidates_by_global[global_id].add(lang)
                    lang_map[local_id] = global_id
                self.lang_local_to_global[lang] = lang_map
                self.global_to_lang_local[lang] = {global_id: local_id for local_id, global_id in lang_map.items()}

            self.vocab_size = len(self.vocabulary)
            logging.info(f"Aggregate vocab size (dedup): {self.vocab_size}")

            self.tokenizer = DummyTokenizer(self.vocabulary)

            self.langs_by_token_id: Dict[int, Union[str, None]] = {}
            self.tokenizers_by_token_id: Dict[int, Union[TokenizerSpecBase, None]] = {}
            self.offset_token_ids_by_token_id: Dict[int, Union[int, None]] = {}
            for global_id in range(self.vocab_size):
                lang_candidates = sorted(self.lang_candidates_by_global.get(global_id, []))
                default_lang = lang_candidates[0] if lang_candidates else None
                self.langs_by_token_id[global_id] = default_lang
                if default_lang is not None:
                    self.tokenizers_by_token_id[global_id] = self.tokenizers_dict[default_lang]
                    self.offset_token_ids_by_token_id[global_id] = self.global_to_lang_local[default_lang][global_id]
                else:
                    self.tokenizers_by_token_id[global_id] = None
                    self.offset_token_ids_by_token_id[global_id] = None

        def text_to_tokens(self, text, lang_id):
            tokenizer = self.tokenizers_dict[lang_id]
            return tokenizer.text_to_tokens(text)

        def text_to_ids(self, text, lang_id):
            tokenizer = self.tokenizers_dict[lang_id]
            token_ids = tokenizer.text_to_ids(text)
            lang_map = self.lang_local_to_global[lang_id]
            return [lang_map[t] for t in token_ids]

        def tokens_to_text(self, tokens, lang_id):
            tokenizer = self.tokenizers_dict[lang_id]
            return tokenizer.decode_pieces(tokens)

        def ids_to_text(self, ids):
            if isinstance(ids, (np.ndarray, torch.Tensor)):
                ids = ids.tolist()
            tokens = [self.vocabulary[i] for i in ids]
            return ''.join(tokens).replace('▁', ' ')

        def token_to_id(self, token, lang_id):
            tokenizer = self.tokenizers_dict[lang_id]
            local_id = tokenizer.token_to_id(token)
            if local_id >= 0:
                return self.lang_local_to_global[lang_id][local_id]
            return self._token_to_global[token]

        def ids_to_tokens(self, ids):
            if isinstance(ids, (np.ndarray, torch.Tensor)):
                ids = ids.tolist()
            return [self.vocabulary[i] for i in ids]

        def ids_to_text_and_langs(self, ids):
            result = []
            for idx in ids:
                token = self.vocabulary[idx]
                lang = self.langs_by_token_id.get(idx)
                result.append({'char': token.replace('▁', ' ').strip(), 'lang': lang})
            return result

        def ids_to_words_and_langs(self, ids):
            words_and_langs = []
            current_ids = []
            for idx in ids:
                token = self.vocabulary[idx]
                if token.startswith('▁') and current_ids:
                    word = ''.join(self.vocabulary[i] for i in current_ids).replace('▁', ' ').strip()
                    lang = self.ids_to_lang(current_ids)
                    words_and_langs.append({'word': word, 'lang': lang})
                    current_ids = []
                current_ids.append(idx)
            if current_ids:
                word = ''.join(self.vocabulary[i] for i in current_ids).replace('▁', ' ').strip()
                lang = self.ids_to_lang(current_ids)
                words_and_langs.append({'word': word, 'lang': lang})
            return words_and_langs

        def ids_to_lang(self, ids):
            lang_counts: Dict[str, int] = {}
            for idx in ids:
                lang = self.langs_by_token_id.get(idx)
                if lang is None:
                    continue
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            if not lang_counts:
                return None
            return max(lang_counts.items(), key=lambda item: item[1])[0]

        def tokens_to_ids(self, tokens: Union[str, List[str]], langs: Union[str, List[str]]) -> Union[int, List[int]]:
            if isinstance(tokens, str):
                tokens = [tokens]
            if isinstance(langs, str):
                langs = [langs]
            ids = []
            for token, lang in zip(tokens, langs):
                tokenizer = self.tokenizers_dict[lang]
                local_id = tokenizer.token_to_id(token)
                if local_id >= 0:
                    ids.append(self.lang_local_to_global[lang][local_id])
                else:
                    ids.append(self._token_to_global[token])
            return ids if len(ids) > 1 else ids[0]

        def get_bos(self, lang_id: str) -> int:
            tokenizer = self.tokenizers_dict[lang_id]
            local_id = tokenizer.bos
            return self.lang_local_to_global[lang_id][local_id]

        def get_eos(self, lang_id: str) -> int:
            tokenizer = self.tokenizers_dict[lang_id]
            local_id = tokenizer.eos
            return self.lang_local_to_global[lang_id][local_id]

        @property
        def vocab(self):
            return self.vocabulary

    nemo_agg.AggregateTokenizer = DedupAggregateTokenizer
    if nemo_tokenizers_pkg is not None:
        nemo_tokenizers_pkg.AggregateTokenizer = DedupAggregateTokenizer


class CustomEncDecCTCModelBPE(EncDecCTCModelBPE):
    def setup_custom_loss(self):
        self.use_keyword_loss = self.cfg.get('use_keyword_loss', False)
        self.keyword_loss_weight = self.cfg.get('keyword_loss_weight', 0.3)
        self.keyword_loss_warmup_steps = self.cfg.get('keyword_loss_warmup_steps', 0)
        self.keyword_token_ids = set()
        self._val_family_stats = defaultdict(lambda: {'errors': 0.0, 'words': 0})
        self._validation_dataset_ref = None

        self.use_family_loss_weights = self.cfg.get('use_family_loss_weights', False)
        self.family_loss_weights = None

        if self.use_family_loss_weights:
            logging.info("Using language family weighted loss. Overriding CTC loss reduction to 'none'.")
            if not hasattr(self, 'loss') or not isinstance(self.loss, CTCLoss):
                self.loss = CTCLoss(
                    num_classes=self.decoder.num_classes, zero_infinity=True, reduction='mean_batch'
                )
            
            # Re-create loss with reduction='none' to get per-sample losses
            self.loss = CTCLoss(
                num_classes=self.decoder.num_classes,
                zero_infinity=self.loss.ctc_loss.zero_infinity,
                reduction='none'
            )

    def set_keyword_token_ids(self, keyword_ids):
        self.keyword_token_ids = set(keyword_ids)
        logging.info(f"Set {len(self.keyword_token_ids)} keyword token IDs for custom loss.")

    def set_family_loss_weights(self, weights: Dict[str, float]):
        """Stores the pre-computed language family weights."""
        if self.use_family_loss_weights:
            self.family_loss_weights = weights
            logging.info("Successfully set language family loss weights on the model.")

    def training_step(self, batch, batch_idx):
        sample_ids = None
        if len(batch) == 5:  # audio, audio_len, transcript, transcript_len, sample_id
            audio_signal, audio_signal_len, transcript, transcript_len, sample_ids = batch
        else:
            audio_signal, audio_signal_len, transcript, transcript_len = batch

        log_probs, encoded_len, greedy_predictions = self.forward(
            input_signal=audio_signal, input_signal_length=audio_signal_len
        )
        
        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )

        # --- Language Family Weighted Loss ---
        if self.use_family_loss_weights:
            if sample_ids is not None and self.family_loss_weights:
                # `loss_value` is unreduced here (per-sample)
                dataset = self._train_dl.dataset
                batch_weights_list = [
                    self.family_loss_weights.get(_family_name_for_lang(dataset.language_ids[idx.item()]), 1.0)
                    for idx in sample_ids
                ]
                weights_tensor = torch.tensor(
                    batch_weights_list, device=loss_value.device, dtype=loss_value.dtype
                )
                
                # Apply weights and then compute the mean
                loss_value = (loss_value * weights_tensor).mean()
                self.log('family_weighted_loss', loss_value, on_step=True, on_epoch=False, prog_bar=True)
            else:
                # Fallback: if weights or sample_ids are missing, just take the mean
                loss_value = loss_value.mean()
        
        # --- Keyword Loss (applied after family weighting) ---
        if self.use_keyword_loss and self.training and len(self.keyword_token_ids) > 0:
            current_step = self.trainer.global_step
            if self.keyword_loss_warmup_steps > 0 and current_step < self.keyword_loss_warmup_steps:
                current_keyword_loss_weight = (
                    current_step / self.keyword_loss_warmup_steps
                ) * self.keyword_loss_weight
            else:
                current_keyword_loss_weight = self.keyword_loss_weight

            self.log('current_keyword_loss_weight', current_keyword_loss_weight, on_step=True, on_epoch=False, prog_bar=True)

            keyword_targets = []
            keyword_target_lengths = []
            for i in range(transcript.size(0)):
                target = transcript[i][: transcript_len[i]]
                keyword_target = [
                    token_id.item() for token_id in target if token_id.item() in self.keyword_token_ids
                ]

                if len(keyword_target) > 0:
                    keyword_targets.append(
                        torch.tensor(keyword_target, dtype=torch.long, device=transcript.device)
                    )
                    keyword_target_lengths.append(len(keyword_target))

            if keyword_targets:
                keyword_targets_padded = torch.nn.utils.rnn.pad_sequence(
                    keyword_targets, batch_first=True, padding_value=self.tokenizer.pad_id
                )
                keyword_target_lengths_tensor = torch.tensor(
                    keyword_target_lengths, dtype=torch.long, device=transcript.device
                )

                if keyword_targets_padded.numel() > 0:
                    keyword_loss = self.loss(
                        log_probs=log_probs,
                        targets=keyword_targets_padded,
                        input_lengths=encoded_len,
                        target_lengths=keyword_target_lengths_tensor,
                    )
                    loss_value = (
                        1 - current_keyword_loss_weight
                    ) * loss_value + current_keyword_loss_weight * keyword_loss

        self.log('train_loss', loss_value)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])

        return {'loss': loss_value}

    def _get_dataset_for_prefix(self, prefix: str):
        if prefix == 'val':
            dataset = getattr(self, '_validation_dataset_ref', None)
            if dataset is None:
                data_loader = getattr(self, '_validation_dl', None)
                dataset = getattr(data_loader, 'dataset', None) if data_loader is not None else None
                self._validation_dataset_ref = dataset
            return dataset
        return None

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._val_family_stats = defaultdict(lambda: {'errors': 0.0, 'words': 0})

    def _accumulate_family_wer(self, sample_ids, log_probs, encoded_len, transcript, transcript_len):
        dataset = self._get_dataset_for_prefix('val')
        if dataset is None or not hasattr(dataset, 'language_ids') or sample_ids is None:
            return

        if isinstance(sample_ids, torch.Tensor):
            sample_indices = sample_ids.detach().cpu().view(-1).tolist()
        else:
            sample_indices = [int(idx) for idx in sample_ids]

        if not sample_indices:
            return

        with torch.no_grad():
            try:
                hypotheses = self.decoding.ctc_decoder_predictions_tensor(
                    decoder_outputs=log_probs.detach(),
                    decoder_lengths=encoded_len.detach(),
                    return_hypotheses=False,
                )
            except RuntimeError:
                hypotheses = self.decoding.ctc_decoder_predictions_tensor(
                    decoder_outputs=log_probs.detach().cpu(),
                    decoder_lengths=encoded_len.detach().cpu(),
                    return_hypotheses=False,
                )

            targets_cpu = transcript.detach().cpu()
            target_lens_cpu = transcript_len.detach().cpu()

            limit = min(len(sample_indices), len(hypotheses), targets_cpu.size(0))

            for idx in range(limit):
                sample_index = sample_indices[idx]
                if sample_index >= len(dataset.language_ids):
                    continue
                lang_code = dataset.language_ids[sample_index]
                family = _family_name_for_lang(lang_code)
                hyp_obj = hypotheses[idx]
                hyp_text = hyp_obj.text if hasattr(hyp_obj, 'text') else str(hyp_obj)
                target_tokens = targets_cpu[idx][: target_lens_cpu[idx]].tolist()
                reference_text = self._decode_target_tokens(target_tokens)
                wer_value, words, *_ = word_error_rate_detail(
                    [hyp_text], [reference_text], use_cer=self.wer.use_cer
                )
                if not math.isfinite(wer_value) or words <= 0:
                    continue
                errors = wer_value * words
                stats = self._val_family_stats[family]
                stats['errors'] += float(errors)
                stats['words'] += int(round(words))

    def _log_family_metrics(self, prefix: str):
        if not getattr(self, '_val_family_stats', None):
            return

        aggregated = {}
        device = self.device if isinstance(self.device, torch.device) else torch.device('cpu')

        for family, stats in self._val_family_stats.items():
            tensor = torch.tensor([stats['errors'], stats['words']], dtype=torch.float32, device=device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            aggregated[family] = tensor.cpu()

        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        total_errors = 0.0
        total_words = 0.0
        for family, tensor in aggregated.items():
            errors = float(tensor[0].item())
            words = float(tensor[1].item())
            if words <= 0:
                continue
            total_errors += errors
            total_words += words
            self.log(f"{prefix}_wer_{family}", errors / words, prog_bar=False, sync_dist=False)

        if total_words > 0:
            self.log(f"{prefix}_wer_combined", total_errors / total_words, prog_bar=False, sync_dist=False)

    def on_validation_epoch_end(self):
        result = super().on_validation_epoch_end()
        self._log_family_metrics('val')
        self._val_family_stats = defaultdict(lambda: {'errors': 0.0, 'words': 0})
        return result

    def _decode_target_tokens(self, token_ids: List[int]) -> str:
        decoding = getattr(self, 'decoding', None)
        if decoding is not None:
            if hasattr(decoding, 'decode_ids_to_str'):
                return decoding.decode_ids_to_str(token_ids)
            tokens = None
            if hasattr(decoding, 'decode_ids_to_tokens'):
                tokens = decoding.decode_ids_to_tokens(token_ids)
            if tokens is not None:
                if tokens and isinstance(tokens[0], str):
                    return ''.join(tokens).replace('▁', ' ').strip()
                if hasattr(decoding, 'decode_tokens_to_str'):
                    return decoding.decode_tokens_to_str(tokens)
                return ''.join(tokens).replace('▁', ' ').strip()
        tokenizer = getattr(self, 'tokenizer', None)
        if tokenizer is not None:
            if hasattr(tokenizer, 'ids_to_text'):
                return tokenizer.ids_to_text(token_ids)
            if hasattr(tokenizer, 'ids_to_tokens'):
                tokens = tokenizer.ids_to_tokens(token_ids)
                return ''.join(tokens).replace('▁', ' ').strip()
        return ' '.join(str(t) for t in token_ids)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        sample_ids = None

        if isinstance(batch, DALIOutputs):
            signal, signal_len, transcript, transcript_len = batch
            core_batch = batch
        else:
            if len(batch) == 6:
                signal, signal_len, transcript, transcript_len, _, sample_ids = batch
            elif len(batch) == 5:
                signal, signal_len, transcript, transcript_len, extra = batch
                if torch.is_tensor(extra):
                    if extra.dtype.is_floating_point:
                        sample_ids = None
                    else:
                        sample_ids = extra
                elif isinstance(extra, (float, np.floating)):
                    sample_ids = None
                else:
                    sample_ids = extra
            else:
                signal, signal_len, transcript, transcript_len = batch
            core_batch = (signal, signal_len, transcript, transcript_len)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        if isinstance(core_batch, DALIOutputs) and core_batch.has_processed_signal:
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions = self.forward(
                input_signal=signal, input_signal_length=signal_len
            )

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        loss_value, metrics = self.add_interctc_losses(
            loss_value,
            transcript,
            transcript_len,
            compute_wer=True,
            log_wer_num_denom=True,
            log_prefix="val_",
        )

        self.wer.update(
            predictions=log_probs,
            targets=transcript,
            targets_lengths=transcript_len,
            predictions_lengths=encoded_len,
        )
        wer, wer_num, wer_denom = self.wer.compute()
        self.wer.reset()
        metrics.update({'val_loss': loss_value, 'val_wer_num': wer_num, 'val_wer_denom': wer_denom, 'val_wer': wer})

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32, device=log_probs.device))

        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if sample_ids is not None:
            self._accumulate_family_wer(sample_ids, log_probs, encoded_len, transcript, transcript_len)

        if isinstance(self.trainer.val_dataloaders, list) and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)

        return metrics


class BalancedLanguageBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, temperature=0.2, seed=42, lang_to_family_map: Dict[str, str] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.temperature = temperature
        self.seed = seed
        self.epoch = 0
        self.lang_to_family_map = {k.upper(): v for k, v in (lang_to_family_map or {}).items()}

        self.num_samples = len(self.dataset)
        self.world_size = 1
        self.rank = 0
        self._refresh_distributed_context()
        
        family_counts: Dict[str, int] = {}
        self.sample_families: List[str] = []
        for lang_id in self.dataset.language_ids:
            family = self._resolve_family(lang_id)
            self.sample_families.append(family)
            family_counts[family] = family_counts.get(family, 0) + 1
        
        self.family_counts = family_counts
        self.families = list(self.family_counts.keys())
        print("ALL LANGUAGE FAMILIES")
        print(self.families)
        print("ALL FAMILY COUNTS")
        print(self.family_counts)

        dataset_weights = getattr(self.dataset, 'sample_keyphrase_weights', None)
        if dataset_weights is not None and len(dataset_weights) == len(self.dataset.language_ids):
            self.sample_weights = np.asarray(dataset_weights, dtype=np.float32)
        else:
            self.sample_weights = np.ones(len(self.dataset.language_ids), dtype=np.float32)
        
        # Calculate sampling probabilities with temperature
        total_samples = len(self.dataset)
        weights = np.array([count / total_samples for count in self.family_counts.values()]) if total_samples > 0 else np.array([])
        if weights.size > 0:
            temp_weights = weights ** (1 / self.temperature)
            self.family_sample_probs = temp_weights / np.sum(temp_weights)
        else:
            self.family_sample_probs = np.array([])
        
        family_indices_map = {family: [] for family in self.families}
        family_weight_map = {family: [] for family in self.families}
        for idx, family in enumerate(self.sample_families):
            if family in family_indices_map:
                family_indices_map[family].append(idx)
                family_weight_map[family].append(float(self.sample_weights[idx]))
        self.family_indices = {
            family: np.array(indices, dtype=np.uint32) for family, indices in family_indices_map.items()
        }
        self.family_weights = {
            family: np.array(weights if weights else [1.0] * len(family_indices_map[family]), dtype=np.float32)
            for family, weights in family_weight_map.items()
        }
        
        self.num_batches_per_epoch = self.num_samples // self.batch_size
        
        # Adjust for distributed training
        self.num_samples_per_rank = self._calculate_samples_per_rank()
            
    def __iter__(self):
        self._refresh_distributed_context()

        # Seed with epoch to ensure different shuffling each epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        np.random.seed(self.seed + self.epoch)

        # 1. Create the full list of indices for the epoch based on language sampling probabilities
        num_samples_per_family = np.round(self.family_sample_probs * self.num_samples).astype(int)
        
        # Adjust to ensure the total number of samples is correct
        diff = self.num_samples - np.sum(num_samples_per_family)
        if diff != 0 and len(self.family_sample_probs) > 0:
            num_samples_per_family[np.argmax(self.family_sample_probs)] += diff

        epoch_indices_list = []
        for i, family in enumerate(self.families):
            num_family_samples = num_samples_per_family[i]
            if num_family_samples > 0:
                replace = len(self.family_indices[family]) < num_family_samples
                weights = self.family_weights.get(family)
                if weights is None or len(weights) == 0:
                    probs = None
                else:
                    weights = weights.astype(np.float64)
                    total = np.sum(weights)
                    if total <= 0:
                        probs = None
                    else:
                        probs = weights / total
                indices = np.random.choice(
                    self.family_indices[family],
                    num_family_samples,
                    replace=replace,
                    p=probs,
                )
                epoch_indices_list.append(indices)
        
        if not epoch_indices_list:
            self.epoch += 1
            return iter([])

        epoch_indices = np.concatenate(epoch_indices_list)
        
        # 2. Shuffle the indices for the entire epoch
        np.random.shuffle(epoch_indices)

        # 3. Yield batches for the current rank without creating a list of all batches
        num_batches = len(epoch_indices) // self.batch_size

        for batch_idx in range(self.rank, num_batches, self.world_size):
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            yield epoch_indices[start_idx:end_idx].tolist()

        self.epoch += 1

    def __len__(self):
        self._refresh_distributed_context()
        num_batches = self.num_samples // self.batch_size
        if num_batches == 0:
            return 0
        return math.ceil(num_batches / max(self.world_size, 1))

    def _calculate_samples_per_rank(self):
        world_size = max(self.world_size, 1)
        if world_size <= 1:
            return self.num_samples
        samples = self.num_samples // world_size
        if self.num_samples % world_size != 0:
            samples += 1
        return samples

    def _refresh_distributed_context(self):
        if not dist.is_available():
            self.world_size = 1
            self.rank = 0
            return

        try:
            if dist.is_initialized():
                self.world_size = max(int(dist.get_world_size()), 1)
                self.rank = int(dist.get_rank())
            else:
                self.world_size = 1
                self.rank = 0
        except (RuntimeError, ValueError):
            # During dataloader workers spawn, the default process group might not be ready yet.
            self.world_size = 1
            self.rank = 0
        self.num_samples_per_rank = self._calculate_samples_per_rank()

    def _resolve_family(self, lang_id: str) -> str:
        if not lang_id:
            return "UNKNOWN"
        lang_upper = str(lang_id).upper()
        if lang_upper in self.lang_to_family_map:
            return self.lang_to_family_map[lang_upper]
        if lang_upper in LANG_TO_FAMILY:
            return LANG_TO_FAMILY[lang_upper]
        return f"Singleton_{lang_upper}"


class RobustAudioToBPEDataset(audio_to_text.AudioToBPEDataset):
    skip_audio_validation_default: bool = False
    keyphrase_oversample_factor_default: float = 0.0
    default_lang_field: str = 'lang'

    def __init__(self, manifest_filepath, *args, **kwargs):
        allowed_langs = kwargs.pop('allowed_langs', None)
        if allowed_langs is not None:
            allowed_langs = {lang.upper() for lang in allowed_langs}
        lang_field = kwargs.pop('lang_field', None)
        if not lang_field:
            lang_field = getattr(self, 'default_lang_field', 'lang')

        # Build a normalized temporary manifest that guarantees language casing
        kept_lines = 0
        processed_lines = 0
        tmp_manifest = tempfile.NamedTemporaryFile('w', delete=False, suffix='.json', encoding='utf-8')
        tmp_path = Path(tmp_manifest.name)
        progress_every = int(os.environ.get('ROBUST_DATASET_PROGRESS_EVERY', '50000'))
        skip_audio_validation = self.skip_audio_validation_default or str(manifest_filepath).endswith('.validated.json')
        with open(manifest_filepath, 'r', encoding='utf-8') as src:
            for raw_line in src:
                processed_lines += 1
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Manifest contains malformed JSON at line {processed_lines}: {raw_line[:100]}"
                    ) from exc

                audio_file = data.get('audio_filepath')
                if not audio_file:
                    raise RuntimeError("Manifest entry is missing 'audio_filepath'")

                if not skip_audio_validation:
                    audio_path = Path(audio_file)
                    if not audio_path.exists():
                        raise FileNotFoundError(f"Audio file not found for manifest entry: {audio_file}")

                lang_value = data.get(lang_field)
                if not lang_value and lang_field != 'lang':
                    lang_value = data.get('lang')
                if not lang_value:
                    raise RuntimeError(
                        f"Manifest entry missing language field '{lang_field}' (and fallback 'lang') at line {processed_lines}"
                    )

                lang_value = str(lang_value).upper()
                data['lang'] = lang_value
                if lang_field != 'lang':
                    data[lang_field] = lang_value
                if not skip_audio_validation:
                    try:
                        AudioSegment.from_file(audio_file)
                    except Exception as audio_exc:
                        raise RuntimeError(
                            f"Failed to decode audio file '{audio_file}' referenced in manifest: {audio_exc}"
                        ) from audio_exc

                tmp_manifest.write(json.dumps(data, ensure_ascii=False) + '\n')
                kept_lines += 1

                if progress_every > 0 and processed_lines % progress_every == 0:
                    logging.info(
                        "Manifest prep progress [%s]: processed=%d kept=%d",
                        manifest_filepath,
                        processed_lines,
                        kept_lines,
                    )
        tmp_manifest.close()

        if kept_lines == 0:
            raise RuntimeError(
                f"No usable manifest entries remained after filtering for language IDs. Ensure your manifests contain a '{lang_field}' field for each sample."
            )

        logging.info(
            "Finished preparing dataset manifest %s: total_rows=%d kept=%d",
            manifest_filepath,
            processed_lines,
            kept_lines,
        )

        # Initialize the parent dataset with the filtered manifest
        super().__init__(manifest_filepath=str(tmp_path), *args, **kwargs)

        # Track the temporary manifest for optional cleanup
        self._filtered_manifest_path = tmp_path
        self.lang_field = lang_field

        # Build language ids aligned with the filtered manifest collection
        self.language_ids = []
        new_collection = []
        for item in self.manifest_processor.collection:
            lang_id = getattr(item, 'lang', None)
            if not lang_id and self.lang_field and self.lang_field != 'lang':
                lang_id = getattr(item, self.lang_field, None)
            if not lang_id:
                raise RuntimeError("Manifest sample is missing language metadata after normalization")
            lang_id = lang_id.upper()
            self.language_ids.append(lang_id)
            new_collection.append(item)

        self.manifest_processor.collection = new_collection

        factor = float(self.keyphrase_oversample_factor_default)
        if factor > 0.0 and new_collection:
            weights = []
            for item in new_collection:
                text = getattr(item, 'text', '') or ''
                tokens = text.split()
                entity_count = sum(1 for tok in tokens if tok.upper().startswith('ENTITY_'))
                end_count = sum(1 for tok in tokens if tok.upper() == 'END')
                score = entity_count + end_count
                weights.append(1.0 + factor * score)
            self.sample_keyphrase_weights = np.array(weights, dtype=np.float32)
        else:
            self.sample_keyphrase_weights = np.ones(len(new_collection), dtype=np.float32)

    def __del__(self):
        # Attempt to remove the temporary manifest file when the dataset is garbage collected
        tmp_path = getattr(self, '_filtered_manifest_path', None)
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as exc:
            if index < len(self.manifest_processor.collection):
                sample = self.manifest_processor.collection[index]
                audio_path = getattr(sample, 'audio_filepath', 'unknown')
            else:
                audio_path = 'unknown'
            logging.warning(
                f"Skipping dataset sample at index {index} with audio '{audio_path}' due to error: {exc}"
            )
            return None


# Monkey-patch the original NeMo class with our robust version.
# This is necessary because the model's setup helper functions are hardcoded
# to use the base AudioToBPEDataset and ignore the `_target_` config parameter.
audio_to_text.AudioToBPEDataset = RobustAudioToBPEDataset


def patched_speech_collate_fn(batch, pad_id):
    """
    A patched version of _speech_collate_fn that correctly handles mixed
    mono/stereo audio and pads all signals to a consistent length, and filters out problematic samples.
    """
    has_weight = False
    has_sample_ids = False
    for item in batch:
        if item is None:
            continue
        tuple_len = len(item)
        if tuple_len >= 6:
            has_weight = True
            has_sample_ids = True
            break
        if tuple_len >= 5:
            candidate = item[4]
            if torch.is_tensor(candidate):
                if candidate.dtype.is_floating_point:
                    has_weight = True
                else:
                    has_sample_ids = True
            elif isinstance(candidate, (float, np.floating)):
                has_weight = True
            elif isinstance(candidate, (int, np.integer)):
                has_sample_ids = True
        break

    original_len = len(batch)
    batch = [item for item in batch if item is not None]
    if len(batch) < original_len:
        logging.warning(f"Skipped {original_len - len(batch)} samples in a batch.")

    if not batch:
        empty_audio = torch.empty(0, 0, dtype=torch.float32)
        empty_lengths = torch.tensor([], dtype=torch.long)
        empty_transcripts = torch.empty(0, 0, dtype=torch.long)
        outputs = [empty_audio, empty_lengths, empty_transcripts, empty_lengths.clone()]
        if has_weight:
            outputs.append(torch.tensor([], dtype=torch.float32))
        if has_sample_ids:
            outputs.append(torch.tensor([], dtype=torch.long))
        return tuple(outputs)
    
    if isinstance(batch[0], list):
        batch = [item for sublist in batch for item in sublist]

    audio_signal, audio_lengths, transcript, transcript_lengths = [], [], [], []
    weights = [] if has_weight else None
    sample_ids = [] if has_sample_ids else None

    for i, sample in enumerate(batch):
        sig = sample[0]
        # Always process as raw audio, handling multi-channel cases.
        if sig.ndim > 1:
            # Audio is multi-channel, convert to mono by averaging channels.
            sig = torch.mean(sig, dim=-1)

        audio_signal.append(sig.squeeze())
        audio_lengths.append(sample[1])
        transcript.append(sample[2])
        transcript_lengths.append(sample[3])
        if has_weight and has_sample_ids and len(sample) >= 6:
            weights.append(sample[4])
            sample_idx = sample[5]
        elif has_weight and len(sample) >= 5 and not has_sample_ids:
            weights.append(sample[4])
            sample_idx = None
        elif has_sample_ids and len(sample) >= 5 and not has_weight:
            sample_idx = sample[4]
        else:
            sample_idx = None

        if has_sample_ids and sample_idx is not None:
            if torch.is_tensor(sample_idx):
                sample_idx = int(sample_idx.item())
            else:
                sample_idx = int(sample_idx)
            sample_ids.append(sample_idx)
        if has_weight and weights is not None and len(weights) > 0:
            if torch.is_tensor(weights[-1]):
                weights[-1] = float(weights[-1].item())

    # Padding logic for 1D raw audio
    max_len = max(audio_lengths) if audio_lengths else 0
    audio_signal_padded = []
    for sig in audio_signal:
        sig_padded = torch.zeros(max_len, dtype=sig.dtype, device=sig.device)
        len_to_copy = min(sig.size(0), max_len)
        sig_padded[:len_to_copy] = sig[:len_to_copy]
        audio_signal_padded.append(sig_padded)

    audio_signal = torch.stack(audio_signal_padded) if audio_signal_padded else torch.tensor([])
    audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)

    transcript_lengths = torch.tensor(transcript_lengths, dtype=torch.long)
    transcript = nn.utils.rnn.pad_sequence(transcript, batch_first=True, padding_value=pad_id)

    outputs = [audio_signal, audio_lengths, transcript, transcript_lengths]
    if has_weight:
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        outputs.append(weights_tensor)
    if has_sample_ids:
        sample_ids_tensor = torch.tensor(sample_ids, dtype=torch.long)
        outputs.append(sample_ids_tensor)

    return tuple(outputs)


# Monkey-patch the collate function
audio_to_text._speech_collate_fn = patched_speech_collate_fn


def train_sentencepiece_tokenizer(
    manifest_file,
    tokenizer_folder,
    special_tokens=None,
    vocab_size=5000,
    character_coverage=None,
    extra_options=None,
):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Starting the tokenizer training process")
    logging.info(f"Requested vocab_size={vocab_size}")

    # Step 1: Read the manifest file and extract text data
    def read_manifest(manifest_path):
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
        return [json.loads(line)['text'] for line in lines]
    
    logging.info("Reading manifest file")
    text_data = read_manifest(manifest_file)
    logging.info(f"Extracted {len(text_data)} sentences from the manifest file")
    
    # Step 2: Save the extracted text to a temporary file
    if not os.path.exists(tokenizer_folder):
        os.makedirs(tokenizer_folder)
    
    temp_text_file = os.path.join(tokenizer_folder, 'text_data.txt')
    logging.info(f"Saving extracted text to {temp_text_file}")
    with open(temp_text_file, 'w') as f:
        for sentence in text_data:
            f.write(sentence + '\n')
    
    # Step 3: Train the SentencePiece tokenizer with special tokens if provided
    model_prefix = os.path.join(tokenizer_folder, 'tokenizer')
    
    # Prepare special tokens string
    special_tokens = special_tokens or []
    filtered_special_tokens = [token for token in special_tokens if token and token.strip()]
    logging.info(f"Using {len(filtered_special_tokens)} special tokens")
    
    # Check if there are any duplicate tokens after case normalization
    token_set = set()
    unique_tokens = []
    for token in filtered_special_tokens:
        token_upper = token.upper()
        if token_upper not in token_set:
            token_set.add(token_upper)
            unique_tokens.append(token_upper)
        else:
            logging.warning(f"Duplicate token after case normalization: {token}")
                
    print("\n\nMY UNIQUE TOKENS\n\n")
    print(unique_tokens)

    user_defined_symbols = ','.join(unique_tokens) if unique_tokens else ''

    # Retry loop to adapt vocabulary size when SentencePiece complains
    sp_train_kwargs = {
        'input': temp_text_file,
        'model_prefix': model_prefix,
        'max_sentence_length': 8000,
    }
    if character_coverage is not None:
        sp_train_kwargs['character_coverage'] = character_coverage
    if extra_options:
        sp_train_kwargs.update(extra_options)
    if user_defined_symbols:
        sp_train_kwargs['user_defined_symbols'] = user_defined_symbols

    current_vocab_size = vocab_size
    min_vocab_size = len(unique_tokens) + 1  # safety guard
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        logging.info(f"[Attempt {attempt}/{max_attempts}] Training SentencePiece with vocab_size={current_vocab_size}")
        try:
            spm.SentencePieceTrainer.train(vocab_size=current_vocab_size, **sp_train_kwargs)
            break
        except RuntimeError as exc:
            msg = str(exc)
            logging.error(f"SentencePiece training failed: {msg}")
            # Too small
            match_small = re.search(r"Vocabulary size is smaller than required_chars\. (\d+) vs (\d+)", msg)
            if match_small:
                requested = int(match_small.group(1))
                required = int(match_small.group(2))
                current_vocab_size = max(required + 1, current_vocab_size * 2)
                logging.info(f"Increasing vocab size from {requested} to {current_vocab_size}")
                continue
            # Too large
            match_large = re.search(r"Vocabulary size too high \((\d+)\).*value <= (\d+)", msg)
            if match_large:
                requested = int(match_large.group(1))
                limit = int(match_large.group(2))
                current_vocab_size = max(min(limit, current_vocab_size - 1), min_vocab_size)
                logging.info(f"Reducing vocab size from {requested} to {current_vocab_size}")
                continue
            # Unknown error – re-raise
            raise
    else:
        raise RuntimeError("SentencePiece training failed after multiple attempts; see logs for details.")
    
    # Step 4: Return the paths to the tokenizer model and vocab files
    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"

    logging.info(f"Tokenizer training completed")
    logging.info(f"Model file: {model_file}")
    logging.info(f"Vocab file: {vocab_file}")

    # Step 5: Create a vocab.txt file
    vocab_txt_file = os.path.join(tokenizer_folder, 'vocab.txt')
    logging.info(f"Creating vocab.txt file at {vocab_txt_file}")
    with open(vocab_file, 'r') as vf, open(vocab_txt_file, 'w') as vtf:
        for line in vf:
            token = line.split('\t')[0]
            vtf.write(token + '\n')
    
    logging.info(f"vocab.txt file created at {vocab_txt_file}")
    
    return model_file, vocab_file, vocab_txt_file


def extract_langs_and_special_tokens(
    manifest_filepath,
    special_token_prefixes=None,
    extra_manifest_paths=None,
    lang_field: str = 'lang',
):
    """
    Extracts unique language IDs and special tokens from one or more manifest files.
    """
    langs = set()
    counter = Counter()

    strip_chars = ".,;:!?\"'()[]{}"
    prefixes = tuple(prefix.upper() for prefix in (special_token_prefixes or []))

    manifest_paths = [manifest_filepath]
    if extra_manifest_paths:
        manifest_paths.extend(extra_manifest_paths)

    for path in manifest_paths:
        if not path:
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)

                    lang = data.get(lang_field)
                    if not lang and lang_field != 'lang':
                        lang = data.get('lang')
                    if lang:
                        langs.add(str(lang).upper())

                    text = data.get('text', '')
                    for word in text.split():
                        token = word.strip(strip_chars)
                        if not token:
                            continue
                        token_upper = token.upper()

                        if prefixes:
                            if any(token_upper.startswith(prefix) for prefix in prefixes):
                                counter[token_upper] += 1
                        else:
                            if token_upper.isupper() and '_' in token_upper:
                                counter[token_upper] += 1
        except FileNotFoundError:
            logging.warning(f"Manifest path not found while extracting languages: {path}")

    if counter:
        logging.info(f"Identified {len(counter)} candidate special tokens")

    special_tokens = [tok for tok, _ in counter.most_common()]

    if 'END' in special_tokens:
        special_tokens = [tok for tok in special_tokens if tok != 'END']
    special_tokens.append('END')

    return langs, special_tokens


def build_aggregate_vocab_from_tokenizers(tokenizer_langs_config):
    aggregate_vocab: List[str] = []
    seen_tokens = set()
    for lang, lang_cfg in tokenizer_langs_config.items():
        vocab_file = Path(lang_cfg['dir']) / 'vocab.txt'
        if vocab_file.exists():
            with open(vocab_file, 'r', encoding='utf-8') as vf:
                tokens = [line.split('\t')[0] for line in vf]
        else:
            sp_processor = spm.SentencePieceProcessor()
            sp_processor.Load(str(Path(lang_cfg['dir']) / 'tokenizer.model'))
            tokens = [sp_processor.id_to_piece(idx) for idx in range(sp_processor.get_piece_size())]
        for token in tokens:
            if token in seen_tokens:
                continue
            aggregate_vocab.append(token)
            seen_tokens.add(token)
    return aggregate_vocab


def _family_name_for_lang(lang: str) -> str:
    lang = lang.upper()
    return LANG_TO_FAMILY.get(lang, f"Singleton_{lang}")


def _sanitize_family_name(family: str) -> str:
    return family.replace(' ', '_').lower()


def train_aggregate_tokenizer(
    cfg,
    langs,
    special_tokens,
    extra_manifest_paths: List[str] = None,
    lang_field: str = 'lang',
):
    """
    Trains aggregate tokenizer components by creating one tokenizer per language family.
    Languages that do not belong to a configured family are treated as singleton families.
    """
    ordered_langs = sorted({lang.upper() for lang in langs})
    logging.info(f"Starting aggregate tokenizer training for languages: {ordered_langs}")
    logging.info(f"Special token count: {len(special_tokens)}")

    # Create a temporary directory to store family manifests
    temp_manifest_dir = Path(cfg.training.data_dir) / "temp_manifests"
    temp_manifest_dir.mkdir(exist_ok=True)

    for existing_manifest in temp_manifest_dir.glob("*_manifest.json"):
        try:
            existing_manifest.unlink()
        except OSError:
            logging.debug(f"Could not remove stale manifest {existing_manifest}")

    manifest_filepath = Path(cfg.training.data_dir) / cfg.training.train_manifest

    family_to_langs: Dict[str, Set[str]] = {}
    language_family_map: Dict[str, str] = {}
    family_files: Dict[str, IO[str]] = {}
    family_line_counts: Dict[str, int] = {}

    def _write_family_line(lang_key: str, raw_line: str) -> None:
        family = _family_name_for_lang(lang_key)
        language_family_map[lang_key] = family
        family_to_langs.setdefault(family, set()).add(lang_key)

        if family not in family_files:
            family_manifest_path = temp_manifest_dir / f"{family}_manifest.json"
            family_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            family_files[family] = open(family_manifest_path, "w", encoding='utf-8')
        family_files[family].write(raw_line)
        family_line_counts[family] = family_line_counts.get(family, 0) + 1

    try:
        with open(manifest_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                lang = data.get(lang_field)
                if not lang and lang_field != 'lang':
                    lang = data.get('lang')
                if not lang:
                    continue

                lang_key = str(lang).upper()
                _write_family_line(lang_key, line)

        if extra_manifest_paths:
            for extra_path in extra_manifest_paths:
                if not extra_path:
                    continue
                try:
                    with open(extra_path, 'r', encoding='utf-8') as extra_f:
                        for line in extra_f:
                            data = json.loads(line)
                            lang = data.get(lang_field)
                            if not lang and lang_field != 'lang':
                                lang = data.get('lang')
                            if not lang:
                                continue
                            lang_key = str(lang).upper()
                            _write_family_line(lang_key, line)
                except FileNotFoundError:
                    logging.warning(f"Extra manifest path not found while aggregating tokenizers: {extra_path}")
    finally:
        for handle in family_files.values():
            handle.close()

    for lang in ordered_langs:
        family = _family_name_for_lang(lang)
        language_family_map.setdefault(lang, family)
        family_to_langs.setdefault(family, set()).add(lang)

    dynamic_params = cfg.model.dynamic_tokenizer_params
    non_special_quota = dynamic_params.get('non_special_tokens_per_lang', 256)
    if non_special_quota < 0:
        non_special_quota = 0
    non_special_overrides = dynamic_params.get('non_special_tokens_per_lang_overrides', {})

    special_tokens_list = sorted(special_tokens)
    logging.info(
        "Shared special token count: %d; base non-special quota per family: %d",
        len(special_tokens_list),
        non_special_quota,
    )

    tokenizer_langs_config: Dict[str, Dict[str, str]] = {}
    lang_vocab_tokens: Dict[str, List[str]] = {}
    character_coverage = cfg.model.dynamic_tokenizer_params.get('character_coverage', None)
    coverage_overrides = cfg.model.dynamic_tokenizer_params.get('character_coverage_overrides', {})
    tokenizer_options = cfg.model.dynamic_tokenizer_params.get('tokenizer_options') or {}

    for family in sorted(family_to_langs.keys()):
        manifest_path = temp_manifest_dir / f"{family}_manifest.json"
        langs_in_family = sorted(family_to_langs[family])
        if family_line_counts.get(family, 0) == 0:
            logging.warning(f"No manifest entries for family {family}; skipping tokenizer training.")
            continue

        tokenizer_dir_name = f"{cfg.model.dynamic_tokenizer_params.dir_prefix}{_sanitize_family_name(family)}"
        tokenizer_dir = Path(cfg.model.model_root) / tokenizer_dir_name

        family_quota = non_special_overrides.get(family, non_special_quota)
        target_vocab_size = len(special_tokens_list) + family_quota
        logging.info(
            "Family %s (languages=%s): target vocab size %d (special=%d, quota=%d)",
            family,
            langs_in_family,
            target_vocab_size,
            len(special_tokens_list),
            family_quota,
        )

        family_options = dict(tokenizer_options.get(family, {})) if isinstance(tokenizer_options, dict) else {}
        if coverage_overrides.get(family, character_coverage) is None and 'character_coverage' in family_options:
            # allow override to explicitly disable coverage if set to null
            pass
        train_sentencepiece_tokenizer(
            manifest_file=str(manifest_path),
            tokenizer_folder=str(tokenizer_dir),
            special_tokens=special_tokens_list,
            vocab_size=target_vocab_size,
            character_coverage=coverage_overrides.get(family, character_coverage),
            extra_options=family_options,
        )

        vocab_file = tokenizer_dir / 'vocab.txt'
        if vocab_file.exists():
            with open(vocab_file, 'r', encoding='utf-8') as vf:
                family_tokens = [line.split('\t')[0] for line in vf]
        else:
            logging.warning(
                f"Missing vocab.txt for family {family} at {vocab_file}; using SentencePiece processor to recover tokens."
            )
            sp_processor = spm.SentencePieceProcessor()
            sp_processor.Load(str(tokenizer_dir / 'tokenizer.model'))
            family_tokens = [sp_processor.id_to_piece(idx) for idx in range(sp_processor.get_piece_size())]

        for lang in langs_in_family:
            lang_vocab_tokens[lang] = family_tokens
            tokenizer_langs_config[lang] = {
                'type': cfg.model.dynamic_tokenizer_params.type,
                'dir': str(tokenizer_dir),
            }

    missing_lang_configs = sorted(set(ordered_langs) - set(tokenizer_langs_config.keys()))
    if missing_lang_configs:
        logging.warning(
            "Skipped tokenizer training for languages without usable samples: %s",
            missing_lang_configs,
        )

    aggregate_vocab: List[str] = []
    seen_tokens: Set[str] = set()

    for lang in sorted(lang_vocab_tokens.keys()):
        for token in lang_vocab_tokens[lang]:
            if token in seen_tokens:
                continue
            aggregate_vocab.append(token)
            seen_tokens.add(token)

    logging.info(f"Aggregate vocabulary (dedup) size: {len(aggregate_vocab)}")

    aggregate_vocab_path = store_aggregate_vocabulary(cfg, aggregate_vocab)
    logging.info(f"Wrote deduplicated aggregate vocabulary to {aggregate_vocab_path}")

    return tokenizer_langs_config, special_tokens_list, aggregate_vocab, language_family_map


def setup_tokenizer(cfg, tokenizer_path):
    """Return tokenizer configuration based on aggregate tokenizer setting."""
    if cfg.model.get('use_aggregate_tokenizer', False):
        logging.info("Using aggregate tokenizer")
        return {'type': 'agg', 'langs': cfg.model.tokenizer_langs}

    logging.info("Using a single tokenizer")
    return {'dir': tokenizer_path, 'type': 'bpe', 'bpe_dropout': 0.05}


def _resolve_manifest_path(manifest_value: str, data_dir: Path) -> Path:
    if not manifest_value:
        raise ValueError("Manifest value is empty")
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = data_dir / manifest_value
    return manifest_path.expanduser().resolve()


def _relativize_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def validate_manifests(cfg) -> Dict[str, Dict[str, int]]:
    """Validate configured manifests and update config to point at cleaned copies."""
    data_dir = Path(cfg.training.data_dir).expanduser().resolve()
    manifest_specs = []
    workers = cfg.training.get('validation_workers', None)

    train_manifest = cfg.training.get('train_manifest')
    if train_manifest:
        manifest_specs.append(('train_manifest', None, _resolve_manifest_path(train_manifest, data_dir)))

    test_manifest = cfg.training.get('test_manifest')
    if test_manifest:
        manifest_specs.append(('test_manifest', None, _resolve_manifest_path(test_manifest, data_dir)))

    extra_manifests = cfg.training.get('tokenizer_extra_manifests') or []
    for idx, manifest_name in enumerate(extra_manifests):
        try:
            manifest_path = _resolve_manifest_path(manifest_name, data_dir)
        except ValueError:
            continue
        manifest_specs.append(('tokenizer_extra_manifests', idx, manifest_path))

    validated_cache: Dict[Path, Dict[str, int]] = {}
    results: Dict[str, Dict[str, int]] = {}

    for field, index, manifest_path in manifest_specs:
        if manifest_path in validated_cache:
            stats = validated_cache[manifest_path]
        else:
            output_path = determine_output_path(manifest_path)
            invalid_log_path = manifest_path.with_suffix('.invalid.json')
            logging.info(
                "Validating manifest field=%s idx=%s input=%s output=%s invalid_log=%s",
                field,
                index,
                manifest_path,
                output_path,
                invalid_log_path,
            )
            kept, dropped = run_validate_manifest(
                manifest_path,
                output_path,
                invalid_log_path,
                workers=workers,
            )
            stats = {'kept': kept, 'dropped': dropped, 'output_path': output_path}
            validated_cache[manifest_path] = stats

        if index is None:
            with open_dict(cfg):
                cfg.training[field] = _relativize_path(stats['output_path'], data_dir)
        else:
            manifests = list(cfg.training.get(field, []))
            if index < len(manifests):
                manifests[index] = _relativize_path(stats['output_path'], data_dir)
                with open_dict(cfg):
                    cfg.training[field] = manifests

        results[str(manifest_path)] = {'kept': stats['kept'], 'dropped': stats['dropped']}

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train aggregate tokenizers and ASR model")
    parser.add_argument(
        "--mode",
        choices=["both", "tokenizer", "train", "validate_data"],
        default="both",
        help="Choose whether to (re)train tokenizers, the ASR model, or both sequentially.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to the training YAML config. Defaults to config/config_peoplespeech.yml",
    )
    parser.add_argument(
        "--no-save-config",
        action="store_true",
        help="Do not persist tokenizer metadata back to the config file after training tokenizers.",
    )
    return parser.parse_args()


def save_updated_config(cfg, path):
    container = OmegaConf.to_container(cfg, resolve=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(container, f, sort_keys=False)


def train_model(cfg):
    lang_field = cfg.training.get('lang_field', 'lang')
    RobustAudioToBPEDataset.default_lang_field = lang_field
    tokenizer_langs = load_tokenizer_langs(cfg)
    if not isinstance(tokenizer_langs, dict) or not tokenizer_langs:
        raise RuntimeError(
            "No tokenizer_langs mapping found in config. Run with --mode tokenizer first, or add the mapping manually."
        )

    shared_special_tokens = load_shared_special_tokens(cfg)
    language_family_map = cfg.model.get('language_family_map', {})
    if language_family_map:
        logging.info(
            "Using language family assignments for %d languages",
            len(language_family_map),
        )
    aggregate_vocab = load_aggregate_vocabulary(cfg)
    if not aggregate_vocab:
        logging.warning("Aggregate vocabulary missing from config; rebuilding from tokenizer directories.")
        aggregate_vocab = build_aggregate_vocab_from_tokenizers(tokenizer_langs)
        store_aggregate_vocabulary(cfg, aggregate_vocab)

    lang_list = sorted(tokenizer_langs.keys())
    logging.info(f"Training model with aggregate tokenizer covering languages: {lang_list}")

    model_root = Path(cfg.model.model_root)
    model_path = model_root / cfg.model.model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find base model checkpoint at {model_path}")

    # Sanity-check tokenizer contents
    if shared_special_tokens:
        for lang, lang_cfg in tokenizer_langs.items():
            tok_model_path = Path(lang_cfg['dir']) / 'tokenizer.model'
            if not tok_model_path.exists():
                logging.warning(f"Tokenizer model missing for language {lang}: {tok_model_path}")
                continue
            processor = spm.SentencePieceProcessor()
            processor.Load(str(tok_model_path))
            unk_id = processor.unk_id() if hasattr(processor, 'unk_id') else 0
            missing = []
            for tok in shared_special_tokens:
                tok_id = processor.piece_to_id(tok)
                if tok_id == unk_id:
                    missing.append(tok)
            if missing:
                logging.warning(
                    f"Tokenizer {lang} missing {len(missing)} shared special tokens; first few: {missing[:10]}"
                )

    train_manifest = str(Path(cfg.training.data_dir) / cfg.training.train_manifest)
    val_manifest = str(Path(cfg.training.data_dir) / cfg.training.test_manifest)

    tokenizer_entry = {'type': 'agg', 'langs': tokenizer_langs}
    if shared_special_tokens:
        tokenizer_entry['special_tokens'] = shared_special_tokens

    base_cfg = ASRModel.restore_from(restore_path=str(model_path), return_config=True)
    with open_dict(base_cfg):
        base_cfg.use_keyword_loss = cfg.training.get('use_keyword_loss', False)
        base_cfg.keyword_loss_weight = cfg.training.get('keyword_loss_weight', 0.3)
        base_cfg.keyword_loss_warmup_steps = cfg.training.get('keyword_loss_warmup_steps', 0)

        base_cfg.train_ds.manifest_filepath = train_manifest
        base_cfg.train_ds.batch_size = cfg.training.batch_size
        base_cfg.train_ds.max_duration = cfg.training.max_duration
        base_cfg.train_ds.shuffle = True
        base_cfg.train_ds.is_tarred = False
        base_cfg.train_ds.tarred_audio_filepaths = None
        base_cfg.train_ds.num_workers = cfg.training.num_workers
        base_cfg.train_ds.pin_memory = cfg.training.pin_memory
        base_cfg.train_ds.lang_field = lang_field
        base_cfg.train_ds.return_sample_id = True # Needed for family loss weights

        base_cfg.validation_ds.manifest_filepath = val_manifest
        base_cfg.validation_ds.batch_size = cfg.training.batch_size
        base_cfg.validation_ds.max_duration = cfg.training.max_duration
        base_cfg.validation_ds.shuffle = False
        base_cfg.validation_ds.num_workers = cfg.training.num_workers
        base_cfg.validation_ds.pin_memory = cfg.training.pin_memory
        base_cfg.validation_ds.lang_field = lang_field
        base_cfg.validation_ds.return_sample_id = True

        if 'manifest_processor' not in base_cfg.train_ds:
            base_cfg.train_ds.manifest_processor = {}
        if 'additional_fields' not in base_cfg.train_ds.manifest_processor:
            base_cfg.train_ds.manifest_processor.additional_fields = []
        fields_to_add = {'lang'}
        if lang_field:
            fields_to_add.add(lang_field)
        for field_name in fields_to_add:
            if field_name and field_name not in base_cfg.train_ds.manifest_processor.additional_fields:
                base_cfg.train_ds.manifest_processor.additional_fields.append(field_name)

        base_cfg.train_ds.allowed_langs = lang_list
        base_cfg.validation_ds.allowed_langs = lang_list

        if 'augmentor' in base_cfg.train_ds:
            del base_cfg.train_ds.augmentor

    model = CustomEncDecCTCModelBPE.restore_from(str(model_path), override_config_path=base_cfg, strict=True)
    model.setup_custom_loss()

    tokenizer_cfg = OmegaConf.create(tokenizer_entry)
    #logging.info("Applying deduplicated aggregate tokenizer via change_vocabulary().")
    #model.change_vocabulary(tokenizer_cfg, 'agg')

    aggregate_vocab = list(model.decoder.vocabulary)
    store_aggregate_vocabulary(cfg, aggregate_vocab)

    with open_dict(model.cfg):
        model.cfg.tokenizer = tokenizer_entry
        model.cfg.train_ds.allowed_langs = lang_list
        model.cfg.validation_ds.allowed_langs = lang_list
        model.cfg.train_ds.lang_field = lang_field
        model.cfg.validation_ds.lang_field = lang_field
        model.cfg.validation_ds.return_sample_id = True
        model.cfg.decoder.vocabulary = aggregate_vocab
        model.cfg.decoder.num_classes = len(aggregate_vocab)

    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    model.setup_multiple_test_data(model.cfg.validation_ds)
    model._validation_dataset_ref = getattr(model._validation_dl, 'dataset', None)

    # --- Calculate and set language family loss weights ---
    if cfg.training.get('use_family_loss_weights'):
        logging.info("Calculating language family loss weights...")
        train_dataset = model._train_dl.dataset
        family_counts = defaultdict(int)
        for lang_id in train_dataset.language_ids:
            family = _family_name_for_lang(lang_id)
            family_counts[family] += 1
        
        total_samples = sum(family_counts.values())
        num_families = len(family_counts)
        
        if total_samples > 0 and num_families > 0:
            # Normalized inverse frequency weighting
            weights = {
                fam: total_samples / (num_families * count)
                for fam, count in family_counts.items()
            }
            logging.info("Calculated language family loss weights: %s", weights)
            model.set_family_loss_weights(weights)
        else:
            logging.warning("Could not calculate family loss weights: no samples or families found.")

    train_dataset = model._train_dl.dataset
    train_batch_sampler = BalancedLanguageBatchSampler(
        train_dataset,
        cfg.training.batch_size,
        lang_to_family_map=language_family_map,
    )
    model._train_dl = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=train_dataset.collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        )
        
    logging.info("Manually creating and injecting audio augmentor for training.")
    noise_perturb = WhiteNoisePerturbation(min_level=-90, max_level=-46)
    shift_perturb = ShiftPerturbation(min_shift_ms=100.0, max_shift_ms=500.0)
    augmentor = AudioAugmentor(perturbations=[
        (1.0, noise_perturb),
    (1.0, shift_perturb),
    ])
        
    if hasattr(model, '_train_dl') and model._train_dl is not None:
        model._train_dl.dataset.augmentor = augmentor
    else:
        logging.warning("Could not find the training dataloader to inject augmentor.")

    spec_cfg = cfg.training.get('spec_augment')
    if spec_cfg and hasattr(model.cfg, 'spec_augment'):
        with open_dict(model.cfg):
            if 'time_masks' in spec_cfg:
                model.cfg.spec_augment.time_masks = spec_cfg['time_masks']
            if 'time_width' in spec_cfg:
                model.cfg.spec_augment.time_width = spec_cfg['time_width']
        model.spec_augmentation = model.from_config_dict(model.cfg.spec_augment)

    optim_cfg = cfg.training.get('optim')
    if optim_cfg:
        with open_dict(model.cfg):
            if 'lr' in optim_cfg:
                model.cfg.optim.lr = optim_cfg['lr']
            if 'weight_decay' in optim_cfg:
                model.cfg.optim.weight_decay = optim_cfg['weight_decay']
            if 'sched' in optim_cfg and hasattr(model.cfg.optim, 'sched'):
                if 'warmup_steps' in optim_cfg['sched']:
                    model.cfg.optim.sched.warmup_steps = optim_cfg['sched']['warmup_steps']
    model.setup_optimization(model.cfg.optim)

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    configured_devices = cfg.training.get('devices', None)
    if accelerator == 'cpu':
        devices = configured_devices if isinstance(configured_devices, int) and configured_devices > 0 else 1
    else:
        devices = configured_devices if configured_devices is not None else -1

    tensorboard_logger = TensorBoardLogger(
        save_dir=str(Path(cfg.experiment.exp_dir).expanduser()),
        name=cfg.experiment.exp_name,
    )

    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=devices,
        max_steps=cfg.training.max_steps,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        enable_checkpointing=False,
        logger=tensorboard_logger,
        log_every_n_steps=50,
        use_distributed_sampler=False,
    )

    if accelerator == 'gpu':
        if devices == -1 or (isinstance(devices, int) and devices != 1):
            trainer_kwargs['strategy'] = 'ddp'

    val_interval = cfg.experiment.get('every_n_train_steps')
    if val_interval and val_interval > 0:
        trainer_kwargs['val_check_interval'] = val_interval

    accumulate_grad_batches = cfg.training.get('accumulate_grad_batches', None)
    if accumulate_grad_batches and accumulate_grad_batches > 1:
        trainer_kwargs['accumulate_grad_batches'] = accumulate_grad_batches

    trainer = pl.Trainer(**trainer_kwargs)
    model.set_trainer(trainer)

    callback_params = exp_manager.CallbackParams(
        monitor=cfg.experiment.monitor,
        mode=cfg.experiment.mode,
        always_save_nemo=cfg.experiment.always_save_nemo,
        save_top_k=cfg.experiment.get('save_top_k', 1),
    )
    exp_cfg = exp_manager.ExpManagerConfig(
        exp_dir=None,
        name=None,
        checkpoint_callback_params=callback_params,
        create_checkpoint_callback=True,
        create_tensorboard_logger=False,
        create_wandb_logger=False,
        create_mlflow_logger=False,
        create_dllogger_logger=False,
    )
    exp_cfg = OmegaConf.structured(exp_cfg)
    exp_manager.exp_manager(trainer, exp_cfg)

    logging.info("Starting model training...")
    logging.info("Model class: %s", model.__class__)
    logging.info("LightningModule class from pytorch_lightning: %s", pl.LightningModule)
    try:
        import lightning.pytorch as L

        logging.info(
            "LightningModule class from lightning.pytorch: %s", getattr(L, "LightningModule", None)
        )
        logging.info(
            "isinstance(model, pytorch_lightning.LightningModule): %s",
            isinstance(model, pl.LightningModule),
        )
        lightning_module_cls = getattr(L, "LightningModule", None)
        if lightning_module_cls is not None:
            logging.info(
                "isinstance(model, lightning.pytorch.LightningModule): %s",
                isinstance(model, lightning_module_cls),
            )
    except Exception as exc:
        logging.warning("Failed to inspect lightning module classes: %s", exc)

    trainer.fit(model)
    logging.info("Model training complete.")


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, 'config', 'config_peoplespeech.yml')
    config_file_path = args.config or default_config

    with open(config_file_path, 'r', encoding='utf-8') as f:
        cfg = OmegaConf.create(yaml.safe_load(f))

    lang_field = cfg.training.get('lang_field', 'lang')
    RobustAudioToBPEDataset.default_lang_field = lang_field
    language_families_cfg = cfg.model.get('language_families')
    language_families = OmegaConf.to_container(language_families_cfg, resolve=True) if language_families_cfg else None
    set_language_families(language_families)

    RobustAudioToBPEDataset.skip_audio_validation_default = bool(cfg.training.get('skip_audio_validation', False))
    RobustAudioToBPEDataset.keyphrase_oversample_factor_default = float(cfg.training.get('keyphrase_oversample_factor', 0.0))

    run_validation = args.mode in ('validate_data', 'both')
    run_tokenizers = args.mode in ('both', 'tokenizer')
    run_training = args.mode in ('both', 'train')

    if run_validation:
        logging.info("Validating manifests prior to downstream steps...")
        validation_stats = validate_manifests(cfg)
        for manifest, stats in validation_stats.items():
            logging.info(
                "Validated manifest %s: kept=%d dropped=%d",
                manifest,
                stats['kept'],
                stats['dropped'],
            )
        if not args.no_save_config:
            save_updated_config(cfg, config_file_path)
        if args.mode == 'validate_data':
            logging.info("Validation-only mode complete.")
            return

    if run_tokenizers:
        logging.info("Training tokenizer...")
        if cfg.model.get('use_aggregate_tokenizer', False):
            manifest_path = os.path.join(cfg.training.data_dir, cfg.training.train_manifest)
            special_token_prefixes = cfg.model.get('special_token_prefixes', None)
            extra_manifests: List[str] = []
            test_manifest = cfg.training.get('test_manifest')
            if test_manifest:
                test_manifest_path = os.path.join(cfg.training.data_dir, test_manifest)
                extra_manifests.append(test_manifest_path)

            tokenizer_extra_manifests = cfg.training.get('tokenizer_extra_manifests')
            if tokenizer_extra_manifests:
                for manifest_name in tokenizer_extra_manifests:
                    if not manifest_name:
                        continue
                    manifest_path_candidate = (
                        manifest_name
                        if os.path.isabs(manifest_name)
                        else os.path.join(cfg.training.data_dir, manifest_name)
                    )
                    extra_manifests.append(manifest_path_candidate)

            langs, special_tokens = extract_langs_and_special_tokens(
                manifest_path,
                special_token_prefixes,
                extra_manifest_paths=extra_manifests,
                lang_field=lang_field,
            )
            tokenizer_langs_config, shared_special_tokens, aggregate_vocab, language_family_map = train_aggregate_tokenizer(
                cfg,
                langs,
                special_tokens,
                extra_manifest_paths=extra_manifests,
                lang_field=lang_field,
            )
            store_tokenizer_langs(cfg, tokenizer_langs_config)
            store_shared_special_tokens(cfg, shared_special_tokens)
            store_aggregate_vocabulary(cfg, aggregate_vocab)
            with open_dict(cfg):
                cfg.model.language_family_map = language_family_map
            if not args.no_save_config:
                save_updated_config(cfg, config_file_path)
            logging.info("Tokenizer training complete. Continue with model training as needed.")
        else:
            tokenizer_path = os.path.join(cfg.model.model_root, cfg.model.new_tokenizer_folder)
            character_coverage = cfg.model.get('character_coverage', None)
            tokenizer_options = cfg.model.get('tokenizer_options') or {}
            train_sentencepiece_tokenizer(
                manifest_file=os.path.join(cfg.training.data_dir, cfg.training.train_manifest),
                tokenizer_folder=tokenizer_path,
                special_tokens=[],
                vocab_size=cfg.model.get('vocab_size', 1024),
                character_coverage=character_coverage,
                extra_options=tokenizer_options if isinstance(tokenizer_options, dict) else {},
            )
            logging.info("Tokenizer training complete. Continue with model training as needed.")

        if args.mode == 'tokenizer':
            return

    if run_training:
        train_model(cfg)


if __name__ == "__main__":
    main()
