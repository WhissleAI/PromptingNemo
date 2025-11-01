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
import pytorch_lightning as pl
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
import torch.nn as nn
import torch.nn.functional as F
import nemo
from nemo.collections.asr.data import audio_to_text
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
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
# No longer need the manifest processor import
# from nemo.collections.asr.data.audio_to_text import ASRManifestProcessor as ManifestProcessor
from collections import Counter
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


def set_language_families(language_families: Dict[str, List[str]]) -> None:
    """Configure language family mapping from config."""

    if not language_families:
        raise ValueError("language_families mapping is required but was empty")

    normalized: Dict[str, List[str]] = {}
    mapping: Dict[str, str] = {}

    for family, langs in language_families.items():
        if not isinstance(langs, list):
            raise ValueError(f"Expected list of languages for family '{family}', got {type(langs)}")
        normalized_langs = []
        for lang in langs:
            if not lang:
                continue
            lang_upper = str(lang).upper()
            normalized_langs.append(lang_upper)
            mapping[lang_upper] = family
        normalized[family] = normalized_langs

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

    def set_keyword_token_ids(self, keyword_ids):
        self.keyword_token_ids = set(keyword_ids)
        logging.info(f"Set {len(self.keyword_token_ids)} keyword token IDs for custom loss.")

    def training_step(self, batch, batch_idx):
        audio_signal, audio_signal_len, transcript, transcript_len = batch
        log_probs, encoded_len, greedy_predictions = self.forward(
            input_signal=audio_signal, input_signal_length=audio_signal_len
        )

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )

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


class BalancedLanguageBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, temperature=0.2, seed=42, lang_to_family_map: Dict[str, str] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.temperature = temperature
        self.seed = seed
        self.epoch = 0
        self.lang_to_family_map = {k.upper(): v for k, v in (lang_to_family_map or {}).items()}

        if not dist.is_available() or not dist.is_initialized():
            self.world_size = 1
            self.rank = 0
        else:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        
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
        
        # Calculate sampling probabilities with temperature
        total_samples = len(self.dataset)
        weights = np.array([count / total_samples for count in self.family_counts.values()]) if total_samples > 0 else np.array([])
        if weights.size > 0:
            temp_weights = weights ** (1 / self.temperature)
            self.family_sample_probs = temp_weights / np.sum(temp_weights)
        else:
            self.family_sample_probs = np.array([])
        
        family_indices_map = {family: [] for family in self.families}
        for idx, family in enumerate(self.sample_families):
            if family in family_indices_map:
                family_indices_map[family].append(idx)
        self.family_indices = {
            family: np.array(indices, dtype=np.uint32) for family, indices in family_indices_map.items()
        }
        
        self.num_samples = len(self.dataset)
        self.num_batches_per_epoch = self.num_samples // self.batch_size
        
        # Adjust for distributed training
        self.num_samples_per_rank = self.num_samples // self.world_size
        if self.num_samples % self.world_size != 0:
            self.num_samples_per_rank += 1
            
    def __iter__(self):
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
                indices = np.random.choice(self.family_indices[family], num_family_samples, replace=replace)
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
        num_batches = self.num_samples // self.batch_size
        return num_batches // self.world_size

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

    def __init__(self, manifest_filepath, *args, **kwargs):
        allowed_langs = kwargs.pop('allowed_langs', None)
        if allowed_langs is not None:
            allowed_langs = {lang.upper() for lang in allowed_langs}

        # Build a filtered temporary manifest that keeps only entries with a language tag
        kept_lines = 0
        skipped_lines = 0
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
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed manifest line: {raw_line[:100]}...")
                    skipped_lines += 1
                    continue

                audio_file = data.get('audio_filepath')
                if not audio_file:
                    logging.warning("Skipping manifest entry without audio_filepath")
                    skipped_lines += 1
                    continue

                if not skip_audio_validation:
                    audio_path = Path(audio_file)
                    if not audio_path.exists():
                        logging.warning(f"Skipping missing audio file for manifest entry: {audio_file}")
                        skipped_lines += 1
                        continue

                lang_value = data.get('lang')
                if not lang_value:
                    skipped_lines += 1
                    continue

                lang_value = str(lang_value).upper()
                if allowed_langs is not None and lang_value not in allowed_langs:
                    continue

                data['lang'] = lang_value
                if not skip_audio_validation:
                    try:
                        AudioSegment.from_file(audio_file)
                    except Exception as audio_exc:
                        logging.warning(
                            f"Skipping audio file that failed to decode ({audio_file}): {audio_exc}"
                        )
                        skipped_lines += 1
                        continue

                tmp_manifest.write(json.dumps(data, ensure_ascii=False) + '\n')
                kept_lines += 1

                if progress_every > 0 and processed_lines % progress_every == 0:
                    logging.info(
                        "Manifest prep progress [%s]: processed=%d kept=%d skipped=%d",
                        manifest_filepath,
                        processed_lines,
                        kept_lines,
                        skipped_lines,
                    )
        tmp_manifest.close()

        if kept_lines == 0:
            raise RuntimeError(
                "No usable manifest entries remained after filtering for language IDs. Ensure your manifests contain a 'lang' field for each sample."
            )

        if skipped_lines > 0:
            logging.info(f"Filtered out {skipped_lines} manifest lines without a language tag while preparing aggregate dataset.")
        logging.info(
            "Finished preparing dataset manifest %s: total_rows=%d kept=%d skipped=%d",
            manifest_filepath,
            processed_lines,
            kept_lines,
            skipped_lines,
        )

        # Initialize the parent dataset with the filtered manifest
        super().__init__(manifest_filepath=str(tmp_path), *args, **kwargs)

        # Track the temporary manifest for optional cleanup
        self._filtered_manifest_path = tmp_path

        # Build language ids aligned with the filtered manifest collection
        self.language_ids = []
        new_collection = []
        for item in self.manifest_processor.collection:
            lang_id = getattr(item, 'lang', None)
            if not lang_id:
                continue
            if allowed_langs is not None and lang_id.upper() not in allowed_langs:
                continue
            lang_id = lang_id.upper()
            self.language_ids.append(lang_id)
            new_collection.append(item)

        self.manifest_processor.collection = new_collection

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
    for item in batch:
        if item is not None:
            has_weight = len(item) > 4
            break

    original_len = len(batch)
    batch = [item for item in batch if item is not None]
    if len(batch) < original_len:
        logging.warning(f"Skipped {original_len - len(batch)} samples in a batch.")

    if not batch:
        if has_weight:
            return (
                torch.empty(0, 0, dtype=torch.float32),
                torch.tensor([], dtype=torch.long),
                torch.empty(0, 0, dtype=torch.long),
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.float32),
            )
        else:
            return (
                torch.empty(0, 0, dtype=torch.float32),
                torch.tensor([], dtype=torch.long),
                torch.empty(0, 0, dtype=torch.long),
                torch.tensor([], dtype=torch.long),
            )
    
    if isinstance(batch[0], list):
        batch = [item for sublist in batch for item in sublist]

    audio_signal, audio_lengths, transcript, transcript_lengths = [], [], [], []
    if has_weight:
        weights = []

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
        if has_weight:
            weights.append(sample[4])

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

    if has_weight:
        weights = torch.tensor(weights, dtype=torch.float32)
        return audio_signal, audio_lengths, transcript, transcript_lengths, weights

    return audio_signal, audio_lengths, transcript, transcript_lengths


# Monkey-patch the collate function
audio_to_text._speech_collate_fn = patched_speech_collate_fn


def train_sentencepiece_tokenizer(manifest_file, tokenizer_folder, special_tokens=None, vocab_size=5000, character_coverage=0.9995):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Starting the tokenizer training process")
    logging.info(f"Requested vocab_size={vocab_size}, character_coverage={character_coverage}")

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
        'character_coverage': character_coverage,
        'max_sentence_length': 8000,
    }
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


def extract_langs_and_special_tokens(manifest_filepath, special_token_prefixes=None, extra_manifest_paths=None):
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


def train_aggregate_tokenizer(cfg, langs, special_tokens, extra_manifest_paths: List[str] = None):
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

    special_tokens_list = sorted(special_tokens)
    logging.info(
        "Shared special token count: %d; base non-special quota per family: %d",
        len(special_tokens_list),
        non_special_quota,
    )

    tokenizer_langs_config: Dict[str, Dict[str, str]] = {}
    lang_vocab_tokens: Dict[str, List[str]] = {}

    for family in sorted(family_to_langs.keys()):
        manifest_path = temp_manifest_dir / f"{family}_manifest.json"
        langs_in_family = sorted(family_to_langs[family])
        if family_line_counts.get(family, 0) == 0:
            logging.warning(f"No manifest entries for family {family}; skipping tokenizer training.")
            continue

        tokenizer_dir_name = f"{cfg.model.dynamic_tokenizer_params.dir_prefix}{_sanitize_family_name(family)}"
        tokenizer_dir = Path(cfg.model.model_root) / tokenizer_dir_name

        target_vocab_size = len(special_tokens_list) + non_special_quota
        logging.info(
            "Family %s (languages=%s): target vocab size %d (special=%d, quota=%d)",
            family,
            langs_in_family,
            target_vocab_size,
            len(special_tokens_list),
            non_special_quota,
        )

        train_sentencepiece_tokenizer(
            manifest_file=str(manifest_path),
            tokenizer_folder=str(tokenizer_dir),
            special_tokens=special_tokens_list,
            vocab_size=target_vocab_size,
            character_coverage=cfg.model.dynamic_tokenizer_params.character_coverage,
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
        base_cfg.train_ds.lang_field = 'lang'

        base_cfg.validation_ds.manifest_filepath = val_manifest
        base_cfg.validation_ds.batch_size = cfg.training.batch_size
        base_cfg.validation_ds.max_duration = cfg.training.max_duration
        base_cfg.validation_ds.shuffle = False
        base_cfg.validation_ds.num_workers = cfg.training.num_workers
        base_cfg.validation_ds.pin_memory = cfg.training.pin_memory
        base_cfg.validation_ds.lang_field = 'lang'

        if 'manifest_processor' not in base_cfg.train_ds:
            base_cfg.train_ds.manifest_processor = {}
        if 'additional_fields' not in base_cfg.train_ds.manifest_processor:
            base_cfg.train_ds.manifest_processor.additional_fields = []
        if 'lang' not in base_cfg.train_ds.manifest_processor.additional_fields:
            base_cfg.train_ds.manifest_processor.additional_fields.append('lang')

        base_cfg.train_ds.allowed_langs = lang_list
        base_cfg.validation_ds.allowed_langs = lang_list

        if 'augmentor' in base_cfg.train_ds:
            del base_cfg.train_ds.augmentor

    model = ASRModel.restore_from(str(model_path), override_config_path=base_cfg, strict=True)
    model.__class__ = CustomEncDecCTCModelBPE
    model.setup_custom_loss()

    tokenizer_cfg = OmegaConf.create(tokenizer_entry)
    logging.info("Applying deduplicated aggregate tokenizer via change_vocabulary().")
    model.change_vocabulary(tokenizer_cfg, 'agg')

    aggregate_vocab = list(model.decoder.vocabulary)
    store_aggregate_vocabulary(cfg, aggregate_vocab)

    with open_dict(model.cfg):
        model.cfg.tokenizer = tokenizer_entry
        model.cfg.train_ds.allowed_langs = lang_list
        model.cfg.validation_ds.allowed_langs = lang_list
        model.cfg.train_ds.lang_field = 'lang'
        model.cfg.validation_ds.lang_field = 'lang'
        model.cfg.decoder.vocabulary = aggregate_vocab
        model.cfg.decoder.num_classes = len(aggregate_vocab)

    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    model.setup_multiple_test_data(model.cfg.validation_ds)

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

    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=devices,
        max_steps=cfg.training.max_steps,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        enable_checkpointing=False,
        logger=False,
        log_every_n_steps=50,
        use_distributed_sampler=False,
    )

    if accelerator == 'gpu':
        if devices == -1 or (isinstance(devices, int) and devices != 1):
            trainer_kwargs['strategy'] = 'ddp'

    val_interval = cfg.experiment.get('every_n_train_steps')
    if val_interval and val_interval > 0:
        trainer_kwargs['val_check_interval'] = val_interval

    trainer = pl.Trainer(**trainer_kwargs)
    model.set_trainer(trainer)

    callback_params = exp_manager.CallbackParams(
        monitor=cfg.experiment.monitor,
        mode=cfg.experiment.mode,
        always_save_nemo=cfg.experiment.always_save_nemo,
        save_top_k=cfg.experiment.get('save_top_k', 1),
    )
    exp_cfg = exp_manager.ExpManagerConfig(
        exp_dir=cfg.experiment.exp_dir,
        name=cfg.experiment.exp_name,
        checkpoint_callback_params=callback_params,
        create_checkpoint_callback=True,
    )
    exp_cfg = OmegaConf.structured(exp_cfg)
    exp_manager.exp_manager(trainer, exp_cfg)

    logging.info("Starting model training...")
    trainer.fit(model)
    logging.info("Model training complete.")


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, 'config', 'config_peoplespeech.yml')
    config_file_path = args.config or default_config

    with open(config_file_path, 'r', encoding='utf-8') as f:
        cfg = OmegaConf.create(yaml.safe_load(f))

    language_families_cfg = cfg.model.get('language_families')
    language_families = OmegaConf.to_container(language_families_cfg, resolve=True) if language_families_cfg else None
    set_language_families(language_families)

    RobustAudioToBPEDataset.skip_audio_validation_default = bool(cfg.training.get('skip_audio_validation', False))

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
            )
            tokenizer_langs_config, shared_special_tokens, aggregate_vocab, language_family_map = train_aggregate_tokenizer(
                cfg,
                langs,
                special_tokens,
                extra_manifest_paths=extra_manifests,
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
            train_sentencepiece_tokenizer(
                manifest_file=os.path.join(cfg.training.data_dir, cfg.training.train_manifest),
                tokenizer_folder=tokenizer_path,
                special_tokens=[],
                vocab_size=cfg.model.get('vocab_size', 1024),
                character_coverage=cfg.model.get('character_coverage', 0.9995),
            )
            logging.info("Tokenizer training complete. Continue with model training as needed.")

        if args.mode == 'tokenizer':
            return

    if run_training:
        train_model(cfg)


if __name__ == "__main__":
    main()
