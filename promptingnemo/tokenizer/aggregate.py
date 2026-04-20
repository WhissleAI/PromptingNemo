"""Aggregate tokenizer training: per-family SentencePiece models + dedup merge."""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, IO, List, Set

import sentencepiece as spm
from omegaconf import open_dict

from promptingnemo.tokenizer.config import (
    LANG_TO_FAMILY,
    store_aggregate_vocabulary,
)
from promptingnemo.tokenizer.sentencepiece import train_sentencepiece_tokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _family_name_for_lang(lang: str) -> str:
    lang = lang.upper()
    return LANG_TO_FAMILY.get(lang, f"Singleton_{lang}")


def _sanitize_family_name(family: str) -> str:
    return family.replace(' ', '_').lower()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
