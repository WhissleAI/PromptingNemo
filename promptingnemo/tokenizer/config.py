"""Language family configuration and tokenizer metadata persistence."""

import logging
import yaml
from pathlib import Path
from typing import Dict, List

from omegaconf import OmegaConf, open_dict


# ---------------------------------------------------------------------------
# Module-level state: language family mapping
# ---------------------------------------------------------------------------
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
