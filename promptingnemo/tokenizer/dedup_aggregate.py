"""Deduplicated Aggregate Tokenizer that patches NeMo's AggregateTokenizer."""

import logging
from typing import Dict, List, Set, Union

import numpy as np
import torch

try:
    from nemo.collections.common.tokenizers import aggregate_tokenizer as nemo_agg
    from nemo.collections.common import tokenizers as nemo_tokenizers_pkg
except ImportError:
    nemo_agg = None
    nemo_tokenizers_pkg = None


# Sentinel: replaced after class definition if NeMo is available
class DedupAggregateTokenizer:
    """Placeholder that raises a helpful error when NeMo is not installed."""
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "DedupAggregateTokenizer requires NeMo. Install with: pip install promptingnemo[train]"
        )


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
            self._extended_special_tokens: Dict[str, int] = {}
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

        def extend_vocabulary(self, new_tokens: List[str]):
            added = []
            for token in new_tokens:
                if token in self._token_to_global:
                    continue
                global_id = len(self.vocabulary)
                vocab_token = '\u2581' + token
                self.vocabulary.append(vocab_token)
                self._token_to_global[vocab_token] = global_id
                self._extended_special_tokens[token] = global_id
                self.lang_candidates_by_global[global_id] = set()
                self.langs_by_token_id[global_id] = None
                self.tokenizers_by_token_id[global_id] = None
                self.offset_token_ids_by_token_id[global_id] = None
                added.append(token)
            self.vocab_size = len(self.vocabulary)
            self.tokenizer = DummyTokenizer(self.vocabulary)
            if added:
                logging.info(f"Extended tokenizer with {len(added)} new special tokens. Vocab: {self.vocab_size}")

        def text_to_tokens(self, text, lang_id=None):
            if lang_id is None:
                lang_id = next(iter(self.tokenizers_dict))
            tokenizer = self.tokenizers_dict[lang_id]
            return tokenizer.text_to_tokens(text)

        def text_to_ids(self, text, lang_id=None):
            if lang_id is None:
                lang_id = next(iter(self.tokenizers_dict))
            tokenizer = self.tokenizers_dict[lang_id]
            lang_map = self.lang_local_to_global[lang_id]
            if not self._extended_special_tokens:
                token_ids = tokenizer.text_to_ids(text)
                return [lang_map[t] for t in token_ids]
            words = text.split()
            global_ids = []
            buffer = []
            for word in words:
                if word in self._extended_special_tokens:
                    if buffer:
                        local_ids = tokenizer.text_to_ids(' '.join(buffer))
                        global_ids.extend(lang_map[t] for t in local_ids)
                        buffer = []
                    global_ids.append(self._extended_special_tokens[word])
                else:
                    buffer.append(word)
            if buffer:
                local_ids = tokenizer.text_to_ids(' '.join(buffer))
                global_ids.extend(lang_map[t] for t in local_ids)
            return global_ids

        def tokens_to_text(self, tokens, lang_id=None):
            if lang_id is not None:
                tokenizer = self.tokenizers_dict[lang_id]
                return tokenizer.decode_pieces(tokens)
            return ''.join(tokens).replace('\u2581', ' ').strip()

        def token_to_id(self, token, lang_id):
            tokenizer = self.tokenizers_dict[lang_id]
            local_id = tokenizer.token_to_id(token)
            if local_id >= 0:
                return self.lang_local_to_global[lang_id][local_id]
            return self._token_to_global[token]

        def ids_to_tokens(self, ids):
            if isinstance(ids, (np.ndarray, torch.Tensor)):
                ids = ids.tolist()
            return [self.vocabulary[i] for i in ids if i < len(self.vocabulary)]

        def ids_to_text(self, ids):
            if isinstance(ids, (np.ndarray, torch.Tensor)):
                ids = ids.tolist()
            tokens = [self.vocabulary[i] for i in ids if i < len(self.vocabulary)]
            return ''.join(tokens).replace('\u2581', ' ')

        def ids_to_text_and_langs(self, ids):
            result = []
            for idx in ids:
                if idx >= len(self.vocabulary):
                    continue
                token = self.vocabulary[idx]
                lang = self.langs_by_token_id.get(idx)
                result.append({'char': token.replace('\u2581', ' ').strip(), 'lang': lang})
            return result

        def ids_to_words_and_langs(self, ids):
            words_and_langs = []
            current_ids = []
            for idx in ids:
                if idx >= len(self.vocabulary):
                    continue
                token = self.vocabulary[idx]
                if token.startswith('\u2581') and current_ids:
                    word = ''.join(self.vocabulary[i] for i in current_ids).replace('\u2581', ' ').strip()
                    lang = self.ids_to_lang(current_ids)
                    words_and_langs.append({'word': word, 'lang': lang})
                    current_ids = []
                current_ids.append(idx)
            if current_ids:
                word = ''.join(self.vocabulary[i] for i in current_ids).replace('\u2581', ' ').strip()
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

    # Monkey-patch NeMo's aggregate tokenizer with the dedup version
    nemo_agg.AggregateTokenizer = DedupAggregateTokenizer
    if nemo_tokenizers_pkg is not None:
        nemo_tokenizers_pkg.AggregateTokenizer = DedupAggregateTokenizer
