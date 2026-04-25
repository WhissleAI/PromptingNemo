"""MetaTokenizer: lightweight wrapper using STT-meta-1B aggregate vocabulary.

Loads per-family SentencePiece models from the model directory and maps local
token IDs to the aggregate vocabulary, producing the same global IDs as the
ASR model. No NeMo dependency — just sentencepiece + vocab file.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import sentencepiece as spm

from promptingnemo.data.tag_parser import is_tag

log = logging.getLogger(__name__)


class MetaTokenizer:
    """Aggregate tokenizer for text tagger using STT-meta-1B vocabulary.

    Args:
        tokenizers_dir: Directory containing per-family tokenizer subdirectories
            (tokenizer_english/, tokenizer_indo_aryan/, etc.), each with tokenizer.model.
        aggregate_vocab_path: Path to aggregate_vocab.txt from the meta-1B model.
        default_family: Default language family for encoding (default: ENGLISH).
    """

    def __init__(
        self,
        tokenizers_dir: str,
        aggregate_vocab_path: str,
        default_family: str = 'ENGLISH',
    ):
        self.default_family = default_family

        self.vocabulary: List[str] = []
        self._token_to_id: Dict[str, int] = {}
        with open(aggregate_vocab_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                token = line.strip().split('\t')[0]
                self.vocabulary.append(token)
                self._token_to_id[token] = i

        self._vocab_size = len(self.vocabulary)
        self.blank_id = self._vocab_size

        self._sp_models: Dict[str, spm.SentencePieceProcessor] = {}
        self._local_to_global: Dict[str, Dict[int, int]] = {}

        tokenizers_path = Path(tokenizers_dir)
        for family_dir in sorted(tokenizers_path.iterdir()):
            if not family_dir.is_dir() or not family_dir.name.startswith('tokenizer_'):
                continue
            sp_model_path = family_dir / 'tokenizer.model'
            if not sp_model_path.exists():
                continue

            family = family_dir.name.replace('tokenizer_', '').upper()
            sp = spm.SentencePieceProcessor()
            sp.Load(str(sp_model_path))
            self._sp_models[family] = sp

            mapping: Dict[int, int] = {}
            unmapped = 0
            for local_id in range(sp.GetPieceSize()):
                piece = sp.IdToPiece(local_id)
                global_id = self._token_to_id.get(piece)
                if global_id is not None:
                    mapping[local_id] = global_id
                else:
                    unmapped += 1
            self._local_to_global[family] = mapping
            if unmapped > 0:
                log.debug("Family %s: %d/%d pieces unmapped to aggregate vocab",
                          family, unmapped, sp.GetPieceSize())

        self._tag_tokens: Set[str] = {
            token for token in self.vocabulary if is_tag(token)
        }

        log.info(
            "MetaTokenizer: %d vocab, %d families (%s), %d tag tokens, default=%s",
            self._vocab_size, len(self._sp_models),
            ', '.join(sorted(self._sp_models.keys())),
            len(self._tag_tokens), default_family,
        )

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def vocab_size_with_blank(self) -> int:
        return self._vocab_size + 1

    def encode_clean_text(self, text: str, family: str = None) -> List[int]:
        family = family or self.default_family
        sp = self._sp_models[family]
        mapping = self._local_to_global[family]
        local_ids = sp.EncodeAsIds(text)
        return [mapping[lid] for lid in local_ids if lid in mapping]

    def encode_tagged_text(self, tagged_text: str, family: str = None) -> List[int]:
        family = family or self.default_family
        words = tagged_text.split()
        ids: List[int] = []
        text_buffer: List[str] = []

        def flush():
            if text_buffer:
                text = ' '.join(text_buffer)
                ids.extend(self.encode_clean_text(text, family))
                text_buffer.clear()

        for word in words:
            if word in self._tag_tokens:
                flush()
                ids.append(self._token_to_id[word])
            else:
                text_buffer.append(word)
        flush()
        return ids

    def decode(self, ids: List[int]) -> str:
        if not ids:
            return ''
        tokens = [self.vocabulary[i] for i in ids if 0 <= i < self._vocab_size]
        return ''.join(tokens).replace('\u2581', ' ').strip()

    def ids_to_text(self, ids: List[int]) -> str:
        return self.decode(ids)
