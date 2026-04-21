"""Hybrid tokenizer for text CTC tagger: SentencePiece subwords + compositional tag pieces.

The output vocabulary is:
  [SP subwords (0..sp_size-1)] + [tag pieces (sp_size..sp_size+n_tags-1)] + [CTC blank]

During encoding, regular text words go through SentencePiece and tags are
decomposed into compositional pieces mapped to tag vocabulary IDs.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import sentencepiece as spm

from promptingnemo.data.tag_parser import decompose_tag, is_tag, recompose_tag

log = logging.getLogger(__name__)


class TextTaggerTokenizer:
    """Hybrid tokenizer: SentencePiece for text + fixed vocab for tag pieces."""

    def __init__(
        self,
        sp_model_path: str,
        tag_vocab_path: str,
    ):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(sp_model_path)
        self._sp_vocab_size = self.sp.GetPieceSize()

        self.tag_pieces = self._load_tag_vocab(tag_vocab_path)
        self.tag_to_id = {
            piece: self._sp_vocab_size + i
            for i, piece in enumerate(self.tag_pieces)
        }
        self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}

        self._vocab_size = self._sp_vocab_size + len(self.tag_pieces)
        self.blank_id = self._vocab_size

        log.info(
            "TextTaggerTokenizer: sp_vocab=%d, tag_pieces=%d, total=%d (blank=%d)",
            self._sp_vocab_size,
            len(self.tag_pieces),
            self._vocab_size,
            self.blank_id,
        )

    @staticmethod
    def _load_tag_vocab(path: str) -> List[str]:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return data.get('tag_pieces', [])

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def vocab_size_with_blank(self) -> int:
        return self._vocab_size + 1

    def encode_tagged_text(self, tagged_text: str) -> List[int]:
        """Tokenize tagged text: SP for regular words, compositional for tags.

        Segments the text into runs of regular words and tag tokens.
        Regular words are encoded with SentencePiece.
        Tags are decomposed and each piece maps to its tag vocabulary ID.
        Unknown tag pieces are skipped with a warning.
        """
        words = tagged_text.split()
        ids: List[int] = []
        text_buffer: List[str] = []

        def flush_text():
            if text_buffer:
                text = ' '.join(text_buffer)
                sp_ids = self.sp.EncodeAsIds(text)
                ids.extend(sp_ids)
                text_buffer.clear()

        for word in words:
            if is_tag(word):
                flush_text()
                pieces = decompose_tag(word)
                for piece in pieces:
                    pid = self.tag_to_id.get(piece)
                    if pid is not None:
                        ids.append(pid)
                    else:
                        log.debug("Unknown tag piece skipped: %s (from %s)", piece, word)
            else:
                text_buffer.append(word)

        flush_text()
        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to tagged text."""
        parts: List[str] = []
        sp_buffer: List[int] = []

        def flush_sp():
            if sp_buffer:
                text = self.sp.DecodeIds(sp_buffer)
                parts.append(text)
                sp_buffer.clear()

        for token_id in ids:
            if token_id == self.blank_id:
                continue
            if token_id in self.id_to_tag:
                flush_sp()
                parts.append(self.id_to_tag[token_id])
            elif 0 <= token_id < self._sp_vocab_size:
                sp_buffer.append(token_id)
            # else: unknown id, skip

        flush_sp()

        # Reconstruct: join parts, collapse whitespace
        raw = ' '.join(parts)
        return ' '.join(raw.split())

    def ids_to_text(self, ids: List[int]) -> str:
        return self.decode(ids)

    def save(self, output_dir: str):
        """Save tokenizer config for reloading."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            'sp_model_path': str(self.sp.serialized_model_proto()),
            'tag_pieces': self.tag_pieces,
            'sp_vocab_size': self._sp_vocab_size,
        }
        with open(out / 'text_tagger_tokenizer.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
