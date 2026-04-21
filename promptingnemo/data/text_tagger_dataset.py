"""Dataset for text CTC tagger training.

Reads tagged text from JSONL manifests. For each sample:
  - Parses tagged text into (clean_text, tagged_text)
  - Converts clean_text to character IDs (input to encoder)
  - Converts tagged_text to subword+tag token IDs (CTC target)
"""

import json
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from promptingnemo.data.tag_parser import parse_tagged_text

log = logging.getLogger(__name__)

PAD_CHAR = '<pad>'
UNK_CHAR = '<unk>'
SPECIAL_CHARS = [PAD_CHAR, UNK_CHAR]


class TextTaggerDataset(Dataset):
    """Dataset for text CTC tagger.

    Args:
        manifest_filepath: Path to JSONL manifest with 'text' field.
        tokenizer: TextTaggerTokenizer instance (encodes tagged text to token IDs).
        char_to_id: Mapping from character to integer ID.
        max_text_length: Maximum character length of clean text (longer samples skipped).
        upsample_factor: Encoder upsampling factor; samples where
            chars * upsample_factor < target_tokens are skipped.
    """

    def __init__(
        self,
        manifest_filepath: str,
        tokenizer,
        char_to_id: Dict[str, int],
        max_text_length: int = 512,
        upsample_factor: int = 2,
    ):
        self.tokenizer = tokenizer
        self.char_to_id = char_to_id
        self.unk_id = char_to_id.get(UNK_CHAR, 1)
        self.pad_id = char_to_id.get(PAD_CHAR, 0)
        self.max_text_length = max_text_length
        self.upsample_factor = upsample_factor

        self.samples: List[Dict] = []
        skipped_long = 0
        skipped_ctc = 0
        skipped_empty = 0

        with open(manifest_filepath, encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                text = entry.get('text', '')
                if not text.strip():
                    skipped_empty += 1
                    continue

                clean_text, tagged_text = parse_tagged_text(text)
                if not clean_text.strip():
                    skipped_empty += 1
                    continue

                if len(clean_text) > max_text_length:
                    skipped_long += 1
                    continue

                token_ids = tokenizer.encode_tagged_text(tagged_text)
                if not token_ids:
                    skipped_empty += 1
                    continue

                input_len = len(clean_text) * upsample_factor
                if input_len < len(token_ids):
                    skipped_ctc += 1
                    continue

                self.samples.append({
                    'clean_text': clean_text,
                    'token_ids': token_ids,
                })

        log.info(
            "TextTaggerDataset: %d samples from %s "
            "(skipped: %d long, %d CTC-constraint, %d empty)",
            len(self.samples), manifest_filepath,
            skipped_long, skipped_ctc, skipped_empty,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        clean_text = sample['clean_text']
        token_ids = sample['token_ids']

        char_ids = [self.char_to_id.get(ch, self.unk_id) for ch in clean_text]

        char_tensor = torch.tensor(char_ids, dtype=torch.long)
        char_len = torch.tensor(len(char_ids), dtype=torch.long)
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        token_len = torch.tensor(len(token_ids), dtype=torch.long)

        return char_tensor, char_len, token_tensor, token_len


def text_tagger_collate_fn(batch, pad_char_id: int = 0, pad_token_id: int = 0):
    """Collate for TextTaggerDataset. Pads char and token sequences."""
    batch = [item for item in batch if item is not None]
    if not batch:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty, empty, empty

    char_seqs, char_lens, token_seqs, token_lens = zip(*batch)

    char_lens = torch.stack(list(char_lens))
    token_lens = torch.stack(list(token_lens))
    char_batch = nn.utils.rnn.pad_sequence(
        list(char_seqs), batch_first=True, padding_value=pad_char_id
    )
    token_batch = nn.utils.rnn.pad_sequence(
        list(token_seqs), batch_first=True, padding_value=pad_token_id
    )

    return char_batch, char_lens, token_batch, token_lens
