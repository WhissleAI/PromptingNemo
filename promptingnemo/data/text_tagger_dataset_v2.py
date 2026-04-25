"""Dataset for text CTC tagger v2 (subword input, meta-1B aggregate tokenizer).

Uses the STT-meta-1B aggregate tokenizer for both input and output encoding,
so output token IDs match the ASR model exactly. The model only needs to learn
tag insertion — tokenization is handled by the shared SentencePiece model.

Reads tagged text from JSONL manifests. For each sample:
  - Parses tagged text into (clean_text, tagged_text)
  - Encodes clean_text with aggregate tokenizer → input subword IDs
  - Encodes tagged_text with aggregate tokenizer → target subword + tag IDs
"""

import json
import logging
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from promptingnemo.data.tag_parser import parse_tagged_text

log = logging.getLogger(__name__)


class TextTaggerDatasetV2(Dataset):
    """Dataset for text CTC tagger v2 with aggregate tokenizer.

    Args:
        manifest_filepath: Path to JSONL manifest with 'text' field.
        tokenizer: MetaTokenizer instance with encode_clean_text/encode_tagged_text.
        max_subword_length: Maximum subword token count for clean text input.
        upsample_factor: Encoder upsampling factor; samples where
            n_subwords * upsample_factor < target_tokens are skipped.
    """

    def __init__(
        self,
        manifest_filepath: str,
        tokenizer,
        max_subword_length: int = 256,
        upsample_factor: int = 3,
    ):
        self.tokenizer = tokenizer
        self.max_subword_length = max_subword_length
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

                input_ids = tokenizer.encode_clean_text(clean_text)
                if len(input_ids) > max_subword_length:
                    skipped_long += 1
                    continue

                target_ids = tokenizer.encode_tagged_text(tagged_text)
                if not target_ids:
                    skipped_empty += 1
                    continue

                upsampled_len = len(input_ids) * upsample_factor
                if upsampled_len < len(target_ids):
                    skipped_ctc += 1
                    continue

                self.samples.append({
                    'input_ids': input_ids,
                    'target_ids': target_ids,
                })

        log.info(
            "TextTaggerDatasetV2: %d samples from %s "
            "(skipped: %d long, %d CTC-constraint, %d empty)",
            len(self.samples), manifest_filepath,
            skipped_long, skipped_ctc, skipped_empty,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        input_tensor = torch.tensor(sample['input_ids'], dtype=torch.long)
        input_len = torch.tensor(len(sample['input_ids']), dtype=torch.long)
        target_tensor = torch.tensor(sample['target_ids'], dtype=torch.long)
        target_len = torch.tensor(len(sample['target_ids']), dtype=torch.long)

        return input_tensor, input_len, target_tensor, target_len


def text_tagger_v2_collate_fn(batch, pad_id: int = 0):
    """Collate for TextTaggerDatasetV2. Pads input and target sequences."""
    batch = [item for item in batch if item is not None]
    if not batch:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty, empty, empty

    input_seqs, input_lens, target_seqs, target_lens = zip(*batch)

    input_lens = torch.stack(list(input_lens))
    target_lens = torch.stack(list(target_lens))
    input_batch = nn.utils.rnn.pad_sequence(
        list(input_seqs), batch_first=True, padding_value=pad_id
    )
    target_batch = nn.utils.rnn.pad_sequence(
        list(target_seqs), batch_first=True, padding_value=pad_id
    )

    return input_batch, input_lens, target_batch, target_lens
