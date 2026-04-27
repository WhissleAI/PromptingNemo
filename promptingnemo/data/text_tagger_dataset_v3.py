"""Dataset for text CTC tagger v3 (XLM-R input, aggregate vocab output).

Input: clean text -> XLM-R tokenizer -> input_ids + attention_mask
Output: tagged text -> chunked trailing tags -> MetaTokenizer -> target_ids

Trailing sentence tags (AGE, GENDER, EMOTION, INTENT) are repeated at
regular chunk intervals in the target, teaching the causal model to emit
them at every streaming buffer boundary.
"""

import json
import logging
import random
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from promptingnemo.data.tag_parser import parse_tagged_text
from promptingnemo.data.chunked_tag_utils import build_chunked_target

log = logging.getLogger(__name__)


class TextTaggerDatasetV3(Dataset):
    def __init__(
        self,
        manifest_filepath: str,
        input_tokenizer,
        output_tokenizer,
        max_input_length: int = 128,
        upsample_factor: int = 3,
        chunk_size: int = 8,
        chunk_jitter: int = 3,
    ):
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.max_input_length = max_input_length
        self.upsample_factor = upsample_factor
        self.chunk_size = chunk_size
        self.chunk_jitter = chunk_jitter

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

                input_ids = input_tokenizer(
                    clean_text,
                    max_length=max_input_length,
                    truncation=True,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )['input_ids']
                if len(input_ids) > max_input_length:
                    skipped_long += 1
                    continue

                chunked_tagged = build_chunked_target(tagged_text, chunk_size=chunk_size)
                target_ids = output_tokenizer.encode_tagged_text(chunked_tagged)
                if not target_ids:
                    skipped_empty += 1
                    continue

                upsampled_len = len(input_ids) * upsample_factor
                if upsampled_len < len(target_ids):
                    skipped_ctc += 1
                    continue

                self.samples.append({
                    'input_ids': input_ids,
                    'clean_text': clean_text,
                    'tagged_text': tagged_text,
                })

        log.info(
            "TextTaggerDatasetV3: %d samples from %s "
            "(skipped: %d long, %d CTC-constraint, %d empty) "
            "chunk_size=%d jitter=%d",
            len(self.samples), manifest_filepath,
            skipped_long, skipped_ctc, skipped_empty,
            chunk_size, chunk_jitter,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        cs = self.chunk_size + random.randint(-self.chunk_jitter, self.chunk_jitter)
        cs = max(3, cs)
        chunked_tagged = build_chunked_target(sample['tagged_text'], chunk_size=cs)
        target_ids = self.output_tokenizer.encode_tagged_text(chunked_tagged)

        input_ids = torch.tensor(sample['input_ids'], dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        return input_ids, target_ids


def text_tagger_v3_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty, empty, empty

    input_seqs, target_seqs = zip(*batch)

    input_lengths = [len(s) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    max_input = max(input_lengths)
    max_target = max(target_lengths)

    input_batch = torch.zeros(len(batch), max_input, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_input, dtype=torch.long)
    target_batch = torch.zeros(len(batch), max_target, dtype=torch.long)
    target_len_tensor = torch.tensor(target_lengths, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(input_seqs, target_seqs)):
        input_batch[i, :len(inp)] = inp
        attention_mask[i, :len(inp)] = 1
        target_batch[i, :len(tgt)] = tgt

    return input_batch, attention_mask, target_batch, target_len_tensor
