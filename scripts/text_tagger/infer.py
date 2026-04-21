#!/usr/bin/env python3
"""Inference with a trained Text CTC Tagger.

Takes clean text as input, outputs tagged text with inline meta-tags.
Supports both batch file inference and interactive mode.

Usage:
    python infer.py --checkpoint model.ckpt --data-dir /path/to/data --text "hello world"
    python infer.py --checkpoint model.ckpt --data-dir /path/to/data --input test.json --output tagged.json
    python infer.py --checkpoint model.ckpt --data-dir /path/to/data --interactive
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from promptingnemo.models.text_ctc_model import TextCTCTagger
from promptingnemo.tokenizer.text_tagger_tokenizer import TextTaggerTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


class TextTaggerInference:
    """Wrapper for Text CTC Tagger inference."""

    def __init__(self, checkpoint_path: str, data_dir: str, device: str = 'cpu'):
        self.device = torch.device(device)
        data_path = Path(data_dir)

        # Load char vocab
        with open(data_path / 'char_vocab.json', encoding='utf-8') as f:
            char_data = json.load(f)
        self.char_to_id = {}
        for i, special in enumerate(char_data.get('special', ['<pad>', '<unk>'])):
            self.char_to_id[special] = i
        offset = len(self.char_to_id)
        for i, ch in enumerate(char_data['chars']):
            self.char_to_id[ch] = offset + i
        self.unk_id = self.char_to_id.get('<unk>', 1)

        # Load tokenizer
        self.tokenizer = TextTaggerTokenizer(
            str(data_path / 'sp_text_tagger.model'),
            str(data_path / 'tag_vocab.json'),
        )

        # Load model
        self.model = TextCTCTagger.load_from_checkpoint(
            checkpoint_path, map_location=self.device
        )
        self.model.eval()
        self.model.to(self.device)
        log.info("Model loaded from %s", checkpoint_path)

    @torch.no_grad()
    def tag_text(self, text: str) -> str:
        """Tag a single text string. Returns tagged text."""
        char_ids = [self.char_to_id.get(ch, self.unk_id) for ch in text]
        char_tensor = torch.tensor([char_ids], dtype=torch.long, device=self.device)
        char_lengths = torch.tensor([len(char_ids)], dtype=torch.long, device=self.device)

        log_probs, encoded_lengths = self.model(char_tensor, char_lengths)
        decoded = self.model._greedy_decode(log_probs, encoded_lengths)
        return self.tokenizer.decode(decoded[0])

    @torch.no_grad()
    def tag_batch(self, texts: list) -> list:
        """Tag a batch of texts."""
        return [self.tag_text(t) for t in texts]


def main():
    parser = argparse.ArgumentParser(description='Text CTC Tagger Inference')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--data-dir', required=True, help='Data dir with char_vocab.json, tag_vocab.json, sp model')
    parser.add_argument('--text', default=None, help='Single text to tag')
    parser.add_argument('--input', default=None, help='JSONL file with text to tag')
    parser.add_argument('--output', default=None, help='Output JSONL file')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    args = parser.parse_args()

    engine = TextTaggerInference(args.checkpoint, args.data_dir, args.device)

    if args.text:
        result = engine.tag_text(args.text)
        print(f"Input:  {args.text}")
        print(f"Tagged: {result}")

    elif args.input:
        out_f = open(args.output, 'w', encoding='utf-8') if args.output else sys.stdout
        with open(args.input, encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                text = entry.get('clean_text', entry.get('text', ''))
                tagged = engine.tag_text(text)
                entry['tagged_text'] = tagged
                out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        if args.output:
            out_f.close()
            log.info("Results written to %s", args.output)

    elif args.interactive:
        print("Text CTC Tagger - Interactive Mode (type 'quit' to exit)")
        while True:
            try:
                text = input("\nInput: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() == 'quit':
                break
            result = engine.tag_text(text)
            print(f"Tagged: {result}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
