#!/usr/bin/env python3
"""Train the Text CTC Tagger model.

Usage:
    python train.py --config conf/default.yaml
    python train.py --config conf/default.yaml --resume /path/to/checkpoint.ckpt
"""

import argparse
import json
import logging
import sys
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from promptingnemo.data.text_tagger_dataset import (
    TextTaggerDataset,
    text_tagger_collate_fn,
)
from promptingnemo.data.tag_parser import strip_tags
from promptingnemo.models.text_ctc_model import TextCTCTagger
from promptingnemo.tokenizer.text_tagger_tokenizer import TextTaggerTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def load_char_vocab(char_vocab_path: str):
    """Load character vocabulary and return char_to_id mapping."""
    with open(char_vocab_path, encoding='utf-8') as f:
        data = json.load(f)

    char_to_id = {}
    for i, special in enumerate(data.get('special', ['<pad>', '<unk>'])):
        char_to_id[special] = i

    offset = len(char_to_id)
    for i, ch in enumerate(data['chars']):
        char_to_id[ch] = offset + i

    return char_to_id


def train(cfg: dict, resume_from: str = None):
    data_dir = Path(cfg['data']['data_dir'])

    # Load tokenizer
    sp_model_path = str(data_dir / 'sp_text_tagger.model')
    tag_vocab_path = str(data_dir / 'tag_vocab.json')
    tokenizer = TextTaggerTokenizer(sp_model_path, tag_vocab_path)

    # Load char vocab
    char_vocab_path = str(data_dir / 'char_vocab.json')
    char_to_id = load_char_vocab(char_vocab_path)

    # Model config
    model_cfg = dict(cfg.get('model', {}))
    model_cfg['char_vocab_size'] = len(char_to_id)
    model_cfg['vocab_size'] = tokenizer.vocab_size

    # Training config overrides
    train_cfg = cfg.get('training', {})
    model_cfg['lr'] = train_cfg.get('lr', 1e-3)
    model_cfg['weight_decay'] = train_cfg.get('weight_decay', 0.01)
    model_cfg['warmup_steps'] = train_cfg.get('warmup_steps', 5000)
    model_cfg['max_steps'] = train_cfg.get('max_steps', 500000)

    upsample_factor = model_cfg.get('upsample_factor', 2)
    max_text_length = model_cfg.get('max_text_length', 512)

    # Datasets
    train_manifest = str(data_dir / cfg['data'].get('train_manifest', 'merged_train.json'))
    val_manifest = str(data_dir / cfg['data'].get('val_manifest', 'merged_valid.json'))

    log.info("Loading training data...")
    train_dataset = TextTaggerDataset(
        train_manifest, tokenizer, char_to_id,
        max_text_length=max_text_length,
        upsample_factor=upsample_factor,
    )
    log.info("Loading validation data...")
    val_dataset = TextTaggerDataset(
        val_manifest, tokenizer, char_to_id,
        max_text_length=max_text_length,
        upsample_factor=upsample_factor,
    )

    batch_size = train_cfg.get('batch_size', 256)
    num_workers = train_cfg.get('num_workers', 8)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=text_tagger_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=text_tagger_collate_fn,
    )

    # Model
    model = TextCTCTagger(model_cfg)

    # Callbacks
    exp_cfg = cfg.get('experiment', {})
    exp_dir = Path(exp_cfg.get('exp_dir', './experiments'))
    exp_name = exp_cfg.get('exp_name', 'text_ctc_tagger')
    save_dir = exp_dir / exp_name

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=str(save_dir / 'checkpoints'),
            filename='{step}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            every_n_train_steps=train_cfg.get('save_every_n_steps', 5000),
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
        pl.callbacks.TQDMProgressBar(refresh_rate=50),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_steps=train_cfg.get('max_steps', 500000),
        accelerator='auto',
        devices=train_cfg.get('devices', 1),
        precision=train_cfg.get('precision', '16-mixed'),
        accumulate_grad_batches=train_cfg.get('accumulate_grad_batches', 1),
        gradient_clip_val=train_cfg.get('gradient_clip_val', 1.0),
        val_check_interval=train_cfg.get('val_check_interval', 5000),
        log_every_n_steps=train_cfg.get('log_every_n_steps', 100),
        default_root_dir=str(save_dir),
        callbacks=callbacks,
        enable_checkpointing=True,
    )

    log.info("Starting training: %s", exp_name)
    log.info("  Model params: %d", sum(p.numel() for p in model.parameters()))
    log.info("  Train samples: %d", len(train_dataset))
    log.info("  Val samples: %d", len(val_dataset))
    log.info("  Batch size: %d", batch_size)
    log.info("  Char vocab: %d", len(char_to_id))
    log.info("  Output vocab: %d (+1 blank = %d)", tokenizer.vocab_size, tokenizer.vocab_size_with_blank)

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from)

    log.info("Training complete.")


def main():
    parser = argparse.ArgumentParser(description='Train Text CTC Tagger')
    parser.add_argument('--config', required=True, help='YAML config file')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume_from=args.resume)


if __name__ == '__main__':
    main()
