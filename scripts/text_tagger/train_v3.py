#!/usr/bin/env python3
"""Train Text CTC Tagger v3 (XLM-R encoder + CTC over aggregate vocab)."""

import argparse
import logging
import sys
from pathlib import Path

import lightning.pytorch as pl
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from promptingnemo.data.text_tagger_dataset_v3 import (
    TextTaggerDatasetV3,
    text_tagger_v3_collate_fn,
)
from promptingnemo.models.text_ctc_model_v3 import TextCTCTaggerV3
from promptingnemo.tokenizer.meta_tokenizer import MetaTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def train(cfg: dict, resume_from: str = None):
    data_dir = Path(cfg['data']['data_dir'])
    tok_cfg = cfg['tokenizer']

    encoder_name = cfg['model'].get('encoder_name', 'xlm-roberta-base')
    log.info("Loading XLM-R tokenizer: %s", encoder_name)
    input_tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    log.info("Loading MetaTokenizer for output vocab...")
    output_tokenizer = MetaTokenizer(
        tokenizers_dir=tok_cfg['tokenizers_dir'],
        aggregate_vocab_path=tok_cfg['aggregate_vocab_path'],
        default_family=tok_cfg.get('default_family', 'ENGLISH'),
    )

    model_cfg = dict(cfg.get('model', {}))
    model_cfg['output_vocab_size'] = output_tokenizer.vocab_size

    train_cfg = cfg.get('training', {})
    model_cfg['lr'] = train_cfg.get('lr', 5e-4)
    model_cfg['weight_decay'] = train_cfg.get('weight_decay', 0.01)
    model_cfg['warmup_steps'] = train_cfg.get('warmup_steps', 2000)
    model_cfg['max_steps'] = train_cfg.get('max_steps', 200000)

    upsample_factor = model_cfg.get('upsample_factor', 3)
    max_input_length = model_cfg.get('max_input_length', 128)

    train_manifest = str(data_dir / cfg['data'].get('train_manifest', 'merged_train.json'))
    val_manifest = str(data_dir / cfg['data'].get('val_manifest', 'merged_valid.json'))

    chunk_size = model_cfg.get('chunk_size', 8)
    chunk_jitter = model_cfg.get('chunk_jitter', 3)

    log.info("Loading training data (chunk_size=%d, jitter=%d)...", chunk_size, chunk_jitter)
    train_dataset = TextTaggerDatasetV3(
        train_manifest, input_tokenizer, output_tokenizer,
        max_input_length=max_input_length,
        upsample_factor=upsample_factor,
        chunk_size=chunk_size,
        chunk_jitter=chunk_jitter,
    )
    log.info("Loading validation data (no chunking for clean eval)...")
    val_dataset = TextTaggerDatasetV3(
        val_manifest, input_tokenizer, output_tokenizer,
        max_input_length=max_input_length,
        upsample_factor=upsample_factor,
        chunk_size=1000,
        chunk_jitter=0,
    )

    batch_size = train_cfg.get('batch_size', 64)
    num_workers = train_cfg.get('num_workers', 8)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=text_tagger_v3_collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=text_tagger_v3_collate_fn,
    )

    model = TextCTCTaggerV3(model_cfg)
    model.set_output_tokenizer(output_tokenizer)

    exp_cfg = cfg.get('experiment', {})
    exp_dir = Path(exp_cfg.get('exp_dir', './experiments'))
    exp_name = exp_cfg.get('exp_name', 'text_ctc_tagger_v3')
    save_dir = exp_dir / exp_name

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=str(save_dir / 'checkpoints'),
            filename='{step}-{val_loss:.4f}',
            monitor='val_loss', mode='min', save_top_k=3,
            every_n_train_steps=train_cfg.get('save_every_n_steps', 5000),
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
        pl.callbacks.TQDMProgressBar(refresh_rate=50),
    ]

    trainer = pl.Trainer(
        max_steps=train_cfg.get('max_steps', 200000),
        accelerator='auto', devices=train_cfg.get('devices', 1),
        precision=train_cfg.get('precision', '16-mixed'),
        accumulate_grad_batches=train_cfg.get('accumulate_grad_batches', 4),
        gradient_clip_val=train_cfg.get('gradient_clip_val', 1.0),
        val_check_interval=train_cfg.get('val_check_interval', 5000),
        log_every_n_steps=train_cfg.get('log_every_n_steps', 100),
        default_root_dir=str(save_dir),
        callbacks=callbacks, enable_checkpointing=True,
    )

    log.info("Starting training: %s", exp_name)
    log.info("  Encoder: %s (frozen=%s, unfreeze_top=%d)",
             encoder_name, model_cfg.get('freeze_encoder', True),
             model_cfg.get('unfreeze_top_n', 0))
    log.info("  Output vocab: %d (+1 blank)", output_tokenizer.vocab_size)
    log.info("  Train samples: %d, Val samples: %d", len(train_dataset), len(val_dataset))
    log.info("  Batch size: %d x %d accum = %d effective",
             batch_size, train_cfg.get('accumulate_grad_batches', 4),
             batch_size * train_cfg.get('accumulate_grad_batches', 4))

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from)
    log.info("Training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, resume_from=args.resume)


if __name__ == '__main__':
    main()
