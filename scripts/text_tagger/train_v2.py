#!/usr/bin/env python3
"""Train the Text CTC Tagger v2 model (Conformer + meta-1B aggregate tokenizer).

Usage:
    python train_v2.py --config conf/config_v2.yaml
    python train_v2.py --config conf/config_v2.yaml --resume /path/to/checkpoint.ckpt
"""

import argparse
import logging
import sys
from pathlib import Path

import lightning.pytorch as pl
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from promptingnemo.data.text_tagger_dataset_v2 import (
    TextTaggerDatasetV2,
    text_tagger_v2_collate_fn,
)
from promptingnemo.models.text_ctc_model_v2 import TextCTCTaggerV2
from promptingnemo.tokenizer.meta_tokenizer import MetaTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def train(cfg: dict, resume_from: str = None):
    data_dir = Path(cfg['data']['data_dir'])
    tok_cfg = cfg['tokenizer']

    tokenizer = MetaTokenizer(
        tokenizers_dir=tok_cfg['tokenizers_dir'],
        aggregate_vocab_path=tok_cfg['aggregate_vocab_path'],
        default_family=tok_cfg.get('default_family', 'ENGLISH'),
    )

    model_cfg = dict(cfg.get('model', {}))
    model_cfg['vocab_size'] = tokenizer.vocab_size

    train_cfg = cfg.get('training', {})
    model_cfg['lr'] = train_cfg.get('lr', 1e-3)
    model_cfg['weight_decay'] = train_cfg.get('weight_decay', 0.01)
    model_cfg['warmup_steps'] = train_cfg.get('warmup_steps', 5000)
    model_cfg['max_steps'] = train_cfg.get('max_steps', 500000)

    upsample_factor = model_cfg.get('upsample_factor', 3)
    max_subword_length = model_cfg.get('max_input_length', 256)

    train_manifest = str(data_dir / cfg['data'].get('train_manifest', 'merged_train.json'))
    val_manifest = str(data_dir / cfg['data'].get('val_manifest', 'merged_valid.json'))

    log.info("Loading training data...")
    train_dataset = TextTaggerDatasetV2(
        train_manifest, tokenizer,
        max_subword_length=max_subword_length,
        upsample_factor=upsample_factor,
    )
    log.info("Loading validation data...")
    val_dataset = TextTaggerDatasetV2(
        val_manifest, tokenizer,
        max_subword_length=max_subword_length,
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
        collate_fn=text_tagger_v2_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=text_tagger_v2_collate_fn,
    )

    model = TextCTCTaggerV2(model_cfg)
    model.set_tokenizer(tokenizer)

    exp_cfg = cfg.get('experiment', {})
    exp_dir = Path(exp_cfg.get('exp_dir', './experiments'))
    exp_name = exp_cfg.get('exp_name', 'text_ctc_tagger_v2')
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
    log.info("  Vocab size: %d (+1 blank = %d)", tokenizer.vocab_size, tokenizer.vocab_size_with_blank)
    log.info("  Upsample factor: %d", upsample_factor)
    log.info("  Causal (streaming): %s", model_cfg.get('causal', True))

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from)

    log.info("Training complete.")


def main():
    parser = argparse.ArgumentParser(description='Train Text CTC Tagger v2')
    parser.add_argument('--config', required=True, help='YAML config file')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume_from=args.resume)


if __name__ == '__main__':
    main()
