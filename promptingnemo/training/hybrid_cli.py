"""CLI for Hybrid RNN-T/CTC model training with tag classifier.

Usage:
    python -m promptingnemo.training.hybrid_cli \
        --config recipes/meta_asr/conf/mega_zh_v1_hybrid.yaml \
        --mode train

    python -m promptingnemo.training.hybrid_cli \
        --config recipes/meta_asr/conf/mega_zh_v1_hybrid.yaml \
        --mode tokenizer
"""

import argparse
import logging
import os
import sys

import yaml
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


def main():
    parser = argparse.ArgumentParser(description='Hybrid RNN-T/CTC training with tag classifier')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--mode', required=True, choices=['tokenizer', 'train'],
                        help='tokenizer: build tokenizer; train: run training')
    parser.add_argument('--resume-from', default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg_dict)

    if args.mode == 'tokenizer':
        from promptingnemo.training.cli import main as tokenizer_main
        sys.argv = ['cli', '--config', args.config, '--mode', 'tokenizer']
        tokenizer_main()
    elif args.mode == 'train':
        _meta_asr_dir = '/mnt/nfs/code/PromptingNemo/scripts/asr/meta-asr'
        if _meta_asr_dir not in sys.path:
            sys.path.insert(0, _meta_asr_dir)

        from promptingnemo.training.hybrid_trainer import train_hybrid_model
        train_hybrid_model(cfg, ckpt_path=args.resume_from)


if __name__ == '__main__':
    main()
