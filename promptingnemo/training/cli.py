"""Command-line interface for Meta-ASR training."""

import argparse
import logging
import os
from typing import List

import yaml
from omegaconf import OmegaConf, open_dict

from promptingnemo.tokenizer.config import set_language_families
from promptingnemo.tokenizer.config import (
    store_tokenizer_langs,
    store_shared_special_tokens,
    store_aggregate_vocabulary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train aggregate tokenizers and ASR model")
    parser.add_argument(
        "--mode",
        choices=["both", "tokenizer", "train", "validate_data"],
        default="both",
        help="Choose whether to (re)train tokenizers, the ASR model, or both sequentially.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the training YAML config (required).",
    )
    parser.add_argument(
        "--no-save-config",
        action="store_true",
        help="Do not persist tokenizer metadata back to the config file after training tokenizers.",
    )
    parser.add_argument(
        "--resume_from",
        default=None,
        help="Path to a .ckpt checkpoint to resume training from.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.config:
        raise SystemExit(
            "Error: --config is required. Example:\n"
            "  promptingnemo --config recipes/meta_asr/conf/hindi.yaml --mode both"
        )
    config_file_path = args.config

    with open(config_file_path, 'r', encoding='utf-8') as f:
        cfg = OmegaConf.create(yaml.safe_load(f))

    lang_field = cfg.training.get('lang_field', 'lang')
    language_families_cfg = cfg.model.get('language_families')
    language_families = OmegaConf.to_container(language_families_cfg, resolve=True) if language_families_cfg else None
    set_language_families(language_families)

    # NeMo-dependent imports are deferred so --help works without NeMo
    from promptingnemo.data.dataset import RobustAudioToBPEDataset

    RobustAudioToBPEDataset.default_lang_field = lang_field
    RobustAudioToBPEDataset.skip_audio_validation_default = bool(cfg.training.get('skip_audio_validation', False))
    RobustAudioToBPEDataset.keyphrase_oversample_factor_default = float(cfg.training.get('keyphrase_oversample_factor', 0.0))

    run_validation = args.mode in ('validate_data', 'both')
    run_tokenizers = args.mode in ('both', 'tokenizer')
    run_training = args.mode in ('both', 'train')

    if run_validation:
        from promptingnemo.data.manifest import validate_manifests
        from promptingnemo.training.trainer import save_updated_config

        logging.info("Validating manifests prior to downstream steps...")
        validation_stats = validate_manifests(cfg)
        for manifest, stats in validation_stats.items():
            logging.info(
                "Validated manifest %s: kept=%d dropped=%d",
                manifest,
                stats['kept'],
                stats['dropped'],
            )
        if not args.no_save_config:
            save_updated_config(cfg, config_file_path)
        if args.mode == 'validate_data':
            logging.info("Validation-only mode complete.")
            return

    if run_tokenizers:
        from promptingnemo.tokenizer.sentencepiece import train_sentencepiece_tokenizer
        from promptingnemo.tokenizer.aggregate import extract_langs_and_special_tokens, train_aggregate_tokenizer

        logging.info("Training tokenizer...")
        if cfg.model.get('use_aggregate_tokenizer', False):
            manifest_path = os.path.join(cfg.training.data_dir, cfg.training.train_manifest)
            special_token_prefixes = cfg.model.get('special_token_prefixes', None)
            extra_manifests: List[str] = []
            test_manifest = cfg.training.get('test_manifest')
            if test_manifest:
                test_manifest_path = os.path.join(cfg.training.data_dir, test_manifest)
                extra_manifests.append(test_manifest_path)

            tokenizer_extra_manifests = cfg.training.get('tokenizer_extra_manifests')
            if tokenizer_extra_manifests:
                for manifest_name in tokenizer_extra_manifests:
                    if not manifest_name:
                        continue
                    manifest_path_candidate = (
                        manifest_name
                        if os.path.isabs(manifest_name)
                        else os.path.join(cfg.training.data_dir, manifest_name)
                    )
                    extra_manifests.append(manifest_path_candidate)

            langs, special_tokens = extract_langs_and_special_tokens(
                manifest_path,
                special_token_prefixes,
                extra_manifest_paths=extra_manifests,
                lang_field=lang_field,
            )
            tokenizer_langs_config, shared_special_tokens, aggregate_vocab, language_family_map = train_aggregate_tokenizer(
                cfg,
                langs,
                special_tokens,
                extra_manifest_paths=extra_manifests,
                lang_field=lang_field,
            )
            store_tokenizer_langs(cfg, tokenizer_langs_config)
            store_shared_special_tokens(cfg, shared_special_tokens)
            store_aggregate_vocabulary(cfg, aggregate_vocab)
            with open_dict(cfg):
                cfg.model.language_family_map = language_family_map
            if not args.no_save_config:
                from promptingnemo.training.trainer import save_updated_config
                save_updated_config(cfg, config_file_path)
            logging.info("Tokenizer training complete. Continue with model training as needed.")
        else:
            tokenizer_path = os.path.join(cfg.model.model_root, cfg.model.new_tokenizer_folder)
            character_coverage = cfg.model.get('character_coverage', None)
            tokenizer_options = cfg.model.get('tokenizer_options') or {}
            train_sentencepiece_tokenizer(
                manifest_file=os.path.join(cfg.training.data_dir, cfg.training.train_manifest),
                tokenizer_folder=tokenizer_path,
                special_tokens=[],
                vocab_size=cfg.model.get('vocab_size', 1024),
                character_coverage=character_coverage,
                extra_options=tokenizer_options if isinstance(tokenizer_options, dict) else {},
            )
            logging.info("Tokenizer training complete. Continue with model training as needed.")

        if args.mode == 'tokenizer':
            return

    if run_training:
        from promptingnemo.training.trainer import train_model
        train_model(cfg, ckpt_path=args.resume_from)
