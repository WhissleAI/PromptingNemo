"""Distillation training: compress a 600M teacher into a 35M student.

Usage:
    python scripts/asr/meta-asr/distill.py --config recipes/meta_asr/conf/distill_35m.yaml
    python scripts/asr/meta-asr/distill.py --config /path/to/config.yml --resume_from /path/to/ckpt
"""

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import sentencepiece as spm
import torch
import yaml
try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding, CTCDecodingConfig
from typing import List as _List


class AggregateCTCDecoding(CTCDecoding):
    """CTCDecoding subclass that converts ▁ to spaces for aggregate tokenizers."""

    def decode_tokens_to_str(self, tokens) -> str:
        if tokens and isinstance(tokens[0], int):
            tokens = self.decode_ids_to_tokens(tokens)
        return ''.join(tokens).replace('▁', ' ').strip()
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.parts.preprocessing.perturb import WhiteNoisePerturbation, ShiftPerturbation
from nemo.collections.asr.parts.preprocessing.features import AudioAugmentor
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.utils import logging as nemo_logging
from nemo.utils import exp_manager

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from promptingnemo.models.distill_ctc_model import DistillCTCModel
from promptingnemo.models.ctc_model import CustomEncDecCTCModelBPE
from promptingnemo.models.weight_init import init_student_from_teacher, copy_decoder_weights
from promptingnemo.data.dataset import RobustAudioToBPEDataset
from promptingnemo.data.sampler import BalancedLanguageBatchSampler
from promptingnemo.tokenizer.aggregate import _family_name_for_lang

try:
    from scripts.asr.meta_asr.tag_classifier import (
        build_trailing_tag_maps,
        build_all_special_token_ids,
    )
except ImportError:
    build_trailing_tag_maps = None
    build_all_special_token_ids = None

# Import tokenizer config helpers — try modularized version first, fall back to main.py
try:
    from promptingnemo.tokenizer.config import (
        load_tokenizer_langs,
        load_shared_special_tokens,
        load_aggregate_vocabulary,
        store_aggregate_vocabulary,
    )
    from promptingnemo.tokenizer.aggregate import build_aggregate_vocab_from_tokenizers
except ImportError:
    from scripts.asr.meta_asr.main import (
        load_tokenizer_langs,
        load_shared_special_tokens,
        load_aggregate_vocabulary,
        store_aggregate_vocabulary,
        build_aggregate_vocab_from_tokenizers,
    )

try:
    from promptingnemo.models.decoder import (
        scan_manifest_for_new_tokens,
        extend_decoder_for_new_tokens,
        slim_decoder_for_training,
        scale_down_tag_decoder_weights,
    )
except ImportError:
    from scripts.asr.meta_asr.main import (
        scan_manifest_for_new_tokens,
        extend_decoder_for_new_tokens,
        slim_decoder_for_training,
        scale_down_tag_decoder_weights,
    )

try:
    from scripts.asr.meta_asr.main import set_language_families
except ImportError:
    def set_language_families(families):
        pass

logging.basicConfig(level=logging.INFO, format='[%(levelname)s %(asctime)s] %(message)s')


def _load_teacher(cfg) -> CustomEncDecCTCModelBPE:
    """Load the teacher model from .nemo checkpoint, frozen."""
    model_root = Path(cfg.model.model_root)
    model_path = model_root / cfg.model.model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Teacher model not found: {model_path}")

    base_cfg = ASRModel.restore_from(restore_path=str(model_path), return_config=True)
    teacher = CustomEncDecCTCModelBPE.restore_from(str(model_path), override_config_path=base_cfg, strict=False)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    total_params = sum(p.numel() for p in teacher.parameters())
    logging.info("Teacher loaded: %.1fM params, encoder_dim=%d", total_params / 1e6, teacher.encoder._feat_out)
    return teacher


def _create_student(teacher, cfg) -> DistillCTCModel:
    """Create student model with smaller encoder, initialized from teacher.

    Strategy: restore from teacher .nemo with ORIGINAL config (so state_dict loads
    cleanly and tokenizer paths resolve), then replace encoder+decoder with
    student-sized versions and SVD-init from teacher weights.
    """
    student_enc_cfg = cfg.get('student_encoder', {})
    model_path = str(Path(cfg.model.model_root) / cfg.model.model_name)

    # Step 1: Get teacher config and update data paths (but NOT encoder dims)
    restore_cfg = ASRModel.restore_from(restore_path=model_path, return_config=True)

    with open_dict(restore_cfg):
        restore_cfg.use_keyword_loss = cfg.training.get('use_keyword_loss', False)
        restore_cfg.keyword_loss_weight = cfg.training.get('keyword_loss_weight', 0.3)
        restore_cfg.keyword_loss_warmup_steps = cfg.training.get('keyword_loss_warmup_steps', 0)

        train_manifest = str(Path(cfg.training.data_dir) / cfg.training.train_manifest)
        val_manifest = str(Path(cfg.training.data_dir) / cfg.training.test_manifest)
        lang_field = cfg.training.get('lang_field', 'lang')

        restore_cfg.train_ds.manifest_filepath = train_manifest
        restore_cfg.train_ds.batch_size = cfg.training.batch_size
        restore_cfg.train_ds.max_duration = cfg.training.max_duration
        restore_cfg.train_ds.shuffle = True
        restore_cfg.train_ds.is_tarred = False
        restore_cfg.train_ds.tarred_audio_filepaths = None
        restore_cfg.train_ds.num_workers = cfg.training.num_workers
        restore_cfg.train_ds.pin_memory = cfg.training.pin_memory
        restore_cfg.train_ds.lang_field = lang_field
        restore_cfg.train_ds.return_sample_id = True

        restore_cfg.validation_ds.manifest_filepath = val_manifest
        restore_cfg.validation_ds.batch_size = cfg.training.batch_size
        restore_cfg.validation_ds.max_duration = cfg.training.max_duration
        restore_cfg.validation_ds.shuffle = False
        restore_cfg.validation_ds.num_workers = cfg.training.num_workers
        restore_cfg.validation_ds.pin_memory = cfg.training.pin_memory
        restore_cfg.validation_ds.lang_field = lang_field
        restore_cfg.validation_ds.return_sample_id = True

        if 'manifest_processor' not in restore_cfg.train_ds:
            restore_cfg.train_ds.manifest_processor = {}
        if 'additional_fields' not in restore_cfg.train_ds.manifest_processor:
            restore_cfg.train_ds.manifest_processor.additional_fields = []
        for field_name in {'lang', lang_field}:
            if field_name and field_name not in restore_cfg.train_ds.manifest_processor.additional_fields:
                restore_cfg.train_ds.manifest_processor.additional_fields.append(field_name)

        if 'augmentor' in restore_cfg.train_ds:
            del restore_cfg.train_ds.augmentor

    # Step 2: Restore as DistillCTCModel with teacher-sized weights (loads cleanly)
    student = DistillCTCModel.restore_from(model_path, override_config_path=restore_cfg)
    logging.info("Restored student shell from teacher checkpoint (teacher-sized encoder)")

    # Step 3: Replace encoder with student-sized version
    from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
    d_model = student_enc_cfg.get('d_model', 256)
    num_layers = student_enc_cfg.get('num_layers', 12)
    student_encoder = ConformerEncoder(
        feat_in=restore_cfg.preprocessor.features,
        feat_out=-1,
        n_layers=num_layers,
        d_model=d_model,
        subsampling=student_enc_cfg.get('subsampling', 'dw_striding'),
        subsampling_factor=student_enc_cfg.get('subsampling_factor', 8),
        subsampling_conv_channels=student_enc_cfg.get('subsampling_conv_channels', d_model),
        ff_expansion_factor=student_enc_cfg.get('ff_expansion_factor', 4),
        self_attention_model=student_enc_cfg.get('self_attention_model', 'rel_pos'),
        n_heads=student_enc_cfg.get('n_heads', 4),
        conv_kernel_size=student_enc_cfg.get('conv_kernel_size', 15),
        pos_emb_max_len=student_enc_cfg.get('pos_emb_max_len', 5000),
    )
    student.encoder = student_encoder
    encoder_dim = student.encoder._feat_out

    # Step 4: Decoder setup — either keep teacher's decoder or create student-sized one
    distill_cfg = cfg.get('distillation', {})
    use_teacher_decoder = distill_cfg.get('use_teacher_decoder', False)
    init_strategy = distill_cfg.get('init_strategy', 'evenly_spaced')

    if use_teacher_decoder:
        teacher_dim = teacher.encoder._feat_out
        student.encoder_proj = torch.nn.Conv1d(encoder_dim, teacher_dim, kernel_size=1)
        torch.nn.init.xavier_uniform_(student.encoder_proj.weight)
        torch.nn.init.zeros_(student.encoder_proj.bias)
        freeze_decoder = distill_cfg.get('freeze_decoder', False)
        if freeze_decoder:
            for p in student.decoder.parameters():
                p.requires_grad = False
        dec_params = sum(p.numel() for p in student.decoder.parameters())
        logging.info(
            "Using teacher decoder with encoder_proj: %d → %d (%.1fK proj params, "
            "%.1fM decoder params, frozen=%s)",
            encoder_dim, teacher_dim,
            sum(p.numel() for p in student.encoder_proj.parameters()) / 1e3,
            dec_params / 1e6, freeze_decoder,
        )
    else:
        old_decoder = student.decoder
        old_vocab = list(old_decoder.vocabulary) if hasattr(old_decoder, 'vocabulary') else None
        last_layer = old_decoder.decoder_layers[-1]
        vocab_size = getattr(last_layer, 'out_features', None) or getattr(last_layer, 'out_channels', None)
        from nemo.collections.asr.modules.conv_asr import ConvASRDecoder
        if old_vocab is not None:
            student.decoder = ConvASRDecoder(
                feat_in=encoder_dim, num_classes=len(old_vocab), vocabulary=old_vocab,
            )
        else:
            student.decoder = ConvASRDecoder(feat_in=encoder_dim, num_classes=vocab_size)

    total_params = sum(p.numel() for p in student.parameters())
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    logging.info(
        "Student created: %.1fM total params (%.1fM trainable), encoder_dim=%d, layers=%d, "
        "teacher_decoder=%s, init_strategy=%s",
        total_params / 1e6, trainable / 1e6, encoder_dim, num_layers,
        use_teacher_decoder, init_strategy,
    )

    # Step 5: Initialize student encoder from teacher via SVD
    init_student_from_teacher(student, teacher, strategy=init_strategy)
    if not use_teacher_decoder:
        copy_decoder_weights(student, teacher)

    return student


def distill_model(cfg, ckpt_path=None):
    """Main distillation training loop."""
    lang_field = cfg.training.get('lang_field', 'lang')
    RobustAudioToBPEDataset.default_lang_field = lang_field
    RobustAudioToBPEDataset.skip_audio_validation_default = bool(cfg.training.get('skip_audio_validation', False))
    RobustAudioToBPEDataset.keyphrase_oversample_factor_default = float(cfg.training.get('keyphrase_oversample_factor', 0.0))

    # Monkey-patch NeMo's AudioToBPEDataset so setup_training_data/setup_validation_data
    # use RobustAudioToBPEDataset (which handles aggregate tokenizer lang_id)
    from nemo.collections.asr.data import audio_to_text
    audio_to_text.AudioToBPEDataset = RobustAudioToBPEDataset

    language_families_cfg = cfg.model.get('language_families')
    language_families = OmegaConf.to_container(language_families_cfg, resolve=True) if language_families_cfg else None
    if language_families:
        set_language_families(language_families)

    tokenizer_langs = load_tokenizer_langs(cfg)
    if not isinstance(tokenizer_langs, dict) or not tokenizer_langs:
        raise RuntimeError("No tokenizer_langs mapping found. Run tokenizer training first.")

    shared_special_tokens = load_shared_special_tokens(cfg)
    language_family_map = cfg.model.get('language_family_map', {})

    if language_families:
        families_upper = {f.upper() for f in language_families}
        tokenizer_langs = {k: v for k, v in tokenizer_langs.items() if k.upper() in families_upper}
        if not tokenizer_langs:
            raise RuntimeError(f"No tokenizer_langs matched language_families {language_families}")

    aggregate_vocab = load_aggregate_vocabulary(cfg)
    if not aggregate_vocab:
        aggregate_vocab = build_aggregate_vocab_from_tokenizers(tokenizer_langs)
        store_aggregate_vocabulary(cfg, aggregate_vocab)

    lang_list = sorted(tokenizer_langs.keys())
    logging.info("Distillation with languages: %s", lang_list)

    # --- Load teacher ---
    teacher = _load_teacher(cfg)

    # --- Create student ---
    student = _create_student(teacher, cfg)

    # --- Setup tokenizer and decoder ---
    # NOTE: skip slim_decoder_for_training during distillation — student and teacher
    # must share the same vocabulary for KD logit-level loss to work.
    student.setup_custom_loss()

    train_manifest = str(Path(cfg.training.data_dir) / cfg.training.train_manifest)
    current_vocab = set(student.decoder.vocabulary)
    new_tokens = scan_manifest_for_new_tokens(train_manifest, current_vocab)
    if new_tokens:
        nemo_logging.info(f"Found {len(new_tokens)} new tokens: {new_tokens}")
        extend_decoder_for_new_tokens(student, new_tokens)
        if hasattr(student, 'tokenizer') and hasattr(student.tokenizer, 'extend_vocabulary'):
            student.tokenizer.extend_vocabulary(new_tokens)
        student.setup_custom_loss()

    tag_scale = cfg.training.get('tag_decoder_scale', 0.01)
    if tag_scale < 1.0:
        tag_bias = cfg.training.get('tag_init_bias', -5.0)
        scale_down_tag_decoder_weights(student, scale_factor=tag_scale)

    # --- Setup distillation ---
    distill_cfg = cfg.get('distillation', {})
    student.setup_distillation(teacher, distill_cfg)

    # --- Tag classifier ---
    tag_cls_cfg = cfg.get('tag_classifier', {})
    if tag_cls_cfg and tag_cls_cfg.get('enabled', False):
        from scripts.asr.meta_asr.tag_classifier import TrailingTagClassifier
        tag_cls_categories = tag_cls_cfg.get('categories', None)
        if tag_cls_categories:
            tag_cls_categories = [c.upper() for c in tag_cls_categories]
        tag_cls_weight = tag_cls_cfg.get('weight', 0.1)
        encoder_dim = student.encoder._feat_out
        special_prefixes = cfg.model.get('special_token_prefixes', None)
        student.setup_tag_classifier(
            encoder_dim=encoder_dim,
            vocabulary=list(student.decoder.vocabulary),
            categories=tag_cls_categories,
            weight=tag_cls_weight,
            special_token_prefixes=special_prefixes,
        )

    # --- Entity-aware CTC weighting ---
    entity_weight = cfg.training.get('entity_sample_weight', 0.0)
    if entity_weight > 1.0:
        student.setup_entity_weighting(
            vocabulary=list(student.decoder.vocabulary),
            weight=entity_weight,
        )

    # --- Setup tokenizer config on student ---
    tokenizer_entry = {'type': 'agg', 'langs': tokenizer_langs}
    if shared_special_tokens:
        tokenizer_entry['special_tokens'] = shared_special_tokens

    aggregate_vocab = list(student.decoder.vocabulary)
    store_aggregate_vocabulary(cfg, aggregate_vocab)

    with open_dict(student.cfg):
        student.cfg.tokenizer = tokenizer_entry
        student.cfg.train_ds.allowed_langs = lang_list
        student.cfg.validation_ds.allowed_langs = lang_list
        student.cfg.train_ds.lang_field = lang_field
        student.cfg.validation_ds.lang_field = lang_field
        student.cfg.validation_ds.return_sample_id = True
        student.cfg.decoder.vocabulary = aggregate_vocab
        student.cfg.decoder.num_classes = len(aggregate_vocab)

    decoding_cfg = student.cfg.get('decoding', OmegaConf.create({'strategy': 'greedy'}))
    student.decoding = AggregateCTCDecoding(decoding_cfg=decoding_cfg, vocabulary=aggregate_vocab)
    student.wer = WER(
        decoding=student.decoding,
        use_cer=student._cfg.get('use_cer', False),
        log_prediction=student._cfg.get('log_prediction', True),
    )

    # --- Data loaders ---
    student.setup_training_data(student.cfg.train_ds)
    student.setup_validation_data(student.cfg.validation_ds)
    student.setup_multiple_test_data(student.cfg.validation_ds)
    student._validation_dataset_ref = getattr(student._validation_dl, 'dataset', None)

    if cfg.training.get('use_family_loss_weights'):
        train_dataset = student._train_dl.dataset
        family_counts = defaultdict(int)
        for lang_id in train_dataset.language_ids:
            family = _family_name_for_lang(lang_id)
            family_counts[family] += 1
        total_samples = sum(family_counts.values())
        num_families = len(family_counts)
        if total_samples > 0 and num_families > 0:
            weights = {fam: total_samples / (num_families * count) for fam, count in family_counts.items()}
            logging.info("Family loss weights: %s", weights)
            student.set_family_loss_weights(weights)

    train_dataset = student._train_dl.dataset
    train_batch_sampler = BalancedLanguageBatchSampler(
        train_dataset, cfg.training.batch_size, lang_to_family_map=language_family_map,
    )
    student._train_dl = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=train_dataset.collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
    )

    # Audio augmentation on validation only (student noise is handled in training_step)
    spec_cfg = cfg.training.get('spec_augment')
    if spec_cfg and hasattr(student.cfg, 'spec_augment'):
        with open_dict(student.cfg):
            if 'time_masks' in spec_cfg:
                student.cfg.spec_augment.time_masks = spec_cfg['time_masks']
            if 'time_width' in spec_cfg:
                student.cfg.spec_augment.time_width = spec_cfg['time_width']
        student.spec_augmentation = student.from_config_dict(student.cfg.spec_augment)

    # --- Optimizer ---
    optim_cfg = cfg.training.get('optim')
    if optim_cfg:
        with open_dict(student.cfg):
            if 'lr' in optim_cfg:
                student.cfg.optim.lr = optim_cfg['lr']
            if 'weight_decay' in optim_cfg:
                student.cfg.optim.weight_decay = optim_cfg['weight_decay']
            if 'sched' in optim_cfg and hasattr(student.cfg.optim, 'sched'):
                if 'warmup_steps' in optim_cfg['sched']:
                    student.cfg.optim.sched.warmup_steps = optim_cfg['sched']['warmup_steps']
    student.setup_optimization(student.cfg.optim)

    # --- Trainer ---
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    configured_devices = cfg.training.get('devices', None)
    if accelerator == 'cpu':
        devices = configured_devices if isinstance(configured_devices, int) and configured_devices > 0 else 1
    else:
        devices = configured_devices if configured_devices is not None else 1

    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=devices,
        max_steps=cfg.training.max_steps,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        enable_checkpointing=False,
        logger=False,
        log_every_n_steps=50,
        use_distributed_sampler=False,
    )

    if accelerator == 'gpu' and (devices == -1 or (isinstance(devices, int) and devices > 1)):
        trainer_kwargs['strategy'] = 'ddp'

    val_interval = cfg.experiment.get('every_n_train_steps')
    if val_interval and val_interval > 0:
        trainer_kwargs['val_check_interval'] = val_interval

    accumulate_grad_batches = cfg.training.get('accumulate_grad_batches', None)
    if accumulate_grad_batches and accumulate_grad_batches > 1:
        trainer_kwargs['accumulate_grad_batches'] = accumulate_grad_batches

    trainer = pl.Trainer(**trainer_kwargs)
    # Workaround: protobuf version mismatch causes hparams logging to fail in some containers
    for logger in (trainer.loggers if hasattr(trainer, 'loggers') else [trainer.logger]):
        if logger is not None and hasattr(logger, 'log_hyperparams'):
            logger.log_hyperparams = lambda *a, **kw: None
    student.set_trainer(trainer)

    every_n_train_steps = cfg.experiment.get('every_n_train_steps', None)
    callback_params = exp_manager.CallbackParams(
        monitor=cfg.experiment.monitor,
        mode=cfg.experiment.mode,
        always_save_nemo=cfg.experiment.always_save_nemo,
        save_top_k=cfg.experiment.get('save_top_k', 1),
        every_n_train_steps=every_n_train_steps,
        every_n_epochs=0 if every_n_train_steps else 1,
    )
    # Manually load model weights from latest checkpoint (skip optimizer state
    # to avoid mismatch when teacher param groups changed between runs).
    import glob as _glob
    ckpt_dir = os.path.join(
        cfg.experiment.get('exp_dir', ''), cfg.experiment.get('exp_name', ''), 'checkpoints'
    )
    latest_ckpts = sorted(
        _glob.glob(os.path.join(ckpt_dir, '*-last.ckpt')),
        key=os.path.getmtime,
    )
    if latest_ckpts and not ckpt_path:
        ckpt = torch.load(latest_ckpts[-1], map_location='cpu', weights_only=False)
        result = student.load_state_dict(ckpt.get('state_dict', {}), strict=False)
        logging.info(
            "Loaded model weights from %s (skipping optimizer state). "
            "Missing keys: %d, Unexpected keys: %d",
            os.path.basename(latest_ckpts[-1]),
            len(result.missing_keys), len(result.unexpected_keys),
        )
        del ckpt

    exp_cfg = exp_manager.ExpManagerConfig(
        exp_dir=cfg.experiment.get('exp_dir', None),
        name=cfg.experiment.get('exp_name', None),
        checkpoint_callback_params=callback_params,
        resume_if_exists=False,
        resume_past_end=False,
        resume_ignore_no_checkpoint=True,
        create_checkpoint_callback=True,
        create_tensorboard_logger=True,
        create_wandb_logger=False,
        create_mlflow_logger=False,
        create_dllogger_logger=False,
    )
    exp_cfg = OmegaConf.structured(exp_cfg)
    exp_manager.exp_manager(trainer, exp_cfg)

    # Workaround: protobuf version mismatch causes hparams logging to fail
    for logger in (trainer.loggers if hasattr(trainer, 'loggers') else []):
        if hasattr(logger, 'log_hyperparams'):
            logger.log_hyperparams = lambda *a, **kw: None

    logging.info("Starting distillation training...")
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        result = student.load_state_dict(ckpt.get('state_dict', {}), strict=False)
        logging.info(
            "Loaded weights from resume checkpoint %s (strict=False). "
            "Missing keys: %d, Unexpected keys: %d",
            os.path.basename(ckpt_path),
            len(result.missing_keys), len(result.unexpected_keys),
        )
        if result.missing_keys:
            logging.info("Missing keys (new params, will train from scratch): %s", result.missing_keys[:20])
        del ckpt
        trainer.fit(student)
    else:
        trainer.fit(student)
    logging.info("Distillation training complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge distillation for ASR model compression")
    parser.add_argument("--config", required=True, help="Path to distillation YAML config")
    parser.add_argument("--resume_from", default=None, help="Path to .ckpt checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = OmegaConf.create(yaml.safe_load(f))

    distill_model(cfg, ckpt_path=args.resume_from)


if __name__ == "__main__":
    main()
