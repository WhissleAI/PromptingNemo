"""Hybrid RNN-T/CTC training: uses the native Hybrid model with tag classifier.

Unlike trainer.py which converts the Hybrid model to pure CTC, this trainer
preserves both the RNN-T and CTC decoders. Combined loss:

    total = (1 - ctc_weight) * rnnt_loss + ctc_weight * ctc_loss + tag_weight * tag_cls_loss

Usage:
    python -m promptingnemo.training.hybrid_cli \
        --config recipes/meta_asr/conf/mega_zh_v1_hybrid.yaml \
        --mode train
"""

import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import sentencepiece as spm
import torch
import yaml
import lightning.pytorch as pl
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.collections.asr.parts.preprocessing.perturb import WhiteNoisePerturbation, ShiftPerturbation
from nemo.collections.asr.parts.preprocessing.features import AudioAugmentor
from nemo.utils import logging as nemo_logging
from nemo.utils import exp_manager

from promptingnemo.tokenizer.config import (
    load_tokenizer_langs,
    load_shared_special_tokens,
)
from promptingnemo.tokenizer.aggregate import _family_name_for_lang
from promptingnemo.models.decoder import (
    scan_manifest_for_new_tokens,
    extend_decoder_for_new_tokens,
)
from promptingnemo.data.dataset import RobustAudioToBPEDataset
from promptingnemo.data.sampler import BalancedLanguageBatchSampler


class ValidationMetricsPrinter(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        step = trainer.global_step
        lines = ['', '=' * 70, f'  VALIDATION METRICS @ step {step}', '=' * 70]
        for key in ['val_loss', 'val_rnnt_loss', 'val_ctc_loss', 'val_wer', 'val_wer_ctc',
                     'val_tag_cls_loss']:
            val = metrics.get(key)
            if val is not None:
                if 'wer' in key:
                    lines.append(f'  {key}: {float(val)*100:.2f}%')
                else:
                    lines.append(f'  {key}: {float(val):.4f}')
        for cat in ['AGE', 'GENDER', 'EMOTION', 'INTENT']:
            acc = metrics.get(f'val_tag_acc_{cat}')
            if acc is not None:
                lines.append(f'  {cat} accuracy: {float(acc)*100:.1f}%')
        lines.append('=' * 70)
        logging.info('\n'.join(lines))


def train_hybrid_model(cfg, ckpt_path=None):
    lang_field = cfg.training.get('lang_field', 'lang')
    RobustAudioToBPEDataset.default_lang_field = lang_field

    tokenizer_langs = load_tokenizer_langs(cfg)
    if not isinstance(tokenizer_langs, dict) or not tokenizer_langs:
        raise RuntimeError("No tokenizer_langs mapping found. Run with --mode tokenizer first.")

    shared_special_tokens = load_shared_special_tokens(cfg)
    language_family_map = cfg.model.get('language_family_map', {})
    model_root = Path(cfg.model.model_root)
    model_path = model_root / cfg.model.model_name

    language_families = cfg.model.get('language_families', [])
    lang_list = list(language_families) if language_families else list(tokenizer_langs.keys())

    train_manifest = str(Path(cfg.training.data_dir) / cfg.training.train_manifest)
    val_manifest = str(Path(cfg.training.data_dir) / cfg.training.test_manifest)

    use_tag_classifier = cfg.training.get('use_tag_classifier', False)

    # Load base config from checkpoint
    base_cfg = ASRModel.restore_from(restore_path=str(model_path), return_config=True)

    # Read CTC decoder vocabulary for extending with new tokens
    ctc_vocab = []
    if hasattr(base_cfg, 'aux_ctc') and hasattr(base_cfg.aux_ctc, 'decoder'):
        ctc_vocab = list(base_cfg.aux_ctc.decoder.vocabulary) if hasattr(base_cfg.aux_ctc.decoder, 'vocabulary') else []
    if not ctc_vocab and hasattr(base_cfg, 'decoder') and hasattr(base_cfg.decoder, 'vocabulary'):
        ctc_vocab = list(base_cfg.decoder.vocabulary)
    logging.info("Checkpoint CTC vocabulary: %d tokens", len(ctc_vocab))

    with open_dict(base_cfg):
        # Disable Lhotse (incompatible with manifest_processor for tag classifier)
        if hasattr(base_cfg, 'train_ds') and hasattr(base_cfg.train_ds, 'use_lhotse'):
            base_cfg.train_ds.use_lhotse = False
        if hasattr(base_cfg, 'validation_ds') and hasattr(base_cfg.validation_ds, 'use_lhotse'):
            base_cfg.validation_ds.use_lhotse = False

        # Dataset config
        base_cfg.train_ds.manifest_filepath = train_manifest
        base_cfg.train_ds.batch_size = cfg.training.batch_size
        base_cfg.train_ds.max_duration = cfg.training.max_duration
        base_cfg.train_ds.shuffle = True
        base_cfg.train_ds.is_tarred = False
        base_cfg.train_ds.tarred_audio_filepaths = None
        base_cfg.train_ds.num_workers = cfg.training.num_workers
        base_cfg.train_ds.pin_memory = cfg.training.pin_memory
        base_cfg.train_ds.return_sample_id = True

        base_cfg.validation_ds.manifest_filepath = val_manifest
        base_cfg.validation_ds.batch_size = cfg.training.batch_size
        base_cfg.validation_ds.max_duration = cfg.training.max_duration
        base_cfg.validation_ds.shuffle = False
        base_cfg.validation_ds.num_workers = cfg.training.num_workers
        base_cfg.validation_ds.pin_memory = cfg.training.pin_memory
        base_cfg.validation_ds.return_sample_id = True

        base_cfg.train_ds.allowed_langs = lang_list
        base_cfg.validation_ds.allowed_langs = lang_list
        base_cfg.train_ds.lang_field = lang_field
        base_cfg.validation_ds.lang_field = lang_field

        if 'augmentor' in base_cfg.train_ds:
            del base_cfg.train_ds.augmentor

        # Adapter config
        adapter_cfg = cfg.get('adapter', {})
        if adapter_cfg and adapter_cfg.get('enabled', False):
            base_cfg.encoder._target_ = 'nemo.collections.asr.modules.conformer_encoder.ConformerEncoderAdapter'

    # Restore model as Hybrid (native, no conversion)
    from promptingnemo.models.hybrid_model import CustomEncDecHybridRNNTCTCBPEModel, FlexibleSaveRestoreConnector
    connector = FlexibleSaveRestoreConnector()
    model = CustomEncDecHybridRNNTCTCBPEModel.restore_from(
        str(model_path), override_config_path=base_cfg, strict=False,
        save_restore_connector=connector,
    )
    model.setup_custom_loss()

    # Switch RNN-T loss from warprnnt_numba to pure PyTorch (avoids numba/PTX issues)
    from nemo.collections.asr.losses.rnnt import RNNTLoss
    logging.info("Replacing RNN-T loss with pure PyTorch backend (avoids numba PTX compatibility issues)")
    model.loss = RNNTLoss(
        num_classes=model.joint.num_classes_with_blank,
        reduction='mean_batch',
        loss_name='pytorch',
    )

    # Force disable Lhotse on model's internal config (may differ from base_cfg)
    with open_dict(model.cfg):
        if hasattr(model.cfg, 'train_ds') and hasattr(model.cfg.train_ds, 'use_lhotse'):
            model.cfg.train_ds.use_lhotse = False
        if hasattr(model.cfg, 'validation_ds') and hasattr(model.cfg.validation_ds, 'use_lhotse'):
            model.cfg.validation_ds.use_lhotse = False

    # Extend CTC decoder with new tag tokens from manifest
    if hasattr(model, 'ctc_decoder') and ctc_vocab:
        current_ctc_vocab = set(ctc_vocab)
        new_tokens = scan_manifest_for_new_tokens(
            train_manifest, current_ctc_vocab,
            allowed_prefixes=(
                'ENTITY_', 'INTENT_', 'EMOTION_', 'GENDER_', 'AGE_',
                'DIALECT_', 'KEYWORD_', 'LANG_', 'OTHER_',
            ),
        )
        if new_tokens:
            logging.info("Found %d new tokens in manifest for CTC decoder: %s", len(new_tokens), new_tokens)

    # Adapter setup
    if adapter_cfg and adapter_cfg.get('enabled', False):
        adapter_name = adapter_cfg.get('name', 'lang_adapter')
        adapter_dim = adapter_cfg.get('dim', 128)
        adapter_config = LinearAdapterConfig(
            in_features=model.encoder._feat_out,
            dim=adapter_dim,
            activation=adapter_cfg.get('activation', 'swish'),
            norm_position=adapter_cfg.get('norm_position', 'pre'),
        )
        existing = model.get_enabled_adapters() if hasattr(model, 'get_enabled_adapters') else []
        if adapter_name not in existing:
            model.add_adapter(name=adapter_name, cfg=adapter_config)
            logging.info("Added adapter '%s' (dim=%d)", adapter_name, adapter_dim)
        model.set_enabled_adapters(enabled=True)

        if adapter_cfg.get('unfreeze_encoder', False):
            model.freeze()
            model.encoder.unfreeze()
            logging.info('Unfreezing encoder')
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            logging.info('After unfreeze: %d trainable / %d total params', trainable, total)

    # Use RobustAudioToBPEDataset directly — NeMo's default setup validates every
    # audio file over NFS which takes hours on 882K samples.
    logging.info("Setting up training data (RobustAudioToBPEDataset, skip_audio_validation)...")
    RobustAudioToBPEDataset.skip_audio_validation_default = True

    train_ds = RobustAudioToBPEDataset(
        manifest_filepath=train_manifest,
        tokenizer=model.tokenizer,
        sample_rate=16000,
        max_duration=cfg.training.max_duration,
        min_duration=cfg.training.get('min_duration', 0.1),
        return_sample_id=True,
        lang_field=lang_field,
    )
    model._train_dl = DataLoader(
        dataset=train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=train_ds.collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        drop_last=True,
    )

    logging.info("Training data: %d samples loaded", len(train_ds))

    val_ds = RobustAudioToBPEDataset(
        manifest_filepath=val_manifest,
        tokenizer=model.tokenizer,
        sample_rate=16000,
        max_duration=cfg.training.max_duration,
        min_duration=cfg.training.get('min_duration', 0.1),
        return_sample_id=True,
        lang_field=lang_field,
    )
    model._validation_dl = DataLoader(
        dataset=val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=val_ds.collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
    )

    logging.info("Validation data: %d samples loaded", len(val_ds))
    model._validation_dataset_ref = getattr(model._validation_dl, 'dataset', None)

    # Tag classifier setup
    if use_tag_classifier:
        tag_categories = sorted(cfg.training.get('tag_categories', ['AGE', 'GENDER', 'EMOTION', 'INTENT']))
        tag_weight = cfg.training.get('tag_classifier_weight', 0.5)

        train_dataset_tmp = model._train_dl.dataset
        collection = train_dataset_tmp.manifest_processor.collection
        num_samples = len(collection)

        min_dur = cfg.training.get('min_duration', 0.1)
        max_dur = cfg.training.get('max_duration', None)
        manifest_tags = []
        with open(train_manifest, 'r', encoding='utf-8') as mf:
            for line in mf:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                dur = float(entry.get('duration', 0))
                if min_dur is not None and dur < min_dur:
                    continue
                if max_dur is not None and dur > max_dur:
                    continue
                row = {}
                for cat in tag_categories:
                    row[cat] = int(entry.get(f'tag_{cat.lower()}', 0) or 0)
                manifest_tags.append(row)

        unique_tags = {cat: set() for cat in tag_categories}
        for row in manifest_tags:
            for cat in tag_categories:
                if row[cat] != 0:
                    unique_tags[cat].add(row[cat])

        category_sizes = {}
        for cat in tag_categories:
            category_sizes[cat] = max(unique_tags[cat]) + 1 if unique_tags[cat] else 2

        tag_labels = torch.zeros(num_samples, len(tag_categories), dtype=torch.long)
        for i in range(min(num_samples, len(manifest_tags))):
            for j, cat in enumerate(tag_categories):
                tag_labels[i, j] = manifest_tags[i][cat]

        # Class weights (sqrt inverse frequency)
        class_weights = {}
        for j, cat in enumerate(tag_categories):
            labels_col = tag_labels[:num_samples, j]
            valid_mask = labels_col >= 0
            counts = torch.bincount(labels_col[valid_mask], minlength=category_sizes[cat])
            total = counts.sum().float()
            n_classes = len(counts)
            weights = torch.zeros(n_classes)
            for c in range(n_classes):
                if counts[c] > 0:
                    weights[c] = torch.sqrt(total / (n_classes * counts[c].float()))
            class_weights[cat] = weights
            logging.info("  %s counts: %s", cat, {c: int(v) for c, v in enumerate(counts.tolist())})

        encoder_dim = model.encoder._feat_out
        model.setup_tag_classifier(
            encoder_dim, category_sizes, weight=tag_weight,
            hidden_dim=cfg.training.get('tag_classifier_hidden_dim', 256),
            dropout=cfg.training.get('tag_classifier_dropout', 0.3),
        )
        model.register_buffer('_tag_labels', tag_labels)
        model._tag_class_weights = class_weights
        logging.info("Tag labels: %d samples, %d categories %s", num_samples, len(tag_categories), tag_categories)

        # Validation tag labels
        val_manifest_tags = []
        with open(val_manifest, 'r', encoding='utf-8') as mf:
            for line in mf:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                dur = float(entry.get('duration', 0))
                if min_dur is not None and dur < min_dur:
                    continue
                if max_dur is not None and dur > max_dur:
                    continue
                row = {}
                for cat in tag_categories:
                    row[cat] = int(entry.get(f'tag_{cat.lower()}', 0) or 0)
                val_manifest_tags.append(row)

        val_dataset = model._validation_dl.dataset
        val_num = len(val_dataset.manifest_processor.collection)
        val_tag_labels = torch.zeros(val_num, len(tag_categories), dtype=torch.long)
        for i in range(min(val_num, len(val_manifest_tags))):
            for j, cat in enumerate(tag_categories):
                val_tag_labels[i, j] = val_manifest_tags[i][cat]
        model.register_buffer('_val_tag_labels', val_tag_labels)
        logging.info("Validation tag labels: %d samples", val_num)

        # Oversampling
        oversample_factor = float(cfg.training.get('keyphrase_oversample_factor', 0.0))
        oversample_categories = cfg.training.get('oversample_categories', [])
        if oversample_factor > 0.0 and manifest_tags:
            import numpy as np
            tag_sample_weights = np.ones(num_samples, dtype=np.float32)
            for cat_name in oversample_categories:
                if cat_name not in tag_categories:
                    continue
                j = tag_categories.index(cat_name)
                labels_col = tag_labels[:num_samples, j].numpy()
                counts = np.bincount(labels_col, minlength=category_sizes[cat_name])
                max_count = float(counts[counts > 0].max()) if counts.any() else 1.0
                for i in range(num_samples):
                    c = int(labels_col[i])
                    if c > 0 and counts[c] > 0:
                        ratio = max_count / counts[c]
                        boost = 1.0 + oversample_factor * (np.sqrt(ratio) - 1.0)
                        tag_sample_weights[i] = max(tag_sample_weights[i], boost)
            train_dataset_tmp.sample_keyphrase_weights = tag_sample_weights
            logging.info("Tag oversampling: min=%.2f, max=%.2f, mean=%.2f",
                         tag_sample_weights.min(), tag_sample_weights.max(), tag_sample_weights.mean())

    # Balanced batch sampler
    train_dataset = model._train_dl.dataset
    train_batch_sampler = BalancedLanguageBatchSampler(
        train_dataset, cfg.training.batch_size,
        lang_to_family_map=language_family_map,
    )
    model._train_dl = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=train_dataset.collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
    )

    # Audio augmentation
    aug_cfg = cfg.get('augmentation', {})
    if aug_cfg and aug_cfg.get('enabled', False):
        _meta_asr_dir = '/mnt/nfs/code/PromptingNemo/scripts/asr/meta-asr'
        if _meta_asr_dir not in sys.path:
            sys.path.insert(0, _meta_asr_dir)
        from ambient_noise import build_augmentor_from_config
        augmentor = build_augmentor_from_config(aug_cfg)
    else:
        noise_perturb = WhiteNoisePerturbation(min_level=-90, max_level=-46)
        shift_perturb = ShiftPerturbation(min_shift_ms=100.0, max_shift_ms=500.0)
        augmentor = AudioAugmentor(perturbations=[(1.0, noise_perturb), (1.0, shift_perturb)])
    if hasattr(model, '_train_dl') and model._train_dl is not None:
        model._train_dl.dataset.augmentor = augmentor

    # Spec augment
    spec_cfg = cfg.training.get('spec_augment')
    if spec_cfg and hasattr(model.cfg, 'spec_augment'):
        with open_dict(model.cfg):
            if 'time_masks' in spec_cfg:
                model.cfg.spec_augment.time_masks = spec_cfg['time_masks']
            if 'time_width' in spec_cfg:
                model.cfg.spec_augment.time_width = spec_cfg['time_width']
        model.spec_augmentation = model.from_config_dict(model.cfg.spec_augment)

    # Optimizer
    optim_cfg = cfg.training.get('optim')
    if optim_cfg:
        with open_dict(model.cfg):
            if 'lr' in optim_cfg:
                model.cfg.optim.lr = optim_cfg['lr']
            if 'weight_decay' in optim_cfg:
                model.cfg.optim.weight_decay = optim_cfg['weight_decay']
            if 'sched' in optim_cfg and hasattr(model.cfg.optim, 'sched'):
                sched_override = optim_cfg['sched']
                if 'name' in sched_override and sched_override['name'] != model.cfg.optim.sched.get('name'):
                    for k in ('d_model',):
                        if k in model.cfg.optim.sched:
                            del model.cfg.optim.sched[k]
                for key in ('name', 'warmup_steps', 'min_lr', 'max_steps', 'warmup_ratio'):
                    if key in sched_override:
                        model.cfg.optim.sched[key] = sched_override[key]
            logging.info("Optimizer: lr=%s, sched=%s", model.cfg.optim.lr,
                         OmegaConf.to_container(model.cfg.optim.sched))
    model.setup_optimization(model.cfg.optim)

    # Trainer
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = cfg.training.get('devices', None)
    if accelerator == 'cpu':
        devices = devices if isinstance(devices, int) and devices > 0 else 1
    else:
        devices = devices if devices is not None else -1

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
    if accelerator == 'gpu' and (devices == -1 or (isinstance(devices, int) and devices != 1)):
        trainer_kwargs['strategy'] = 'ddp'

    val_interval = cfg.experiment.get('every_n_train_steps')
    if val_interval and val_interval > 0:
        trainer_kwargs['val_check_interval'] = val_interval

    accumulate = cfg.training.get('accumulate_grad_batches', None)
    if accumulate and accumulate > 1:
        trainer_kwargs['accumulate_grad_batches'] = accumulate

    trainer = pl.Trainer(**trainer_kwargs, callbacks=[ValidationMetricsPrinter()])
    model.set_trainer(trainer)

    every_n = cfg.experiment.get('every_n_train_steps', None)
    callback_params = exp_manager.CallbackParams(
        monitor=cfg.experiment.monitor,
        mode=cfg.experiment.mode,
        always_save_nemo=cfg.experiment.always_save_nemo,
        save_top_k=cfg.experiment.get('save_top_k', 1),
        every_n_train_steps=every_n,
        every_n_epochs=0 if every_n else 1,
    )
    exp_cfg = exp_manager.ExpManagerConfig(
        exp_dir=cfg.experiment.get('exp_dir', None),
        name=cfg.experiment.get('exp_name', None),
        checkpoint_callback_params=callback_params,
        resume_if_exists=cfg.experiment.get('resume_if_exists', False),
        resume_past_end=cfg.experiment.get('resume_past_end', False),
        resume_ignore_no_checkpoint=True,
        create_checkpoint_callback=True,
        create_tensorboard_logger=True,
    )
    exp_cfg = OmegaConf.structured(exp_cfg)
    exp_manager.exp_manager(trainer, exp_cfg)

    logging.info("Starting Hybrid RNN-T/CTC training...")
    logging.info("Model: %s", model.__class__.__name__)
    logging.info("Loss: (1-%.2f)*RNNT + %.2f*CTC + %.1f*TagClassifier",
                 model.ctc_loss_weight, model.ctc_loss_weight,
                 cfg.training.get('tag_classifier_weight', 0.0))
    trainer.fit(model, ckpt_path=ckpt_path)
    logging.info("Hybrid training complete.")
