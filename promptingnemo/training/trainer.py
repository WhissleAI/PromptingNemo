"""Model training orchestration: checkpoint loading, data setup, and training loop."""

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import sentencepiece as spm
import torch
import yaml
import lightning.pytorch as pl
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding, CTCDecodingConfig
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.parts.preprocessing.perturb import WhiteNoisePerturbation, ShiftPerturbation
from nemo.collections.asr.parts.preprocessing.features import AudioAugmentor
from nemo.utils import logging as nemo_logging
from nemo.utils import exp_manager

from promptingnemo.tokenizer.config import (
    load_tokenizer_langs,
    load_shared_special_tokens,
    load_aggregate_vocabulary,
    store_aggregate_vocabulary,
)
from promptingnemo.tokenizer.aggregate import (
    build_aggregate_vocab_from_tokenizers,
    _family_name_for_lang,
)
from promptingnemo.models.ctc_model import CustomEncDecCTCModelBPE
from promptingnemo.models.decoder import (
    scan_manifest_for_new_tokens,
    extend_decoder_for_new_tokens,
    slim_decoder_for_training,
    scale_down_tag_decoder_weights,
)
from promptingnemo.data.dataset import RobustAudioToBPEDataset
from promptingnemo.data.sampler import BalancedLanguageBatchSampler


def save_updated_config(cfg, path):
    container = OmegaConf.to_container(cfg, resolve=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(container, f, sort_keys=False)


def train_model(cfg, ckpt_path=None):
    lang_field = cfg.training.get('lang_field', 'lang')
    RobustAudioToBPEDataset.default_lang_field = lang_field
    tokenizer_langs = load_tokenizer_langs(cfg)
    if not isinstance(tokenizer_langs, dict) or not tokenizer_langs:
        raise RuntimeError(
            "No tokenizer_langs mapping found in config. Run with --mode tokenizer first, or add the mapping manually."
        )

    shared_special_tokens = load_shared_special_tokens(cfg)
    language_family_map = cfg.model.get('language_family_map', {})
    if language_family_map:
        logging.info(
            "Using language family assignments for %d languages",
            len(language_family_map),
        )

    language_families = cfg.model.get('language_families', [])
    if language_families:
        families_upper = {f.upper() for f in language_families}
        all_langs = list(tokenizer_langs.keys())
        tokenizer_langs = {k: v for k, v in tokenizer_langs.items() if k.upper() in families_upper}
        if not tokenizer_langs:
            raise RuntimeError(
                f"No tokenizer_langs matched language_families {language_families}. "
                f"Available: {all_langs}"
            )
        logging.info(
            f"Filtered tokenizer_langs to target families {sorted(families_upper)}: "
            f"{sorted(tokenizer_langs.keys())} (removed {len(all_langs) - len(tokenizer_langs)} non-target)"
        )

    aggregate_vocab = load_aggregate_vocabulary(cfg)
    if not aggregate_vocab:
        logging.warning("Aggregate vocabulary missing from config; rebuilding from tokenizer directories.")
        aggregate_vocab = build_aggregate_vocab_from_tokenizers(tokenizer_langs)
        store_aggregate_vocabulary(cfg, aggregate_vocab)

    lang_list = sorted(tokenizer_langs.keys())
    logging.info(f"Training model with aggregate tokenizer covering languages: {lang_list}")

    model_root = Path(cfg.model.model_root)
    model_path = model_root / cfg.model.model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find base model checkpoint at {model_path}")

    # Sanity-check tokenizer contents
    if shared_special_tokens:
        for lang, lang_cfg in tokenizer_langs.items():
            tok_model_path = Path(lang_cfg['dir']) / 'tokenizer.model'
            if not tok_model_path.exists():
                logging.warning(f"Tokenizer model missing for language {lang}: {tok_model_path}")
                continue
            processor = spm.SentencePieceProcessor()
            processor.Load(str(tok_model_path))
            unk_id = processor.unk_id() if hasattr(processor, 'unk_id') else 0
            missing = []
            for tok in shared_special_tokens:
                tok_id = processor.piece_to_id(tok)
                if tok_id == unk_id:
                    missing.append(tok)
            if missing:
                logging.warning(
                    f"Tokenizer {lang} missing {len(missing)} shared special tokens; first few: {missing[:10]}"
                )

    train_manifest = str(Path(cfg.training.data_dir) / cfg.training.train_manifest)
    val_manifest = str(Path(cfg.training.data_dir) / cfg.training.test_manifest)

    tokenizer_entry = {'type': 'agg', 'langs': tokenizer_langs}
    if shared_special_tokens:
        tokenizer_entry['special_tokens'] = shared_special_tokens

    base_cfg = ASRModel.restore_from(restore_path=str(model_path), return_config=True)
    with open_dict(base_cfg):
        adapter_cfg_section = cfg.get('adapter', {})
        if adapter_cfg_section and adapter_cfg_section.get('enabled', False):
            base_cfg.encoder._target_ = 'nemo.collections.asr.modules.conformer_encoder.ConformerEncoderAdapter'

        base_cfg.use_keyword_loss = cfg.training.get('use_keyword_loss', False)
        base_cfg.keyword_loss_weight = cfg.training.get('keyword_loss_weight', 0.3)
        base_cfg.keyword_loss_warmup_steps = cfg.training.get('keyword_loss_warmup_steps', 0)

        base_cfg.train_ds.manifest_filepath = train_manifest
        base_cfg.train_ds.batch_size = cfg.training.batch_size
        base_cfg.train_ds.max_duration = cfg.training.max_duration
        base_cfg.train_ds.shuffle = True
        base_cfg.train_ds.is_tarred = False
        base_cfg.train_ds.tarred_audio_filepaths = None
        base_cfg.train_ds.num_workers = cfg.training.num_workers
        base_cfg.train_ds.pin_memory = cfg.training.pin_memory
        base_cfg.train_ds.lang_field = lang_field
        base_cfg.train_ds.return_sample_id = True  # Needed for family loss weights

        base_cfg.validation_ds.manifest_filepath = val_manifest
        base_cfg.validation_ds.batch_size = cfg.training.batch_size
        base_cfg.validation_ds.max_duration = cfg.training.max_duration
        base_cfg.validation_ds.shuffle = False
        base_cfg.validation_ds.num_workers = cfg.training.num_workers
        base_cfg.validation_ds.pin_memory = cfg.training.pin_memory
        base_cfg.validation_ds.lang_field = lang_field
        base_cfg.validation_ds.return_sample_id = True

        if 'manifest_processor' not in base_cfg.train_ds:
            base_cfg.train_ds.manifest_processor = {}
        if 'additional_fields' not in base_cfg.train_ds.manifest_processor:
            base_cfg.train_ds.manifest_processor.additional_fields = []
        fields_to_add = {'lang'}
        if lang_field:
            fields_to_add.add(lang_field)
        for field_name in fields_to_add:
            if field_name and field_name not in base_cfg.train_ds.manifest_processor.additional_fields:
                base_cfg.train_ds.manifest_processor.additional_fields.append(field_name)

        base_cfg.train_ds.allowed_langs = lang_list
        base_cfg.validation_ds.allowed_langs = lang_list

        if 'augmentor' in base_cfg.train_ds:
            del base_cfg.train_ds.augmentor

    model = CustomEncDecCTCModelBPE.restore_from(str(model_path), override_config_path=base_cfg, strict=True)
    model.setup_custom_loss()

    if language_families:
        slim_decoder_for_training(model, language_families)
        model.setup_custom_loss()

    current_vocab = set(model.decoder.vocabulary)
    new_tokens = scan_manifest_for_new_tokens(train_manifest, current_vocab)
    if new_tokens:
        nemo_logging.info(f"Found {len(new_tokens)} special tokens in training data missing from model vocabulary: {new_tokens}")
        extend_decoder_for_new_tokens(model, new_tokens)
        if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'extend_vocabulary'):
            model.tokenizer.extend_vocabulary(new_tokens)
        model.setup_custom_loss()

    scale_down_tag_decoder_weights(model, scale_factor=0.01)

    adapter_cfg = cfg.get('adapter', {})
    if adapter_cfg and adapter_cfg.get('enabled', False):
        adapter_name = adapter_cfg.get('name', 'lang_adapter')
        adapter_dim = adapter_cfg.get('dim', 128)
        adapter_act = adapter_cfg.get('activation', 'swish')
        adapter_norm = adapter_cfg.get('norm_position', 'pre')
        adapter_config = LinearAdapterConfig(
            in_features=model.encoder._feat_out,
            dim=adapter_dim,
            activation=adapter_act,
            norm_position=adapter_norm,
        )
        existing = model.get_enabled_adapters() if hasattr(model, 'get_enabled_adapters') else []
        if adapter_name in existing:
            nemo_logging.info("Adapter '%s' already exists in checkpoint -- reusing", adapter_name)
        else:
            model.add_adapter(name=adapter_name, cfg=adapter_config)
            nemo_logging.info("Added adapter '%s' (dim=%d) to encoder layers", adapter_name, adapter_dim)
        model.set_enabled_adapters(enabled=True)
        if adapter_cfg.get('unfreeze_decoder', False):
            model.encoder.freeze()
            model.decoder.unfreeze()
        else:
            model.freeze()

    tokenizer_cfg = OmegaConf.create(tokenizer_entry)
    #logging.info("Applying deduplicated aggregate tokenizer via change_vocabulary().")
    #model.change_vocabulary(tokenizer_cfg, 'agg')

    aggregate_vocab = list(model.decoder.vocabulary)
    store_aggregate_vocabulary(cfg, aggregate_vocab)

    with open_dict(model.cfg):
        model.cfg.tokenizer = tokenizer_entry
        model.cfg.train_ds.allowed_langs = lang_list
        model.cfg.validation_ds.allowed_langs = lang_list
        model.cfg.train_ds.lang_field = lang_field
        model.cfg.validation_ds.lang_field = lang_field
        model.cfg.validation_ds.return_sample_id = True
        model.cfg.decoder.vocabulary = aggregate_vocab
        model.cfg.decoder.num_classes = len(aggregate_vocab)

    decoding_cfg = model.cfg.get('decoding', OmegaConf.create({'strategy': 'greedy'}))
    model.decoding = CTCDecoding(decoding_cfg=decoding_cfg, vocabulary=aggregate_vocab)
    model.wer = WER(
        decoding=model.decoding,
        use_cer=model._cfg.get('use_cer', False),
        log_prediction=model._cfg.get('log_prediction', True),
    )
    nemo_logging.info(f"Re-initialized decoding/WER with {len(aggregate_vocab)}-token vocabulary")

    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    model.setup_multiple_test_data(model.cfg.validation_ds)
    model._validation_dataset_ref = getattr(model._validation_dl, 'dataset', None)

    # --- Calculate and set language family loss weights ---
    if cfg.training.get('use_family_loss_weights'):
        logging.info("Calculating language family loss weights...")
        train_dataset = model._train_dl.dataset
        family_counts = defaultdict(int)
        for lang_id in train_dataset.language_ids:
            family = _family_name_for_lang(lang_id)
            family_counts[family] += 1

        total_samples = sum(family_counts.values())
        num_families = len(family_counts)

        if total_samples > 0 and num_families > 0:
            # Normalized inverse frequency weighting
            weights = {
                fam: total_samples / (num_families * count)
                for fam, count in family_counts.items()
            }
            logging.info("Calculated language family loss weights: %s", weights)
            model.set_family_loss_weights(weights)
        else:
            logging.warning("Could not calculate family loss weights: no samples or families found.")

    train_dataset = model._train_dl.dataset
    train_batch_sampler = BalancedLanguageBatchSampler(
        train_dataset,
        cfg.training.batch_size,
        lang_to_family_map=language_family_map,
    )
    model._train_dl = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=train_dataset.collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        )

    logging.info("Manually creating and injecting audio augmentor for training.")
    noise_perturb = WhiteNoisePerturbation(min_level=-90, max_level=-46)
    shift_perturb = ShiftPerturbation(min_shift_ms=100.0, max_shift_ms=500.0)
    augmentor = AudioAugmentor(perturbations=[
        (1.0, noise_perturb),
    (1.0, shift_perturb),
    ])

    if hasattr(model, '_train_dl') and model._train_dl is not None:
        model._train_dl.dataset.augmentor = augmentor
    else:
        logging.warning("Could not find the training dataloader to inject augmentor.")

    spec_cfg = cfg.training.get('spec_augment')
    if spec_cfg and hasattr(model.cfg, 'spec_augment'):
        with open_dict(model.cfg):
            if 'time_masks' in spec_cfg:
                model.cfg.spec_augment.time_masks = spec_cfg['time_masks']
            if 'time_width' in spec_cfg:
                model.cfg.spec_augment.time_width = spec_cfg['time_width']
        model.spec_augmentation = model.from_config_dict(model.cfg.spec_augment)

    optim_cfg = cfg.training.get('optim')
    if optim_cfg:
        with open_dict(model.cfg):
            if 'lr' in optim_cfg:
                model.cfg.optim.lr = optim_cfg['lr']
            if 'weight_decay' in optim_cfg:
                model.cfg.optim.weight_decay = optim_cfg['weight_decay']
            if 'sched' in optim_cfg and hasattr(model.cfg.optim, 'sched'):
                if 'warmup_steps' in optim_cfg['sched']:
                    model.cfg.optim.sched.warmup_steps = optim_cfg['sched']['warmup_steps']
    model.setup_optimization(model.cfg.optim)

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    configured_devices = cfg.training.get('devices', None)
    if accelerator == 'cpu':
        devices = configured_devices if isinstance(configured_devices, int) and configured_devices > 0 else 1
    else:
        devices = configured_devices if configured_devices is not None else -1

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

    if accelerator == 'gpu':
        if devices == -1 or (isinstance(devices, int) and devices != 1):
            trainer_kwargs['strategy'] = 'ddp'

    val_interval = cfg.experiment.get('every_n_train_steps')
    if val_interval and val_interval > 0:
        trainer_kwargs['val_check_interval'] = val_interval

    accumulate_grad_batches = cfg.training.get('accumulate_grad_batches', None)
    if accumulate_grad_batches and accumulate_grad_batches > 1:
        trainer_kwargs['accumulate_grad_batches'] = accumulate_grad_batches

    trainer = pl.Trainer(**trainer_kwargs)
    model.set_trainer(trainer)

    every_n_train_steps = cfg.experiment.get('every_n_train_steps', None)
    callback_params = exp_manager.CallbackParams(
        monitor=cfg.experiment.monitor,
        mode=cfg.experiment.mode,
        always_save_nemo=cfg.experiment.always_save_nemo,
        save_top_k=cfg.experiment.get('save_top_k', 1),
        every_n_train_steps=every_n_train_steps,
        every_n_epochs=0 if every_n_train_steps else 1,
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
        create_wandb_logger=False,
        create_mlflow_logger=False,
        create_dllogger_logger=False,
    )
    exp_cfg = OmegaConf.structured(exp_cfg)
    exp_manager.exp_manager(trainer, exp_cfg)

    logging.info("Starting model training...")
    logging.info("Model class: %s", model.__class__)
    logging.info("LightningModule class from pytorch_lightning: %s", pl.LightningModule)
    try:
        import lightning.pytorch as L

        logging.info(
            "LightningModule class from lightning.pytorch: %s", getattr(L, "LightningModule", None)
        )
        logging.info(
            "isinstance(model, pytorch_lightning.LightningModule): %s",
            isinstance(model, pl.LightningModule),
        )
        lightning_module_cls = getattr(L, "LightningModule", None)
        if lightning_module_cls is not None:
            logging.info(
                "isinstance(model, lightning.pytorch.LightningModule): %s",
                isinstance(model, lightning_module_cls),
            )
    except Exception as exc:
        logging.warning("Failed to inspect lightning module classes: %s", exc)

    trainer.fit(model, ckpt_path=ckpt_path)
    logging.info("Model training complete.")
