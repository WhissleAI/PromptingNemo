"""Model training orchestration: checkpoint loading, data setup, and training loop."""

import json
import logging
import os
import sys
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
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding, CTCDecodingConfig, CTCBPEDecoding
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


def _restore_checkpoint_decoder(model, connector, ckpt_decoder_vocab):
    """Replace model's decoder with the checkpoint's original decoder.

    After restore_from with FlexibleSaveRestoreConnector, the decoder may have
    been created with the wrong vocabulary size (from the tokenizer). This
    function rebuilds the decoder using the checkpoint's actual vocabulary and
    weights, extends the tokenizer to match, and re-initialises decoding/loss.
    """
    from nemo.collections.asr.modules.conv_asr import ConvASRDecoder
    from nemo.collections.asr.losses.ctc import CTCLoss

    skipped = getattr(connector, '_skipped_state', {})
    decoder_weights = {}
    for key, tensor in skipped.items():
        if key.startswith('decoder.'):
            decoder_weights[key.replace('decoder.', '', 1)] = tensor

    if not decoder_weights:
        current_vocab = list(model.decoder.vocabulary)
        if len(current_vocab) == len(ckpt_decoder_vocab):
            logging.info("Decoder already matches checkpoint (%d tokens) — no replacement needed", len(current_vocab))
            return
        logging.warning("No skipped decoder weights found and vocab sizes differ (%d vs %d)", len(current_vocab), len(ckpt_decoder_vocab))
        return

    feat_in = model.encoder._feat_out
    ckpt_num_classes = len(ckpt_decoder_vocab)

    new_decoder = ConvASRDecoder(
        feat_in=feat_in,
        num_classes=ckpt_num_classes,
        vocabulary=ckpt_decoder_vocab,
    )
    new_decoder.load_state_dict(decoder_weights, strict=True)
    model.decoder = new_decoder

    with open_dict(model.cfg):
        model.cfg.decoder.vocabulary = ckpt_decoder_vocab
        model.cfg.decoder.num_classes = ckpt_num_classes

    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'extend_vocabulary'):
        tokenizer_vocab_set = set()
        if hasattr(model.tokenizer, 'vocabulary'):
            tokenizer_vocab_set = set(model.tokenizer.vocabulary)
        elif hasattr(model.tokenizer, 'tokenizer') and hasattr(model.tokenizer.tokenizer, 'get_vocab'):
            tokenizer_vocab_set = set(model.tokenizer.tokenizer.get_vocab().keys())
        spm_size = len(tokenizer_vocab_set)
        extra_tokens_raw = []
        for tok in ckpt_decoder_vocab:
            if tok not in tokenizer_vocab_set:
                raw = tok.lstrip('▁')
                extra_tokens_raw.append(raw)
        if extra_tokens_raw:
            model.tokenizer.extend_vocabulary(extra_tokens_raw)
            logging.info(
                "Extended tokenizer from %d to %d tokens (+%d from decoder vocabulary)",
                spm_size, spm_size + len(extra_tokens_raw), len(extra_tokens_raw),
            )

    decoding_cfg = model.cfg.get('decoding', OmegaConf.create({'strategy': 'greedy'}))
    model.decoding = CTCBPEDecoding(decoding_cfg=decoding_cfg, tokenizer=model.tokenizer)
    model.wer = WER(
        decoding=model.decoding,
        use_cer=model._cfg.get('use_cer', False),
        log_prediction=model._cfg.get('log_prediction', True),
    )

    model.loss = CTCLoss(
        num_classes=ckpt_num_classes,
        zero_infinity=True,
        reduction='mean_batch',
    )

    logging.info(
        "Restored checkpoint decoder: %d classes (+1 blank), "
        "loaded %d weight tensors, decoder weights shape: %s",
        ckpt_num_classes, len(decoder_weights),
        {k: list(v.shape) for k, v in decoder_weights.items()},
    )


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

    use_tag_classifier = cfg.training.get('use_tag_classifier', False)

    base_cfg = ASRModel.restore_from(restore_path=str(model_path), return_config=True)

    ckpt_decoder_vocab = list(base_cfg.decoder.vocabulary) if hasattr(base_cfg.decoder, 'vocabulary') else []
    if ckpt_decoder_vocab:
        logging.info("Checkpoint decoder vocabulary: %d tokens", len(ckpt_decoder_vocab))

    if use_tag_classifier and ckpt_decoder_vocab:
        import tarfile
        ckpt_tok_dir = model_root / '_ckpt_tokenizer'
        if not (ckpt_tok_dir / 'tokenizer.model').exists():
            ckpt_tok_dir.mkdir(parents=True, exist_ok=True)
            target_suffixes = ('tokenizer.model', 'tokenizer.vocab', 'vocab.txt')
            with tarfile.open(str(model_path), 'r') as tar:
                for member in tar.getmembers():
                    basename = os.path.basename(member.name)
                    for suffix in target_suffixes:
                        if basename.endswith(suffix):
                            f = tar.extractfile(member)
                            if f:
                                with open(ckpt_tok_dir / suffix, 'wb') as dst:
                                    dst.write(f.read())
                            break
            logging.info("Extracted checkpoint tokenizer to %s: %s", ckpt_tok_dir, os.listdir(ckpt_tok_dir))

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
        base_cfg.train_ds.return_sample_id = True

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

        if use_tag_classifier:
            tag_categories = cfg.training.get('tag_categories', ['AGE', 'GENDER', 'EMOTION', 'INTENT'])
            for cat in tag_categories:
                field_name = f'tag_{cat.lower()}'
                if field_name not in base_cfg.train_ds.manifest_processor.additional_fields:
                    base_cfg.train_ds.manifest_processor.additional_fields.append(field_name)

        base_cfg.train_ds.allowed_langs = lang_list
        base_cfg.validation_ds.allowed_langs = lang_list

        if 'augmentor' in base_cfg.train_ds:
            del base_cfg.train_ds.augmentor

        if use_tag_classifier and ckpt_decoder_vocab:
            if hasattr(base_cfg, 'tokenizer') and hasattr(base_cfg.tokenizer, 'langs'):
                for lang in base_cfg.tokenizer.langs:
                    lang_entry = base_cfg.tokenizer.langs[lang]
                    lang_entry.dir = str(ckpt_tok_dir)
                    lang_entry.type = 'bpe'
            else:
                base_cfg.tokenizer = OmegaConf.create({
                    'type': 'agg',
                    'langs': {'ENGLISH': {'type': 'bpe', 'dir': str(ckpt_tok_dir)}},
                })
            base_cfg.decoder.num_classes = -1
            logging.info(
                "Dual-head mode: using checkpoint's %d-token tokenizer, "
                "decoder.num_classes=-1 (will restore %d-class decoder after loading)",
                len(os.listdir(ckpt_tok_dir)), len(ckpt_decoder_vocab),
            )
        else:
            if hasattr(base_cfg, 'tokenizer') and hasattr(base_cfg.tokenizer, 'langs'):
                for lang in base_cfg.tokenizer.langs:
                    lang_entry = base_cfg.tokenizer.langs[lang]
                    if not lang_entry.get('type'):
                        lang_entry.type = tokenizer_langs.get(lang, {}).get('type', 'bpe')
            else:
                base_cfg.tokenizer = OmegaConf.create(tokenizer_entry)

            if hasattr(base_cfg, 'decoder'):
                base_cfg.decoder.num_classes = cfg.model.vocab_size

    from promptingnemo.models.ctc_model import FlexibleSaveRestoreConnector
    connector = FlexibleSaveRestoreConnector()
    model = CustomEncDecCTCModelBPE.restore_from(
        str(model_path), override_config_path=base_cfg, strict=False,
        save_restore_connector=connector,
    )
    model.setup_custom_loss()

    if use_tag_classifier and ckpt_decoder_vocab:
        _restore_checkpoint_decoder(model, connector, ckpt_decoder_vocab)

        current_vocab = set(model.decoder.vocabulary)
        new_tokens = scan_manifest_for_new_tokens(
            train_manifest, current_vocab,
            allowed_prefixes=(
                'ENTITY_', 'INTENT_', 'EMOTION_', 'GENDER_', 'AGE_',
                'DIALECT_', 'KEYWORD_', 'LANG_', 'OTHER_',
                'ROLE_', 'BEHAVIOR_', 'EVAL_',
            ),
        )
        if new_tokens:
            nemo_logging.info(
                "Extending decoder/tokenizer with %d new tags from training data: %s",
                len(new_tokens), new_tokens,
            )
            extend_decoder_for_new_tokens(model, new_tokens)
            if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'extend_vocabulary'):
                model.tokenizer.extend_vocabulary(new_tokens)
            updated_vocab = list(model.decoder.vocabulary)
            decoding_cfg = model.cfg.get('decoding', OmegaConf.create({'strategy': 'greedy'}))
            model.decoding = CTCBPEDecoding(decoding_cfg=decoding_cfg, tokenizer=model.tokenizer)
            model.wer = WER(
                decoding=model.decoding,
                use_cer=model._cfg.get('use_cer', False),
                log_prediction=model._cfg.get('log_prediction', True),
            )
            from nemo.collections.asr.losses.ctc import CTCLoss as _CTCLoss
            model.loss = _CTCLoss(
                num_classes=len(updated_vocab),
                zero_infinity=True,
                reduction='mean_batch',
            )
        model.setup_custom_loss()

    if language_families and not use_tag_classifier:
        slim_decoder_for_training(model, language_families)
        model.setup_custom_loss()

    if not use_tag_classifier:
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
        elif adapter_cfg.get('unfreeze_encoder', False):
            model.freeze()
            model.encoder.unfreeze()
            nemo_logging.info('Unfreezing encoder (unfreeze_encoder=True)')
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            nemo_logging.info(f'After encoder unfreeze: {trainable:,} trainable / {total:,} total params')
        else:
            model.freeze()

    if use_tag_classifier:
        with open_dict(model.cfg):
            model.cfg.train_ds.allowed_langs = lang_list
            model.cfg.validation_ds.allowed_langs = lang_list
            model.cfg.train_ds.lang_field = lang_field
            model.cfg.validation_ds.lang_field = lang_field
            model.cfg.validation_ds.return_sample_id = True
        orig_vocab = list(model.decoder.vocabulary)
        nemo_logging.info(f"Dual-head mode: keeping original tokenizer/decoder ({len(orig_vocab)} tokens)")
    else:
        tokenizer_cfg = OmegaConf.create(tokenizer_entry)

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

    if use_tag_classifier:
        tag_categories = sorted(cfg.training.get('tag_categories', ['AGE', 'GENDER', 'EMOTION', 'INTENT']))
        tag_weight = cfg.training.get('tag_classifier_weight', 0.5)
        train_dataset_tmp = model._train_dl.dataset
        collection = train_dataset_tmp.manifest_processor.collection
        num_samples = len(collection)

        train_manifest = os.path.join(cfg.training.data_dir, cfg.training.train_manifest)
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
                else:
                    weights[c] = 0.0
            class_weights[cat] = weights
            nemo_logging.info(
                "  %s class weights: %s (counts: %s)",
                cat, {c: f"{w:.3f}" for c, w in enumerate(weights.tolist())},
                {c: int(v) for c, v in enumerate(counts.tolist())},
            )

        encoder_dim = model.encoder._feat_out
        tag_hidden_dim = cfg.training.get('tag_classifier_hidden_dim', 256)
        tag_dropout = cfg.training.get('tag_classifier_dropout', 0.3)
        model.setup_tag_classifier(
            encoder_dim, category_sizes, weight=tag_weight,
            hidden_dim=tag_hidden_dim, dropout=tag_dropout,
        )
        model.register_buffer('_tag_labels', tag_labels)
        model._tag_class_weights = class_weights
        nemo_logging.info(
            "Pre-computed %d tag labels for %d categories %s, sizes=%s",
            num_samples, len(tag_categories), tag_categories, category_sizes,
        )

        # --- Tag-based oversampling: boost minority class samples ---
        oversample_factor = float(cfg.training.get('keyphrase_oversample_factor', 0.0))
        oversample_categories = cfg.training.get('oversample_categories', ['BEHAVIOR', 'EVAL'])
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
            train_dataset_tmp = model._train_dl.dataset
            existing = getattr(train_dataset_tmp, 'sample_keyphrase_weights', None)
            if existing is not None and len(existing) == num_samples:
                train_dataset_tmp.sample_keyphrase_weights = existing * tag_sample_weights
            else:
                train_dataset_tmp.sample_keyphrase_weights = tag_sample_weights
            top5 = sorted(enumerate(tag_sample_weights), key=lambda x: -x[1])[:5]
            nemo_logging.info(
                "Tag oversampling (factor=%.1f, cats=%s): "
                "min=%.2f, max=%.2f, mean=%.2f, top5=%s",
                oversample_factor, oversample_categories,
                tag_sample_weights.min(), tag_sample_weights.max(),
                tag_sample_weights.mean(),
                [(idx, f"{w:.1f}") for idx, w in top5],
            )

        val_manifest_path = os.path.join(cfg.training.data_dir, cfg.training.test_manifest)
        val_dataset = model._validation_dl.dataset
        val_collection = val_dataset.manifest_processor.collection
        val_num = len(val_collection)
        val_manifest_tags = []
        with open(val_manifest_path, 'r', encoding='utf-8') as mf:
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

        val_tag_labels = torch.zeros(val_num, len(tag_categories), dtype=torch.long)
        for i in range(min(val_num, len(val_manifest_tags))):
            for j, cat in enumerate(tag_categories):
                val_tag_labels[i, j] = val_manifest_tags[i][cat]
        model.register_buffer('_val_tag_labels', val_tag_labels)
        nemo_logging.info(
            "Pre-computed %d validation tag labels for %d categories",
            val_num, len(tag_categories),
        )

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

    aug_cfg = cfg.get('augmentation', {})
    if aug_cfg and aug_cfg.get('enabled', False):
        _meta_asr_dir = '/mnt/nfs/code/PromptingNemo/scripts/asr/meta-asr'
        if _meta_asr_dir not in sys.path:
            sys.path.insert(0, _meta_asr_dir)
        from ambient_noise import build_augmentor_from_config
        logging.info("Building augmentor from config: %d perturbation types",
                      len(aug_cfg.get('perturbations', [])))
        augmentor = build_augmentor_from_config(aug_cfg)
    else:
        logging.info("Using default audio augmentor (white noise + shift).")
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
                sched_override = optim_cfg['sched']
                if 'name' in sched_override and sched_override['name'] != model.cfg.optim.sched.get('name'):
                    noam_only_keys = {'d_model'}
                    for k in noam_only_keys:
                        if k in model.cfg.optim.sched:
                            del model.cfg.optim.sched[k]
                for key in ('name', 'warmup_steps', 'min_lr', 'max_steps', 'warmup_ratio', 'd_model'):
                    if key in sched_override:
                        model.cfg.optim.sched[key] = sched_override[key]
            logging.info("Optimizer config after override: lr=%s, sched=%s",
                         model.cfg.optim.lr, OmegaConf.to_container(model.cfg.optim.sched))
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
