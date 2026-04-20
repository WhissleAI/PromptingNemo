#!/usr/bin/env python3
"""Audio-Visual ASR training entry point.

Loads a YAML config, optionally overrides the SNR setting, and trains an
AV-CTC model using the PromptingNemo framework. The model combines a
Conformer audio encoder with CLIP visual features through a Transformer
fusion module, and is trained with CTC loss on transcripts that include
an appended noise class label.

Usage:
    python train.py --config conf/av_conformer_ctc.yaml --snr rand --gpus 1
    python train.py --config conf/audio_only_baseline.yaml --snr 10.0 --gpus 2
    python train.py --config conf/av_conformer_ctc.yaml --resume /path/to/ckpt
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Audio-Visual ASR model (Conformer + CLIP + Transformer fusion)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., conf/av_conformer_ctc.yaml)",
    )
    parser.add_argument(
        "--snr",
        type=str,
        default=None,
        help=(
            "Override SNR ratio. Use 'rand' for uniform random sampling "
            "from [snr_min, snr_max], or a float value (e.g., '10.0') for "
            "fixed-SNR training. Defaults to the value in the config file."
        ),
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for training (default: 1)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint (.ckpt) to resume training from",
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        default=None,
        help="Override the training manifest path from the config",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default=None,
        help="Override the validation manifest path from the config",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="Override the experiment output directory from the config",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default=None,
        help="Override the tokenizer directory path from the config",
    )
    return parser.parse_args()


def _validate_required_field(cfg, field_path: str, field_name: str):
    """Check that a dotted config path is set (not '???')."""
    val = cfg
    for part in field_path.split("."):
        val = getattr(val, part, "???")
    if val == "???":
        logger.error(
            "Required config field '%s' (%s) is not set. "
            "Set it in the config file or via the corresponding CLI flag.",
            field_path,
            field_name,
        )
        sys.exit(1)


def build_av_model(cfg):
    """Build the Audio-Visual CTC model from config.

    Loads the pretrained Conformer encoder, configures the Transformer fusion
    module and CTC decoder, and applies adapter settings.
    """
    import torch
    import torch.nn as nn
    from omegaconf import OmegaConf, open_dict
    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding, CTCDecodingConfig
    from nemo.collections.asr.metrics.wer import WER

    # Resolve the base model
    a_model_name = cfg.get("a_model_name", "")
    if a_model_name.startswith("BPE:"):
        nemo_model_name = a_model_name.split(":", 1)[1]
    else:
        nemo_model_name = a_model_name

    logger.info("Loading pretrained audio encoder: %s", nemo_model_name)

    # Load pretrained model config and weights
    base_model = ASRModel.from_pretrained(nemo_model_name, return_config=False)
    base_cfg = base_model.cfg.copy()

    # Update data configs from our recipe config
    with open_dict(base_cfg):
        base_cfg.train_ds.manifest_filepath = cfg.train_ds.manifest_filepath
        base_cfg.train_ds.batch_size = cfg.train_ds.batch_size
        base_cfg.train_ds.max_duration = cfg.train_ds.max_duration
        base_cfg.train_ds.shuffle = cfg.train_ds.get("shuffle", True)
        base_cfg.train_ds.num_workers = cfg.train_ds.get("num_workers", 8)
        base_cfg.train_ds.pin_memory = cfg.train_ds.get("pin_memory", True)

        base_cfg.validation_ds.manifest_filepath = cfg.validation_ds.manifest_filepath
        base_cfg.validation_ds.batch_size = cfg.validation_ds.batch_size
        base_cfg.validation_ds.max_duration = cfg.validation_ds.max_duration
        base_cfg.validation_ds.shuffle = False
        base_cfg.validation_ds.num_workers = cfg.validation_ds.get("num_workers", 8)
        base_cfg.validation_ds.pin_memory = cfg.validation_ds.get("pin_memory", True)

        # Store AV-specific config sections
        base_cfg.use_video_modality = cfg.get("use_video_modality", True)
        base_cfg.av_encoder = cfg.av_encoder
        base_cfg.v_model = cfg.v_model
        base_cfg.train_ds.video_frame_rate = cfg.train_ds.video_frame_rate
        base_cfg.train_ds.get_vid_feats = cfg.train_ds.get("get_vid_feats", True)
        base_cfg.train_ds.get_zero_vid_feats = cfg.train_ds.get("get_zero_vid_feats", False)
        base_cfg.train_ds.override_snr_ratio = cfg.train_ds.get("override_snr_ratio", "rand")
        base_cfg.train_ds.snr_min = cfg.train_ds.get("snr_min", -5.0)
        base_cfg.train_ds.snr_max = cfg.train_ds.get("snr_max", 5.0)

    # The AVEncDecCTCModelBPE class extends the base NeMo CTC model with:
    # - A Transformer fusion encoder (av_encoder)
    # - Video feature projection layer
    # - Modified forward pass that concatenates audio + video features
    try:
        from promptingnemo.models.av_ctc_model import AVEncDecCTCModelBPE
    except ImportError:
        logger.error(
            "Could not import AVEncDecCTCModelBPE. "
            "Ensure promptingnemo is installed with AV support: "
            "pip install -e '.[train]' from the repo root.\n"
            "The AV model class should be at promptingnemo/models/av_ctc_model.py"
        )
        sys.exit(1)

    model = AVEncDecCTCModelBPE(cfg=base_cfg)

    # Apply adapter configuration
    adapter_cfg = cfg.get("adapter", {})
    if adapter_cfg and adapter_cfg.get("enabled", False):
        from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig

        adapter_name = adapter_cfg.get("name", "linear_adapter")
        adapter_dim = adapter_cfg.get("dim", 64)
        adapter_act = adapter_cfg.get("activation", "swish")
        adapter_norm = adapter_cfg.get("norm_position", "pre")

        adapter_config = LinearAdapterConfig(
            in_features=model.encoder._feat_out,
            dim=adapter_dim,
            activation=adapter_act,
            norm_position=adapter_norm,
        )

        existing = model.get_enabled_adapters() if hasattr(model, "get_enabled_adapters") else []
        if adapter_name in existing:
            logger.info("Adapter '%s' already exists -- reusing", adapter_name)
        else:
            model.add_adapter(name=adapter_name, cfg=adapter_config)
            logger.info("Added adapter '%s' (dim=%d) to encoder", adapter_name, adapter_dim)
        model.set_enabled_adapters(enabled=True)

    # Setup tokenizer
    tokenizer_cfg = cfg.get("tokenizer", {})
    if tokenizer_cfg:
        tokenizer_dir = tokenizer_cfg.get("dir", "")
        tokenizer_type = tokenizer_cfg.get("type", "bpe")
        if tokenizer_dir:
            model.change_vocabulary(
                new_tokenizer_dir=tokenizer_dir,
                new_tokenizer_type=tokenizer_type,
            )
            logger.info("Updated tokenizer from %s (type=%s)", tokenizer_dir, tokenizer_type)

    # Reinitialize decoding and WER with the updated vocabulary
    vocab = list(model.decoder.vocabulary)
    decoding_cfg = model.cfg.get("decoding", OmegaConf.create({"strategy": "greedy"}))
    model.decoding = CTCDecoding(decoding_cfg=decoding_cfg, vocabulary=vocab)
    model.wer = WER(
        decoding=model.decoding,
        use_cer=model._cfg.get("use_cer", False),
        log_prediction=model._cfg.get("log_prediction", True),
    )
    logger.info("Initialized decoding/WER with %d-token vocabulary", len(vocab))

    return model


def setup_av_training(model, cfg):
    """Setup data loaders and optimizer for AV training."""
    from omegaconf import open_dict

    # Setup data
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)

    # Setup optimizer
    optim_cfg = cfg.get("optim", {})
    if optim_cfg:
        with open_dict(model.cfg):
            if "name" in optim_cfg:
                model.cfg.optim.name = optim_cfg.name
            if "lr" in optim_cfg:
                model.cfg.optim.lr = optim_cfg.lr
            if "betas" in optim_cfg:
                model.cfg.optim.betas = optim_cfg.betas
            if "weight_decay" in optim_cfg:
                model.cfg.optim.weight_decay = optim_cfg.weight_decay
            if "sched" in optim_cfg and hasattr(model.cfg.optim, "sched"):
                sched = optim_cfg.sched
                if "name" in sched:
                    model.cfg.optim.sched.name = sched.name
                if "warmup_steps" in sched:
                    model.cfg.optim.sched.warmup_steps = sched.warmup_steps
                if "d_model" in sched:
                    model.cfg.optim.sched.d_model = sched.d_model
                if "min_lr" in sched:
                    model.cfg.optim.sched.min_lr = sched.min_lr

    model.setup_optimization(model.cfg.optim)

    # Setup spec augmentation
    spec_cfg = cfg.get("spec_augment", {})
    if spec_cfg and hasattr(model.cfg, "spec_augment"):
        with open_dict(model.cfg):
            for key in ["freq_masks", "freq_width", "time_masks", "time_width"]:
                if key in spec_cfg:
                    setattr(model.cfg.spec_augment, key, spec_cfg[key])
        model.spec_augmentation = model.from_config_dict(model.cfg.spec_augment)
        logger.info("Configured spec augmentation")


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    # Deferred imports so --help works without heavy dependencies
    import yaml
    from omegaconf import OmegaConf, open_dict

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = OmegaConf.create(yaml.safe_load(f))

    # --- Apply CLI overrides ---
    with open_dict(cfg):
        if args.snr is not None:
            snr_value = args.snr if args.snr == "rand" else float(args.snr)
            cfg.train_ds.override_snr_ratio = snr_value
            logger.info("Overriding train SNR ratio to: %s", snr_value)

        cfg.trainer.devices = args.gpus

        if args.train_manifest:
            cfg.train_ds.manifest_filepath = args.train_manifest
        if args.val_manifest:
            cfg.validation_ds.manifest_filepath = args.val_manifest
        if args.exp_dir:
            cfg.experiment.exp_dir = args.exp_dir
        if args.tokenizer_dir:
            cfg.tokenizer.dir = args.tokenizer_dir

    # Validate required fields
    for field_path, field_name in [
        ("train_ds.manifest_filepath", "train manifest"),
        ("validation_ds.manifest_filepath", "validation manifest"),
        ("experiment.exp_dir", "experiment directory"),
        ("tokenizer.dir", "tokenizer directory"),
    ]:
        _validate_required_field(cfg, field_path, field_name)

    # --- Log configuration summary ---
    use_video = cfg.get("use_video_modality", False)
    snr_setting = cfg.train_ds.get("override_snr_ratio", "N/A")
    model_variant = "AV" if use_video else "Audio-only"
    snr_label = "UNI-SNR" if snr_setting == "rand" else f"SNR={snr_setting}"

    logger.info("=" * 60)
    logger.info("  Audio-Visual ASR Training")
    logger.info("  Config      : %s", config_path)
    logger.info("  Model       : %s (%s)", model_variant, snr_label)
    logger.info("  GPUs        : %d", args.gpus)
    logger.info("  Video       : %s", "enabled" if use_video else "disabled")
    logger.info("  Train data  : %s", cfg.train_ds.manifest_filepath)
    logger.info("  Val data    : %s", cfg.validation_ds.manifest_filepath)
    logger.info("  Experiment  : %s/%s", cfg.experiment.exp_dir, cfg.experiment.exp_name)
    if args.resume:
        logger.info("  Resume from : %s", args.resume)
    logger.info("=" * 60)

    # --- Import heavy dependencies and train ---
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import torch
    import lightning.pytorch as pl
    from nemo.utils import exp_manager

    # Build model
    model = build_av_model(cfg)

    # Setup Lightning trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = cfg.trainer.devices if accelerator == "gpu" else 1

    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=devices,
        max_steps=cfg.trainer.max_steps,
        accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 1),
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", 1.0),
        gradient_clip_algorithm=cfg.trainer.get("gradient_clip_algorithm", "norm"),
        val_check_interval=cfg.experiment.get("every_n_train_steps", 5000),
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 50),
        precision=cfg.trainer.get("precision", "16-mixed"),
        enable_checkpointing=False,
        logger=False,
        use_distributed_sampler=False,
    )

    if accelerator == "gpu" and (devices == -1 or (isinstance(devices, int) and devices > 1)):
        trainer_kwargs["strategy"] = "ddp"

    trainer = pl.Trainer(**trainer_kwargs)
    model.set_trainer(trainer)

    # Experiment manager (checkpointing + TensorBoard)
    every_n_train_steps = cfg.experiment.get("every_n_train_steps", None)
    callback_params = exp_manager.CallbackParams(
        monitor=cfg.experiment.monitor,
        mode=cfg.experiment.mode,
        always_save_nemo=cfg.experiment.always_save_nemo,
        save_top_k=cfg.experiment.get("save_top_k", 3),
        every_n_train_steps=every_n_train_steps,
        every_n_epochs=0 if every_n_train_steps else 1,
    )
    exp_cfg = exp_manager.ExpManagerConfig(
        exp_dir=cfg.experiment.get("exp_dir", None),
        name=cfg.experiment.get("exp_name", None),
        checkpoint_callback_params=callback_params,
        resume_if_exists=cfg.experiment.get("resume_if_exists", False),
        resume_past_end=cfg.experiment.get("resume_past_end", False),
        resume_ignore_no_checkpoint=cfg.experiment.get("resume_ignore_no_checkpoint", True),
        create_checkpoint_callback=True,
        create_tensorboard_logger=True,
        create_wandb_logger=False,
        create_mlflow_logger=False,
        create_dllogger_logger=False,
    )
    exp_cfg = OmegaConf.structured(exp_cfg)
    exp_manager.exp_manager(trainer, exp_cfg)

    # Setup data loaders and optimizer
    setup_av_training(model, cfg)

    # Train
    logger.info("Starting training...")
    trainer.fit(model, ckpt_path=args.resume)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
