# Meta-ASR Pipeline Guide

This document describes the end-to-end workflow for preparing multilingual
speech data, training language-family-aware tokenizers, and fine-tuning NeMo ASR
models with the meta-ASR scripts in this directory. The tooling is designed for
NVIDIA's NeMo containers, but the instructions apply to any environment with
compatible GPU drivers and the required Python dependencies.

---

## Quick Start

```bash
# 1. Validate manifests
python main.py --mode validate_data \
  --config config/config_peoplespeech.yml

# 2. Train the language-family tokenizers + aggregate vocabulary
python main.py --mode tokenizer \
  --config config/config_peoplespeech.yml

# 3. Fine-tune the ASR model with the aggregate tokenizer
python main.py --mode train \
  --config config/config_peoplespeech.yml

# Run all stages sequentially
python main.py --mode both \
  --config config/config_peoplespeech.yml
```

---

## Environment and Prerequisites

- NVIDIA GPU with up-to-date drivers.
- Docker (recommended) or a Python environment with NeMo, PyTorch Lightning,
  and project requirements installed.
- Training manifests in NeMo JSONL format containing `audio_filepath`, `text`,
  and uppercase language identifiers (`lang` or the `lang_field` specified in
  the config).
- Base `.nemo` checkpoint referenced in `config/config_peoplespeech.yml`.

### Recommended Container Launch

```bash
docker run --gpus all -it --rm \
  -v /path/to/PromptingNemo:/workspace/PromptingNemo \
  -v /path/to/datasets:/external/home/ksingla/data \
  -v /path/to/pretrained:/workspace/pretrained \
  nvcr.io/nvidia/nemo:24.05.01 \
  bash
```

Inside the container:

```bash
cd /workspace/PromptingNemo/scripts/asr/meta_asr
pip install -r ../../../../requirements.txt  # adjust path if needed
```

---

## Pipeline Stages

### 1. Manifest Validation

The validator scans all manifests declared in the YAML configuration (`train`,
`test`, and any `tokenizer_extra_manifests`). For each manifest it:

- Validates JSON syntax and required fields.
- Confirms the audio files exist and can be decoded via NeMo's `AudioSegment`.
- Writes a `<name>.validated.json` companion file containing only valid entries.
- Logs failures (line number, error, payload) to `<name>.invalid.json`.

Use `--no-save-config` to prevent automatic updates of manifest paths inside the
configuration.

#### Standalone Validator

```bash
python validate_data.py --manifest /path/to/manifest.json \
  --output /path/to/manifest.validated.json \
  --log-invalid /path/to/manifest.invalid.json \
  --workers 16
```

Parallelism can be controlled with the `--workers` flag or by setting
`training.validation_workers` in the YAML when using the unified CLI.

### 2. Tokenizer Training (Language-Family Aware)

`--mode tokenizer` trains SentencePiece tokenizers for each configured language
family, aggregates their vocabularies, and persists the metadata. Prerequisites
are the validated manifests produced in the previous stage.

The tokenizer step:

- Groups languages by the `model.language_families` mapping or creates
  singleton families when no mapping exists.
- Trains per-family tokenizers with the settings in
  `model.dynamic_tokenizer_params`.
- Collects shared special tokens (`special_token_prefixes`).
- Generates a deduplicated aggregate vocabulary and stores it at
  `aggregate_vocabulary_path`.
- Updates the YAML with `tokenizer_langs`, `shared_special_tokens`, and
  `aggregate_vocabulary` unless `--no-save-config` is supplied.

### 3. Model Fine-Tuning with Aggregate Tokenizer

`--mode train` restores the base `.nemo` checkpoint and applies the aggregate
tokenizer produced above. The training stage:

- Calls `change_vocabulary` to swap in the aggregate tokenizer.
- Rebuilds the training/validation dataloaders with language-family-aware
  sampling and optional augmentations.
- Enables keyword-aware loss when configured.
- Logs training and validation metrics through a TensorBoard logger.

---

## Feature Highlights

- **Deduplicated Aggregate Tokenizer**: merges per-family tokenizers and
  maintains a consistent set of shared special tokens.
- **BalancedLanguageBatchSampler**: performs temperature-controlled,
  language-family-aware sampling that respects distributed training contexts.
- **RobustAudioToBPEDataset**: normalizes language identifiers, validates audio
  files, and skips problematic samples during training.
- **Keyword-Aware Loss (optional)**: blends base CTC with a keyword-focused
  objective using warm-up and tunable weights.
- **Configurable Augmentation**: injects white noise and time-shift
  perturbations via `AudioAugmentor`.
- **Lightning Integration**: uses `lightning.pytorch` for distributed training,
  gradient accumulation, and experiment management.

---

## Loss Customization and Training Tricks

### Keyword-Aware CTC Blending

Setting `training.use_keyword_loss` to `true` activates a blended loss that
combines the standard CTC objective with an auxiliary term computed on tokens
tagged with the configured keyword prefixes (typically `KEYWORD_`, set via
`model.special_token_prefixes`). The blend ratio follows a linear warm-up:

- `training.keyword_loss_weight` defines the target weight applied to the
  keyword-specific CTC once warm-up completes.
- `training.keyword_loss_warmup_steps` controls the number of global steps used
  to ramp the keyword weight from 0 to the configured target, avoiding early
  training instability.

### Key-Phrase Oversampling

`training.keyphrase_oversample_factor` increases the sampling probability of
utterances containing keyword tokens. Values greater than `0.0` bias batches
toward key-phrase-rich audio, which is especially useful when the keyword loss
is enabled.

### Balanced Language Scheduling

`BalancedLanguageBatchSampler` tempers sampling across language families to
prevent any single group from dominating multi-lingual training. Critical
inputs include:

- `model.language_families` for the language-to-family mapping.
- `training.batch_size` (per device) and `training.accumulate_grad_batches`
  (global batch sizing).
- The sampler's temperature parameter (`0.2` by default in code) controls how
  aggressively it equalizes family frequencies; adjust inside the sampler if a
  more uniform or skewed schedule is desired.

---

## Configuration Reference (excerpt)

| YAML Path | Purpose | Notes |
| --- | --- | --- |
| `model.model_root` | Base checkpoint directory | Must contain the `.nemo` file referenced by `model_name`. |
| `model.dynamic_tokenizer_params` | SentencePiece training parameters | `non_special_tokens_per_lang` and overrides control vocab size per family. |
| `model.language_families` | Language→family mapping | Update to add or regroup languages before tokenizer training. |
| `training.lang_field` | Manifest key containing the language code | Defaults to `lang`. Must match manifest content. |
| `training.batch_size` | Per-device batch size | Combine with `accumulate_grad_batches` to control effective batch. |
| `training.accumulate_grad_batches` | Gradient accumulation | Useful when scaling across GPUs. |
| `training.use_keyword_loss` | Enable keyword-aware loss blending | Set to `true` to mix keyword CTC with the base loss. |
| `training.keyword_loss_weight` | Keyword loss mixing weight | Final blend factor applied after warm-up; typical range 0.1–0.7. |
| `training.keyword_loss_warmup_steps` | Keyword loss warm-up horizon | Number of steps to ramp the keyword weight from 0 to the target. |
| `training.keyphrase_oversample_factor` | Sampling boost for key phrases | Values > 0 increase sampling frequency for utterances containing keyword tokens. |
| `training.max_steps` | Maximum optimization steps | Set to a large number when using validation-based early stopping. |
| `training.spec_augment` | SpecAugment parameters | Controls time masking behaviour. |
| `experiment.exp_dir` / `experiment.exp_name` | Logging and checkpoint root | TensorBoard summaries are written under this path. |
| `experiment.every_n_train_steps` | Validation frequency | Also used by exp_manager to schedule checkpoints. |

For a complete list, inspect `config/config_peoplespeech.yml` and the helper
functions in `nemo_adapter_with_langid.py`.

---

## Troubleshooting

- **Missing language family mapping**: ensure every language in the manifests is
  present in `model.language_families`; the tokenizer stage logs any unmapped
  codes.
- **Audio decode warnings**: inspect the `.invalid.json` files generated during
  validation; they include the first 100 characters of the invalid record and
  the exception message.
- **Slow multi-GPU throughput**: confirm the custom batch sampler is active.
  `BalancedLanguageBatchSampler` now refreshes its distributed context each
  epoch to avoid duplicated batches across ranks.
- **Logger conflicts**: the trainer instantiates its own TensorBoard logger.
  When adjusting logging, ensure `experiment` settings align with trainer
  configuration to avoid exp_manager errors.

---

## Support Scripts

| Script | Purpose |
| --- | --- |
| `nemo_adapter_with_langid.py` | Unified CLI for validation, tokenizer training, and model fine-tuning. |
| `validate_data.py` | Standalone manifest cleaner that writes validated copies and rejection logs. |
| `inferece_asr.py` | FastAPI-based web server for interactive and API-driven transcription. |

---

## Interactive Transcription with the Inference Server

This project includes a web-based inference server that provides an interactive UI for transcribing audio files or recording directly from a microphone.

### Prerequisites

Install the required web server and audio processing libraries:

```bash
pip install fastapi uvicorn python-multipart librosa
```

### Running the Server

Launch the server by pointing it to your fine-tuned `.nemo` checkpoint.

```bash
python inferece_asr.py \
    --model-path /path/to/your/finetuned_model.nemo \
    --gpu-id 0
```

- `--model-path`: **(Required)** The full path to your trained `.nemo` model file.
- `--gpu-id`: The GPU device ID to use for inference (default: `0`).

Once the server is running, you can access the web interface by navigating to `http://localhost:8000` in your browser.

### Production Setup (Nginx and SSL)

For production use, it is highly recommended to run the server behind a reverse proxy like Nginx and to secure it with an SSL certificate (e.g., from Let's Encrypt). This enables HTTPS, which is required by modern browsers for microphone access.

Refer to standard guides for setting up Nginx and Certbot for a FastAPI application on your specific operating system.

---

Following the steps above will produce validated manifests, language-family
tokenizers, and a fine-tuned NeMo ASR model tailored to multilingual data while
keeping the pipeline reproducible and easy to extend.

