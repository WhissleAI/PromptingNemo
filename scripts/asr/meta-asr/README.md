# Meta-ASR Pipeline Guide

This directory contains the orchestration scripts for building and fine-tuning
NeMo ASR models with dynamic, language-family-aware tokenizers. The workflow is
designed to run inside an NVIDIA NeMo Docker image, but the commands below also
apply to any Python environment with NeMo, PyTorch Lightning, and the project
dependencies installed.

## Prerequisites

- NVIDIA GPU with the appropriate drivers.
- Docker (optional, but recommended).
- Training manifests in NeMo JSONL format with `audio_filepath`, `text`, and
  uppercase `lang` fields.
- Base ASR `.nemo` checkpoint referenced in `config/config_peoplespeech.yml`.

### Suggested Docker Command

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
pip install -r ../../../../requirements.txt  # adjust if needed
```

## Workflow Overview

1. **Validate manifests** (new): produce cleaned copies with only decodable audio.
2. **Train tokenizers**: build SentencePiece tokenizers per language family and
   assemble the aggregate vocabulary.
3. **Fine-tune the ASR model** with the aggregate tokenizer and balanced
   sampling.

Each stage is triggered via `nemo_adapter_with_langid.py` by selecting an
appropriate `--mode`.

## 1. Manifest Validation

The validator scans all manifests referenced in
`config/config_peoplespeech.yml` (`train_manifest`, `test_manifest`, and any
entries in `tokenizer_extra_manifests`). For every manifest it:

- Confirms JSON syntax.
- Verifies the audio file exists and can be opened with NeMo's
  `AudioSegment` helpers.
- Writes a `.validated.json` sibling file containing only the good entries.
- Logs rejected entries (line number, error, and payload) to
  `<original>.invalid.json`.

Run validation only:

```bash
python nemo_adapter_with_langid.py --mode validate_data \
  --config config/config_peoplespeech.yml
```

By default the config is updated to point at the validated manifests (use
`--no-save-config` to keep the YAML unchanged).

### Standalone Validator

Alternatively, use the smaller helper:

```bash
python validate_data.py --manifest /path/to/manifest.json \
  --output /path/to/manifest.validated.json \
  --log-invalid /path/to/manifest.invalid.json \
  --workers 16

Control parallelism either with the `--workers` flag (standalone script) or by
setting `training.validation_workers` in the YAML config when using
`nemo_adapter_with_langid.py`.
```

## 2. Tokenizer Training (Language-Family Aware)

After validation, retrain tokenizers grouped by language family:

```bash
python nemo_adapter_with_langid.py --mode tokenizer \
  --config config/config_peoplespeech.yml
```

> Note: `--mode tokenizer` assumes the manifests referenced in the config are
> already validated. Run `--mode validate_data` first whenever the manifests
> change.

Key features:

- **Language family buckets** (e.g., Germanic, Romance, Indo_Aryan, etc.)
  trained in parallel; singleton families are created when a language lacks an
  explicit mapping.
- Shared special tokens collected from manifests.
- Deduplicated aggregate vocabulary stored alongside the tokenizer metadata.
- Config automatically records `tokenizer_langs`, `shared_special_tokens`, and
  `aggregate_vocabulary`.

## 3. Model Fine-Tuning with Aggregate Tokenizer

Launch fine-tuning:

```bash
python nemo_adapter_with_langid.py --mode train \
  --config config/config_peoplespeech.yml
```

The training step automatically pulls tokenizer metadata from
`tokenizer_langs_path`, `shared_special_tokens_path`, and
`aggregate_vocabulary_path` in the config.

Highlights of the training setup:

- **Aggregate tokenizer** rebuilt from the family tokenizers and applied via
  `change_vocabulary`.
- **BalancedLanguageBatchSampler** samples batches by language family with a
  temperature-controlled schedule, keeping multilingual training balanced.
- **RobustAudioToBPEDataset** filters manifest entries without `lang`, ensures
  audio readability, and skips problematic samples on-the-fly during data
  loading.
- Optional keyword loss, augmentation, and optimizer tweaks inherited from the
  config file.

To execute the entire pipeline sequentially (validate → tokenizer → train):

```bash
python nemo_adapter_with_langid.py --mode both \
  --config config/config_peoplespeech.yml
```

## Configuration Notes

- `training.data_dir` is treated as the base for relative manifest paths.
- `training.validation_workers` controls the number of parallel validation
  workers (defaults to CPU count when omitted).
- `model.language_families` now defines the language→family mapping consumed by
  `nemo_adapter_with_langid.py`; adjust this block to change tokenizer buckets.
- `training.skip_audio_validation` skips per-run audio decoding checks inside
  the dataset loader (set to `true` when feeds already went through
  `validate_data`).
- Validated manifests are written next to the originals with the suffix
  `.validated.json`. The `.invalid.json` log files are informational; delete
  them if not needed.
- Set `--no-save-config` to prevent YAML updates if you prefer to manage the
  manifest paths manually.

## Troubleshooting

- **Missing language mapping**: the tokenizer logs any languages without usable
  samples; ensure manifests contain the expected `lang` codes and family
  mappings (`LANG_FAMILIES` in `nemo_adapter_with_langid.py`).
- **Audio decode warnings**: check `<manifest>.invalid.json` for the rejected
  entries. ffmpeg logs can reveal corrupt or unsupported files.
- **Batch sampler + DDP**: the trainer forces `use_distributed_sampler=False`
  because batching is already controlled by the custom sampler. Adjust only if
  you rework the sampling strategy.

## Optional Customization

- Extend `LANG_FAMILIES` for new language groups.
- Adjust `dynamic_tokenizer_params.non_special_tokens_per_lang` to tune per
  family vocab sizes.
- Add more manifests to `training.tokenizer_extra_manifests` to improve special
  token coverage.

## Support Scripts Summary

| Script | Purpose |
| --- | --- |
| `validate_data.py` | Standalone manifest cleaner; keeps only readable audio entries. |
| `nemo_adapter_with_langid.py` | Unified CLI for data validation, tokenizer training, and model fine-tuning. |

By running the pipeline in the order above, you can reliably fine-tune NeMo
models on multilingual datasets with robust tokenization and safe data loading.

