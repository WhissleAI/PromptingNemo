# Recipes

Ready-to-run training and data preparation pipelines for PromptingNemo models.
Each recipe directory contains entry-point scripts, YAML configs, and
supporting utilities for a specific task.

## Recipe Index

| Recipe | Description | Entry Point |
|--------|-------------|-------------|
| [`meta_asr/`](meta_asr/) | Full Meta-ASR training pipeline | `run.sh` / `train.py` |
| [`av_asr/`](av_asr/) | Audio-Visual ASR for noisy speech | `run.sh` / `train.py` |
| [`data_prep/`](data_prep/) | Manifest normalization and validation | `normalize_manifest.py` |
| [`espnet_asr/`](espnet_asr/) | ESPnet-W bridge for Meta-ASR | (see README) |
| [`text_tagger/`](text_tagger/) | Text-only CTC tagger configs | `conf/default.yaml` |

## meta_asr/ -- Meta-ASR Training

The primary training recipe. Trains multilingual, metadata-enriched ASR models
that output transcription and structured tags (entities, intents, emotions,
gender, age, keywords, dialect, language ID) in a single CTC pass.

**Supported languages:** Hindi, Bengali, Gujarati, Marathi, Punjabi, Kannada,
Malayalam, Mandarin, English, Slavic family, Indo-Aryan family, and more.

**Key features:**
- Aggregate multi-language tokenizer with language-family routing
- Adapter-based fine-tuning (Linear, Bottleneck, Attention adapters)
- Keyword loss for better tag prediction
- Balanced language-family batch sampling
- Checkpoint resumption

```bash
cd recipes/meta_asr

# Quick start: tokenizer + training
./run.sh --lang hindi --mode both

# Or use the Python entry point directly
python train.py --config conf/hindi.yaml --mode both

# Resume from a checkpoint
python train.py --config conf/hindi.yaml --mode train \
    --resume_from /path/to/checkpoint.ckpt
```

**Config files** (`conf/`):

| Config | Language / Purpose |
|--------|--------------------|
| `hindi.yaml` | Hindi (AI4Bharat) |
| `bengali.yaml` | Bengali |
| `gujarati.yaml` | Gujarati |
| `marathi.yaml` | Marathi |
| `punjabi.yaml` | Punjabi |
| `kannada.yaml` | Kannada |
| `malayalam.yaml` | Malayalam |
| `indo_aryan.yaml` | Indo-Aryan multi-language |
| `mandarin.yaml` | Mandarin Chinese (Meta-1B base) |
| `mandarin_aishell3.yaml` | Mandarin Chinese (Parakeet base) |
| `english.yaml` | English (PeopleSpeech, multilingual tokenizer) |
| `slavic.yaml` | Slavic languages (adapter-based, CommonVoice) |
| `vils.yaml` | VILS dataset (English) |
| `wellness.yaml` | Wellness domain (English) |
| `pretrain_meta.yaml` | Hindi pretraining continuation |

See [`meta_asr/README.md`](meta_asr/README.md) for the full guide.

## av_asr/ -- Audio-Visual ASR (EMNLP 2025)

Trains Audio-Visual ASR models that use CLIP visual features from video frames
to contextualize noise sources, improving speech recognition accuracy in noisy
conditions.

Based on: "Visual-Aware Speech Recognition for Noisy Scenarios" (Darur & Singla,
EMNLP 2025).

**Architecture:** Conformer audio encoder + CLIP ViT-L/14 visual encoder, fused
via a Transformer cross-modal attention module, trained end-to-end with CTC loss.
Noise class labels (`<N1>`, `<N2>`, ...) are appended to transcripts.

```bash
cd recipes/av_asr

# Data preparation: download VANS dataset + extract CLIP features
./run.sh --stage 0

# Audio-Visual training
python train.py --config conf/av_conformer_ctc.yaml --gpus 2

# Audio-only baseline
python train.py --config conf/audio_only_baseline.yaml

# Override SNR setting
python train.py --config conf/av_conformer_ctc.yaml --snr rand
```

**Local utilities** (`local/`):
- `prepare_vans.py` -- download and prepare the VANS dataset
- `extract_features.py` -- extract CLIP visual features from video frames

See [`av_asr/README.md`](av_asr/README.md) for the full training guide and
architecture description.

## data_prep/ -- Manifest Normalization

Normalize and validate NeMo-format JSONL manifests before training. Fixes
common annotation issues: tag typos (`EMOTION_HAP` to `EMOTION_HAPPY`,
`GER_` to `GENDER_`), forbidden tags, whitespace artifacts, and concatenated
tag strings.

```bash
# Normalize a manifest
python recipes/data_prep/normalize_manifest.py \
    --input-manifest data/train.json \
    --output-manifest data/train_normalized.json
```

For the full annotation pipeline (segmentation, diarization, ASR, NER/intent/emotion
tagging), use [whissle-annotator](https://github.com/WhissleAI/whissle-annotator).

See [`data_prep/README.md`](data_prep/README.md) for manifest format
documentation and programmatic usage.

## espnet_asr/ -- ESPnet-W Bridge

Reference documentation for training Meta-ASR models via
[espnet-w](https://github.com/WhissleAI/espnet-w), WhissleAI's fork of ESPnet.
This provides an alternative training backend using ESPnet's stage-based
`egs2/` recipe system.

Both PromptingNemo and espnet-w produce models with the same Meta-ASR tag
vocabulary -- choose based on your workflow preferences.

See [`espnet_asr/README.md`](espnet_asr/README.md) for setup instructions and a
comparison of the two approaches.

## text_tagger/ -- Text-Only Tagger Configs

YAML configs for training text-only CTC taggers that predict inline tags from
character input (no audio). Uses the `TextCTCTagger` model from
`promptingnemo.models`.

**Config files** (`conf/`):

| Config | Description |
|--------|-------------|
| `default.yaml` | Standard Transformer encoder, non-causal attention |
| `streaming.yaml` | Causal attention for streaming inference |

Training scripts are in `scripts/text_tagger/`. See the
[promptingnemo package README](../promptingnemo/README.md) for model details.

## Directory Structure

```
recipes/
├── meta_asr/               # Meta-ASR training pipeline
│   ├── run.sh              # Shell entry point
│   ├── train.py            # Python entry point
│   └── conf/               # Per-language YAML configs
├── av_asr/                 # Audio-Visual ASR
│   ├── run.sh              # Shell entry point
│   ├── train.py            # Python entry point
│   ├── conf/               # AV model configs
│   └── local/              # Data prep utilities
├── data_prep/              # Manifest normalization
│   └── normalize_manifest.py
├── espnet_asr/             # ESPnet-W bridge (documentation only)
│   └── README.md
└── text_tagger/            # Text tagger configs
    └── conf/
        ├── default.yaml
        └── streaming.yaml
```
