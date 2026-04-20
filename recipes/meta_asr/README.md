# Meta-ASR Training Recipe

Train multilingual, metadata-enriched ASR models using the PromptingNemo framework.
Meta-ASR extends standard CTC-based ASR with structured tags for entities, intents,
emotions, gender, age, keywords, and language identification -- all decoded in a
single pass alongside the transcript.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with 16 GB+ VRAM (T4 minimum; A100 recommended for large configs)
- NeMo toolkit and PromptingNemo installed (`pip install -e .` from the repo root)
- A pretrained `.nemo` checkpoint (e.g., `WhissleAI/STT-meta-1B` from HuggingFace)
- Training data as NeMo-format JSONL manifests with audio files

## Quick Start

```bash
# 1. From the repo root, install PromptingNemo
pip install -e .

# 2. Run training for a specific language
cd recipes/meta_asr
./run.sh --lang hindi --mode both          # tokenizer + training
./run.sh --lang mandarin --mode tokenizer  # only rebuild tokenizer
./run.sh --lang english --mode train       # only train (tokenizer already built)

# 3. Or use a custom config
./run.sh --config conf/custom.yaml --mode train

# 4. Resume from a checkpoint
./run.sh --lang hindi --mode train -- --resume_from /path/to/checkpoint.ckpt
```

## Config Files

Each YAML config in `conf/` defines model paths, data locations, adapter settings,
and experiment parameters. Edit the paths to match your local or GCP storage layout.

| Config | Language / Purpose |
|--------|--------------------|
| `hindi.yaml` | Hindi (AI4Bharat dataset) |
| `bengali.yaml` | Bengali |
| `gujarati.yaml` | Gujarati |
| `marathi.yaml` | Marathi |
| `punjabi.yaml` | Punjabi |
| `kannada.yaml` | Kannada |
| `malayalam.yaml` | Malayalam |
| `indo_aryan.yaml` | Indo-Aryan language family (multi-language) |
| `mandarin.yaml` | Mandarin Chinese (AISHELL-3, Meta-1B base) |
| `mandarin_aishell3.yaml` | Mandarin Chinese (AISHELL-3, Parakeet base) |
| `english.yaml` | English (PeopleSpeech, multilingual meta tokenizer) |
| `slavic.yaml` | Slavic languages (adapter-based, CommonVoice) |
| `vils.yaml` | VILS dataset (English) |
| `wellness.yaml` | Wellness domain (English) |
| `pretrain_meta.yaml` | Hindi pretraining continuation |

## Training Modes

- **both** (default) -- Builds the tokenizer from training data, then trains the model.
- **tokenizer** -- Only builds/rebuilds the aggregate tokenizer.
- **train** -- Only runs model training (assumes tokenizer is already built).
- **validate_data** -- Validates manifests without training.

## GCP Spot Instance Training

The `gcp/` directory contains scripts for running training on GCP spot instances with
persistent disks. See `gcp/README.md` for the full workflow (instance creation, data
download, fine-tuning, TensorBoard monitoring, and model upload).

## Directory Structure

```
recipes/meta_asr/
├── run.sh              # Entry point script
├── train.py            # Python entry point (calls promptingnemo.training.cli)
├── conf/               # YAML config files per language
│   ├── hindi.yaml
│   ├── english.yaml
│   └── ...
├── gcp/                # GCP spot instance training scripts
│   ├── launch-experiment.sh
│   ├── run-finetune.sh
│   └── ...
└── README.md           # This file
```

## Further Reading

- [PromptingNemo main README](../../README.md) -- installation, architecture overview
- [Data preparation recipe](../data_prep/README.md) -- manifest normalization, annotation
- [espnet-w recipe](../espnet_asr/README.md) -- alternative training with ESPnet
