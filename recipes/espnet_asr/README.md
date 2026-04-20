# ESPnet-w ASR Recipe

Reference documentation for using [espnet-w](https://github.com/WhissleAI/espnet-w)
to train Meta-ASR models via the ESPnet framework.

## What is espnet-w?

espnet-w is WhissleAI's fork of ESPnet, extended with support for:

- Aggregate tokenizers (multi-language BPE with language-family routing)
- Meta-ASR tag vocabulary (entities, intents, emotions, gender, age, keywords)
- Keyword loss and family-weighted training
- NeMo-compatible model export

It follows ESPnet's standard `egs2/` recipe structure, making it easy to plug in
new datasets and language families.

## Repository

```
https://github.com/WhissleAI/espnet-w
```

## Setup

```bash
# 1. Clone espnet-w
git clone https://github.com/WhissleAI/espnet-w.git
cd espnet-w

# 2. Install ESPnet and dependencies
pip install -e .
cd tools && make  # builds kaldi-style tools (sph2pipe, etc.)

# 3. Navigate to a recipe
cd egs2/<recipe_name>/asr1
```

## Recipe Structure (egs2/)

Each recipe in `egs2/` follows the standard ESPnet layout:

```
egs2/<dataset>/asr1/
├── run.sh           # Main entry point
├── conf/            # YAML training configs
│   ├── train_asr.yaml
│   ├── decode_asr.yaml
│   └── ...
├── local/           # Dataset-specific scripts (download, prep, scoring)
│   ├── data_prep.sh
│   └── ...
└── README.md
```

## Running a Recipe

```bash
cd egs2/<recipe_name>/asr1

# Full pipeline: data prep -> tokenizer -> training -> decoding -> scoring
./run.sh --stage 1 --stop-stage 13

# Or run individual stages:
./run.sh --stage 1 --stop-stage 1   # data preparation
./run.sh --stage 2 --stop-stage 2   # speed perturbation (optional)
./run.sh --stage 3 --stop-stage 3   # feature extraction
./run.sh --stage 4 --stop-stage 4   # tokenizer training
./run.sh --stage 5 --stop-stage 5   # model training
./run.sh --stage 6 --stop-stage 6   # decoding
./run.sh --stage 7 --stop-stage 7   # scoring
```

## When to Use espnet-w vs. PromptingNemo Recipes

| Aspect | PromptingNemo (`recipes/meta_asr/`) | espnet-w (`egs2/`) |
|--------|-------------------------------------|---------------------|
| Framework | NeMo (PyTorch Lightning) | ESPnet (PyTorch) |
| Config style | Single YAML | Stage-based YAML |
| Strengths | Adapter training, NeMo ecosystem | Full pipeline, kaldi scoring |
| Best for | Quick fine-tuning, production export | Research, ablations, benchmarks |

Both approaches produce models with the same Meta-ASR tag vocabulary. Models can
be converted between formats using the export utilities in PromptingNemo.

## Further Reading

- [espnet-w repository](https://github.com/WhissleAI/espnet-w)
- [ESPnet documentation](https://espnet.github.io/espnet/)
- [Meta-ASR training recipe (PromptingNemo)](../meta_asr/README.md)
- [Data preparation recipe](../data_prep/README.md)
