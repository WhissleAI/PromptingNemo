# Contributing to PromptingNemo

Thanks for your interest in contributing to PromptingNemo. This guide covers development setup, code style, testing, and how to submit changes.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/WhissleAI/PromptingNemo.git
cd PromptingNemo

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

This installs the core library plus `pytest`, `pytest-cov`, `pytest-asyncio`, and `ruff` for linting.

For training-related development, install the full set of dependencies:

```bash
pip install -e ".[all]"
```

## Running Tests

```bash
pytest tests/ -v
```

Tests that require a GPU or large model downloads are marked with `@pytest.mark.slow`. To run only those:

```bash
pytest tests/ -v -m slow
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

- **Line length:** 120 characters
- **Python version:** 3.10+
- **Lint rules:** `E`, `F`, `I`, `W` (with `E501` ignored since we set line-length explicitly)

Run the linter:

```bash
ruff check .
```

Auto-fix issues:

```bash
ruff check --fix .
```

Format code:

```bash
ruff format .
```

## Project Structure

```
promptingnemo/          # Installable library
  tokenizer/            # Aggregate tokenizer training
  models/               # Custom CTC model, slim decoder
  data/                 # Dataset, sampler, normalization
  training/             # Trainer and CLI
  eval/                 # WER + NER F1 scoring
  export/               # ONNX and HuggingFace export

recipes/                # Runnable training recipes
  meta_asr/             # Main training entry point + per-language configs
  data_prep/            # Data preparation utilities
  espnet_asr/           # ESPnet-W integration

scripts/                # Legacy and utility scripts
  asr/meta-asr/         # Config files, GCP tooling, evaluation
  data/                 # Data download and annotation scripts
  eval/                 # Evaluation and benchmarking scripts
  utils/                # Shared utility scripts

tests/                  # Test suite
docs/                   # Documentation
```

## Adding a New Language Recipe

To add training support for a new language:

1. **Prepare the data.** Create a NeMo JSONL manifest with tagged transcriptions. Each line should be a JSON object with `audio_filepath`, `duration`, `text`, and a language field (e.g., `lang` or `lang_family`). Tags should follow the canonical format defined in [docs/tag_schema.md](docs/tag_schema.md).

   ```json
   {"audio_filepath": "/data/audio/001.wav", "duration": 3.2, "text": "ENTITY_PERSON_NAME Rahul END said hello EMOTION_HAPPY AGE_18_30 GENDER_MALE", "lang": "HINDI"}
   ```

2. **Normalize tags.** Run your manifest through `promptingnemo.data.normalize.normalize_text()` to canonicalize tag formats (e.g., `GER_MALE` to `GENDER_MALE`, `EMOTION_HAP` to `EMOTION_HAPPY`).

3. **Create a config file.** Add a YAML config in `recipes/meta_asr/conf/` (e.g., `my_language.yaml`). Use an existing config like `hindi.yaml` or `mandarin.yaml` as a starting point. Key fields to update:
   - `model.model_root` and `model.model_name` -- path to the pretrained checkpoint
   - `model.language_families` -- which language families to include
   - `training.data_dir`, `training.train_manifest`, `training.test_manifest` -- data paths
   - `training.batch_size`, `training.max_steps` -- training hyperparameters
   - `experiment.exp_name` -- experiment name for checkpointing

4. **Test tokenizer training.** Run tokenizer-only mode first to verify the data and vocabulary:
   ```bash
   cd recipes/meta_asr
   python train.py --config conf/my_language.yaml --mode tokenizer
   ```

5. **Train the model.** Run the full pipeline:
   ```bash
   python train.py --config conf/my_language.yaml --mode both
   ```

6. **Evaluate.** Use the evaluation utilities in `promptingnemo/eval/` to compute WER and NER F1 on your test set.

7. **Submit a PR** with:
   - The YAML config in `recipes/meta_asr/conf/`
   - Any new normalization rules in `promptingnemo/data/normalize.py` if the language uses non-standard tag formats
   - A note in the PR description about the dataset source and expected WER/F1

## Pull Request Process

1. **Fork and branch.** Create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-change
   ```

2. **Make your changes.** Follow the code style guidelines above.

3. **Run tests and lint.** Ensure everything passes:
   ```bash
   ruff check .
   pytest tests/ -v
   ```

4. **Write a clear PR description.** Include:
   - What the change does and why
   - How to test it
   - Any breaking changes or migration notes

5. **Submit.** Open a pull request against `main`. A maintainer will review your changes. Address any feedback, and once approved, the PR will be merged.

## Reporting Issues

Open an issue on [GitHub](https://github.com/WhissleAI/PromptingNemo/issues) with:
- A clear description of the problem or feature request
- Steps to reproduce (for bugs)
- Your environment (Python version, OS, GPU type if relevant)
