# Data Preparation Recipe

Prepare, normalize, and validate training manifests for Meta-ASR.

## What Data Prep Involves

1. **Acquiring raw audio + transcripts** -- from HuggingFace datasets, internal
   corpora, or the Whissle annotation pipeline.
2. **Converting to NeMo manifest format** -- JSONL where each line has
   `audio_filepath`, `text`, `duration`, and optionally `lang` / `lang_family`.
3. **Normalizing text** -- fix tag typos (e.g., `EMOTION_HAP` to `EMOTION_HAPPY`),
   remove forbidden tags, collapse whitespace, and strip artifacts.
4. **Validating manifests** -- ensure audio files exist and durations are reasonable.

## Full Annotation Pipeline (whissle-annotator)

For end-to-end annotation with entity/intent/emotion tagging, use the
[whissle-annotator](https://github.com/WhissleAI/whissle-annotator) tool.
It handles:

- Audio segmentation and VAD
- Speaker diarization
- ASR transcription
- NER / intent / emotion tagging via LLMs
- Export to NeMo-compatible JSONL manifests

## Manual Steps: Downloading from HuggingFace

Many WhissleAI datasets are hosted on HuggingFace. To download and convert:

```bash
# Install the HuggingFace CLI
pip install huggingface_hub[cli]

# Download a dataset (e.g., Hindi AI4Bharat)
huggingface-cli download WhissleAI/Meta_STT_HI_AI4Bharat --local-dir ./data/hindi

# For GCP training instances, use the provided helper script:
cd recipes/meta_asr/gcp
./download-data.sh --dataset WhissleAI/Meta_STT_HI_AI4Bharat --lang INDO_ARYAN
```

The downloaded dataset typically includes `train.json` and `valid.json` manifests
plus the audio files referenced within them.

## Normalizing Manifests

PromptingNemo includes a text normalization module that fixes common tag issues
in annotated manifests. Use the provided CLI script:

```bash
# Normalize a single manifest
python recipes/data_prep/normalize_manifest.py \
    --input-manifest data/hindi/train.json \
    --output-manifest data/hindi/train_normalized.json

# The script will:
#   - Fix tag typos (EMOTION_HAP -> EMOTION_HAPPY, GER_ -> GENDER_, etc.)
#   - Normalize AGE_60PLUS -> AGE_60+
#   - Remove invalid emotion/age tags
#   - Collapse whitespace and strip artifacts
#   - Report how many lines were modified
```

### Programmatic Usage

```python
from promptingnemo.data.normalize import normalize_text

raw = "hello EMOTION_HAP GER_MALE AGE_60PLUS world"
cleaned = normalize_text(raw)
# -> "hello EMOTION_HAPPY GENDER_MALE AGE_60+ world"
```

## Manifest Format

Each line in a NeMo manifest JSONL file looks like:

```json
{"audio_filepath": "/data/clips/sample_001.wav", "text": "EMOTION_NEUTRAL GENDER_MALE hello world ENTITY_PERSON john", "duration": 3.45, "lang": "ENGLISH"}
```

Key fields:
- `audio_filepath` -- absolute or relative path to the WAV/FLAC file
- `text` -- transcript with inline meta tags
- `duration` -- audio duration in seconds
- `lang` or `lang_family` -- language identifier (used by aggregate tokenizer)

## Further Reading

- [Meta-ASR training recipe](../meta_asr/README.md)
- [whissle-annotator](https://github.com/WhissleAI/whissle-annotator)
- [PromptingNemo normalize module](../../promptingnemo/data/normalize.py)
