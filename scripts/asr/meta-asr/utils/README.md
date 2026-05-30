# Meta-ASR Utility Scripts

Data downloaders, annotation tools, and manifest utilities for preparing
Meta-ASR training data. These scripts handle the full data lifecycle: downloading
public speech datasets, annotating them with metadata tags, and producing clean
NeMo-format JSONL manifests ready for training.

## Data Downloaders

Scripts that download public speech datasets from HuggingFace, OpenSLR, and
other sources, extract audio as 16 kHz WAV files, and produce NeMo-format
JSONL manifests.

| Script | Description |
|--------|-------------|
| `download_hf_dataset.py` | Generic downloader for any `WhissleAI/Meta_STT_*` HuggingFace dataset; handles single-split and multi-split datasets, auto-generates train/valid splits |
| `download_meta_sets_hf.py` | Batch downloader for multiple WhissleAI Meta-STT datasets from HuggingFace with parallel audio extraction |
| `download_en_people.py` | Download PeopleSpeech audio and join with `WhissleAI/Meta_STT_EN_Set1` annotations by parquet shard/row index |
| `download_en_digits.py` | Download English digit/number speech (Google Speech Commands v2 + Free Spoken Digit Dataset) for numeric ASR coverage |
| `download_en_in_tech.py` | Download `WhissleAI/Meta_STT_EN-IN_Tech_Interviews` with tag normalization and canonical inline META tag assembly |
| `download_hinglish.py` | Download Hinglish (Hindi-English code-switching) datasets (MUCS + IndicVoices) as 16 kHz WAV with NeMo manifests |
| `download_indicvoices_raw.py` | Download raw IndicVoices / Kathbath / IndicVoices-R parquet data for a single Indic language with metadata preservation |
| `download_madasr.py` | Download `WhissleAI/Meta_STT_MADASR2.0_train_lg` (8 Indic languages) with inline tag bug fixes during extraction |
| `download_multilingual_cv.py` | Download CommonVoice audio for European, Slavic, and English languages; match with WhissleAI metadata annotations |
| `download_slavic_cv.py` | Download Slavic CommonVoice audio and build NeMo manifests from `WhissleAI/Meta_STT_SLAVIC_CommonVoice` |
| `download_zh_aishell3.py` | Download `WhissleAI/Meta_STT_ZH_AIShell3` Chinese speech with embedded metatags |
| `download_zh_datasets.py` | Download Chinese ASR datasets (CommonVoice zh-CN, AISHELL-1, KeSpeech, MagicData) without inline tags |

**Example -- download a HuggingFace dataset:**

```bash
python download_hf_dataset.py \
    --dataset WhissleAI/Meta_STT_ZH_AIShell3 \
    --output-dir /mnt/training/data/zh_aishell3 \
    --lang MANDARIN
```

## Annotation Tools

Scripts that add or fix metadata tags (AGE, GENDER, EMOTION, INTENT, ENTITY)
in existing NeMo manifests using GPU classification models or LLM-based annotation.

| Script | Description |
|--------|-------------|
| `annotate_audio_tags.py` | Annotate manifests with AGE, GENDER, EMOTION tags using wav2vec2 (age/gender) and HuBERT (emotion) GPU models; same pipeline as whissle-annotator |
| `annotate_indic_meta.py` | Full annotation pipeline for Indic speech manifests: AGE/GENDER from metadata or HuBERT, EMOTION from HuBERT, ENTITY/INTENT from Gemini |
| `annotate_with_gemini.py` | Fill missing INTENT tags in META-ASR manifests using Gemini 2.5 Flash; batches samples for efficient API usage |
| `annotate_misc_interview.py` | MISC-based behavioral annotation for tech interview data using Gemini 2.5 Flash; applies Motivational Interviewing Skill Codes |
| `add_langid_into_manifest.py` | Add language ID field to manifest entries using the Lingua language detection library |

**Example -- annotate a manifest with audio-based tags:**

```bash
python annotate_audio_tags.py \
    --manifest /data/hindi/train.json \
    --output /data/hindi/train.tagged \
    --batch-size 32 \
    --device cuda
```

## Manifest Tools

Scripts for merging, normalizing, splitting, validating, and analyzing
NeMo-format JSONL manifests.

| Script | Description |
|--------|-------------|
| `normalize_annotations.py` | Normalize META-ASR annotations across all multilingual datasets: fix GER_ to GENDER_ prefix, truncated tag names, missing lang_family fields, and missing intent tags |
| `normalize_aishell3.py` | Normalize AISHELL-3 Chinese manifests: fix lang codes, add lang_family, insert default EMOTION_NEUTRAL |
| `merge_english_manifests.py` | Merge English training manifests (Meta_STT_EN_Set1 + CommonVoice + digits) with digit-to-word normalization |
| `merge_gujarati_manifests.py` | Merge annotated Gujarati manifests from multiple datasets with deduplication by audio_filepath |
| `integrate_hindi.py` | Integrate Hindi Set1 data into the multilingual directory structure with path fixes and lang_family assignment |
| `split_multi_speaker.py` | Split multi-speaker segments into single-speaker samples with per-segment re-annotation via Gemini |
| `validate_data.py` | Validate manifest entries by ensuring audio files are readable; parallel validation with progress reporting |
| `manifest_stats.py` | Compute distribution statistics (AGE, GENDER, EMOTION, INTENT, language) across all manifests in a directory |
| `inferece_asr.py` | Run ASR inference on a manifest using a NeMo checkpoint and write predictions |
| `custom_models.py` | Legacy custom CTC model definition (superseded by `promptingnemo.models.ctc_model`) |

**Example -- normalize annotations across datasets:**

```bash
# Dry run to see what would change
python normalize_annotations.py \
    --data-root /mnt/nfs/data/multilingual_v1/raw \
    --dry-run

# Apply fixes
python normalize_annotations.py \
    --data-root /mnt/nfs/data/multilingual_v1/raw
```

**Example -- validate audio files in a manifest:**

```bash
python validate_data.py \
    --manifest /data/train.jsonl \
    --workers 16
```

## Typical Workflow

A common end-to-end data preparation workflow:

```bash
# 1. Download a dataset from HuggingFace
python download_hf_dataset.py \
    --dataset WhissleAI/Meta_STT_HI_Set1 \
    --output-dir /data/hindi \
    --lang HINDI --family INDO_ARYAN

# 2. Annotate with audio-based tags (if not already present)
python annotate_audio_tags.py \
    --manifest /data/hindi/train.jsonl \
    --output /data/hindi/train_tagged.jsonl

# 3. Fill missing intent tags with Gemini
python annotate_with_gemini.py \
    --data-root /data/hindi

# 4. Normalize all annotations
python normalize_annotations.py \
    --data-root /data/hindi

# 5. Validate audio files
python validate_data.py \
    --manifest /data/hindi/train.jsonl

# 6. Check distribution stats
python manifest_stats.py /data/hindi

# 7. Train with the recipe
cd ../../recipes/meta_asr
python train.py --config conf/hindi.yaml --mode both
```
