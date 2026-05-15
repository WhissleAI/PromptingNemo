#!/usr/bin/env python3
"""
push_to_hf.py — Upload a .nemo model + auto-generated model card to HuggingFace.

Usage:
    python push_to_hf.py \
        --model /path/to/model.nemo \
        --repo WhissleAI/STT-meta-HI \
        --eval-summary /path/to/eval_summary.json \
        --token $HF_TOKEN \
        [--fleurs-wer 0.2197]
"""
import argparse
import json
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import yaml


def extract_model_info(nemo_path):
    """Extract architecture info from .nemo archive."""
    info = {}
    with tarfile.open(nemo_path) as t:
        for m in t.getmembers():
            if "model_config" in m.name:
                f = t.extractfile(m)
                config = yaml.safe_load(f)
                enc = config.get("encoder", {})
                dec = config.get("decoder", {})

                info["encoder_layers"] = enc.get("n_layers")
                info["d_model"] = enc.get("d_model")
                info["n_heads"] = enc.get("n_heads")
                info["feat_in"] = enc.get("feat_in", 80)
                info["subsampling"] = enc.get("subsampling")
                info["encoder_target"] = enc.get("_target_", "")

                vocab = dec.get("vocabulary", [])
                info["vocab_size"] = len(vocab)
                info["num_classes"] = dec.get("num_classes", len(vocab))

                tags = {}
                for v in vocab:
                    for prefix in ["AGE_", "GENDER_", "EMOTION_", "INTENT_", "ENTITY_"]:
                        if v.startswith(prefix):
                            cat = prefix.rstrip("_")
                            tags.setdefault(cat, []).append(v)
                            break
                    if v == "END":
                        tags.setdefault("END", []).append(v)
                info["tag_categories"] = {k: sorted(v) for k, v in sorted(tags.items())}
                break

    file_size = os.path.getsize(nemo_path)
    info["file_size_mb"] = round(file_size / 1024 / 1024, 1)
    return info


def generate_model_card(model_info, eval_summary, fleurs_wer, repo_id, model_filename):
    """Generate README.md content for HuggingFace model card."""
    # Build tag tables
    tag_sections = []
    if eval_summary and "tags" in eval_summary:
        for tag_name, tag_data in sorted(eval_summary["tags"].items()):
            lines = [f"### {tag_name}"]
            lines.append("")
            lines.append(f"**Accuracy: {tag_data['accuracy']:.1%}** | Macro F1: {tag_data['macro_f1']:.1%} | Weighted F1: {tag_data['weighted_f1']:.1%}")
            lines.append("")

            # Per-class table
            lines.append("| Label | Precision | Recall | F1 | Support |")
            lines.append("|-------|-----------|--------|-----|---------|")
            for label, metrics in sorted(tag_data.get("per_class", {}).items()):
                if metrics.get("support", 0) > 0:
                    lines.append(
                        f"| {label} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['support']} |"
                    )
            lines.append("")

            # Confusion matrix
            if "confusion_matrix" in tag_data and "labels" in tag_data:
                labels = tag_data["labels"]
                conf = tag_data["confusion_matrix"]
                if len(labels) <= 10:
                    lines.append("<details>")
                    lines.append(f"<summary>Confusion Matrix</summary>")
                    lines.append("")
                    short_labels = [l.split("_", 1)[-1] if "_" in l else l for l in labels]
                    header = "| | " + " | ".join(short_labels) + " |"
                    sep = "|---|" + "|".join(["---"] * len(labels)) + "|"
                    lines.append(header)
                    lines.append(sep)
                    for ri, row_label in enumerate(short_labels):
                        row = f"| **{row_label}** | " + " | ".join(str(conf[ri][ci]) for ci in range(len(labels))) + " |"
                        lines.append(row)
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")

            tag_sections.append("\n".join(lines))

    # Entity metrics
    entity_section = ""
    if eval_summary and "entity" in eval_summary:
        ent = eval_summary["entity"]
        entity_section = f"""### Entity Detection

| Metric | Value |
|--------|-------|
| Precision | {ent['precision']:.3f} |
| Recall | {ent['recall']:.3f} |
| F1 | {ent['f1']:.3f} |
"""

    wer_val = eval_summary.get("wer", "N/A") if eval_summary else "N/A"
    cer_val = eval_summary.get("cer", "N/A") if eval_summary else "N/A"
    n_samples = eval_summary.get("samples", "N/A") if eval_summary else "N/A"

    wer_str = f"{wer_val:.1%}" if isinstance(wer_val, float) else str(wer_val)
    cer_str = f"{cer_val:.1%}" if isinstance(cer_val, float) else str(cer_val)

    # Tag categories listing
    tag_listing = ""
    if model_info.get("tag_categories"):
        for cat, tokens in model_info["tag_categories"].items():
            if cat == "END":
                continue
            if cat == "ENTITY":
                tag_listing += f"- **{cat}**: {len(tokens)} types (ENTITY_PERSON_NAME, ENTITY_CITY, ENTITY_ORGANIZATION, ...)\n"
            else:
                tag_listing += f"- **{cat}**: {', '.join(tokens)}\n"

    fleurs_row = ""
    if fleurs_wer:
        fleurs_row = f"| FLEURS Hindi (test) | {fleurs_wer:.1%} | — | 418 |"

    card = f"""---
language:
- hi
license: apache-2.0
tags:
- automatic-speech-recognition
- speech
- nemo
- conformer
- ctc
- hindi
- speech-tagging
- entity-recognition
- emotion-detection
datasets:
- custom
pipeline_tag: automatic-speech-recognition
model-index:
- name: {repo_id.split('/')[-1]}
  results:
  - task:
      type: automatic-speech-recognition
    metrics:
    - name: WER
      type: wer
      value: {wer_val if isinstance(wer_val, float) else 0}
---

# {repo_id.split('/')[-1]} — Hindi Speech Tagger

A Conformer-CTC model for Hindi automatic speech recognition with **inline entity tagging**, **speaker attribute detection** (age, gender, emotion), and **intent classification** — all in a single forward pass.

## Model Details

| Property | Value |
|----------|-------|
| Architecture | Conformer CTC (NeMo) |
| Encoder Layers | {model_info.get('encoder_layers', 'N/A')} |
| Hidden Size (d_model) | {model_info.get('d_model', 'N/A')} |
| Attention Heads | {model_info.get('n_heads', 'N/A')} |
| Vocabulary Size | {model_info.get('vocab_size', 'N/A')} |
| Input Features | {model_info.get('feat_in', 80)} mel-filterbanks |
| Sample Rate | 16 kHz |
| Language | Hindi |
| Model Size | {model_info.get('file_size_mb', 'N/A')} MB |

## Supported Tag Categories

{tag_listing}
- **END**: Delimiter token for entity span boundaries

## Evaluation Results

### ASR Performance

| Dataset | WER | CER | Samples |
|---------|-----|-----|---------|
| Internal validation set | {wer_str} | {cer_str} | {n_samples} |
{fleurs_row}

### Tag Classification

{chr(10).join(tag_sections)}

{entity_section}

## Usage

```python
from nemo.collections.asr.models import EncDecCTCModelBPE

# Load model
model = EncDecCTCModelBPE.restore_from("{model_filename}")
model.eval()

# Transcribe
transcriptions = model.transcribe(["audio.wav"])
print(transcriptions[0])
# Example output: "ENTITY_PERSON_NAME नरेंद्र मोदी END ने कहा कि AGE_45_60 GENDER_MALE EMOTION_NEUTRAL INTENT_INFORM"
```

### Extracting Tags

```python
import re

text = transcriptions[0]

# Extract trailing tags
age = re.search(r"\\b(AGE_\\S+)\\b", text)
gender = re.search(r"\\b(GENDER_\\S+)\\b", text)
emotion = re.search(r"\\b(EMOTION_\\S+)\\b", text)
intent = re.search(r"\\b(INTENT_\\S+)\\b", text)

# Extract inline entities
entities = re.findall(r"(ENTITY_\\S+)\\s+(.*?)\\s+END", text)
# [("ENTITY_PERSON_NAME", "नरेंद्र मोदी")]

# Get clean transcript (tags removed)
clean = re.sub(r"\\b(?:AGE_\\S+|GENDER_\\S+|EMOTION_\\S+|INTENT_\\S+|ENTITY_\\S+|END)\\b", "", text)
clean = " ".join(clean.split())
```

## Training

- **Base model**: [WhissleAI/STT-meta-1B](https://huggingface.co/WhissleAI/STT-meta-1B) (Conformer 600M, multilingual)
- **Fine-tuning data**: 223K Hindi utterances with speech tags (entity, age, gender, emotion, intent annotations)
- **Optimizer**: AdamW, cosine annealing LR schedule
- **Training**: Full fine-tuning on NVIDIA A100 40GB

## Citation

```bibtex
@misc{{whissle-stt-meta-hi,
  title={{STT-meta-HI: Hindi Speech Tagger with Entity Recognition}},
  author={{Whissle AI}},
  year={{2026}},
  url={{https://huggingface.co/{repo_id}}}
}}
```

## License

Apache 2.0
"""
    return card


def upload_with_retry(api, path, repo_id, path_in_repo, token, max_retries=3):
    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=token,
            )
            print(f"  Uploaded {path_in_repo}")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                delay = 5 * (2 ** attempt)
                print(f"  Retry {attempt + 1}/{max_retries} in {delay}s: {e}")
                time.sleep(delay)
            else:
                print(f"  FAILED {path_in_repo}: {e}")
                return False


def main():
    parser = argparse.ArgumentParser(description="Push .nemo model to HuggingFace")
    parser.add_argument("--model", required=True, help="Path to .nemo file")
    parser.add_argument("--repo", required=True, help="HF repo ID (e.g. WhissleAI/STT-meta-HI)")
    parser.add_argument("--eval-summary", help="Path to eval_summary.json")
    parser.add_argument("--token", required=True, help="HuggingFace API token")
    parser.add_argument("--fleurs-wer", type=float, help="FLEURS Hindi WER (optional)")
    parser.add_argument("--eval-xlsx", help="Path to eval_results.xlsx to upload")
    args = parser.parse_args()

    from huggingface_hub import HfApi, create_repo
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi()

    # Ensure repo exists
    try:
        api.repo_info(args.repo)
        print(f"Repository {args.repo} exists")
    except RepositoryNotFoundError:
        print(f"Creating {args.repo}")
        create_repo(args.repo, token=args.token, private=False)

    # Extract model info
    print("Extracting model info from .nemo archive...")
    model_info = extract_model_info(args.model)
    print(f"  Encoder: {model_info.get('encoder_layers')} layers, d_model={model_info.get('d_model')}")
    print(f"  Vocab: {model_info.get('vocab_size')} tokens, {model_info.get('file_size_mb')} MB")

    # Load eval summary
    eval_summary = None
    if args.eval_summary and os.path.exists(args.eval_summary):
        with open(args.eval_summary) as f:
            eval_summary = json.load(f)
        print(f"  Eval: WER={eval_summary.get('wer')}, CER={eval_summary.get('cer')}")

    # Generate model card
    model_filename = os.path.basename(args.model)
    readme_content = generate_model_card(
        model_info, eval_summary, args.fleurs_wer, args.repo, model_filename,
    )

    # Write README to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(readme_content)
        readme_path = f.name

    # Upload files
    print(f"\nUploading to {args.repo}...")
    upload_with_retry(api, readme_path, args.repo, "README.md", args.token)
    upload_with_retry(api, args.model, args.repo, model_filename, args.token)

    if args.eval_xlsx and os.path.exists(args.eval_xlsx):
        upload_with_retry(api, args.eval_xlsx, args.repo, "eval_results.xlsx", args.token)

    if args.eval_summary and os.path.exists(args.eval_summary):
        upload_with_retry(api, args.eval_summary, args.repo, "eval_summary.json", args.token)

    os.unlink(readme_path)
    print(f"\nDone! Model published at https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
