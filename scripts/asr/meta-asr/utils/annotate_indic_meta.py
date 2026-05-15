"""
Annotate raw Indic speech manifests with META-ASR tags:
  AGE, GENDER (from parquet metadata or HuBERT), EMOTION (HuBERT), ENTITY/INTENT (Gemini).

Designed for Gujarati but works for any Indic language by adjusting --lang.

Usage:
  # Full pipeline (emotion + entity/intent)
  python annotate_indic_meta.py \
      --manifest /mnt/nfs/data/gujarati_v1/raw/indicvoices/train.json \
      --output /mnt/nfs/data/gujarati_v1/raw/indicvoices/train_nemo.jsonl \
      --lang gujarati --batch-size 10

  # Skip emotion extraction (use default NEUTRAL)
  python annotate_indic_meta.py \
      --manifest ... --output ... --lang gujarati --skip-emotion
"""
import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CANONICAL_INTENTS = [
    "INFORM", "QUESTION", "COMMAND", "REQUEST", "EXCLAIM",
    "OPINION", "EXPLAIN", "DESCRIBE", "STATEMENT",
]

ENTITY_TYPES_STR = """PERSON_NAME, ORGANIZATION, LOCATION, ADDRESS, CITY, STATE, COUNTRY,
CURRENCY, PRICE, DATE, TIME, DURATION, EVENT, TASK, ACTION_ITEM, PRIORITY,
FEEDBACK, COMPLAINT, QUESTION, RESPONSE, REMINDER, NOTE, ANNOUNCEMENT,
SCHEDULE, ORDER_NUMBER, PAYMENT_METHOD, PAYMENT_AMOUNT, BANK_NAME,
SYMPTOM, DIAGNOSIS, MEDICATION, DOSAGE, PRESCRIPTION, HOSPITAL_NAME,
DEPARTMENT, PRODUCT, SERVICE, FOOD_ITEM, CUISINE, OCCUPATION, SKILL,
MEASUREMENT, WEATHER_CONDITION, TEMPERATURE"""


def normalize_age_group(age_group):
    if not age_group:
        return "UNKNOWN"
    return str(age_group).replace("-", "_").replace(" ", "_").upper()


def normalize_gender(gender):
    if not gender:
        return "UNKNOWN"
    return str(gender).upper().strip()


class EmotionExtractor:
    def __init__(self, device="cpu"):
        import torch
        from transformers import pipeline as hf_pipeline
        self.classifier = hf_pipeline(
            "audio-classification",
            model="superb/hubert-large-superb-er",
            device=0 if device == "cuda" and torch.cuda.is_available() else -1,
        )

    def predict_batch(self, audio_paths, batch_size=8):
        results = []
        for i in range(0, len(audio_paths), batch_size):
            batch = audio_paths[i:i + batch_size]
            try:
                preds = self.classifier(batch, return_all_scores=False)
                for p in preds:
                    top = p[0] if isinstance(p, list) else p
                    results.append(top["label"].upper())
            except Exception as e:
                logger.warning("Emotion extraction failed for batch %d: %s", i, e)
                results.extend(["NEUTRAL"] * len(batch))
        return results


def build_gemini_prompt(sentences, lang="gujarati"):
    lang_title = lang.title()
    return f'''You are an expert linguistic annotator for {lang_title} text.
You will receive a list of {lang_title} sentences. Each sentence already includes metadata tags (AGE_*, GENDER_*, EMOTION_*) at the end.

Your task:
1. PRESERVE existing AGE_, GENDER_, EMOTION_ tags exactly as they appear at the end.
2. Identify entities in the transcription text and wrap them with ENTITY_<TYPE> ... END tags.
3. Add one INTENT_<TYPE> tag at the absolute end.
4. Return a JSON array of annotated strings, same order, same count.

Entity types: [{ENTITY_TYPES_STR}]
Intent types: [{", ".join(CANONICAL_INTENTS)}]

Sentences to annotate:
{json.dumps(sentences, ensure_ascii=False)}'''


def annotate_with_gemini(sentences, lang, api_key, retries=3):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = build_gemini_prompt(sentences, lang)

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()

            if text.startswith("```json"):
                text = text[len("```json"):].strip()
            if text.startswith("```"):
                text = text[len("```"):].strip()
            if text.endswith("```"):
                text = text[:-3].strip()

            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                text = match.group(0)

            results = json.loads(text)
            if isinstance(results, list) and len(results) == len(sentences):
                return results

            logger.warning("Gemini returned %d results for %d inputs (attempt %d)",
                           len(results) if isinstance(results, list) else -1,
                           len(sentences), attempt + 1)
        except Exception as e:
            logger.warning("Gemini annotation failed (attempt %d): %s", attempt + 1, e)
            if "rate" in str(e).lower():
                time.sleep(10 * (attempt + 1))
            else:
                time.sleep(2 * (attempt + 1))

    return sentences


def process_manifest(manifest_path, output_path, lang, batch_size, skip_emotion, gemini_key):
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    logger.info("Loaded %d records from %s", len(records), manifest_path)

    already_done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    already_done.add(r.get("audio_filepath"))
                except json.JSONDecodeError:
                    pass
        logger.info("Resuming: %d records already annotated", len(already_done))

    pending = [r for r in records if r.get("audio_filepath") not in already_done]
    logger.info("%d records remaining to process", len(pending))

    if not pending:
        return

    emotion_extractor = None
    if not skip_emotion:
        logger.info("Loading HuBERT emotion model...")
        emotion_extractor = EmotionExtractor()

    total_written = len(already_done)

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(pending) + batch_size - 1) // batch_size

        # Step 1: Build base text with AGE/GENDER/EMOTION
        audio_paths = [r["audio_filepath"] for r in batch]

        if not skip_emotion and emotion_extractor:
            emotions = emotion_extractor.predict_batch(audio_paths)
        else:
            emotions = ["NEUTRAL"] * len(batch)

        base_texts = []
        for r, emotion in zip(batch, emotions):
            text = r.get("text", "").strip()
            age = normalize_age_group(r.get("age_group", ""))
            gender = normalize_gender(r.get("gender", ""))
            base_text = f"{text} AGE_{age} GENDER_{gender} EMOTION_{emotion}"
            base_texts.append(base_text)

        # Step 2: Gemini entity/intent annotation
        if gemini_key:
            annotated = annotate_with_gemini(base_texts, lang, gemini_key)
        else:
            annotated = [t + " INTENT_INFORM" for t in base_texts]

        # Step 3: Write results
        with open(output_path, "a", encoding="utf-8") as f:
            for r, ann_text in zip(batch, annotated):
                out = {
                    "audio_filepath": r["audio_filepath"],
                    "text": ann_text,
                    "duration": r.get("duration", 0),
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

        total_written += len(batch)
        logger.info("Batch %d/%d done. Total written: %d", batch_num, total_batches, total_written)

    logger.info("Annotation complete: %d total records in %s", total_written, output_path)


def main():
    parser = argparse.ArgumentParser(description="Annotate Indic speech manifests with meta-ASR tags")
    parser.add_argument("--manifest", required=True, help="Input NeMo manifest (JSONL)")
    parser.add_argument("--output", required=True, help="Output annotated manifest (JSONL)")
    parser.add_argument("--lang", default="gujarati", help="Language name for prompts")
    parser.add_argument("--batch-size", type=int, default=10, help="Gemini annotation batch size")
    parser.add_argument("--skip-emotion", action="store_true", help="Skip HuBERT emotion, use NEUTRAL")
    parser.add_argument("--gemini-key", default=None, help="Gemini API key (or set GEMINI_API_KEY env)")
    args = parser.parse_args()

    gemini_key = args.gemini_key or os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        logger.warning("No GEMINI_API_KEY — entity/intent annotation will be skipped, using default INTENT_INFORM")

    process_manifest(args.manifest, args.output, args.lang, args.batch_size, args.skip_emotion, gemini_key)


if __name__ == "__main__":
    main()
