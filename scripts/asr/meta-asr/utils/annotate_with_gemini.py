"""
Annotate missing INTENT tags in META-ASR manifests using Gemini 2.5 Flash.

Finds samples that have AGE/GENDER/EMOTION inline tags but are missing INTENT_*,
batches them to Gemini for classification, and inserts the predicted intent tag.

Usage:
  # Dry-run: report counts only
  python annotate_with_gemini.py --data-root /mnt/nfs/data/multilingual_v1/raw --dry-run

  # Annotate all manifests
  export GEMINI_API_KEY=your_key
  python annotate_with_gemini.py --data-root /mnt/nfs/data/multilingual_v1/raw --batch-size 50

  # Single file
  python annotate_with_gemini.py --file /mnt/nfs/data/multilingual_v1/raw/cv_euro/train.json

  # Also fix malformed entity tags
  python annotate_with_gemini.py --data-root /mnt/nfs/data/multilingual_v1/raw --fix-entities
"""
import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CANONICAL_INTENTS = [
    "INFORM", "QUESTION", "COMMAND", "REQUEST", "EXCLAIM",
    "OPINION", "EXPLAIN", "DESCRIBE", "STATEMENT",
]

HAS_META_TAGS_RE = re.compile(r'\b(AGE_\S+|GENDER_\S+|EMOTION_\S+)\b')
HAS_INTENT_RE = re.compile(r'\bINTENT_\S+')
META_TAG_RE = re.compile(r'\b(AGE_\S+|GENDER_\S+|EMOTION_\S+|INTENT_\S+|ENTITY_\S+|END)\b')

ENTITY_GLUED_RE = re.compile(r'\bENTITY_(CITY|ORGANIZATION|PERSON|LOCATION|DATE|TIME|NUMBER)END\b')
ENTITY_GENDER_RE = re.compile(r'\bENTITY_GENDER\S*\b')


def strip_meta_tags(text: str) -> str:
    cleaned = META_TAG_RE.sub('', text)
    return re.sub(r'\s+', ' ', cleaned).strip()


def needs_intent(text: str) -> bool:
    return bool(HAS_META_TAGS_RE.search(text)) and not bool(HAS_INTENT_RE.search(text))


def fix_entity_tags(text: str) -> tuple[str, bool]:
    changed = False
    if ENTITY_GLUED_RE.search(text):
        text = ENTITY_GLUED_RE.sub(r'ENTITY_\1 END', text)
        changed = True
    if ENTITY_GENDER_RE.search(text):
        text = ENTITY_GENDER_RE.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()
        changed = True
    return text, changed


def classify_intents_batch(texts: list[str], model, retries: int = 5) -> list[str]:
    numbered = "\n".join(f"{i+1}. \"{t}\"" for i, t in enumerate(texts))
    prompt = (
        "Classify the intent of each transcript below. "
        "Return ONLY a JSON array of intent labels, one per transcript, in order.\n"
        f"Valid labels: {', '.join(CANONICAL_INTENTS)}\n\n"
        f"{numbered}"
    )

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            raw = response.text.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            intents = json.loads(raw)
            if not isinstance(intents, list) or len(intents) != len(texts):
                logging.warning(f"Gemini returned {len(intents) if isinstance(intents, list) else 'non-list'}, expected {len(texts)}")
                return ["STATEMENT"] * len(texts)
            validated = []
            for intent in intents:
                intent_upper = intent.upper().replace("INTENT_", "")
                if intent_upper in CANONICAL_INTENTS:
                    validated.append(intent_upper)
                else:
                    validated.append("STATEMENT")
            return validated
        except json.JSONDecodeError as e:
            logging.error(f"JSON parse error from Gemini: {e}\nRaw: {raw[:200]}")
            return ["STATEMENT"] * len(texts)
        except Exception as e:
            wait = min(2 ** (attempt + 1) * 2, 60)
            logging.warning(f"Gemini API error (attempt {attempt+1}/{retries}): {e}, retrying in {wait}s")
            time.sleep(wait)

    logging.error(f"Failed batch after {retries} retries, using STATEMENT")
    return ["STATEMENT"] * len(texts)


def insert_intent_tag(text: str, intent: str) -> str:
    emotion_match = re.search(r'\bEMOTION_\S+', text)
    if emotion_match:
        pos = emotion_match.end()
        return text[:pos] + f" INTENT_{intent}" + text[pos:]
    gender_match = re.search(r'\bGENDER_\S+', text)
    if gender_match:
        pos = gender_match.end()
        return text[:pos] + f" INTENT_{intent}" + text[pos:]
    age_match = re.search(r'\bAGE_\S+', text)
    if age_match:
        pos = age_match.end()
        return text[:pos] + f" INTENT_{intent}" + text[pos:]
    return text + f" INTENT_{intent}"


def process_manifest(filepath: str, model, batch_size: int, dry_run: bool,
                     fix_entities: bool, max_workers: int = 20) -> dict:
    stats = defaultdict(int)
    stats["file"] = filepath

    with open(filepath, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    stats["total"] = len(raw_lines)

    progress_file = filepath + ".annotate_progress.json"
    completed_indices = set()
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            progress = json.load(f)
            completed_indices = set(progress.get("completed", []))
        logging.info(f"  Resuming: {len(completed_indices)} already annotated")

    samples = []
    for i, line in enumerate(raw_lines):
        line = line.strip()
        if not line:
            samples.append((i, None, line))
            continue
        try:
            sample = json.loads(line)
            samples.append((i, sample, line))
        except json.JSONDecodeError:
            stats["json_errors"] += 1
            samples.append((i, None, line))

    need_intent_indices = []
    for i, sample, raw in samples:
        if sample and needs_intent(sample.get("text", "")) and i not in completed_indices:
            need_intent_indices.append(i)

    stats["missing_intent"] = len(need_intent_indices) + len(completed_indices)
    stats["to_annotate"] = len(need_intent_indices)

    if fix_entities:
        entity_fixes = 0
        for i, sample, raw in samples:
            if sample and sample.get("text"):
                fixed_text, changed = fix_entity_tags(sample["text"])
                if changed:
                    sample["text"] = fixed_text
                    samples[i] = (i, sample, raw)
                    entity_fixes += 1
        stats["entity_fixes"] = entity_fixes

    if dry_run:
        stats["annotated"] = 0
        return dict(stats)

    if not need_intent_indices and (not fix_entities or stats.get("entity_fixes", 0) == 0):
        return dict(stats)

    # Build all batches upfront
    all_batches = []
    for batch_start in range(0, len(need_intent_indices), batch_size):
        batch_indices = need_intent_indices[batch_start:batch_start + batch_size]
        batch_texts = []
        for idx in batch_indices:
            _, sample, _ = samples[idx]
            clean = strip_meta_tags(sample["text"])
            if len(clean) > 500:
                clean = clean[:500]
            batch_texts.append(clean if clean else "...")
        all_batches.append((batch_indices, batch_texts))

    total_batches = len(all_batches)
    annotated = 0
    lock = threading.Lock()

    def process_batch(batch_item):
        batch_indices, batch_texts = batch_item
        intents = classify_intents_batch(batch_texts, model)
        return batch_indices, intents

    if total_batches > 0:
        logging.info(f"  Annotating {len(need_intent_indices)} samples in {total_batches} batches ({max_workers} parallel workers)")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_batch, b): i for i, b in enumerate(all_batches)}

        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                batch_indices, intents = future.result()
            except Exception as e:
                logging.error(f"Batch {batch_idx} failed: {e}")
                batch_indices = all_batches[batch_idx][0]
                intents = ["STATEMENT"] * len(batch_indices)

            with lock:
                for idx, intent in zip(batch_indices, intents):
                    i, sample, raw = samples[idx]
                    sample["text"] = insert_intent_tag(sample["text"], intent)
                    samples[idx] = (i, sample, raw)
                    completed_indices.add(i)
                    annotated += 1

                if annotated % 5000 < batch_size:
                    with open(progress_file, 'w') as f:
                        json.dump({"completed": list(completed_indices)}, f)
                    logging.info(f"  Progress: {annotated}/{len(need_intent_indices)} annotated ({annotated*100//len(need_intent_indices)}%)")

    stats["annotated"] = annotated

    backup = filepath + ".pre_annotate.bak"
    if not os.path.exists(backup):
        os.rename(filepath, backup)
    else:
        os.remove(filepath)

    with open(filepath, 'w', encoding='utf-8') as f:
        for i, sample, raw in samples:
            if sample:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            else:
                if raw:
                    f.write(raw + "\n")

    if os.path.exists(progress_file):
        os.remove(progress_file)

    stats["written"] = True
    return dict(stats)


def find_manifests(data_root: str) -> list[str]:
    manifests = []
    for root, dirs, files in os.walk(data_root):
        dirs[:] = [d for d in dirs if d != 'audio']
        for f in sorted(files):
            if (f.endswith('.json') and f != 'dataset_info.json'
                    and not f.endswith('.bak')
                    and '.annotate_progress' not in f
                    and '.pre_annotate' not in f):
                manifests.append(os.path.join(root, f))
    return manifests


def main():
    parser = argparse.ArgumentParser(description="Annotate missing INTENT tags with Gemini")
    parser.add_argument("--data-root", help="Root of raw data dir")
    parser.add_argument("--file", help="Process single manifest file")
    parser.add_argument("--batch-size", type=int, default=50, help="Texts per Gemini call")
    parser.add_argument("--dry-run", action="store_true", help="Report only, no changes")
    parser.add_argument("--fix-entities", action="store_true", help="Also fix malformed entity tags")
    parser.add_argument("--gemini-key", help="Gemini API key (or set GEMINI_API_KEY env)")
    parser.add_argument("--workers", type=int, default=20, help="Parallel Gemini API calls (default 20)")
    args = parser.parse_args()

    if not args.data_root and not args.file:
        parser.error("Provide --data-root or --file")

    model = None
    if not args.dry_run:
        api_key = args.gemini_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logging.error("Set GEMINI_API_KEY env or pass --gemini-key")
            sys.exit(1)
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        logging.info("Gemini 2.5 Flash initialized")

    if args.file:
        manifests = [args.file]
    else:
        manifests = find_manifests(args.data_root)

    logging.info(f"Found {len(manifests)} manifest files")
    if args.dry_run:
        logging.info("DRY RUN — no files will be modified\n")

    total_missing = 0
    total_annotated = 0
    total_entity_fixes = 0

    for mf in manifests:
        stats = process_manifest(mf, model, args.batch_size, args.dry_run, args.fix_entities,
                                 max_workers=args.workers)
        missing = stats.get("missing_intent", 0)
        annotated = stats.get("annotated", 0)
        entity_fixes = stats.get("entity_fixes", 0)
        total_missing += missing
        total_annotated += annotated
        total_entity_fixes += entity_fixes

        rel = os.path.relpath(mf, args.data_root) if args.data_root else mf
        parts = [f"{stats['total']} samples"]
        if missing > 0:
            parts.append(f"{missing} missing intent")
        if annotated > 0:
            parts.append(f"{annotated} annotated")
        if entity_fixes > 0:
            parts.append(f"{entity_fixes} entity fixes")
        if stats.get("json_errors"):
            parts.append(f"{stats['json_errors']} JSON errors")

        status = "OK" if missing == 0 and entity_fixes == 0 else ""
        if stats.get("written"):
            status = "(written)"
        logging.info(f"  {rel}: {', '.join(parts)} {status}")

    logging.info(f"\n{'='*60}")
    logging.info(f"Total manifests: {len(manifests)}")
    logging.info(f"Total missing INTENT: {total_missing}")
    logging.info(f"Total annotated: {total_annotated}")
    if total_entity_fixes:
        logging.info(f"Total entity fixes: {total_entity_fixes}")
    if args.dry_run:
        logging.info("Re-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
