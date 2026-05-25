"""
Split multi-speaker segments into single-speaker samples.

Reads an annotated MISC manifest and splits segments where Gemini detected
speaker changes. Each sub-segment gets its own ROLE, BEHAVIOR, EVAL, etc.
via fresh Gemini re-annotation. Audio is split proportionally using NeMo
manifest offset/duration fields (no physical file splitting needed).

Usage:
  export GEMINI_API_KEY=your_key
  python split_multi_speaker.py \
    --input /mnt/nfs/data/en_in_tech_interviews/train_misc.json \
    --output /mnt/nfs/data/en_in_tech_interviews/train_misc_split.json \
    --workers 10
"""
import argparse
import json
import logging
import os
import re
import sys
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("split_multi_speaker.log")]
)

VALID_ROLES = ["INTERVIEWER", "INTERVIEWEE"]
ALL_BEHAVIORS = [
    "QUESTION_OPEN", "QUESTION_CLOSED", "INFORM", "REFLECT", "AFFIRM",
    "DIRECT", "ADVISE", "CONFRONT", "STRUCTURE", "FACILITATE", "EVALUATE", "WARN",
    "EXPLAIN", "REASON", "COMMIT", "ABILITY", "QUESTION", "ACKNOWLEDGE",
    "THINK_ALOUD", "EXPRESS", "FOLLOW_NEUTRAL", "SUPPORT", "REFRAME",
    "RAISE_CONCERN", "FILLER",
]
EVAL_CODES = ["CORRECT", "INCORRECT", "PARTIAL", "PROBE", "HINT", "SKIP", "NONE"]
ENTITY_TYPES = ["TECHNOLOGY", "CONCEPT", "SYSTEM", "METRIC", "COMPANY", "ROLE", "PROJECT", "ACRONYM"]

META_TAG_RE = re.compile(
    r'\b(AGE_\S+|GENDER_\S+|EMOTION_\S+|BEHAVIOR_\S+|ENTITY_\S+|ROLE_\S+|KEYWORD_\S+|EVAL_\S+|INTENT_\S+|SPEAKER_CHANGE)\b'
)

def strip_all_tags(text: str) -> str:
    cleaned = META_TAG_RE.sub('', text)
    return re.sub(r'\s+', ' ', cleaned).strip()


ANNOTATE_PROMPT = """You are an expert behavioral coder for tech interviews.

Given a SHORT text segment from a tech interview, classify it:

- role: INTERVIEWER or INTERVIEWEE
- behavior: one of: QUESTION_OPEN, QUESTION_CLOSED, INFORM, REFLECT, AFFIRM, DIRECT, ADVISE, CONFRONT, STRUCTURE, FACILITATE, EVALUATE, WARN, EXPLAIN, REASON, COMMIT, ABILITY, QUESTION, ACKNOWLEDGE, THINK_ALOUD, EXPRESS, FOLLOW_NEUTRAL, SUPPORT, REFRAME, RAISE_CONCERN, FILLER
- eval: CORRECT, INCORRECT, PARTIAL, PROBE, HINT, SKIP, or NONE (NONE for all interviewee speech and non-evaluative interviewer speech)
- emotion: NEUTRAL, HAPPY, SAD, ANGRY, FEAR, SURPRISE, or DISGUST
- age: 20_30, 30_45, 45_60, 60PLUS, or CHILD
- gender: MALE, FEMALE, or OTHER
- entities: list of {type, text} where type is TECHNOLOGY/CONCEPT/SYSTEM/METRIC/COMPANY/ROLE/PROJECT/ACRONYM
- confidence_level: 1-5
- fluency_score: 1-5
- technical_depth: 1-5
- communication_clarity: 1-5
- interview_stage: introduction, technical, behavioral, or closing

Return ONLY a JSON array, one object per utterance. No markdown fences."""


def build_split_prompt(texts: list[str]) -> str:
    numbered = "\n".join(f'{i+1}. "{t}"' for i, t in enumerate(texts))
    return f"""{ANNOTATE_PROMPT}

Utterances:

{numbered}"""


def call_gemini_batch(texts: list[str], model, retries: int = 5) -> list[dict]:
    prompt = build_split_prompt(texts)
    default = {
        "role": "INTERVIEWEE", "behavior": "EXPLAIN", "eval": "NONE",
        "emotion": "NEUTRAL", "age": "30_45", "gender": "MALE",
        "entities": [], "confidence_level": 3, "fluency_score": 3,
        "technical_depth": 3, "communication_clarity": 3,
        "interview_stage": "technical",
    }

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            raw = response.text.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            results = json.loads(raw)

            if not isinstance(results, list) or len(results) != len(texts):
                logging.warning(f"Gemini returned {len(results) if isinstance(results, list) else 'non-list'}, expected {len(texts)}")
                return [dict(default) for _ in texts]

            validated = []
            for r in results:
                v = dict(default)
                if isinstance(r, dict):
                    role = str(r.get("role", "")).upper()
                    v["role"] = role if role in VALID_ROLES else "INTERVIEWEE"

                    behavior = str(r.get("behavior", "")).upper().replace("BEHAVIOR_", "").replace("INTENT_", "")
                    v["behavior"] = behavior if behavior in ALL_BEHAVIORS else "EXPLAIN"

                    eval_code = str(r.get("eval", "NONE")).upper().replace("EVAL_", "")
                    v["eval"] = eval_code if eval_code in EVAL_CODES else "NONE"

                    emotion = str(r.get("emotion", "NEUTRAL")).upper().replace("EMOTION_", "")
                    v["emotion"] = emotion if emotion in ["NEUTRAL", "HAPPY", "SAD", "ANGRY", "FEAR", "SURPRISE", "DISGUST"] else "NEUTRAL"

                    age = str(r.get("age", "30_45")).upper().replace("AGE_", "")
                    v["age"] = age if age in ["20_30", "30_45", "45_60", "60PLUS", "CHILD"] else "30_45"

                    gender = str(r.get("gender", "MALE")).upper().replace("GENDER_", "")
                    v["gender"] = gender if gender in ["MALE", "FEMALE", "OTHER"] else "MALE"

                    entities = r.get("entities", [])
                    v["entities"] = [
                        e for e in entities
                        if isinstance(e, dict) and e.get("type", "").upper() in ENTITY_TYPES
                    ]

                    for field in ["confidence_level", "fluency_score", "technical_depth", "communication_clarity"]:
                        val = r.get(field, 3)
                        v[field] = max(1, min(5, int(val))) if isinstance(val, (int, float)) else 3

                    stage = str(r.get("interview_stage", "technical")).lower()
                    v["interview_stage"] = stage if stage in ("introduction", "technical", "behavioral", "closing") else "technical"

                validated.append(v)
            return validated

        except json.JSONDecodeError as e:
            logging.warning(f"JSON parse error (attempt {attempt+1}): {e}")
            if attempt == retries - 1:
                return [dict(default) for _ in texts]
        except Exception as e:
            wait = min(2 ** (attempt + 1) * 2, 60)
            logging.warning(f"Gemini error (attempt {attempt+1}/{retries}): {e}, retry in {wait}s")
            time.sleep(wait)

    return [dict(default) for _ in texts]


def build_sample_text(clean_text: str, annotation: dict) -> str:
    """Build manifest text with inline entities + end-of-line classifier tags."""
    parts = [clean_text]

    for ent in annotation.get("entities", []):
        ent_type = ent.get("type", "").upper()
        if ent_type and ent_type in ENTITY_TYPES:
            parts.append(f"ENTITY_{ent_type}")

    parts.append(f"AGE_{annotation['age']}")
    parts.append(f"GENDER_{annotation['gender']}")
    parts.append(f"EMOTION_{annotation['emotion']}")
    parts.append(f"ROLE_{annotation['role']}")
    parts.append(f"BEHAVIOR_{annotation['behavior']}")
    parts.append(f"EVAL_{annotation['eval']}")

    return " ".join(parts)


def split_segment(sample: dict, idx: int) -> list[dict]:
    """Split a multi-speaker sample into single-speaker sub-samples."""
    speaker_changes = sample.get("speaker_changes", [])
    if not speaker_changes:
        return [sample]

    clean_text = strip_all_tags(sample.get("text", ""))
    words = clean_text.split()
    total_words = len(words)

    if total_words == 0:
        return [sample]

    duration = sample.get("duration", 0)
    offset = sample.get("offset", 0.0)
    audio_filepath = sample.get("audio_filepath", "")

    # Build split points (sorted, deduplicated, within bounds)
    split_points = sorted(set(
        sc for sc in speaker_changes
        if isinstance(sc, (int, float)) and 0 < int(sc) < total_words
    ))

    if not split_points:
        return [sample]

    # Create sub-segments
    boundaries = [0] + [int(sp) for sp in split_points] + [total_words]
    sub_samples = []

    for i in range(len(boundaries) - 1):
        start_word = boundaries[i]
        end_word = boundaries[i + 1]

        sub_text = " ".join(words[start_word:end_word])
        if not sub_text.strip():
            continue

        # Proportional audio split
        start_frac = start_word / total_words
        end_frac = end_word / total_words
        sub_offset = offset + start_frac * duration
        sub_duration = (end_frac - start_frac) * duration

        # Skip very short segments (< 0.3s)
        if sub_duration < 0.3:
            continue

        sub_sample = {
            "audio_filepath": audio_filepath,
            "offset": round(sub_offset, 3),
            "duration": round(sub_duration, 3),
            "text": sub_text,  # placeholder — will be rebuilt after annotation
            "lang": sample.get("lang", "EN"),
            "lang_family": sample.get("lang_family", "ENGLISH"),
            "_clean_text": sub_text,
            "_parent_idx": idx,
            "_segment_idx": i,
            "_needs_annotation": True,
        }
        sub_samples.append(sub_sample)

    return sub_samples if sub_samples else [sample]


def main():
    parser = argparse.ArgumentParser(description="Split multi-speaker segments")
    parser.add_argument("--input", required=True, help="Input MISC-annotated manifest")
    parser.add_argument("--output", required=True, help="Output split manifest")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--gemini-key", help="Gemini API key (or GEMINI_API_KEY env)")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    args = parser.parse_args()

    # Read input
    logging.info(f"Reading {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    logging.info(f"Total samples: {len(lines)}")

    # Phase 1: Split multi-speaker segments
    all_samples = []
    needs_annotation = []
    multi_speaker_count = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        sample = json.loads(line)
        speaker_changes = sample.get("speaker_changes", [])

        if speaker_changes:
            multi_speaker_count += 1
            sub_samples = split_segment(sample, i)
            for sub in sub_samples:
                if sub.get("_needs_annotation"):
                    needs_annotation.append((len(all_samples), sub))
                all_samples.append(sub)
        else:
            # Single speaker — strip any stray SPEAKER_CHANGE from text
            text = sample.get("text", "")
            if "SPEAKER_CHANGE" in text:
                text = text.replace("SPEAKER_CHANGE", "").strip()
                text = re.sub(r'\s+', ' ', text)
                sample["text"] = text
            all_samples.append(sample)

    logging.info(f"Multi-speaker segments: {multi_speaker_count}")
    logging.info(f"After splitting: {len(all_samples)} total samples (+{len(all_samples) - len(lines)} new)")
    logging.info(f"Sub-segments needing annotation: {len(needs_annotation)}")

    # Phase 2: Annotate split sub-segments with Gemini
    if needs_annotation:
        api_key = args.gemini_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logging.error("Set GEMINI_API_KEY env or pass --gemini-key")
            sys.exit(1)
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(args.gemini_model)
        logging.info(f"Gemini model: {args.gemini_model}")

        # Build batches
        batches = []
        for batch_start in range(0, len(needs_annotation), args.batch_size):
            batch = needs_annotation[batch_start:batch_start + args.batch_size]
            batch_texts = []
            for sample_idx, sub in batch:
                text = sub.get("_clean_text", strip_all_tags(sub.get("text", "")))
                if len(text) > 500:
                    text = text[:500]
                batch_texts.append(text if text else "...")
            batches.append((batch, batch_texts))

        logging.info(f"Annotating {len(needs_annotation)} sub-segments in {len(batches)} batches")

        annotated_count = 0
        lock = threading.Lock()

        def process_batch(batch_item):
            batch, batch_texts = batch_item
            return batch, call_gemini_batch(batch_texts, model)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_batch, b): i for i, b in enumerate(batches)}

            for future in as_completed(futures):
                try:
                    batch, annotations = future.result()
                except Exception as e:
                    logging.error(f"Batch failed: {e}")
                    continue

                with lock:
                    for (sample_idx, sub), annotation in zip(batch, annotations):
                        clean_text = sub.get("_clean_text", strip_all_tags(sub.get("text", "")))
                        sub["text"] = build_sample_text(clean_text, annotation)
                        sub["speaker_role"] = annotation["role"].lower()
                        sub["misc_behavior"] = annotation["behavior"]
                        sub["misc_eval"] = annotation["eval"]
                        sub["confidence_level"] = annotation["confidence_level"]
                        sub["fluency_score"] = annotation["fluency_score"]
                        sub["technical_depth"] = annotation["technical_depth"]
                        sub["communication_clarity"] = annotation["communication_clarity"]
                        sub["interview_stage"] = annotation["interview_stage"]
                        sub["entities"] = [
                            {"type": e["type"], "text": e["text"]}
                            for e in annotation.get("entities", [])
                        ]
                        sub["behavioral_keywords"] = []
                        sub["speaker_changes"] = []

                        # Clean up internal fields
                        for key in ["_clean_text", "_parent_idx", "_segment_idx", "_needs_annotation"]:
                            sub.pop(key, None)

                        all_samples[sample_idx] = sub
                        annotated_count += 1

                    if annotated_count % 500 < args.batch_size:
                        logging.info(f"Annotated: {annotated_count}/{len(needs_annotation)}")

        logging.info(f"Annotation complete: {annotated_count} sub-segments")

    # Phase 3: Clean all samples and write output
    logging.info(f"Writing {len(all_samples)} samples to {args.output}")

    stats = Counter()
    with open(args.output, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            # Clean up any remaining internal fields
            for key in ["_clean_text", "_parent_idx", "_segment_idx", "_needs_annotation"]:
                sample.pop(key, None)

            # Remove speaker_changes from output (no longer needed)
            sample.pop("speaker_changes", None)

            stats[sample.get("speaker_role", "unknown")] += 1
            stats[f"behavior_{sample.get('misc_behavior', 'unknown')}"] += 1
            stats[f"eval_{sample.get('misc_eval', 'unknown')}"] += 1

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("SPLIT SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Input samples: {len(lines)}")
    logging.info(f"Multi-speaker segments split: {multi_speaker_count}")
    logging.info(f"Output samples: {len(all_samples)}")
    logging.info(f"Net new samples: {len(all_samples) - len(lines)}")

    role_stats = {k: v for k, v in stats.items() if k in ["interviewer", "interviewee", "unknown"]}
    logging.info(f"\nRoles: {dict(role_stats)}")

    behavior_stats = {k: v for k, v in stats.items() if k.startswith("behavior_")}
    logging.info(f"\nTop behaviors:")
    for k, v in sorted(behavior_stats.items(), key=lambda x: -x[1])[:10]:
        logging.info(f"  {k}: {v}")

    eval_stats = {k: v for k, v in stats.items() if k.startswith("eval_")}
    logging.info(f"\nEvals: {dict(eval_stats)}")


if __name__ == "__main__":
    main()
