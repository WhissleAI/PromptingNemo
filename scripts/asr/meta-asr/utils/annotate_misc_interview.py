"""
MISC-based behavioral annotation for tech interview data using Gemini 2.5 Flash.

Applies the Motivational Interviewing Skill Code (MISC 2.0, Miller et al. 2003)
framework adapted for tech interviews. Each utterance is annotated with:

  Inline tags (ASR model predicts these at inference):
    - ROLE_INTERVIEWER / ROLE_INTERVIEWEE
    - BEHAVIOR_* (25 MISC-based behavioral codes)
    - EVAL_* (7 interviewer evaluation codes)
    - SPEAKER_CHANGE (inline within text at speaker transitions)
    - ENTITY_* (tech domain entities)
    - KEYWORD_* (behavioral markers)
    - AGE_*, GENDER_*, EMOTION_* (kept from original)

  Manifest metadata (Gemini-only, for downstream analytics / VILS):
    - confidence_level (1-5)
    - fluency_score (1-5)
    - technical_depth (1-5)
    - communication_clarity (1-5)
    - interview_stage (introduction/technical/behavioral/closing)

Usage:
  export GEMINI_API_KEY=your_key
  python annotate_misc_interview.py \\
    --input /mnt/nfs/data/en_in_tech_interviews/train_dualhead.json \\
    --output /mnt/nfs/data/en_in_tech_interviews/train_misc.json \\
    --batch-size 20 --workers 10
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
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("annotate_misc.log")]
)

# ---------------------------------------------------------------------------
# MISC 2.0 Adapted Taxonomy for Tech Interviews
# ---------------------------------------------------------------------------

INTERVIEWER_BEHAVIORS = [
    "QUESTION_OPEN",
    "QUESTION_CLOSED",
    "INFORM",
    "REFLECT",
    "AFFIRM",
    "DIRECT",
    "ADVISE",
    "CONFRONT",
    "STRUCTURE",
    "FACILITATE",
    "EVALUATE",
    "WARN",
]

INTERVIEWEE_BEHAVIORS = [
    "EXPLAIN",
    "REASON",
    "COMMIT",
    "ABILITY",
    "QUESTION",
    "ACKNOWLEDGE",
    "THINK_ALOUD",
    "EXPRESS",
]

SHARED_BEHAVIORS = [
    "FOLLOW_NEUTRAL",
    "SUPPORT",
    "REFRAME",
    "RAISE_CONCERN",
    "FILLER",
]

ALL_BEHAVIORS = INTERVIEWER_BEHAVIORS + INTERVIEWEE_BEHAVIORS + SHARED_BEHAVIORS
VALID_ROLES = ["INTERVIEWER", "INTERVIEWEE"]

EVAL_CODES = [
    "CORRECT",
    "INCORRECT",
    "PARTIAL",
    "PROBE",
    "HINT",
    "SKIP",
    "NONE",
]

ENTITY_TYPES = [
    "TECHNOLOGY",
    "CONCEPT",
    "SYSTEM",
    "METRIC",
    "COMPANY",
    "ROLE",
    "PROJECT",
    "ACRONYM",
]

KEYWORD_CATEGORIES = {
    "CONFIDENCE": [
        "definitely", "absolutely", "certainly", "clearly", "obviously",
        "I'm sure", "I'm certain", "I know", "I believe", "without doubt",
    ],
    "HEDGING": [
        "maybe", "perhaps", "I think", "sort of", "kind of", "probably",
        "I guess", "might be", "could be", "not sure", "I'm not certain",
    ],
    "STRUCTURE": [
        "first", "second", "third", "finally", "in summary", "to summarize",
        "let me walk through", "the approach would be", "step by step",
        "on one hand", "on the other hand", "moving on",
    ],
    "FILLER_WORD": [
        "um", "uh", "like", "you know", "basically", "actually",
        "so yeah", "right", "okay so",
    ],
}

META_TAG_RE = re.compile(
    r'\b(AGE_\S+|GENDER_\S+|EMOTION_\S+|INTENT_\S+|BEHAVIOR_\S+|ENTITY_\S+|ROLE_\S+|KEYWORD_\S+|EVAL_\S+|SPEAKER_CHANGE)\b'
)

def strip_all_tags(text: str) -> str:
    cleaned = META_TAG_RE.sub('', text)
    return re.sub(r'\s+', ' ', cleaned).strip()

def extract_existing_tags(text: str) -> dict:
    tags = {}
    for match in META_TAG_RE.finditer(text):
        tag = match.group(0)
        prefix = tag.split('_')[0]
        tags[prefix] = tag
    return tags


# ---------------------------------------------------------------------------
# Gemini Prompt Construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert behavioral coder trained in the MISC 2.0
(Motivational Interviewing Skill Code) framework by Miller et al. (2003),
adapted for tech interview conversations.

Your task: annotate each utterance from a tech interview transcript with
structured behavioral codes. The interviews are between a technical
interviewer and a job candidate (interviewee).

## Speaker Role Detection
Determine if the speaker is:
- INTERVIEWER: asks questions, sets problems, evaluates, manages flow
- INTERVIEWEE: answers questions, explains solutions, demonstrates knowledge

## Behavior Taxonomy (MISC 2.0 Adapted)

### Interviewer Behaviors:
- QUESTION_OPEN: Open question inviting wide elaboration ("Tell me about...", "How would you...")
- QUESTION_CLOSED: Yes/no or specific-answer question ("Did you use X?", "How many?")
- INFORM: Giving context, explaining the problem setup, providing information
- REFLECT: Paraphrasing or summarizing what the candidate said
- AFFIRM: Positive feedback, appreciation, encouragement ("Good answer", "That's right")
- DIRECT: Instructions, commands ("Now implement...", "Write the code for...")
- ADVISE: Hints, suggestions, nudging toward solution ("What about using a hash map?")
- CONFRONT: Challenging an answer, probing deeper ("But wouldn't that be O(n²)?")
- STRUCTURE: Managing flow — greetings, transitions, time management ("Let's move on")
- FACILITATE: Backchannels ("Mm-hmm", "Okay", "Go on", "Right")
- EVALUATE: Explicit assessment ("That's correct", "Not quite", "Partial answer")
- WARN: Constraints to consider ("Remember, this needs to handle edge cases")

### Interviewee Behaviors:
- EXPLAIN: Detailed technical explanation — the primary answer mode
- REASON: Justifying an approach, giving rationale ("Because Redis gives us sub-ms latency")
- COMMIT: Stating a design decision ("I would use a binary search tree here")
- ABILITY: Expressing capability or uncertainty ("I've worked with React" / "I'm not sure")
- QUESTION: Asking for clarification ("Can I assume sorted input?")
- ACKNOWLEDGE: Confirming understanding ("I see", "Got it", "Makes sense")
- THINK_ALOUD: Verbalizing thought process ("Let me think... if we use a trie...")
- EXPRESS: Emotional expression, realization ("Oh, I see!", "That's interesting")

### Shared Behaviors (either role):
- FOLLOW_NEUTRAL: Non-committal, following along
- SUPPORT: Sympathetic, encouraging ("That's a tough one", "Good point")
- REFRAME: Offering a new perspective on what was said
- RAISE_CONCERN: Pointing out a potential problem ("But what about thread safety?")
- FILLER: Pleasantries, non-substantive ("Good morning", "Nice to meet you")

## Interviewer Evaluation (EVAL)

When the interviewer is evaluating or reacting to a candidate's answer, classify:
- CORRECT: Confirms the answer is right ("Yes, exactly", "That's correct")
- INCORRECT: Indicates the answer is wrong ("No, that's not right", "Not quite")
- PARTIAL: Partially correct ("You're on the right track but...", "Almost")
- PROBE: Digs deeper to test depth ("Can you elaborate?", "What about edge cases?")
- HINT: Provides a clue or nudge ("Think about what happens when the list is empty")
- SKIP: Moves on without resolution, implicitly abandoning the topic
- NONE: No evaluation happening — use for all interviewee utterances and non-evaluative interviewer utterances

## Speaker Change Detection

If the utterance contains speech from MULTIPLE speakers (a speaker transition
happens mid-utterance), identify the character positions where each speaker
change occurs. Return these as a list of word indices (0-based) where the new
speaker begins. For example, if the text is:

  "the answer is binary search okay and what about the worst case"

And the interviewer says "okay and what about the worst case" starting at word 5,
return speaker_changes: [5]

If there is no speaker change, return speaker_changes: []

## Entity Types (extract from text):
- TECHNOLOGY: programming languages, frameworks, tools (Python, React, AWS, Kubernetes)
- CONCEPT: algorithms, data structures, patterns (binary search, linked list, microservices)
- SYSTEM: infrastructure components (load balancer, cache, database, message queue)
- METRIC: performance numbers, complexity (O(n log n), 99.9%, 10ms latency)
- COMPANY: companies, products (Google, MongoDB, Redis Labs)
- ROLE: job titles (backend engineer, SRE, tech lead)
- PROJECT: specific projects/features mentioned
- ACRONYM: technical acronyms (API, SDK, CI/CD, DNS, SQL)

## Behavioral Assessment (1-5 scale):
- confidence_level: How confident the speaker sounds (1=very uncertain, 5=very confident)
- fluency_score: Verbal fluency — minimal fillers/hesitations (1=many disfluencies, 5=very fluent)
- technical_depth: Depth of technical content (1=surface level, 5=expert depth)
- communication_clarity: How clearly the point is communicated (1=unclear, 5=crystal clear)

## Interview Stage:
- introduction: greetings, introductions, warm-up
- technical: core technical questions and problem-solving
- behavioral: soft-skill or experience questions
- closing: wrap-up, questions for interviewer, farewell
"""

def build_batch_prompt(texts: list[str]) -> str:
    numbered = "\n".join(f'{i+1}. "{t}"' for i, t in enumerate(texts))
    return f"""{SYSTEM_PROMPT}

## Instructions

For each utterance below, return a JSON array where each element has:
{{
  "role": "INTERVIEWER" or "INTERVIEWEE",
  "behavior": one of the behavior codes listed above,
  "eval": one of "CORRECT", "INCORRECT", "PARTIAL", "PROBE", "HINT", "SKIP", "NONE",
  "speaker_changes": [list of 0-based word indices where speaker changes, or empty],
  "entities": [{{ "type": "TECHNOLOGY|CONCEPT|SYSTEM|METRIC|COMPANY|ROLE|PROJECT|ACRONYM", "text": "extracted text" }}],
  "confidence_level": 1-5,
  "fluency_score": 1-5,
  "technical_depth": 1-5,
  "communication_clarity": 1-5,
  "interview_stage": "introduction|technical|behavioral|closing",
  "behavioral_keywords": ["hedging_phrase1", "confidence_marker1", ...]
}}

Important:
- "eval" should be "NONE" for all interviewee utterances and non-evaluative interviewer speech
- "speaker_changes" should be [] unless you detect multiple speakers in one utterance
- Return ONLY the JSON array, no markdown fences, no explanation.

## Utterances to annotate:

{numbered}"""


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------

def call_gemini_batch(texts: list[str], model, retries: int = 5) -> list[dict]:
    prompt = build_batch_prompt(texts)
    default = {
        "role": "INTERVIEWEE", "behavior": "EXPLAIN", "eval": "NONE",
        "speaker_changes": [], "entities": [],
        "confidence_level": 3, "fluency_score": 3, "technical_depth": 3,
        "communication_clarity": 3, "interview_stage": "technical",
        "behavioral_keywords": []
    }

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            raw = response.text.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            results = json.loads(raw)

            if not isinstance(results, list) or len(results) != len(texts):
                logging.warning(
                    f"Gemini returned {len(results) if isinstance(results, list) else type(results).__name__}, "
                    f"expected {len(texts)}, using defaults"
                )
                return [dict(default) for _ in texts]

            validated = []
            for r in results:
                v = dict(default)
                if isinstance(r, dict):
                    role = str(r.get("role", "")).upper()
                    v["role"] = role if role in VALID_ROLES else "INTERVIEWEE"

                    behavior = str(r.get("behavior", "")).upper()
                    behavior = behavior.replace("BEHAVIOR_", "").replace("INTENT_", "")
                    v["behavior"] = behavior if behavior in ALL_BEHAVIORS else "EXPLAIN"

                    eval_code = str(r.get("eval", "NONE")).upper()
                    eval_code = eval_code.replace("EVAL_", "")
                    v["eval"] = eval_code if eval_code in EVAL_CODES else "NONE"

                    speaker_changes = r.get("speaker_changes", [])
                    if isinstance(speaker_changes, list):
                        v["speaker_changes"] = [
                            int(sc) for sc in speaker_changes
                            if isinstance(sc, (int, float)) and sc >= 0
                        ]
                    else:
                        v["speaker_changes"] = []

                    entities = r.get("entities", [])
                    v["entities"] = [
                        e for e in entities
                        if isinstance(e, dict) and e.get("type", "").upper() in ENTITY_TYPES
                    ]

                    for field in ["confidence_level", "fluency_score",
                                  "technical_depth", "communication_clarity"]:
                        val = r.get(field, 3)
                        v[field] = max(1, min(5, int(val))) if isinstance(val, (int, float)) else 3

                    stage = str(r.get("interview_stage", "technical")).lower()
                    v["interview_stage"] = stage if stage in (
                        "introduction", "technical", "behavioral", "closing"
                    ) else "technical"

                    v["behavioral_keywords"] = r.get("behavioral_keywords", [])
                    if not isinstance(v["behavioral_keywords"], list):
                        v["behavioral_keywords"] = []

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


# ---------------------------------------------------------------------------
# Manifest Processing
# ---------------------------------------------------------------------------

def rebuild_text_with_tags(clean_text: str, existing_tags: dict,
                           annotation: dict) -> str:
    """Rebuild the text line with BEHAVIOR/EVAL/SPEAKER_CHANGE tags."""
    words = clean_text.split()

    # Insert SPEAKER_CHANGE markers inline at detected positions
    speaker_changes = sorted(annotation.get("speaker_changes", []), reverse=True)
    for sc_idx in speaker_changes:
        if 0 < sc_idx < len(words):
            words.insert(sc_idx, "SPEAKER_CHANGE")

    parts = [" ".join(words)]

    # Entity inline tags
    for ent in annotation.get("entities", []):
        ent_type = ent.get("type", "").upper()
        if ent_type and ent_type in ENTITY_TYPES:
            parts.append(f"ENTITY_{ent_type}")

    # Classification tags at end
    age_tag = existing_tags.get("AGE", "AGE_30_45")
    gender_tag = existing_tags.get("GENDER", "GENDER_MALE")
    emotion_tag = existing_tags.get("EMOTION", "EMOTION_NEUTRAL")
    role_tag = f"ROLE_{annotation['role']}"
    behavior_tag = f"BEHAVIOR_{annotation['behavior']}"
    eval_tag = f"EVAL_{annotation['eval']}"

    parts.extend([age_tag, gender_tag, emotion_tag, role_tag, behavior_tag, eval_tag])

    return " ".join(parts)


def process_manifest(input_path: str, output_path: str, model,
                     batch_size: int, dry_run: bool, max_workers: int) -> dict:
    stats = Counter()

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    stats["total"] = len(lines)

    # Load checkpoint
    progress_file = output_path + ".progress.json"
    completed = {}
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            completed = json.load(f)
        logging.info(f"Resuming: {len(completed)} already annotated")
        stats["resumed"] = len(completed)

    # Parse all samples
    samples = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            sample = json.loads(line)
            samples.append((i, sample))
        except json.JSONDecodeError:
            stats["json_errors"] += 1

    # Find unannotated samples
    to_annotate = [(i, s) for i, s in samples if str(i) not in completed]
    stats["to_annotate"] = len(to_annotate)

    if dry_run:
        logging.info(f"DRY RUN: {len(to_annotate)} samples need annotation out of {len(samples)}")
        return dict(stats)

    # Build batches
    all_batches = []
    for batch_start in range(0, len(to_annotate), batch_size):
        batch = to_annotate[batch_start:batch_start + batch_size]
        batch_texts = []
        for idx, sample in batch:
            text = strip_all_tags(sample.get("text", ""))
            if len(text) > 500:
                text = text[:500]
            batch_texts.append(text if text else "...")
        all_batches.append((batch, batch_texts))

    total_batches = len(all_batches)
    annotated_count = len(completed)
    lock = threading.Lock()

    logging.info(f"Annotating {len(to_annotate)} samples in {total_batches} batches "
                 f"({max_workers} workers, batch_size={batch_size})")

    def process_batch(batch_item):
        batch, batch_texts = batch_item
        return batch, call_gemini_batch(batch_texts, model)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_batch, b): i for i, b in enumerate(all_batches)}

        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                batch, annotations = future.result()
            except Exception as e:
                logging.error(f"Batch {batch_idx} failed: {e}")
                continue

            with lock:
                for (idx, sample), annotation in zip(batch, annotations):
                    completed[str(idx)] = annotation
                    annotated_count += 1
                    stats[f"role_{annotation['role'].lower()}"] += 1
                    stats[f"behavior_{annotation['behavior'].lower()}"] += 1
                    stats[f"eval_{annotation['eval'].lower()}"] += 1
                    stats[f"stage_{annotation['interview_stage']}"] += 1
                    if annotation.get("speaker_changes"):
                        stats["has_speaker_change"] += 1

                # Checkpoint every 2000 samples
                if annotated_count % 2000 < batch_size:
                    with open(progress_file, 'w') as f:
                        json.dump(completed, f)
                    total_done = annotated_count
                    pct = total_done * 100 // stats["total"]
                    logging.info(f"Progress: {total_done}/{stats['total']} ({pct}%)")

    # Final checkpoint
    with open(progress_file, 'w') as f:
        json.dump(completed, f)

    stats["annotated"] = annotated_count

    # Apply all annotations and write output
    sample_map = {i: s for i, s in samples}
    for str_idx, annotation in completed.items():
        idx = int(str_idx)
        if idx in sample_map:
            sample = sample_map[idx]
            existing_tags = extract_existing_tags(sample.get("text", ""))
            clean_text = strip_all_tags(sample.get("text", ""))

            sample["text"] = rebuild_text_with_tags(
                clean_text, existing_tags, annotation
            )
            sample["lang_family"] = sample.get("lang_family", "ENGLISH")

            sample["speaker_role"] = annotation["role"].lower()
            sample["misc_behavior"] = annotation["behavior"]
            sample["misc_eval"] = annotation["eval"]
            sample["speaker_changes"] = annotation.get("speaker_changes", [])
            sample["confidence_level"] = annotation.get("confidence_level", 3)
            sample["fluency_score"] = annotation.get("fluency_score", 3)
            sample["technical_depth"] = annotation.get("technical_depth", 3)
            sample["communication_clarity"] = annotation.get("communication_clarity", 3)
            sample["interview_stage"] = annotation.get("interview_stage", "technical")
            sample["behavioral_keywords"] = annotation.get("behavioral_keywords", [])
            sample["entities"] = [
                {"type": e["type"], "text": e["text"]}
                for e in annotation.get("entities", [])
            ]

    with open(output_path, 'w', encoding='utf-8') as f:
        for i in sorted(sample_map.keys()):
            f.write(json.dumps(sample_map[i], ensure_ascii=False) + "\n")

    if os.path.exists(progress_file):
        os.remove(progress_file)

    stats["written"] = len(sample_map)

    logging.info("\n" + "=" * 60)
    logging.info("ANNOTATION SUMMARY")
    logging.info("=" * 60)

    for prefix in ["role_", "behavior_", "eval_", "stage_"]:
        group = {k: v for k, v in stats.items() if k.startswith(prefix)}
        if group:
            logging.info(f"\n{prefix.rstrip('_').upper()} distribution:")
            for k, v in sorted(group.items(), key=lambda x: -x[1]):
                logging.info(f"  {k}: {v}")

    sc_count = stats.get("has_speaker_change", 0)
    logging.info(f"\nSpeaker changes detected in: {sc_count} utterances")

    return dict(stats)


def main():
    parser = argparse.ArgumentParser(
        description="MISC 2.0 behavioral annotation for tech interview data"
    )
    parser.add_argument("--input", required=True, help="Input manifest (JSONL)")
    parser.add_argument("--output", required=True, help="Output manifest (JSONL)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Utterances per Gemini call (default 20)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel Gemini API workers (default 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report stats only, no annotation")
    parser.add_argument("--gemini-key", help="Gemini API key (or GEMINI_API_KEY env)")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash",
                        help="Gemini model name (default: gemini-2.5-flash)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)

    model = None
    if not args.dry_run:
        api_key = args.gemini_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logging.error("Set GEMINI_API_KEY env or pass --gemini-key")
            sys.exit(1)
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(args.gemini_model)
        logging.info(f"Gemini model: {args.gemini_model}")

    stats = process_manifest(
        args.input, args.output, model,
        args.batch_size, args.dry_run, args.workers
    )

    logging.info(f"\nDone. Total: {stats.get('total', 0)}, "
                 f"Annotated: {stats.get('annotated', 0)}, "
                 f"Written: {stats.get('written', 0)}")


if __name__ == "__main__":
    main()
