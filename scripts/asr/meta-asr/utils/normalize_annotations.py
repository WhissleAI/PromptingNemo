#!/usr/bin/env python3
"""Normalize META-ASR annotations across all multilingual datasets.

Fixes:
  1. GER_ → GENDER_ prefix
  2. Truncated/inconsistent emotion tags → canonical names
  3. Inconsistent intent tags → canonical names
  4. Missing lang_family field
  5. lang="MULTI" → actual language detection
  6. Missing intent tags on samples

Usage:
  python normalize_annotations.py --data-root /mnt/nfs/data/multilingual_v1/raw --dry-run
  python normalize_annotations.py --data-root /mnt/nfs/data/multilingual_v1/raw
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


GENDER_MAP = {
    "GER_MALE": "GENDER_MALE",
    "GER_FEMALE": "GENDER_FEMALE",
    "GER_OTHER": "GENDER_OTHER",
    "GER_M": "GENDER_MALE",
    "GER_F": "GENDER_FEMALE",
    "GERDER_MALE": "GENDER_MALE",
    "GERDER_FEMALE": "GENDER_FEMALE",
    "GERDER_OTHER": "GENDER_OTHER",
    "GERDER_M": "GENDER_MALE",
    "GERDER_F": "GENDER_FEMALE",
}

EMOTION_MAP = {
    "EMOTION_NEU": "EMOTION_NEUTRAL",
    "EMOTION_ANG": "EMOTION_ANGRY",
    "EMOTION_HAP": "EMOTION_HAPPY",
    "EMOTION_JOY": "EMOTION_HAPPY",
    "EMOTION_SADNESS": "EMOTION_SAD",
    "EMOTION_ANGER": "EMOTION_ANGRY",
}

INTENT_MAP = {
    "INTENT_INFORMATIONAL": "INTENT_INFORM",
    "INTENT_EXCLAMATION": "INTENT_EXCLAIM",
    "INTENT_INSTRUCT": "INTENT_COMMAND",
}

CANONICAL_INTENT_SET = {
    "INFORM", "QUESTION", "COMMAND", "REQUEST", "EXCLAIM",
    "OPINION", "EXPLAIN", "DESCRIBE", "STATEMENT",
}

INTENT_COLLAPSE = {
    "SUGGEST": "REQUEST", "SUGGESTION": "REQUEST", "RECOMMEND": "REQUEST",
    "RECOMMENDATION": "REQUEST", "PROPOSE": "REQUEST", "PROPOSAL": "REQUEST",
    "OFFER": "REQUEST", "INVITE": "REQUEST", "PERSUADE": "REQUEST",
    "ENCOURAGE": "REQUEST", "MOTIVATE": "REQUEST", "INSPIRE": "REQUEST",
    "PLEA": "REQUEST", "APPEAL": "REQUEST", "PRAY": "REQUEST",
    "ANNOUNCE": "INFORM", "REPORT": "INFORM", "STATE": "INFORM",
    "AGREE": "INFORM", "AFFIRM": "INFORM", "ACKNOWLEDGE": "INFORM",
    "ACKNOWLEDGEMENT": "INFORM", "CONFIRM": "INFORM", "PROMISE": "INFORM",
    "ASSURE": "INFORM", "PLAN": "INFORM", "UPDATE": "INFORM",
    "WARN": "INFORM", "WARNING": "INFORM", "ALERT": "INFORM",
    "NOTIFICATION": "INFORM", "REMINDER": "INFORM", "REMIND": "INFORM",
    "ANNOUNCEMENT": "INFORM", "INFORMATION": "INFORM", "INFORMATIVE": "INFORM",
    "AFFIRMATION": "INFORM", "AFFIRMATIVE": "INFORM",
    "NARRATE": "DESCRIBE", "NARRATIVE": "DESCRIBE", "RECOUNT": "DESCRIBE",
    "REMEMBER": "DESCRIBE", "RECALL": "DESCRIBE", "REMINISCE": "DESCRIBE",
    "ANECDOTE": "DESCRIBE", "TELL_STORY": "DESCRIBE", "DESCRIPTION": "DESCRIBE",
    "DESCRIPTIVE": "DESCRIBE", "DESCRIBE_ACTION": "DESCRIBE",
    "RECOLLECT": "DESCRIBE", "EXPERIENCE": "DESCRIBE", "NARRATION": "DESCRIBE",
    "ANECDOTAL": "DESCRIBE", "PERSONAL_EXPERIENCE": "DESCRIBE",
    "ARGUE": "OPINION", "DEBATE": "OPINION", "DISAGREE": "OPINION",
    "PREDICT": "OPINION", "SPECULATE": "OPINION", "SPECULATION": "OPINION",
    "HOPE": "OPINION", "HYPOTHETICAL": "OPINION", "HYPOTHESIS": "OPINION",
    "HYPOTHESIZE": "OPINION", "PREFER": "OPINION", "PREFERENCE": "OPINION",
    "EXPRESS_OPINION": "OPINION", "REFLECT": "OPINION", "REFLECTIVE": "OPINION",
    "REFLECTION": "OPINION", "DENY": "OPINION", "REFUSE": "OPINION",
    "REJECT": "OPINION", "REFUSAL": "OPINION", "NEGATE": "OPINION",
    "NEGATION": "OPINION", "OPPOSE": "OPINION", "OBJECT": "OPINION",
    "CONTRADICT": "OPINION", "DISPUTE": "OPINION",
    "COMPLAIN": "OPINION", "COMPLAINT": "OPINION", "CRITICIZE": "OPINION",
    "CRITICISM": "OPINION", "CRITIQUE": "OPINION", "ACCUSE": "OPINION",
    "ACCUSATION": "OPINION", "PROTEST": "OPINION",
    "PRAISE": "OPINION", "COMPLIMENT": "OPINION", "APPRECIATE": "OPINION",
    "CONGRATULATE": "OPINION", "SUPPORT": "OPINION",
    "COMPARE": "OPINION", "COMPARISON": "OPINION", "CONTRAST": "OPINION",
    "EVALUATE": "OPINION", "ASSESS": "OPINION", "ANALYZE": "OPINION",
    "REVIEW": "OPINION", "FEEDBACK": "OPINION",
    "GREETING": "EXCLAIM", "GREET": "EXCLAIM", "FAREWELL": "EXCLAIM",
    "WELCOME": "EXCLAIM", "CLOSE": "EXCLAIM",
    "THANK": "EXCLAIM", "THANK_YOU": "EXCLAIM", "THANKS": "EXCLAIM",
    "THANKING": "EXCLAIM", "THANKYOU": "EXCLAIM",
    "EXPRESS_GRATITUDE": "EXCLAIM", "EXPRESSION_OF_GRATITUDE": "EXCLAIM",
    "APOLOGIZE": "EXCLAIM", "APOLOGY": "EXCLAIM",
    "WISH": "EXCLAIM", "EXPRESS_WISH": "EXCLAIM",
    "EXPRESS_EMOTION": "EXCLAIM", "EXPRESSION": "EXCLAIM", "EXPRESS": "EXCLAIM",
    "EXPRESS_FEELING": "EXCLAIM", "EXPRESS_AFFECTION": "EXCLAIM",
    "EXPRESS_SYMPATHY": "EXCLAIM", "EXPRESS_HOPE": "EXCLAIM",
    "EXPRESS_FEAR": "EXCLAIM", "EXPRESS_DESIRE": "EXCLAIM",
    "EXPRESS_CONCERN": "EXCLAIM", "EXPRESS_REGRET": "EXCLAIM",
    "EXPRESS_APPROVAL": "EXCLAIM", "EXPRESS_CONDOLENCES": "EXCLAIM",
    "CONDOLENCE": "EXCLAIM",
    "ASSERT": "STATEMENT", "ASSERTION": "STATEMENT", "DECLARE": "STATEMENT",
    "DECLARATIVE": "STATEMENT", "STATE_FACT": "STATEMENT",
    "CONCLUDE": "STATEMENT", "SUMMARY": "STATEMENT", "SUMMARIZE": "STATEMENT",
    "DEFINE": "STATEMENT", "DEFINITION": "STATEMENT",
    "OBSERVE": "STATEMENT", "OBSERVATION": "STATEMENT",
    "COMMENT": "STATEMENT", "LIST": "STATEMENT",
    "INSTRUCTION": "COMMAND", "DIRECTIVE": "COMMAND",
    "DEMAND": "COMMAND", "ORDER": "COMMAND", "PROHIBIT": "COMMAND",
    "PLAY_MUSIC": "COMMAND", "MUSIC_REQUEST": "COMMAND", "PLAY_SONG": "COMMAND",
    "PLAY_MEDIA": "COMMAND", "MEDIA_REQUEST": "COMMAND", "PLAY_REQUEST": "COMMAND",
    "TASK": "COMMAND", "ACTION": "COMMAND", "CALL_TO_ACTION": "COMMAND",
    "WEATHER_QUERY": "QUESTION", "INQUIRY": "QUESTION", "INQUIRE": "QUESTION",
    "QUERY": "QUESTION", "REQUEST_INFORMATION": "QUESTION",
    "VERIFY": "QUESTION", "TEST": "QUESTION",
    "CLARIFY": "QUESTION", "UNDERSTAND": "QUESTION",
    "EXPLANATION": "EXPLAIN", "EXPLANATORY": "EXPLAIN",
    "REASON": "EXPLAIN", "JUSTIFY": "EXPLAIN",
    "UNKNOWN": "STATEMENT", "OTHER": "STATEMENT", "INCOMPLETE": "STATEMENT",
    "RESPONSE": "STATEMENT", "ANSWER": "STATEMENT", "RESPOND": "STATEMENT",
    "DIALOG": "STATEMENT", "TELL": "STATEMENT",
    "CONFESS": "STATEMENT", "INTRODUCE": "STATEMENT", "INTRODUCTION": "STATEMENT",
    "DECIDE": "STATEMENT", "CONSIDER": "STATEMENT", "THINK": "STATEMENT",
    "ACCEPT": "STATEMENT", "APPROVE": "STATEMENT",
    "CORRECT": "INFORM", "ADVISE": "INFORM", "ADVICE": "INFORM",
    "NEGOTIATE": "REQUEST", "CHALLENGE": "QUESTION",
    "REASSURE": "INFORM", "INSULT": "OPINION", "THREAT": "COMMAND",
    "THREATEN": "COMMAND",
    "CONDITIONAL": "STATEMENT", "SPECULATIVE": "OPINION",
    "PREDICTION": "OPINION", "EXPECTATION": "OPINION",
    "DESIRE": "REQUEST", "NEED": "REQUEST",
    "TRAVEL": "COMMAND", "PROBLEM": "INFORM", "PROBLEM_SOLVING": "INFORM",
    "INFER": "OPINION", "QUOTE": "DESCRIBE",
    "MEMORY": "DESCRIBE", "UNDERSTANDING": "INFORM",
    "INTENTION": "INFORM", "ADDRESS": "INFORM", "CONCERN": "OPINION",
    "DENIAL": "OPINION", "AGREEMENT": "INFORM",
    "PERSONAL": "DESCRIBE", "DESCRIPTIVE_ACTION": "DESCRIBE",
}

EMOTION_GARBAGE = {
    "INFORM", "QUESTION", "WISH", "THANK", "OPINION",
    "EXPLAIN", "AUDIO", "DISAGREE", "COMPLAINT",
}

AGE_RANGE_MAP = {
    "14-17": "0_18", "14_17": "0_18",
    "18-24": "18_30", "18_24": "18_30",
    "25-30": "18_30", "25_30": "18_30",
    "31-35": "30_45", "31_35": "30_45",
    "36-40": "30_45", "36_40": "30_45",
    "41-45": "45_60", "41_45": "45_60",
    "46-50": "45_60", "46_50": "45_60",
    "51-55": "45_60", "51_55": "45_60",
    "56-60": "45_60", "56_60": "45_60",
    "61-65": "60PLUS", "61_65": "60PLUS",
    "66-70": "60PLUS", "66_70": "60PLUS",
    "14_25": "18_30",
}

LANG_CODE_TO_FAMILY = {
    "EN": "ENGLISH",
    "FR": "EUROPEAN",
    "DE": "EUROPEAN",
    "ES": "EUROPEAN",
    "IT": "EUROPEAN",
    "PT": "EUROPEAN",
    "NL": "EUROPEAN",
    "DA": "EUROPEAN",
    "FI": "EUROPEAN",
    "SV": "EUROPEAN",
    "ET": "EUROPEAN",
    "GL": "EUROPEAN",
    "CA": "EUROPEAN",
    "EU": "EUROPEAN",
    "RO": "EUROPEAN",
    "RU": "SLAVIC",
    "PL": "SLAVIC",
    "CS": "SLAVIC",
    "SK": "SLAVIC",
    "BG": "SLAVIC",
    "UK": "SLAVIC",
    "BE": "SLAVIC",
    "MK": "SLAVIC",
    "KA": "SLAVIC",
    "SR": "SLAVIC",
    "HR": "SLAVIC",
    "SL": "SLAVIC",
    "HI": "INDO_ARYAN",
    "BN": "INDO_ARYAN",
    "MR": "INDO_ARYAN",
    "GU": "INDO_ARYAN",
    "PA": "INDO_ARYAN",
    "UR": "INDO_ARYAN",
    "TA": "INDO_ARYAN",
    "TE": "INDO_ARYAN",
    "KN": "INDO_ARYAN",
    "ML": "INDO_ARYAN",
    "OR": "INDO_ARYAN",
    "AS": "INDO_ARYAN",
    "ZH": "EAST_ASIAN",
    "JA": "EAST_ASIAN",
    "KO": "EAST_ASIAN",
    "MANDARIN": "EAST_ASIAN",
    "BH": "INDO_ARYAN",
    "CH": "INDO_ARYAN",
    "KN": "INDO_ARYAN",
    "MG": "INDO_ARYAN",
    "MT": "INDO_ARYAN",
    "EN_IN": "ENGLISH",
}

FAMILY_NAMES = {"ENGLISH", "EUROPEAN", "SLAVIC", "INDO_ARYAN", "EAST_ASIAN", "MULTI"}

TAG_PATTERN = re.compile(
    r'\b('
    r'GER_(?:MALE|FEMALE|OTHER|M|F)'
    r'|GERDER_(?:MALE|FEMALE|OTHER|M|F)'
    r'|EMOTION_(?:NEU|ANG|HAP|JOY|SADNESS|ANGER)'
    r'|INTENT_(?:INFORMATIONAL|EXCLAMATION|INSTRUCT)'
    r')\b'
)

AGE_RANGE_RE = re.compile(r'\bAGE_(\d+[-]\d+)\b')
NON_CANONICAL_INTENT_RE = re.compile(r'\bINTENT_(\S+)')
NON_CANONICAL_EMOTION_RE = re.compile(r'\bEMOTION_(\S+)')

CANONICAL_EMOTIONS = {
    "EMOTION_NEUTRAL", "EMOTION_HAPPY", "EMOTION_SAD", "EMOTION_ANGRY",
    "EMOTION_FEAR", "EMOTION_DISGUST", "EMOTION_SURPRISE",
}
CANONICAL_INTENTS = {
    "INTENT_INFORM", "INTENT_QUESTION", "INTENT_COMMAND", "INTENT_REQUEST",
    "INTENT_EXCLAIM", "INTENT_OPINION", "INTENT_EXPLAIN", "INTENT_DESCRIBE",
    "INTENT_STATEMENT", "INTENT_THANK", "INTENT_ASSERT",
}

ALL_TAG_MAPS = {**GENDER_MAP, **EMOTION_MAP, **INTENT_MAP}

ENTITY_GLUED_RE = re.compile(r'\bENTITY_(CITY|ORGANIZATION|PERSON|LOCATION|DATE|TIME|NUMBER)END\b')
ENTITY_GENDER_RE = re.compile(r'\bENTITY_GENDER\S*\b')


def fix_entity_tags(text: str) -> tuple[str, bool]:
    """Fix malformed entity tags: ENTITY_CITYEND → ENTITY_CITY END, remove ENTITY_GENDER*."""
    changed = False
    if ENTITY_GLUED_RE.search(text):
        text = ENTITY_GLUED_RE.sub(r'ENTITY_\1 END', text)
        changed = True
    if ENTITY_GENDER_RE.search(text):
        text = ENTITY_GENDER_RE.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()
        changed = True
    return text, changed


def detect_lang_from_text(text: str) -> str:
    """Rough language detection from script used in text."""
    clean = re.sub(r'\b(?:AGE_|GENDER_|EMOTION_|INTENT_|ENTITY_|END)\S*', '', text).strip()
    if not clean:
        return "MULTI"
    cyrillic = len(re.findall(r'[Ѐ-ӿ]', clean))
    devanagari = len(re.findall(r'[ऀ-ॿ]', clean))
    latin = len(re.findall(r'[a-zA-Z]', clean))
    arabic = len(re.findall(r'[؀-ۿ]', clean))
    bengali = len(re.findall(r'[ঀ-৿]', clean))
    total = cyrillic + devanagari + latin + arabic + bengali + 1
    if cyrillic / total > 0.3:
        return "SLAVIC"
    if devanagari / total > 0.3:
        return "INDO_ARYAN"
    if latin / total > 0.3:
        return "ENGLISH"
    if arabic / total > 0.3:
        return "INDO_ARYAN"
    if bengali / total > 0.3:
        return "INDO_ARYAN"
    return "MULTI"


def normalize_text(text: str) -> str:
    """Replace all non-canonical tags with canonical versions."""
    def replace_tag(m):
        return ALL_TAG_MAPS.get(m.group(0), m.group(0))
    text = TAG_PATTERN.sub(replace_tag, text)

    def fix_age_range(m):
        raw = m.group(1)
        mapped = AGE_RANGE_MAP.get(raw)
        return f"AGE_{mapped}" if mapped else m.group(0)
    text = AGE_RANGE_RE.sub(fix_age_range, text)

    def collapse_intent(m):
        intent = m.group(1)
        if intent in CANONICAL_INTENT_SET:
            return m.group(0)
        mapped = INTENT_COLLAPSE.get(intent)
        if mapped:
            return f"INTENT_{mapped}"
        return f"INTENT_STATEMENT"
    text = NON_CANONICAL_INTENT_RE.sub(collapse_intent, text)

    def fix_garbage_emotion(m):
        emotion = m.group(1)
        if emotion in EMOTION_GARBAGE:
            return "EMOTION_NEUTRAL"
        return m.group(0)
    text = NON_CANONICAL_EMOTION_RE.sub(fix_garbage_emotion, text)

    return text


def normalize_sample(sample: dict, dataset_name: str) -> tuple[dict, dict]:
    """Normalize a single manifest line. Returns (normalized_sample, changes_dict)."""
    changes = {}

    # 1. Normalize text tags
    old_text = sample.get("text", "")
    new_text = normalize_text(old_text)
    if new_text != old_text:
        changes["text_tags"] = True
        sample["text"] = new_text

    # 1b. Fix malformed entity tags
    fixed_text, entity_changed = fix_entity_tags(sample["text"])
    if entity_changed:
        changes["entity_fix"] = True
        sample["text"] = fixed_text

    # 2. Fix lang field
    lang = sample.get("lang", "")
    if lang == "MANDARIN":
        sample["lang"] = "ZH"
        lang = "ZH"
        changes["lang_mandarin_fixed"] = True
    if lang == "MULTI":
        detected = detect_lang_from_text(new_text)
        if detected != "MULTI":
            changes["lang_multi_fixed"] = f"MULTI→{detected}"
        sample["lang_family"] = detected
    elif lang in FAMILY_NAMES:
        # lang is already a family name (e.g., EN People uses "ENGLISH")
        sample["lang_family"] = lang
    elif lang.upper() in LANG_CODE_TO_FAMILY:
        family = LANG_CODE_TO_FAMILY[lang.upper()]
        if sample.get("lang_family") != family:
            changes["lang_family_added"] = family
        sample["lang_family"] = family
    else:
        if "lang_family" not in sample:
            detected = detect_lang_from_text(new_text)
            sample["lang_family"] = detected
            changes["lang_family_inferred"] = detected

    return sample, changes


def process_manifest(filepath: str, dry_run: bool = False) -> dict:
    """Process a single manifest file. Returns stats."""
    stats = defaultdict(int)
    stats["file"] = filepath
    lines = []

    with open(filepath, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    stats["total"] = len(raw_lines)
    dataset_name = Path(filepath).stem

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            stats["json_errors"] += 1
            lines.append(line)
            continue

        sample, changes = normalize_sample(sample, dataset_name)

        if changes:
            stats["modified"] += 1
            for k in changes:
                stats[f"change_{k}"] += 1

        lines.append(json.dumps(sample, ensure_ascii=False))

    if not dry_run and stats["modified"] > 0:
        backup = filepath + ".bak"
        if not os.path.exists(backup):
            os.rename(filepath, backup)
        else:
            os.remove(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        stats["written"] = True
    else:
        stats["written"] = False

    return dict(stats)


def find_manifests(data_root: str) -> list[str]:
    """Find all .json manifest files (excluding dataset_info.json)."""
    manifests = []
    for root, dirs, files in os.walk(data_root):
        dirs[:] = [d for d in dirs if d != 'audio']
        for f in sorted(files):
            if f.endswith('.json') and f != 'dataset_info.json':
                manifests.append(os.path.join(root, f))
    return manifests


def main():
    parser = argparse.ArgumentParser(description="Normalize META-ASR annotations")
    parser.add_argument("--data-root", required=True, help="Root of raw data dir")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    parser.add_argument("--file", type=str, help="Process single file instead of scanning")
    args = parser.parse_args()

    if args.file:
        manifests = [args.file]
    else:
        manifests = find_manifests(args.data_root)

    print(f"Found {len(manifests)} manifest files")
    if args.dry_run:
        print("DRY RUN — no files will be modified\n")

    total_modified = 0
    total_samples = 0
    all_changes = defaultdict(int)

    for mf in manifests:
        stats = process_manifest(mf, dry_run=args.dry_run)
        total_samples += stats.get("total", 0)
        modified = stats.get("modified", 0)
        total_modified += modified

        change_keys = [k for k in stats if k.startswith("change_")]
        if modified > 0 or stats.get("json_errors", 0) > 0:
            rel = os.path.relpath(mf, args.data_root)
            print(f"  {rel}: {stats['total']} samples, {modified} modified", end="")
            if stats.get("json_errors"):
                print(f", {stats['json_errors']} JSON errors", end="")
            if change_keys:
                details = ", ".join(f"{k.replace('change_', '')}={stats[k]}" for k in change_keys)
                print(f" [{details}]", end="")
            print(f" {'(written)' if stats.get('written') else ''}")

            for k in change_keys:
                all_changes[k.replace("change_", "")] += stats[k]
        else:
            rel = os.path.relpath(mf, args.data_root)
            print(f"  {rel}: {stats['total']} samples — OK")

    print(f"\n{'='*60}")
    print(f"Total: {total_samples} samples across {len(manifests)} files")
    print(f"Modified: {total_modified} samples")
    if all_changes:
        print("Changes breakdown:")
        for k, v in sorted(all_changes.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
    if args.dry_run:
        print("\nRe-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
