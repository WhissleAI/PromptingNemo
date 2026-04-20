"""Manifest text normalization, tag extraction, and filtering utilities.

Extracted from scripts/asr/meta-asr/clean_manifest_before_training.py.
"""

import re
import unicodedata
from typing import List, Tuple


# --- Canonical tag names (must match pretrained model vocabulary) ---
CANONICAL_AGES = {
    "AGE_0_18", "AGE_18_30", "AGE_30_45", "AGE_45_60", "AGE_60PLUS",
}
CANONICAL_GENDERS = {"GENDER_FEMALE", "GENDER_MALE", "GENDER_OTHER", "GENDER_UNKNOWN"}
CANONICAL_EMOTIONS = {
    "EMOTION_ANGRY", "EMOTION_DISGUST", "EMOTION_FEAR",
    "EMOTION_HAPPY", "EMOTION_NEUTRAL", "EMOTION_SAD", "EMOTION_SURPRISE",
}

ALLOWED_EMOTIONS = CANONICAL_EMOTIONS
FORBIDDEN_AGES = {"AGE_NUMBER", "AGE_STATE"}

# Valid sentence-level tag prefixes (tags that appear at end of utterance)
SENTENCE_TAG_PREFIXES = (
    "AGE_", "GENDER_", "EMOTION_", "INTENT_", "DIALECT_",
)

# Unicode/spacing-safe tag grabbers:
# - Accept anything non-whitespace after the prefix so hidden/odd chars don't break detection
EMOTION_TAG = re.compile(r"EMOTION_[^\s]+", flags=re.UNICODE)
AGE_TAG     = re.compile(r"AGE_[^\s]+",     flags=re.UNICODE)

# Matches any ALL_CAPS token that looks like a meta-tag
META_TAG_PATTERN = re.compile(r'\b[A-Z][A-Z0-9_]*_[A-Z0-9_+.]+\b|^END$')

# Concatenated tags (annotation errors like "ENTITY_STATEAGE_30_45")
CONCAT_TAG_PATTERN = re.compile(
    r'(ENTITY_|OTHER_|INTENT_|EMOTION_|AGE_|GENDER_|DIALECT_)'
    r'.*?(ENTITY_|OTHER_|INTENT_|EMOTION_|AGE_|GENDER_|DIALECT_)'
)


# --- Normalization helpers ---
def normalize_text(t: str) -> str:
    """Normalize common tag typos/abbreviations and whitespace."""
    if not isinstance(t, str):
        return ""
    # Explicit common typos first
    t = t.replace("EMOTION_HAPPYPY", "EMOTION_HAPPY")
    # Map common abbrevs to canonical (prefix-safe)
    t = re.sub(r"EMOTION_HAP(?!PY)\b", "EMOTION_HAPPY", t)        # HAP -> HAPPY
    t = re.sub(r"EMOTION_NEU(?!TRAL)\b", "EMOTION_NEUTRAL", t)    # NEU -> NEUTRAL
    t = re.sub(r"EMOTION_ANG(?!RY)\b", "EMOTION_ANGRY", t)        # ANG -> ANGRY
    t = re.sub(r"EMOTION_INFORM\b", "EMOTION_NEUTRAL", t)         # non-standard
    t = re.sub(r"EMOTION_JOY\b", "EMOTION_HAPPY", t)              # non-standard
    t = re.sub(r"EMOTION_ANGRYER\b", "EMOTION_ANGRY", t)          # typo

    # Gender typos (must come before GER_ replacement)
    t = t.replace("GGENDER_", "GENDER_")
    t = t.replace("GENSION_", "GENDER_")
    t = t.replace("GER_", "GENDER_")

    # Normalize AGE_60+ to AGE_60PLUS (pretrained model vocab form)
    t = t.replace("AGE_60+", "AGE_60PLUS")

    # Strip trailing ", from tags (JSON escaping artifacts)
    t = re.sub(r'(?<=\w)",', '', t)

    # Fix concatenated tags (e.g. "ENTITY_STATEAGE_30_45" → "ENTITY_STATE AGE_30_45")
    t = re.sub(r'(ENTITY_[A-Z_]+)(AGE_)', r'\1 \2', t)
    t = re.sub(r'(OTHER_[A-Z_]+)(OTHER_)', r'\1 \2', t)
    t = re.sub(r'(OTHER_[A-Z_]+)(INTENT_)', r'\1 \2', t)
    t = re.sub(r'(INTENT_[A-Z_]+)(AGE_)', r'\1 \2', t)

    # Collapse funky whitespace (incl. unicode spaces)
    t = " ".join(t.split())
    return t


def clean_rare_tags(text: str, known_tags: set, min_prefix_match: bool = True) -> str:
    """Remove tags from text that aren't in the known_tags set.

    Args:
        text: Normalized text with tags.
        known_tags: Set of valid tag strings (e.g. from pretrained vocabulary).
        min_prefix_match: If True, keep tags whose prefix (e.g. ENTITY_) matches
            a known prefix even if the exact tag is unknown. If False, only keep
            exact matches.

    Returns:
        Text with unknown tags removed.
    """
    words = text.split()
    kept = []
    for w in words:
        if META_TAG_PATTERN.match(w):
            if w in known_tags:
                kept.append(w)
            elif min_prefix_match and any(w.startswith(p) for p in SENTENCE_TAG_PREFIXES):
                kept.append(w)
            # else: drop unknown tag
        else:
            kept.append(w)
    return " ".join(kept)


def extract_tags(text: str) -> Tuple[List[str], List[str]]:
    """Return lists of emotion and age tags from normalized text."""
    norm = normalize_text(text)
    return EMOTION_TAG.findall(norm), AGE_TAG.findall(norm)


def should_keep_line(text: str, debug: bool = False) -> bool:
    """Decide whether to keep a line based on allowed/forbidden tags."""
    norm = normalize_text(text)
    emotions, ages = extract_tags(norm)

    if debug:
        print(f"[DEBUG] text: {norm}")
        print(f"[DEBUG] emotions: {emotions} | ages: {ages}")

    # Keep if at least one emotion is allowed
    if not emotions or not any(e in ALLOWED_EMOTIONS for e in emotions):
        if debug:
            print(f"[DROP] reason=no_allowed_emotion")
            print(f"[DROP] emotions_found: {emotions}")
            chunk, hexes, names = debug_surroundings(norm, "EMOTION_")
            print(f"[DROP] around 'EMOTION_': {repr(chunk)}")
            print(f"[DROP] codepoints: {hexes}")
            print(f"[DROP] unicode-names: {names}")
        return False

    # Drop only if a forbidden age label appears exactly
    if any(a in FORBIDDEN_AGES for a in ages):
        if debug:
            print(f"[DROP] reason=forbidden_age")
            print(f"[DROP] ages_found: {ages}")
        return False

    return True


def debug_surroundings(s: str, needle: str = "EMOTION_") -> Tuple[str, str, str]:
    """Return a small window around `needle` with codepoints and Unicode names."""
    i = s.find(needle)
    if i == -1:
        i = len(s) // 2
    start = max(0, i - 12)
    end = min(len(s), i + 30)
    chunk = s[start:end]
    hexes = " ".join(f"U+{ord(c):04X}" for c in chunk)
    names = " | ".join(f"{c}:{unicodedata.name(c, 'UNKNOWN')}" for c in chunk)
    return chunk, hexes, names
