"""Manifest text normalization, tag extraction, and filtering utilities.

Extracted from scripts/asr/meta-asr/clean_manifest_before_training.py.
"""

import re
import unicodedata
from typing import List, Tuple


# --- Canonical sets ---
ALLOWED_EMOTIONS = {
    "EMOTION_ANGRY",
    "EMOTION_DISGUST",
    "EMOTION_FEAR",
    "EMOTION_HAPPY",
    "EMOTION_NEUTRAL",
    "EMOTION_SAD",
    "EMOTION_SURPRISE",
}
FORBIDDEN_AGES = {"AGE_NUMBER", "AGE_STATE"}

# Unicode/spacing-safe tag grabbers:
# - Accept anything non-whitespace after the prefix so hidden/odd chars don't break detection
EMOTION_TAG = re.compile(r"EMOTION_[^\s]+", flags=re.UNICODE)
AGE_TAG     = re.compile(r"AGE_[^\s]+",     flags=re.UNICODE)


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

    # Gender typo
    t = t.replace("GER_", "GENDER_")

    # Normalize AGE_60PLUS to AGE_60+ (model vocab canonical form)
    t = t.replace("AGE_60PLUS", "AGE_60+")

    # Strip trailing ", from tags (JSON escaping artifacts)
    t = re.sub(r'(?<=\w)",', '', t)

    # Collapse funky whitespace (incl. unicode spaces)
    t = " ".join(t.split())
    return t


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
