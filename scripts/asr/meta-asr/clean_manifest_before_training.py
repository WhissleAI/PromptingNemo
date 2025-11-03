#!/usr/bin/env python3
import argparse
import json
import os
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


# List of all language codes
lang_codes = [
    "EN",  # English
    "DE", "NL", "AF", "SV", "DA", "NB", "NN", "IS", "FR", "ES", "PT", "IT", "RO", "CA", "LA",
    "MS", "ID", "TL", "SW", "XH", "ZU", "ST", "TN", "TS", "SN", "SO", "ET", "SQ", "MI", "LG",
    "GA", "CY", "EU",
    "RU", "PL", "CS", "SK", "SL", "SR", "HR", "BG", "BE", "UK", "MK",
    "HI", "MR", "BN", "GU", "PA", "UR",
    "TA", "TE", "KA",
    "ZH"
]

# Dictionary mapping each code to its language family
lang_family = {
    # English
    "EN": "ENGLISH",

    # European
    "DE": "EUROPEAN", "NL": "EUROPEAN", "AF": "EUROPEAN", "SV": "EUROPEAN", "DA": "EUROPEAN",
    "NB": "EUROPEAN", "NN": "EUROPEAN", "IS": "EUROPEAN", "FR": "EUROPEAN", "ES": "EUROPEAN",
    "PT": "EUROPEAN", "IT": "EUROPEAN", "RO": "EUROPEAN", "CA": "EUROPEAN", "LA": "EUROPEAN",
    "MS": "EUROPEAN", "ID": "EUROPEAN", "TL": "EUROPEAN", "SW": "EUROPEAN", "XH": "EUROPEAN",
    "ZU": "EUROPEAN", "ST": "EUROPEAN", "TN": "EUROPEAN", "TS": "EUROPEAN", "SN": "EUROPEAN",
    "SO": "EUROPEAN", "ET": "EUROPEAN", "SQ": "EUROPEAN", "MI": "EUROPEAN", "LG": "EUROPEAN",
    "GA": "EUROPEAN", "CY": "EUROPEAN", "EU": "EUROPEAN",

    # Slavic
    "RU": "SLAVIC", "PL": "SLAVIC", "CS": "SLAVIC", "SK": "SLAVIC", "SL": "SLAVIC",
    "SR": "SLAVIC", "HR": "SLAVIC", "BG": "SLAVIC", "BE": "SLAVIC", "UK": "SLAVIC", "MK": "SLAVIC",

    # Indo-Aryan
    "HI": "INDO_ARYAN", "MR": "INDO_ARYAN", "BN": "INDO_ARYAN",
    "GU": "INDO_ARYAN", "PA": "INDO_ARYAN", "UR": "INDO_ARYAN",

    # Dravidian
    "TA": "DRAVIDIAN", "TE": "DRAVIDIAN", "KA": "DRAVIDIAN",

    # Mandarin
    "ZH": "MANDARIN"
}

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

    # Collapse funky whitespace (incl. unicode spaces)
    t = " ".join(t.split())
    return t

def extract_tags(text: str) -> Tuple[List[str], List[str]]:
    """Return lists of emotion and age tags from normalized text."""
    norm = normalize_text(text)
    return EMOTION_TAG.findall(norm), AGE_TAG.findall(norm)

def append_lang_tag(text: str, lang: str) -> str:
    """Append LANG_XX once (idempotent) after normalization."""
    norm = normalize_text(text)
    if not lang:
        return norm
    tag = f"LANG_{str(lang).upper()}"
    if f" {tag} " not in f" {norm} ":
        norm = f"{norm} {tag}"
    return norm

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

def main():
    parser = argparse.ArgumentParser(
        description="Clean/filter a JSONL manifest and append LANG_XXX to text."
    )
    parser.add_argument("--input-manifest", required=True, help="Input JSONL")
    parser.add_argument("--output-manifest", required=True, help="Output JSONL (kept)")
    parser.add_argument("--removed-manifest", required=True, help="Output JSONL (removed)")
    parser.add_argument("--no-langid-manifest", required=True, help="Output JSONL (no langid)")
    parser.add_argument("--debug", action="store_true", help="Print debug info/reasons")
    args = parser.parse_args()

    removed_path = args.removed_manifest or os.devnull
    no_langid_path = args.no_langid_manifest or os.devnull

    with open(args.input_manifest, "r", encoding="utf-8") as infile, \
         open(args.output_manifest, "w", encoding="utf-8") as outfile, \
         open(removed_path, "w", encoding="utf-8") as removed, \
         open(no_langid_path, "w", encoding="utf-8") as no_langid:

        for idx, raw in enumerate(infile, start=1):
            line = raw.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                if args.debug:
                    print(f"[DROP] [{idx}] reason=malformed_json line={line[:200]}...")
                removed.write(line + "\n")
                continue

            text = data.get("text", "")

            if not should_keep_line(text, debug=args.debug):
                # Already printed reason in debug; record the JSON
                removed.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            langid = data.get("lang")
            #print(langid)
            
            if langid in lang_codes:                  
                    data["lang_family"] = lang_family[langid]
                    # Keep line → normalize + append LANG_...
                    del data["lang"]
                    #data["text"] = append_lang_tag(text, data.get("lang_family"))
                    outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                if langid == None:
                    #print("None line adding to INDIAN_LOW_RESOURCE")
                    data["lang_family"] = "INDIAN_LOW_RESOURCE"
                    #data["text"] = append_lang_tag(text, data.get("lang_family"))
                    outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                else:
                    #print("Skipping line, langid not in lang_codes")

                    removed.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
