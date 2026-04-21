"""Parse tagged text into (clean_text, tagged_text) pairs and build compositional tag vocabularies.

Tags in WhissleAI meta-ASR data come in two forms:
  1. Trailing: "hello world AGE_30_45 GENDER_MALE EMOTION_HAPPY INTENT_INFORM"
  2. Inline entities: "ENTITY_PERSON_NAME John END went to ENTITY_LOCATION Paris END"

This module handles both, and supports compositional tag decomposition so the
CTC model can generalize to unseen tag combinations.
"""

import json
import re
from collections import Counter
from typing import Dict, List, Set, Tuple

from promptingnemo.data.normalize import (
    META_TAG_PATTERN,
    SENTENCE_TAG_PREFIXES,
    normalize_text,
)

TAG_PREFIXES = (
    'ENTITY_', 'INTENT_', 'EMOTION_', 'GENDER_', 'AGE_',
    'DIALECT_', 'KEYWORD_', 'LANG_', 'OTHER_', 'ROLE_',
    'SPEAKER_', 'TURN_', 'FAMILY_',
)

EXACT_TAG_TOKENS = {'END', 'TURN_CHANGE'}

_TAG_RE = re.compile(
    r'^(?:' + '|'.join(re.escape(p) for p in TAG_PREFIXES) + r')[A-Z0-9_+.<>]*$'
    r'|^(?:' + '|'.join(re.escape(t) for t in EXACT_TAG_TOKENS) + r')$'
)


def is_tag(word: str) -> bool:
    return bool(_TAG_RE.match(word))


def strip_tags(text: str) -> str:
    """Remove all meta-tags from text, keeping only transcription words."""
    words = text.split()
    return ' '.join(w for w in words if not is_tag(w))


def parse_tagged_text(text: str) -> Tuple[str, str]:
    """Return (clean_text, normalized_tagged_text).

    clean_text has all tags and markers removed.
    normalized_tagged_text has tags normalized via normalize_text().
    """
    norm = normalize_text(text)
    clean = strip_tags(norm)
    return clean, norm


def decompose_tag(tag: str) -> List[str]:
    """Break a tag into compositional keyword pieces.

    Examples:
        'INTENT_REPORT_SYMPTOM' -> ['INTENT_', 'REPORT', '_SYMPTOM']
        'EMOTION_HAPPY'         -> ['EMOTION_', 'HAPPY']
        'AGE_30_45'             -> ['AGE_', '30', '_45']
        'END'                   -> ['END']
        'TURN_CHANGE'           -> ['TURN_', 'CHANGE']
        'ENTITY_PERSON_NAME'    -> ['ENTITY_', 'PERSON', '_NAME']
    """
    if tag in EXACT_TAG_TOKENS:
        return [tag]

    for prefix in TAG_PREFIXES:
        if tag.startswith(prefix):
            suffix = tag[len(prefix):]
            pieces = [prefix]
            parts = suffix.split('_')
            for i, part in enumerate(parts):
                if not part:
                    continue
                if i == 0:
                    pieces.append(part)
                else:
                    pieces.append('_' + part)
            return pieces

    return [tag]


def recompose_tag(pieces: List[str]) -> str:
    """Inverse of decompose_tag: join pieces back into a tag string.

    >>> recompose_tag(['INTENT_', 'REPORT', '_SYMPTOM'])
    'INTENT_REPORT_SYMPTOM'
    """
    return ''.join(pieces)


def build_tag_vocabulary(
    manifest_paths: List[str],
    max_tag_pieces: int = 10000,
    min_count: int = 5,
) -> Tuple[List[str], Dict[str, int]]:
    """Scan manifests, decompose all tags, return deduplicated tag pieces.

    Returns:
        tag_pieces: Sorted list of unique tag pieces (prefixes, keywords, exact tokens).
        tag_counts: Mapping from full tag -> count across all manifests.
    """
    tag_counts: Counter = Counter()

    for path in manifest_paths:
        with open(path, encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                text = normalize_text(entry.get('text', ''))
                for word in text.split():
                    if is_tag(word):
                        tag_counts[word] += 1

    piece_counts: Counter = Counter()
    for tag, count in tag_counts.items():
        if count < min_count:
            continue
        for piece in decompose_tag(tag):
            piece_counts[piece] += count

    sorted_pieces = sorted(piece_counts.keys(), key=lambda p: (-piece_counts[p], p))

    if len(sorted_pieces) > max_tag_pieces:
        sorted_pieces = sorted_pieces[:max_tag_pieces]

    return sorted_pieces, dict(tag_counts)


def build_char_vocabulary(
    manifest_paths: List[str],
    min_count: int = 10,
) -> List[str]:
    """Build character vocabulary from clean text in manifests.

    Returns sorted list of characters. Special tokens <pad>, <unk>, <bos>, <eos>
    are NOT included — the caller should prepend them.
    """
    char_counts: Counter = Counter()

    for path in manifest_paths:
        with open(path, encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                text = entry.get('text', '')
                clean = strip_tags(normalize_text(text))
                for ch in clean:
                    char_counts[ch] += 1

    chars = [ch for ch, cnt in char_counts.items() if cnt >= min_count]
    return sorted(chars)
