"""Utilities for chunked/streaming-aware tag target construction.

Splits tagged text into inline entity tags (position-dependent) and
trailing sentence tags (buffer-level), then reconstructs CTC targets
with trailing tags repeated at regular chunk intervals.
"""

import re
from typing import List, Tuple

TAG_PREFIXES = (
    'ENTITY_', 'INTENT_', 'EMOTION_', 'GENDER_', 'AGE_',
    'DIALECT_', 'KEYWORD_', 'LANG_', 'OTHER_', 'ROLE_',
    'SPEAKER_', 'TURN_', 'FAMILY_',
)
EXACT_TAG_TOKENS = {'END', 'TURN_CHANGE'}

TRAILING_PREFIXES = (
    'AGE_', 'GENDER_', 'EMOTION_', 'INTENT_', 'DIALECT_',
    'LANG_', 'SPEAKER_', 'ROLE_', 'FAMILY_',
)


def _is_tag(word: str) -> bool:
    if word in EXACT_TAG_TOKENS:
        return True
    return any(word.startswith(p) for p in TAG_PREFIXES)


def _is_trailing_tag(word: str) -> bool:
    return any(word.startswith(p) for p in TRAILING_PREFIXES)


def split_inline_and_trailing(tagged_text: str) -> Tuple[str, List[str]]:
    """Split tagged text into inline-tagged body and trailing sentence tags.

    Returns:
        body: text with inline entity tags (ENTITY_...END) preserved,
              trailing tags removed
        trailing_tags: list of trailing tag tokens in order
    """
    words = tagged_text.split()

    trailing_tags = []
    i = len(words) - 1
    while i >= 0 and _is_trailing_tag(words[i]):
        trailing_tags.insert(0, words[i])
        i -= 1

    body = ' '.join(words[:i + 1])
    return body, trailing_tags


def build_chunked_target(
    tagged_text: str,
    chunk_size: int = 8,
) -> str:
    """Build a CTC target with trailing tags repeated at chunk intervals.

    Given: "ENTITY_PERSON_NAME John END went to ENTITY_LOCATION Paris END AGE_30_45 GENDER_MALE"
    With chunk_size=3, produces:
      "ENTITY_PERSON_NAME John END went AGE_30_45 GENDER_MALE to ENTITY_LOCATION Paris END AGE_30_45 GENDER_MALE"

    The trailing tags appear after every chunk_size transcription words,
    teaching the causal model to emit them at regular buffer boundaries.

    Args:
        tagged_text: full tagged text with trailing tags at the end
        chunk_size: number of transcription words per chunk before inserting trailing tags
    """
    body, trailing_tags = split_inline_and_trailing(tagged_text)
    if not trailing_tags:
        return tagged_text

    body_words = body.split()
    result = []
    transcript_count = 0

    for word in body_words:
        result.append(word)
        if not _is_tag(word):
            transcript_count += 1
            if transcript_count % chunk_size == 0:
                result.extend(trailing_tags)

    if transcript_count % chunk_size != 0:
        result.extend(trailing_tags)

    return ' '.join(result)
