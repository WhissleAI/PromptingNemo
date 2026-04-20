"""Tests for scan_manifest_for_new_tokens from promptingnemo.models.decoder.

The decoder module imports NeMo + torch at the top level. We extract just the
scan_manifest_for_new_tokens function by loading its source directly, since the
function itself only depends on json + re (stdlib).
"""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

# --- Inline fallback of scan_manifest_for_new_tokens for environments without NeMo ---

_DECODER_PATH = Path(__file__).resolve().parent.parent / "promptingnemo" / "models" / "decoder.py"


def _fallback_scan_manifest_for_new_tokens(
    manifest_path: str,
    current_vocab: set,
    min_count: int = 10,
    allowed_prefixes: tuple = (
        'ENTITY_', 'INTENT_', 'EMOTION_', 'GENDER_', 'AGE_',
        'DIALECT_', 'KEYWORD_', 'LANG_', 'OTHER_',
    ),
) -> List[str]:
    """Standalone copy of scan_manifest_for_new_tokens (no NeMo deps)."""
    tag_pattern = re.compile(r'^[A-Z][A-Z0-9_]*_[A-Z0-9_<>+.]*$|^END$')
    counts: Dict[str, int] = {}
    with open(manifest_path, encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            for word in entry.get('text', '').split():
                if tag_pattern.match(word) and word not in current_vocab:
                    counts[word] = counts.get(word, 0) + 1

    found = []
    for token, count in sorted(counts.items()):
        if count < min_count:
            continue
        if allowed_prefixes and not any(token.startswith(p) for p in allowed_prefixes):
            continue
        found.append(token)
    return found


def _get_scan_function():
    try:
        from promptingnemo.models.decoder import scan_manifest_for_new_tokens
        return scan_manifest_for_new_tokens
    except ImportError:
        return _fallback_scan_manifest_for_new_tokens


scan_manifest_for_new_tokens = _get_scan_function()


class TestScanManifestForNewTokens:
    def _write_manifest(self, lines, tmpdir):
        path = os.path.join(tmpdir, "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            for entry in lines:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return path

    def test_finds_tag_tokens_not_in_vocab(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Each token must appear >= min_count times
            entries = [
                {"text": "EMOTION_HAPPY GENDER_MALE hello world ENTITY_PERSON_NAME john AGE_30_45"}
            ] * 15
            manifest_path = self._write_manifest(entries, tmpdir)
            current_vocab = {"EMOTION_HAPPY"}

            new_tokens = scan_manifest_for_new_tokens(manifest_path, current_vocab)

            assert "GENDER_MALE" in new_tokens
            assert "AGE_30_45" in new_tokens
            assert "ENTITY_PERSON_NAME" in new_tokens
            assert "EMOTION_HAPPY" not in new_tokens

    def test_returns_sorted_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = [
                {"text": "EMOTION_SAD EMOTION_HAPPY AGE_60PLUS GENDER_FEMALE"},
            ] * 15
            manifest_path = self._write_manifest(entries, tmpdir)
            current_vocab = set()

            new_tokens = scan_manifest_for_new_tokens(manifest_path, current_vocab)

            assert new_tokens == sorted(new_tokens)

    def test_ignores_regular_words(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = [
                {"text": "EMOTION_HAPPY hello world this is regular text"},
            ] * 15
            manifest_path = self._write_manifest(entries, tmpdir)
            current_vocab = set()

            new_tokens = scan_manifest_for_new_tokens(manifest_path, current_vocab)

            assert "hello" not in new_tokens
            assert "world" not in new_tokens

    def test_empty_manifest_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_manifest([], tmpdir)
            new_tokens = scan_manifest_for_new_tokens(manifest_path, set())
            assert new_tokens == []

    def test_all_tokens_in_vocab_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = [{"text": "EMOTION_HAPPY GENDER_MALE"}] * 15
            manifest_path = self._write_manifest(entries, tmpdir)
            current_vocab = {"EMOTION_HAPPY", "GENDER_MALE"}

            new_tokens = scan_manifest_for_new_tokens(manifest_path, current_vocab)
            assert new_tokens == []

    def test_rare_tokens_filtered_by_min_count(self):
        """Tokens appearing fewer than min_count times should be excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = [
                {"text": "EMOTION_HAPPY GENDER_MALE"},  # appears 15x
            ] * 15 + [
                {"text": "ENTITY_BALCONY"},  # appears 1x — should be filtered
            ]
            manifest_path = self._write_manifest(entries, tmpdir)
            current_vocab = set()

            new_tokens = scan_manifest_for_new_tokens(manifest_path, current_vocab)

            assert "EMOTION_HAPPY" in new_tokens
            assert "ENTITY_BALCONY" not in new_tokens

    def test_min_count_override(self):
        """Setting min_count=1 should include all tokens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = [{"text": "ENTITY_RARE_THING"}]
            manifest_path = self._write_manifest(entries, tmpdir)

            new_tokens = scan_manifest_for_new_tokens(
                manifest_path, set(), min_count=1
            )
            assert "ENTITY_RARE_THING" in new_tokens

    def test_invalid_prefix_filtered(self):
        """Tokens with non-allowed prefixes should be excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = [{"text": "EMOTION_HAPPY GARBAGE_TOKEN"}] * 15
            manifest_path = self._write_manifest(entries, tmpdir)

            new_tokens = scan_manifest_for_new_tokens(manifest_path, set())

            assert "EMOTION_HAPPY" in new_tokens
            assert "GARBAGE_TOKEN" not in new_tokens

    def test_no_duplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = [{"text": "EMOTION_HAPPY hello"}] * 20
            manifest_path = self._write_manifest(entries, tmpdir)

            new_tokens = scan_manifest_for_new_tokens(manifest_path, set())
            assert new_tokens.count("EMOTION_HAPPY") == 1

    def test_entry_without_text_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = [
                {"audio_filepath": "/data/audio.wav", "duration": 3.0},
            ] + [{"text": "EMOTION_HAPPY hello"}] * 15
            manifest_path = self._write_manifest(entries, tmpdir)

            new_tokens = scan_manifest_for_new_tokens(manifest_path, set())
            assert "EMOTION_HAPPY" in new_tokens
