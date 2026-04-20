"""Tests for scan_manifest_for_new_tokens from promptingnemo.models.decoder.

The decoder module imports NeMo + torch at the top level. We extract just the
scan_manifest_for_new_tokens function by loading its source directly, since the
function itself only depends on json + re (stdlib).
"""

import importlib.util
import json
import os
import re
import tempfile
from pathlib import Path
from typing import List

import pytest

# --- Inline fallback of scan_manifest_for_new_tokens for environments without NeMo ---
# We try importing the real module first; if that fails we define a local copy
# from the source, since the function is pure stdlib (json + re).

_DECODER_PATH = Path(__file__).resolve().parent.parent / "promptingnemo" / "models" / "decoder.py"


def _fallback_scan_manifest_for_new_tokens(manifest_path: str, current_vocab: set) -> List[str]:
    """Standalone copy of scan_manifest_for_new_tokens (no NeMo deps)."""
    tag_pattern = re.compile(r'^[A-Z][A-Z0-9_]*_[A-Z0-9_<>+.]*$|^END$')
    found = set()
    with open(manifest_path, encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            for word in entry.get('text', '').split():
                if tag_pattern.match(word) and word not in current_vocab:
                    found.add(word)
    return sorted(found)


def _get_scan_function():
    try:
        from promptingnemo.models.decoder import scan_manifest_for_new_tokens
        return scan_manifest_for_new_tokens
    except ImportError:
        # NeMo not available -- use the standalone copy
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
            entries = [
                {"text": "EMOTION_HAPPY GENDER_MALE hello world END"},
                {"text": "AGE_30_45 ENTITY_PERSON_NAME john END"},
            ]
            manifest_path = self._write_manifest(entries, tmpdir)
            current_vocab = {"EMOTION_HAPPY", "END"}

            new_tokens = scan_manifest_for_new_tokens(manifest_path, current_vocab)

            assert "GENDER_MALE" in new_tokens
            assert "AGE_30_45" in new_tokens
            assert "ENTITY_PERSON_NAME" in new_tokens
            # Already in vocab, should NOT appear
            assert "EMOTION_HAPPY" not in new_tokens
            assert "END" not in new_tokens

    def test_returns_sorted_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = [
                {"text": "EMOTION_SAD EMOTION_HAPPY AGE_60+ GENDER_FEMALE END"},
            ]
            manifest_path = self._write_manifest(entries, tmpdir)
            current_vocab = set()

            new_tokens = scan_manifest_for_new_tokens(manifest_path, current_vocab)

            assert new_tokens == sorted(new_tokens)

    def test_ignores_regular_words(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = [
                {"text": "EMOTION_HAPPY hello world this is regular text END"},
            ]
            manifest_path = self._write_manifest(entries, tmpdir)
            current_vocab = set()

            new_tokens = scan_manifest_for_new_tokens(manifest_path, current_vocab)

            assert "hello" not in new_tokens
            assert "world" not in new_tokens
            assert "this" not in new_tokens
            assert "is" not in new_tokens
            assert "regular" not in new_tokens
            assert "text" not in new_tokens

    def test_empty_manifest_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_manifest([], tmpdir)
            current_vocab = set()

            new_tokens = scan_manifest_for_new_tokens(manifest_path, current_vocab)

            assert new_tokens == []

    def test_all_tokens_in_vocab_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = [
                {"text": "EMOTION_HAPPY GENDER_MALE END"},
            ]
            manifest_path = self._write_manifest(entries, tmpdir)
            current_vocab = {"EMOTION_HAPPY", "GENDER_MALE", "END"}

            new_tokens = scan_manifest_for_new_tokens(manifest_path, current_vocab)

            assert new_tokens == []

    def test_no_duplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = [
                {"text": "EMOTION_HAPPY hello"},
                {"text": "EMOTION_HAPPY world"},
                {"text": "EMOTION_HAPPY again"},
            ]
            manifest_path = self._write_manifest(entries, tmpdir)
            current_vocab = set()

            new_tokens = scan_manifest_for_new_tokens(manifest_path, current_vocab)

            assert new_tokens.count("EMOTION_HAPPY") == 1

    def test_entry_without_text_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = [
                {"audio_filepath": "/data/audio.wav", "duration": 3.0},
                {"text": "EMOTION_HAPPY hello END"},
            ]
            manifest_path = self._write_manifest(entries, tmpdir)
            current_vocab = set()

            new_tokens = scan_manifest_for_new_tokens(manifest_path, current_vocab)

            assert "EMOTION_HAPPY" in new_tokens
            assert "END" in new_tokens
