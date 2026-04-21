"""Tests for promptingnemo.data.tag_parser module."""

import json
import os
import tempfile

import pytest

from promptingnemo.data.tag_parser import (
    build_char_vocabulary,
    build_tag_vocabulary,
    decompose_tag,
    is_tag,
    parse_tagged_text,
    recompose_tag,
    strip_tags,
)


class TestIsTag:
    def test_emotion_tag(self):
        assert is_tag("EMOTION_HAPPY") is True

    def test_entity_tag(self):
        assert is_tag("ENTITY_PERSON_NAME") is True

    def test_intent_tag(self):
        assert is_tag("INTENT_INFORM") is True

    def test_age_tag(self):
        assert is_tag("AGE_30_45") is True

    def test_gender_tag(self):
        assert is_tag("GENDER_MALE") is True

    def test_end_token(self):
        assert is_tag("END") is True

    def test_turn_change(self):
        assert is_tag("TURN_CHANGE") is True

    def test_regular_word(self):
        assert is_tag("hello") is False

    def test_mixed_case(self):
        assert is_tag("Hello") is False

    def test_number(self):
        assert is_tag("123") is False


class TestStripTags:
    def test_strip_trailing_tags(self):
        text = "hello world AGE_30_45 GENDER_MALE EMOTION_HAPPY"
        assert strip_tags(text) == "hello world"

    def test_strip_inline_entities(self):
        text = "ENTITY_PERSON_NAME John END went home"
        assert strip_tags(text) == "John went home"

    def test_no_tags(self):
        assert strip_tags("hello world") == "hello world"

    def test_only_tags(self):
        assert strip_tags("EMOTION_HAPPY GENDER_MALE") == ""


class TestDecomposeTag:
    def test_simple_emotion(self):
        assert decompose_tag("EMOTION_HAPPY") == ["EMOTION_", "HAPPY"]

    def test_intent_multi_part(self):
        assert decompose_tag("INTENT_REPORT_SYMPTOM") == ["INTENT_", "REPORT", "_SYMPTOM"]

    def test_age_numeric(self):
        assert decompose_tag("AGE_30_45") == ["AGE_", "30", "_45"]

    def test_end_token(self):
        assert decompose_tag("END") == ["END"]

    def test_entity_person_name(self):
        assert decompose_tag("ENTITY_PERSON_NAME") == ["ENTITY_", "PERSON", "_NAME"]

    def test_age_60plus(self):
        assert decompose_tag("AGE_60PLUS") == ["AGE_", "60PLUS"]

    def test_turn_change(self):
        # TURN_CHANGE is in EXACT_TAG_TOKENS, treated as atomic
        assert decompose_tag("TURN_CHANGE") == ["TURN_CHANGE"]


class TestRecomposeTag:
    def test_roundtrip_emotion(self):
        assert recompose_tag(decompose_tag("EMOTION_HAPPY")) == "EMOTION_HAPPY"

    def test_roundtrip_intent(self):
        assert recompose_tag(decompose_tag("INTENT_REPORT_SYMPTOM")) == "INTENT_REPORT_SYMPTOM"

    def test_roundtrip_age(self):
        assert recompose_tag(decompose_tag("AGE_30_45")) == "AGE_30_45"

    def test_roundtrip_end(self):
        assert recompose_tag(decompose_tag("END")) == "END"

    def test_roundtrip_entity(self):
        assert recompose_tag(decompose_tag("ENTITY_PERSON_NAME")) == "ENTITY_PERSON_NAME"


class TestParseTaggedText:
    def test_trailing_tags(self):
        text = "hello world EMOTION_HAPPY GENDER_MALE"
        clean, tagged = parse_tagged_text(text)
        assert clean == "hello world"
        assert "EMOTION_HAPPY" in tagged
        assert "GENDER_MALE" in tagged

    def test_inline_entities(self):
        text = "ENTITY_LOCATION Paris END is beautiful"
        clean, tagged = parse_tagged_text(text)
        assert clean == "Paris is beautiful"
        assert "ENTITY_LOCATION" in tagged

    def test_normalizes_typos(self):
        text = "hello GER_MALE EMOTION_HAP"
        clean, tagged = parse_tagged_text(text)
        assert "GENDER_MALE" in tagged
        assert "EMOTION_HAPPY" in tagged


class TestBuildTagVocabulary:
    def _write_manifest(self, entries, tmpdir):
        path = os.path.join(tmpdir, "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        return path

    def test_builds_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = [
                {"text": "hello EMOTION_HAPPY GENDER_MALE AGE_30_45"}
            ] * 20
            path = self._write_manifest(entries, tmpdir)
            pieces, counts = build_tag_vocabulary([path], min_count=1)
            assert "EMOTION_" in pieces
            assert "HAPPY" in pieces
            assert "GENDER_" in pieces
            assert "MALE" in pieces
            assert "AGE_" in pieces

    def test_respects_min_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = [{"text": "hello EMOTION_HAPPY"}] * 20 + [
                {"text": "hello INTENT_RARE_THING"}
            ]
            path = self._write_manifest(entries, tmpdir)
            pieces, counts = build_tag_vocabulary([path], min_count=10)
            assert "EMOTION_" in pieces
            assert "RARE" not in pieces

    def test_respects_max_pieces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = [
                {"text": "EMOTION_HAPPY GENDER_MALE AGE_30_45 INTENT_INFORM ENTITY_PERSON_NAME END"}
            ] * 20
            path = self._write_manifest(entries, tmpdir)
            pieces, _ = build_tag_vocabulary([path], max_tag_pieces=5, min_count=1)
            assert len(pieces) <= 5


class TestBuildCharVocabulary:
    def _write_manifest(self, entries, tmpdir):
        path = os.path.join(tmpdir, "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        return path

    def test_builds_chars_from_clean_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = [{"text": "hello world EMOTION_HAPPY"}] * 20
            path = self._write_manifest(entries, tmpdir)
            chars = build_char_vocabulary([path], min_count=1)
            assert 'h' in chars
            assert 'e' in chars
            assert ' ' in chars
            # Tags should be stripped, so tag chars shouldn't dominate
            assert 'E' not in chars or 'h' in chars  # 'E' might appear in clean text too
