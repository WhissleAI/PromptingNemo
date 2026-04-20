"""Tests for promptingnemo.data.normalize module.

We load the normalize module directly from its file path to bypass
the data/__init__.py which imports NeMo-dependent modules.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# Load normalize.py directly to avoid data/__init__.py triggering NeMo imports
_NORMALIZE_PATH = Path(__file__).resolve().parent.parent / "promptingnemo" / "data" / "normalize.py"


def _load_normalize_module():
    spec = importlib.util.spec_from_file_location("promptingnemo_data_normalize", _NORMALIZE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    _mod = _load_normalize_module()
    normalize_text = _mod.normalize_text
    extract_tags = _mod.extract_tags
    should_keep_line = _mod.should_keep_line
    ALLOWED_EMOTIONS = _mod.ALLOWED_EMOTIONS
    FORBIDDEN_AGES = _mod.FORBIDDEN_AGES
    _HAS_NORMALIZE = True
except Exception as exc:
    _HAS_NORMALIZE = False
    _IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(not _HAS_NORMALIZE, reason="normalize module could not be loaded")


# ---------------------------------------------------------------------------
# normalize_text tests
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_ger_male_to_gender_male(self):
        result = normalize_text("GER_MALE hello")
        assert "GENDER_MALE" in result
        assert "GER_MALE" not in result

    def test_ger_female_to_gender_female(self):
        result = normalize_text("GER_FEMALE world")
        assert "GENDER_FEMALE" in result
        assert "GER_FEMALE" not in result

    def test_emotion_hap_to_emotion_happy(self):
        result = normalize_text("EMOTION_HAP some text")
        assert "EMOTION_HAPPY" in result
        assert "EMOTION_HAP " not in result

    def test_emotion_neu_to_emotion_neutral(self):
        result = normalize_text("EMOTION_NEU more text")
        assert "EMOTION_NEUTRAL" in result

    def test_emotion_ang_to_emotion_angry(self):
        result = normalize_text("EMOTION_ANG angry text")
        assert "EMOTION_ANGRY" in result

    def test_emotion_happypy_to_emotion_happy(self):
        result = normalize_text("EMOTION_HAPPYPY test")
        assert "EMOTION_HAPPY" in result
        assert "HAPPYPY" not in result

    def test_age_60plus_to_age_60_plus(self):
        result = normalize_text("AGE_60PLUS speaker")
        assert "AGE_60+" in result
        assert "AGE_60PLUS" not in result

    def test_trailing_quote_comma_cleanup(self):
        """Trailing '", ' after a tag should be stripped."""
        result = normalize_text('AGE_STATE", some text')
        assert '",' not in result

    def test_whitespace_collapse(self):
        result = normalize_text("hello    world   test")
        assert result == "hello world test"

    def test_whitespace_collapse_with_tabs_and_newlines(self):
        result = normalize_text("hello\t\tworld\n\ntest")
        assert result == "hello world test"

    def test_non_string_input_returns_empty(self):
        assert normalize_text(None) == ""
        assert normalize_text(123) == ""

    def test_empty_string_returns_empty(self):
        assert normalize_text("") == ""

    def test_no_changes_needed(self):
        text = "EMOTION_HAPPY GENDER_MALE AGE_30_45 hello world"
        result = normalize_text(text)
        assert result == text

    def test_multiple_fixes_combined(self):
        text = "EMOTION_HAP GER_FEMALE AGE_60PLUS hello"
        result = normalize_text(text)
        assert "EMOTION_HAPPY" in result
        assert "GENDER_FEMALE" in result
        assert "AGE_60+" in result

    def test_emotion_happy_not_double_replaced(self):
        """EMOTION_HAPPY should not be modified by the HAP -> HAPPY rule."""
        result = normalize_text("EMOTION_HAPPY")
        assert result == "EMOTION_HAPPY"


# ---------------------------------------------------------------------------
# extract_tags tests
# ---------------------------------------------------------------------------

class TestExtractTags:
    def test_extracts_emotion_tags(self):
        emotions, ages = extract_tags("EMOTION_HAPPY hello world EMOTION_SAD goodbye")
        assert "EMOTION_HAPPY" in emotions
        assert "EMOTION_SAD" in emotions
        assert len(emotions) == 2

    def test_extracts_age_tags(self):
        emotions, ages = extract_tags("EMOTION_HAPPY AGE_30_45 hello AGE_60+")
        assert "AGE_30_45" in ages
        assert "AGE_60+" in ages
        assert len(ages) == 2

    def test_normalizes_before_extracting(self):
        """Tags with typos should be normalized then extracted."""
        emotions, ages = extract_tags("EMOTION_HAP AGE_60PLUS text")
        assert "EMOTION_HAPPY" in emotions
        assert "AGE_60+" in ages

    def test_no_tags_returns_empty(self):
        emotions, ages = extract_tags("just regular text here")
        assert emotions == []
        assert ages == []

    def test_empty_string(self):
        emotions, ages = extract_tags("")
        assert emotions == []
        assert ages == []


# ---------------------------------------------------------------------------
# should_keep_line tests
# ---------------------------------------------------------------------------

class TestShouldKeepLine:
    def test_valid_line_with_allowed_emotion(self):
        assert should_keep_line("EMOTION_HAPPY GENDER_MALE hello world") is True

    def test_valid_line_with_multiple_emotions(self):
        assert should_keep_line("EMOTION_HAPPY EMOTION_SAD hello world") is True

    def test_line_without_emotion_is_dropped(self):
        assert should_keep_line("GENDER_MALE AGE_30_45 hello world") is False

    def test_line_with_forbidden_age_is_dropped(self):
        assert should_keep_line("EMOTION_HAPPY AGE_NUMBER hello world") is False

    def test_line_with_age_state_is_dropped(self):
        assert should_keep_line("EMOTION_HAPPY AGE_STATE hello world") is False

    def test_line_with_allowed_age_is_kept(self):
        assert should_keep_line("EMOTION_HAPPY AGE_30_45 hello world") is True

    def test_normalized_typo_line_is_kept(self):
        """Lines with fixable typos should be kept after normalization."""
        assert should_keep_line("EMOTION_HAP GER_MALE hello world") is True

    def test_line_with_unknown_emotion_is_dropped(self):
        assert should_keep_line("EMOTION_UNKNOWN hello world") is False

    def test_empty_string_is_dropped(self):
        assert should_keep_line("") is False

    def test_all_allowed_emotions_individually(self):
        for emotion in ALLOWED_EMOTIONS:
            assert should_keep_line(f"{emotion} hello world") is True, \
                f"{emotion} should be an allowed emotion"


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    def test_allowed_emotions_contains_expected_set(self):
        expected = {
            "EMOTION_ANGRY", "EMOTION_DISGUST", "EMOTION_FEAR",
            "EMOTION_HAPPY", "EMOTION_NEUTRAL", "EMOTION_SAD", "EMOTION_SURPRISE",
        }
        assert ALLOWED_EMOTIONS == expected

    def test_forbidden_ages_contains_expected_set(self):
        assert FORBIDDEN_AGES == {"AGE_NUMBER", "AGE_STATE"}
