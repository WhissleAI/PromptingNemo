"""Tests for promptingnemo.tokenizer.config module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from promptingnemo.tokenizer.config import (
    LANG_FAMILIES,
    LANG_TO_FAMILY,
    set_language_families,
    resolve_model_path,
)


class TestSetLanguageFamilies:
    def setup_method(self):
        """Clear module-level state before each test."""
        LANG_FAMILIES.clear()
        LANG_TO_FAMILY.clear()

    def test_dict_input_with_family_language_mapping(self):
        families = {
            "indic": ["hi", "bn", "ta"],
            "european": ["en", "fr", "es"],
        }
        set_language_families(families)

        assert "INDIC" in LANG_FAMILIES
        assert "EUROPEAN" in LANG_FAMILIES
        assert sorted(LANG_FAMILIES["INDIC"]) == ["BN", "HI", "TA"]
        assert sorted(LANG_FAMILIES["EUROPEAN"]) == ["EN", "ES", "FR"]

    def test_populates_lang_to_family_correctly(self):
        families = {
            "indic": ["hi", "bn"],
            "european": ["en", "fr"],
        }
        set_language_families(families)

        assert LANG_TO_FAMILY["HI"] == "INDIC"
        assert LANG_TO_FAMILY["BN"] == "INDIC"
        assert LANG_TO_FAMILY["EN"] == "EUROPEAN"
        assert LANG_TO_FAMILY["FR"] == "EUROPEAN"

    def test_raises_value_error_on_empty_input(self):
        with pytest.raises(ValueError, match="required but was empty"):
            set_language_families({})

    def test_raises_value_error_on_none_input(self):
        with pytest.raises(ValueError, match="required but was empty"):
            set_language_families(None)

    def test_list_input_with_string_entries(self):
        families = ["en", "fr", "de"]
        set_language_families(families)

        assert "EN" in LANG_TO_FAMILY
        assert "FR" in LANG_TO_FAMILY
        assert "DE" in LANG_TO_FAMILY

    def test_list_input_with_dict_entries(self):
        families = [
            {"indic": ["hi", "bn"]},
            {"european": ["en"]},
        ]
        set_language_families(families)

        assert "INDIC" in LANG_FAMILIES
        assert "EUROPEAN" in LANG_FAMILIES
        assert LANG_TO_FAMILY["HI"] == "INDIC"

    def test_case_normalization(self):
        """All keys and values should be upper-cased."""
        families = {"Indic": ["hi", "BN", "Ta"]}
        set_language_families(families)

        assert "INDIC" in LANG_FAMILIES
        assert all(lang.isupper() for lang in LANG_FAMILIES["INDIC"])
        assert all(key.isupper() for key in LANG_TO_FAMILY.keys())

    def test_replaces_previous_state(self):
        set_language_families({"indic": ["hi"]})
        assert "INDIC" in LANG_FAMILIES
        assert "HI" in LANG_TO_FAMILY

        set_language_families({"european": ["en"]})
        assert "INDIC" not in LANG_FAMILIES
        assert "HI" not in LANG_TO_FAMILY
        assert "EUROPEAN" in LANG_FAMILIES
        assert "EN" in LANG_TO_FAMILY

    def test_single_language_string_value(self):
        families = {"indic": "hi"}
        set_language_families(families)
        assert LANG_FAMILIES["INDIC"] == ["HI"]
        assert LANG_TO_FAMILY["HI"] == "INDIC"

    def test_invalid_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported language_families type"):
            set_language_families(42)

    def test_invalid_language_iterable_raises_value_error(self):
        with pytest.raises(ValueError, match="Expected iterable"):
            set_language_families({"indic": 42})

    def test_family_with_none_languages_uses_family_as_language(self):
        families = {"EN": None}
        set_language_families(families)
        assert "EN" in LANG_FAMILIES
        assert LANG_TO_FAMILY["EN"] == "EN"


class TestResolveModelPath:
    def test_absolute_path_returned_as_is(self):
        cfg = MagicMock()
        cfg.model.model_root = "/some/root"
        result = resolve_model_path(cfg, "/absolute/path/to/file.txt")
        assert result == Path("/absolute/path/to/file.txt")

    def test_relative_path_resolved_against_model_root(self):
        cfg = MagicMock()
        cfg.model.model_root = "/some/root"
        result = resolve_model_path(cfg, "relative/file.txt")
        assert str(result).endswith("some/root/relative/file.txt")

    def test_relative_path_with_tilde_model_root(self):
        cfg = MagicMock()
        cfg.model.model_root = "~/models"
        result = resolve_model_path(cfg, "file.txt")
        # Should not start with ~ since model_root gets expanded
        assert "~" not in str(result)
        assert str(result).endswith("models/file.txt")
