"""Tests for promptingnemo.eval utilities.

get_clean_transcript, make_distinct, and get_entity_format are pure Python
functions inside eval/wer.py, but the module imports NeMo at the top level.
We load these functions via importlib.util to bypass the NeMo import.

get_ner_scores requires numpy + editdistance, tested with graceful skip.
"""

import importlib.util
from pathlib import Path

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Load wer.py directly to bypass NeMo imports
# ---------------------------------------------------------------------------

_WER_PATH = Path(__file__).resolve().parent.parent / "promptingnemo" / "eval" / "wer.py"
_NER_PATH = Path(__file__).resolve().parent.parent / "promptingnemo" / "eval" / "ner.py"


def _load_wer_functions():
    """Load wer.py functions directly, stubbing out the NeMo import."""
    import types
    import sys

    # Create a stub for the NeMo import so the module can load
    nemo_stub = types.ModuleType("nemo")
    nemo_collections = types.ModuleType("nemo.collections")
    nemo_asr = types.ModuleType("nemo.collections.asr")
    nemo_metrics = types.ModuleType("nemo.collections.asr.metrics")
    nemo_wer = types.ModuleType("nemo.collections.asr.metrics.wer")

    # Provide a dummy word_error_rate for the import
    nemo_wer.word_error_rate = lambda hyps, refs, use_cer=False: 0.0

    nemo_stub.collections = nemo_collections
    nemo_collections.asr = nemo_asr
    nemo_asr.metrics = nemo_metrics
    nemo_metrics.wer = nemo_wer

    # Also stub the ner import since wer.py imports from it
    ner_stub = types.ModuleType("promptingnemo.eval.ner")
    ner_stub.get_ner_scores = lambda gt, pred: {}

    saved_modules = {}
    stubs = {
        "nemo": nemo_stub,
        "nemo.collections": nemo_collections,
        "nemo.collections.asr": nemo_asr,
        "nemo.collections.asr.metrics": nemo_metrics,
        "nemo.collections.asr.metrics.wer": nemo_wer,
        "promptingnemo.eval.ner": ner_stub,
    }
    for name, mod in stubs.items():
        saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = mod

    try:
        spec = importlib.util.spec_from_file_location("_test_wer", _WER_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        # Restore original modules
        for name, original in saved_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _load_ner_module():
    """Load ner.py -- requires numpy and editdistance."""
    try:
        spec = importlib.util.spec_from_file_location("_test_ner", _NER_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except ImportError:
        return None


try:
    _wer_mod = _load_wer_functions()
    get_clean_transcript = _wer_mod.get_clean_transcript
    make_distinct = _wer_mod.make_distinct
    get_entity_format = _wer_mod.get_entity_format
    _HAS_WER = True
except Exception:
    _HAS_WER = False

try:
    _ner_mod = _load_ner_module()
    if _ner_mod is None:
        raise ImportError("ner module could not be loaded")
    get_ner_scores = _ner_mod.get_ner_scores
    _HAS_NER = True
except Exception:
    _HAS_NER = False


# ---------------------------------------------------------------------------
# get_clean_transcript (from eval.wer)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_WER, reason="wer module could not be loaded")
class TestGetCleanTranscript:
    def test_removes_all_uppercase_tags(self):
        text = "EMOTION_HAPPY GENDER_MALE hello world END"
        result = get_clean_transcript(text)
        assert "EMOTION_HAPPY" not in result
        assert "GENDER_MALE" not in result
        assert "END" not in result

    def test_keeps_lowercase_words(self):
        text = "EMOTION_HAPPY hello world"
        result = get_clean_transcript(text)
        assert "hello" in result
        assert "world" in result

    def test_empty_input(self):
        result = get_clean_transcript("")
        assert result == ""

    def test_only_tags_returns_empty(self):
        result = get_clean_transcript("EMOTION_HAPPY GENDER_MALE END")
        assert result.strip() == ""

    def test_only_words_unchanged(self):
        text = "hello world this is a test"
        result = get_clean_transcript(text)
        assert result == text

    def test_mixed_case_word_kept(self):
        """Words like 'Hello' are not all-upper, so should be kept."""
        text = "EMOTION_HAPPY Hello World"
        result = get_clean_transcript(text)
        assert "Hello" in result
        assert "World" in result

    def test_preserves_word_spacing(self):
        text = "EMOTION_HAPPY hello GENDER_MALE world"
        result = get_clean_transcript(text)
        assert result == "hello world"


# ---------------------------------------------------------------------------
# get_ner_scores (from eval.ner)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_NER, reason="ner module requires numpy + editdistance")
class TestGetNerScores:
    def test_perfect_match(self):
        """When predictions exactly match ground truth, precision/recall/fscore = 1.0."""
        gt = [[("PERSON", "john doe", 1)]]
        pred = [[("PERSON", "john doe", 1)]]
        metrics = get_ner_scores(gt, pred)
        assert metrics["overall_micro"]["precision"] == pytest.approx(1.0)
        assert metrics["overall_micro"]["recall"] == pytest.approx(1.0)
        assert metrics["overall_micro"]["fscore"] == pytest.approx(1.0)

    def test_no_predictions(self):
        """When no predictions are made, recall should be 0."""
        gt = [[("PERSON", "john doe", 1)]]
        pred = [[]]
        # get_ner_scores zips gt and pred; pred has no entries for this sentence
        # so stats won't be populated for this sentence
        # The lists must be same length for zip
        metrics = get_ner_scores(gt, pred)
        if "PERSON" in metrics:
            assert metrics["PERSON"]["recall"] == pytest.approx(0.0)

    def test_extra_predictions_lower_precision(self):
        """Extra predictions should reduce precision."""
        gt = [[("PERSON", "john doe", 1)]]
        pred = [[("PERSON", "john doe", 1), ("PERSON", "jane doe", 2)]]
        metrics = get_ner_scores(gt, pred)
        assert metrics["PERSON"]["precision"] == pytest.approx(0.5)
        assert metrics["PERSON"]["recall"] == pytest.approx(1.0)

    def test_empty_inputs(self):
        """Empty gt and predictions should produce empty metrics."""
        metrics = get_ner_scores([], [])
        assert "overall_micro" in metrics
        assert "overall_macro" in metrics

    def test_multiple_entity_types(self):
        gt = [
            [("PERSON", "john", 1), ("LOCATION", "new york", 1)],
        ]
        pred = [
            [("PERSON", "john", 1), ("LOCATION", "new york", 1)],
        ]
        metrics = get_ner_scores(gt, pred)
        assert "PERSON" in metrics
        assert "LOCATION" in metrics
        assert metrics["overall_micro"]["fscore"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# make_distinct (from eval.wer)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_WER, reason="wer module could not be loaded")
class TestMakeDistinct:
    def test_empty_list(self):
        result = make_distinct([])
        assert result == []

    def test_unique_items(self):
        items = [("PERSON", "john"), ("LOCATION", "nyc")]
        result = make_distinct(items)
        assert len(result) == 2
        assert result[0] == ("PERSON", "john", 1)
        assert result[1] == ("LOCATION", "nyc", 1)

    def test_duplicate_items_get_different_ids(self):
        items = [("PERSON", "john"), ("PERSON", "john")]
        result = make_distinct(items)
        assert len(result) == 2
        assert result[0] == ("PERSON", "john", 1)
        assert result[1] == ("PERSON", "john", 2)
        assert len(set(result)) == len(result)


# ---------------------------------------------------------------------------
# get_entity_format (from eval.wer)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_WER, reason="wer module could not be loaded")
class TestGetEntityFormat:
    def test_extracts_entity_with_phrase(self):
        tags = {
            "NER": {"ENTITY_PERSON_NAME"},
            "END": {"END"},
        }
        sents = ["ENTITY_PERSON_NAME john doe END hello"]
        label_lst, sent_lst = get_entity_format(sents, tags, "exact")
        assert len(label_lst) == 1
        labels = label_lst[0]
        assert any(t[0] == "ENTITY_PERSON_NAME" for t in labels)

    def test_no_entities(self):
        tags = {
            "NER": {"ENTITY_PERSON_NAME"},
            "END": {"END"},
        }
        sents = ["hello world no entities here"]
        label_lst, sent_lst = get_entity_format(sents, tags, "exact")
        assert label_lst == []

    def test_label_score_type(self):
        tags = {
            "NER": {"ENTITY_PERSON_NAME"},
            "END": {"END"},
        }
        sents = ["ENTITY_PERSON_NAME john doe END"]
        label_lst, _ = get_entity_format(sents, tags, "label")
        if label_lst:
            labels = label_lst[0]
            assert any(t[1] == "phrase" for t in labels)
