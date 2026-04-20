"""Smoke tests verifying that all package modules can be imported.

Modules that depend on NeMo are skipped gracefully when NeMo is not installed.
Note: The data/__init__.py re-exports NeMo-dependent symbols, so even
data.normalize requires NeMo via the package __init__. Similarly, training.cli
imports data.dataset at module level. These are tested with graceful skips.
"""

import importlib

import pytest


class TestCoreImports:
    """Modules that should always be importable (no NeMo dependency)."""

    def test_import_promptingnemo(self):
        import promptingnemo
        assert hasattr(promptingnemo, "__version__")

    def test_import_tokenizer_config(self):
        from promptingnemo.tokenizer import config
        assert hasattr(config, "set_language_families")
        assert hasattr(config, "LANG_FAMILIES")
        assert hasattr(config, "LANG_TO_FAMILY")
        assert hasattr(config, "resolve_model_path")


class TestNemoDependentImports:
    """Modules that require NeMo or trigger NeMo via transitive imports.
    Skip gracefully if unavailable."""

    def test_import_data_normalize(self):
        """normalize.py itself is pure stdlib, but data/__init__.py pulls in NeMo."""
        try:
            from promptingnemo.data import normalize
        except ImportError:
            pytest.skip("promptingnemo.data.normalize requires NeMo (via data/__init__.py)")
        assert hasattr(normalize, "normalize_text")
        assert hasattr(normalize, "extract_tags")
        assert hasattr(normalize, "should_keep_line")

    def test_import_data_manifest(self):
        try:
            from promptingnemo.data import manifest
        except ImportError:
            pytest.skip("promptingnemo.data.manifest requires NeMo or scripts package")
        assert hasattr(manifest, "validate_manifests")

    def test_import_data_dataset(self):
        try:
            from promptingnemo.data import dataset
        except ImportError:
            pytest.skip("promptingnemo.data.dataset requires NeMo")
        assert hasattr(dataset, "RobustAudioToBPEDataset")
        assert hasattr(dataset, "patched_speech_collate_fn")

    def test_import_data_sampler(self):
        try:
            from promptingnemo.data import sampler
        except ImportError:
            pytest.skip("promptingnemo.data.sampler requires torch")
        assert hasattr(sampler, "BalancedLanguageBatchSampler")

    def test_import_models_decoder(self):
        try:
            from promptingnemo.models import decoder
        except ImportError:
            pytest.skip("promptingnemo.models.decoder requires NeMo + torch")
        assert hasattr(decoder, "scan_manifest_for_new_tokens")
        assert hasattr(decoder, "extend_decoder_for_new_tokens")

    def test_import_models_ctc_model(self):
        try:
            from promptingnemo.models import ctc_model
        except ImportError:
            pytest.skip("promptingnemo.models.ctc_model requires NeMo")
        assert hasattr(ctc_model, "CustomEncDecCTCModelBPE")

    def test_import_tokenizer_sentencepiece(self):
        try:
            from promptingnemo.tokenizer import sentencepiece
        except ImportError:
            pytest.skip("promptingnemo.tokenizer.sentencepiece requires sentencepiece")
        assert hasattr(sentencepiece, "train_sentencepiece_tokenizer")

    def test_import_tokenizer_aggregate(self):
        try:
            from promptingnemo.tokenizer import aggregate
        except ImportError:
            pytest.skip("promptingnemo.tokenizer.aggregate requires sentencepiece + NeMo")
        assert hasattr(aggregate, "extract_langs_and_special_tokens")
        assert hasattr(aggregate, "train_aggregate_tokenizer")

    def test_import_tokenizer_dedup_aggregate(self):
        try:
            from promptingnemo.tokenizer import dedup_aggregate
        except ImportError:
            pytest.skip("promptingnemo.tokenizer.dedup_aggregate requires NeMo")

    def test_import_training_cli(self):
        """cli.py imports data.dataset at module level, which requires NeMo."""
        try:
            from promptingnemo.training import cli
        except ImportError:
            pytest.skip("promptingnemo.training.cli requires NeMo (via data.dataset)")
        assert hasattr(cli, "parse_args")
        assert hasattr(cli, "main")

    def test_import_training_trainer(self):
        try:
            from promptingnemo.training import trainer
        except ImportError:
            pytest.skip("promptingnemo.training.trainer requires NeMo + lightning")
        assert hasattr(trainer, "train_model")
        assert hasattr(trainer, "save_updated_config")

    def test_import_eval_inference(self):
        try:
            from promptingnemo.eval import inference
        except ImportError:
            pytest.skip("promptingnemo.eval.inference requires NeMo + torch")
        assert hasattr(inference, "transcribe_manifest")

    def test_import_eval_wer(self):
        try:
            from promptingnemo.eval import wer
        except ImportError:
            pytest.skip("promptingnemo.eval.wer requires NeMo")
        assert hasattr(wer, "multi_word_error_rate")
        assert hasattr(wer, "get_clean_transcript")

    def test_import_eval_ner(self):
        try:
            from promptingnemo.eval import ner
        except ImportError:
            pytest.skip("promptingnemo.eval.ner requires numpy + editdistance")
        assert hasattr(ner, "get_ner_scores")

    def test_import_export_to_onnx(self):
        try:
            from promptingnemo.export import to_onnx
        except ImportError:
            pytest.skip("promptingnemo.export.to_onnx requires NeMo + torch")
        assert hasattr(to_onnx, "export_nemo_to_onnx")
        assert hasattr(to_onnx, "extract_metadata_from_nemo")

    def test_import_export_to_hf(self):
        try:
            from promptingnemo.export import to_hf
        except ImportError:
            pytest.skip("promptingnemo.export.to_hf requires huggingface_hub")
        assert hasattr(to_hf, "upload_nemo_to_hf")
