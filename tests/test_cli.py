"""Tests for promptingnemo.training.cli.parse_args.

The cli module imports NeMo-dependent modules at the top level,
so we skip all tests if the import fails.
"""

import sys
from unittest.mock import patch

import pytest

try:
    from promptingnemo.training.cli import parse_args
    _HAS_CLI = True
except ImportError:
    _HAS_CLI = False

pytestmark = pytest.mark.skipif(not _HAS_CLI, reason="NeMo or training dependencies not available")


class TestParseArgs:
    def test_default_mode_is_both(self):
        with patch.object(sys, "argv", ["prog"]):
            args = parse_args()
        assert args.mode == "both"

    def test_mode_train(self):
        with patch.object(sys, "argv", ["prog", "--mode", "train"]):
            args = parse_args()
        assert args.mode == "train"

    def test_mode_tokenizer(self):
        with patch.object(sys, "argv", ["prog", "--mode", "tokenizer"]):
            args = parse_args()
        assert args.mode == "tokenizer"

    def test_mode_validate_data(self):
        with patch.object(sys, "argv", ["prog", "--mode", "validate_data"]):
            args = parse_args()
        assert args.mode == "validate_data"

    def test_invalid_mode_raises_error(self):
        with patch.object(sys, "argv", ["prog", "--mode", "invalid"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_config_is_parsed(self):
        with patch.object(sys, "argv", ["prog", "--config", "/path/to/config.yaml"]):
            args = parse_args()
        assert args.config == "/path/to/config.yaml"

    def test_config_defaults_to_none(self):
        with patch.object(sys, "argv", ["prog"]):
            args = parse_args()
        assert args.config is None

    def test_resume_from_is_parsed(self):
        with patch.object(sys, "argv", ["prog", "--resume_from", "/path/to/checkpoint.ckpt"]):
            args = parse_args()
        assert args.resume_from == "/path/to/checkpoint.ckpt"

    def test_resume_from_defaults_to_none(self):
        with patch.object(sys, "argv", ["prog"]):
            args = parse_args()
        assert args.resume_from is None

    def test_no_save_config_flag(self):
        with patch.object(sys, "argv", ["prog", "--no-save-config"]):
            args = parse_args()
        assert args.no_save_config is True

    def test_no_save_config_defaults_to_false(self):
        with patch.object(sys, "argv", ["prog"]):
            args = parse_args()
        assert args.no_save_config is False

    def test_combined_args(self):
        with patch.object(sys, "argv", [
            "prog",
            "--mode", "train",
            "--config", "/my/config.yaml",
            "--resume_from", "/my/checkpoint.ckpt",
            "--no-save-config",
        ]):
            args = parse_args()
        assert args.mode == "train"
        assert args.config == "/my/config.yaml"
        assert args.resume_from == "/my/checkpoint.ckpt"
        assert args.no_save_config is True
