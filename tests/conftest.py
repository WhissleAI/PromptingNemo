"""Shared pytest fixtures for the PromptingNemo test suite."""

import json
from pathlib import Path

import pytest
import yaml


TEST_DATA_DIR = Path(__file__).parent / "test_data"


@pytest.fixture
def sample_manifest_path():
    """Path to the clean sample manifest (5 valid JSONL lines)."""
    return str(TEST_DATA_DIR / "sample_manifest.json")


@pytest.fixture
def dirty_manifest_path():
    """Path to the dirty sample manifest with tag typos and formatting issues."""
    return str(TEST_DATA_DIR / "sample_manifest_dirty.json")


@pytest.fixture
def sample_config_path():
    """Path to the minimal training config YAML."""
    return str(TEST_DATA_DIR / "sample_config.yaml")


@pytest.fixture
def sample_config(sample_config_path):
    """Loaded config dict from the sample YAML."""
    with open(sample_config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def manifest_lines(sample_manifest_path):
    """Parsed list of dicts from the clean manifest."""
    with open(sample_manifest_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def dirty_manifest_lines(dirty_manifest_path):
    """Parsed list of dicts from the dirty manifest."""
    with open(dirty_manifest_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
