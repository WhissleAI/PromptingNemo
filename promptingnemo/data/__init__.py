"""Data loading and processing utilities for PromptingNemo."""

from promptingnemo.data.normalize import normalize_text, extract_tags, should_keep_line


def __getattr__(name):
    if name == "RobustAudioToBPEDataset":
        try:
            from promptingnemo.data.dataset import RobustAudioToBPEDataset
            return RobustAudioToBPEDataset
        except ImportError:
            raise ImportError(
                f"{name} requires NeMo. Install with: pip install promptingnemo[train]"
            ) from None
    if name == "patched_speech_collate_fn":
        try:
            from promptingnemo.data.dataset import patched_speech_collate_fn
            return patched_speech_collate_fn
        except ImportError:
            raise ImportError(
                f"{name} requires NeMo. Install with: pip install promptingnemo[train]"
            ) from None
    if name == "BalancedLanguageBatchSampler":
        try:
            from promptingnemo.data.sampler import BalancedLanguageBatchSampler
            return BalancedLanguageBatchSampler
        except ImportError:
            raise ImportError(
                f"{name} requires NeMo. Install with: pip install promptingnemo[train]"
            ) from None
    if name == "validate_manifests":
        try:
            from promptingnemo.data.manifest import validate_manifests
            return validate_manifests
        except ImportError:
            raise ImportError(
                f"{name} requires NeMo. Install with: pip install promptingnemo[train]"
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
