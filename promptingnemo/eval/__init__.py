"""Evaluation utilities for PromptingNemo."""


def __getattr__(name):
    if name == "AVWordErrorRate":
        try:
            from promptingnemo.eval.av_wer import AVWordErrorRate
            return AVWordErrorRate
        except ImportError:
            raise ImportError(
                f"{name} requires torchmetrics. Install with: pip install promptingnemo[train]"
            ) from None
    if name == "separate_labels_from_text":
        try:
            from promptingnemo.eval.av_wer import separate_labels_from_text
            return separate_labels_from_text
        except ImportError:
            raise ImportError(
                f"{name} requires torchmetrics. Install with: pip install promptingnemo[train]"
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
