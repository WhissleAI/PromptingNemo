"""ASR model definitions for PromptingNemo."""


def __getattr__(name):
    if name == "CustomEncDecCTCModelBPE":
        try:
            from promptingnemo.models.ctc_model import CustomEncDecCTCModelBPE
            return CustomEncDecCTCModelBPE
        except ImportError:
            raise ImportError(
                f"{name} requires NeMo. Install with: pip install promptingnemo[train]"
            ) from None
    if name == "AVEncDecCTCModelBPE":
        try:
            from promptingnemo.models.av_ctc_model import AVEncDecCTCModelBPE
            return AVEncDecCTCModelBPE
        except ImportError:
            raise ImportError(
                f"{name} requires NeMo. Install with: pip install promptingnemo[train]"
            ) from None
    _decoder_names = {
        "scan_manifest_for_new_tokens",
        "extend_decoder_for_new_tokens",
        "slim_decoder_for_training",
        "scale_down_tag_decoder_weights",
    }
    if name in _decoder_names:
        try:
            from promptingnemo.models import decoder
            return getattr(decoder, name)
        except ImportError:
            raise ImportError(
                f"{name} requires NeMo. Install with: pip install promptingnemo[train]"
            ) from None
    if name == "TextCTCTagger":
        from promptingnemo.models.text_ctc_model import TextCTCTagger
        return TextCTCTagger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
