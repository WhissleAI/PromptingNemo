"""Model export utilities for PromptingNemo.

Export NeMo ASR models to ONNX format compatible with Whissle API (api.whissle.ai).
The exported directory contains config.json, vocabulary.json, tokenizer files,
and model.onnx — ready for use with the decoder_onnx inference engine.
"""

from promptingnemo.export.to_onnx import (
    export_nemo_to_onnx,
    extract_metadata_from_nemo,
)

__all__ = ["export_nemo_to_onnx", "extract_metadata_from_nemo"]
