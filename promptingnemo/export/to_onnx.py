"""Export a NeMo ASR model to ONNX format compatible with Whissle API.

Produces the directory structure expected by decoder_onnx (api.whissle.ai):
    config.json          — preprocessor configuration
    vocabulary.json      — vocabulary, blank_id, tokenizer metadata
    model.onnx           — the exported ONNX model
    tokenizer.model      — SentencePiece model (single-language)
    tokenizer_LANG.model — per-language SentencePiece models (aggregate)

Ported from decoder_onnx/src/export.py to keep PromptingNemo self-contained.
"""

import io
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Optional


def export_nemo_to_onnx(
    nemo_model_path: str,
    output_dir: str,
    opset_version: int = 17,
    use_cpu: bool = False,
    half_precision: bool = False,
    metadata_only: bool = False,
) -> Path:
    """Export a .nemo checkpoint to ONNX with all metadata for Whissle API.

    Args:
        nemo_model_path: Path to the .nemo model checkpoint.
        output_dir: Directory where ONNX model, tokenizer, and config are saved.
        opset_version: ONNX opset version (default 17).
        use_cpu: Force CPU for export (lower memory).
        half_precision: Export in FP16 (requires GPU).
        metadata_only: Only extract tokenizer and config, skip ONNX export.

    Returns:
        Path to the output directory.
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cfg = extract_metadata_from_nemo(nemo_model_path, output_dir)

    if metadata_only:
        print(f"Metadata extracted to {output_dir} (--metadata-only, no ONNX export).")
        return Path(output_dir)

    if cfg is None:
        raise RuntimeError(f"Could not extract config from {nemo_model_path}")

    import torch
    import nemo.collections.asr as nemo_asr

    if use_cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    load_path = _patch_nemo_config_compat(nemo_model_path)

    model_target = cfg.get("target", "")
    try:
        if "EncDecCTCModelBPE" in model_target:
            model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
                load_path, map_location=torch.device(device)
            )
        else:
            model = nemo_asr.models.EncDecCTCModel.restore_from(
                load_path, map_location=torch.device(device)
            )
    except Exception as e:
        raise RuntimeError(f"Failed to load NeMo model: {e}") from e

    model.eval()
    model = model.to(device)

    if half_precision and device == "cuda":
        model = model.half()

    onnx_path = os.path.join(output_dir, "model.onnx")
    model.freeze()
    model.export(
        onnx_path,
        onnx_opset_version=opset_version,
        verbose=False,
        check_trace=False,
    )

    print(f"\nExport completed. Output files in {output_dir}:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        if size > 1024 * 1024:
            print(f"  {f} ({size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  {f} ({size / 1024:.1f} KB)")

    return Path(output_dir)


def extract_metadata_from_nemo(nemo_model_path: str, output_dir: str) -> Optional[dict]:
    """Extract tokenizer, vocabulary, and preprocessor config from a .nemo archive.

    This does not load the full model, so it works on machines without a GPU.

    Args:
        nemo_model_path: Path to .nemo model file.
        output_dir: Directory for extracted metadata files.

    Returns:
        The model config dict, or None if no config was found.
    """
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(nemo_model_path, "r") as tar:
            tar.extractall(tmpdir)

        config_file = None
        for root, _dirs, files in os.walk(tmpdir):
            for f in files:
                if f.endswith(".yaml"):
                    config_file = os.path.join(root, f)
                    break
            if config_file:
                break

        if not config_file:
            print("WARNING: No config YAML found in .nemo archive")
            return None

        cfg = _load_config(config_file)
        if cfg is None:
            return None

        _save_preprocessor_config(cfg, output_dir)
        _extract_tokenizers(cfg, tmpdir, output_dir)

    return cfg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config(config_file: str) -> Optional[dict]:
    """Load YAML config, handling NeMo's python-specific tags."""
    try:
        from ruamel.yaml import YAML
        yaml = YAML(typ="safe")
        with open(config_file, "r") as f:
            return yaml.load(f)
    except Exception:
        pass
    # NeMo configs use !!python/object/apply tags (e.g. pathlib.PosixPath).
    # Safe here because the YAML comes from a trusted .nemo archive.
    import yaml
    with open(config_file, "r") as f:
        return yaml.unsafe_load(f)


def _save_preprocessor_config(cfg: dict, output_dir: str) -> None:
    """Write config.json with preprocessor parameters."""
    p = cfg.get("preprocessor", {})
    sample_rate = p.get("sample_rate", 16000)
    window_size = p.get("window_size", 0.02)
    window_stride = p.get("window_stride", 0.01)

    config = {
        "model_type": cfg.get("target", "unknown"),
        "preprocessor": {
            "sample_rate": sample_rate,
            "window_size": window_size,
            "window_stride": window_stride,
            "win_length": int(window_size * sample_rate),
            "hop_length": int(window_stride * sample_rate),
            "n_fft": p.get("n_fft", 512),
            "features": p.get("features", 80),
            "lowfreq": p.get("lowfreq", 0),
            "highfreq": p.get("highfreq", None),
            "preemph": p.get("preemph", 0.97),
            "log": p.get("log", True),
            "log_zero_guard_type": p.get("log_zero_guard_type", "add"),
            "log_zero_guard_value": float(p.get("log_zero_guard_value", 2**-24)),
            "normalize": p.get("normalize", "per_feature"),
            "pad_to": p.get("pad_to", 16),
            "dither": 0.0,
        },
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def _extract_tokenizers(cfg: dict, tmpdir: str, output_dir: str) -> None:
    """Extract tokenizer files and build vocabulary.json."""
    vocab_data: dict = {"vocabulary": [], "tokenizer_type": "bpe"}
    lang_offsets: dict = {}
    offset = 0

    tokenizer_cfg = cfg.get("tokenizer", {})

    if tokenizer_cfg.get("type") == "agg" and "langs" in tokenizer_cfg:
        _extract_aggregate_tokenizer(
            tokenizer_cfg, tmpdir, output_dir, vocab_data, lang_offsets, offset
        )
    else:
        _extract_single_tokenizer(cfg, tmpdir, output_dir, vocab_data)

    # Decoder vocabulary override (e.g. char-level models with explicit vocab)
    decoder_cfg = cfg.get("decoder", {})
    decoder_vocab = decoder_cfg.get("vocabulary", [])
    decoder_num_classes = decoder_cfg.get("num_classes", 0)

    if decoder_vocab and len(decoder_vocab) == decoder_num_classes:
        vocab_data = {
            "vocabulary": decoder_vocab,
            "tokenizer_type": "decoder_vocab",
            "is_aggregate": False,
            "use_decoder_vocab": True,
        }

    vocab_size = len(vocab_data.get("vocabulary", []))
    if vocab_size > 0:
        vocab_data["blank_id"] = vocab_size

    with open(os.path.join(output_dir, "vocabulary.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)


def _extract_aggregate_tokenizer(
    tokenizer_cfg: dict,
    tmpdir: str,
    output_dir: str,
    vocab_data: dict,
    lang_offsets: dict,
    offset: int,
) -> None:
    """Handle aggregate (multi-language) tokenizer extraction."""
    langs_cfg = tokenizer_cfg["langs"]
    vocab_data["is_aggregate"] = True
    vocab_data["tokenizer_type"] = "aggregate_bpe"
    vocab_data["langs"] = list(langs_cfg.keys())

    try:
        import sentencepiece as spm
    except ImportError:
        print("WARNING: sentencepiece not installed, cannot extract aggregate tokenizer")
        return

    for lang, lang_cfg in langs_cfg.items():
        model_path = lang_cfg.get("model_path", "")
        if model_path.startswith("nemo:"):
            model_filename = model_path[5:]
            src = os.path.join(tmpdir, model_filename)

            if os.path.exists(src):
                sp = spm.SentencePieceProcessor()
                sp.Load(src)
                vocab_size = sp.get_piece_size()

                for i in range(vocab_size):
                    vocab_data["vocabulary"].append(sp.id_to_piece(i))

                lang_offsets[lang] = {"offset": offset, "size": vocab_size}
                offset += vocab_size

                dest = os.path.join(output_dir, f"tokenizer_{lang}.model")
                shutil.copy(src, dest)

    vocab_data["lang_offsets"] = lang_offsets


def _extract_single_tokenizer(
    cfg: dict, tmpdir: str, output_dir: str, vocab_data: dict
) -> None:
    """Handle single-language or character-based tokenizer extraction."""
    tokenizer_files = []
    for root, _dirs, files in os.walk(tmpdir):
        for f in files:
            if f.endswith(".model"):
                tokenizer_files.append(os.path.join(root, f))

    if len(tokenizer_files) >= 1:
        src = tokenizer_files[0]
        dest = os.path.join(output_dir, "tokenizer.model")
        shutil.copy(src, dest)
        vocab_data["is_aggregate"] = False

        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.Load(src)
            for i in range(sp.get_piece_size()):
                vocab_data["vocabulary"].append(sp.id_to_piece(i))
        except ImportError:
            pass
    else:
        decoder_cfg = cfg.get("decoder", {})
        labels = cfg.get("labels", decoder_cfg.get("vocabulary", []))
        if labels:
            vocab_data["vocabulary"] = labels
            vocab_data["tokenizer_type"] = "char"
            vocab_data["is_aggregate"] = False


def _patch_nemo_config_compat(nemo_model_path: str) -> str:
    """Patch .nemo config for NeMo 2.x compatibility.

    NeMo 1.x configs contain ``measure_cfg`` and ``DEPRECATED`` values
    that cause errors in NeMo 2.x. This repacks the archive with those
    keys stripped. Returns the original path if no patching is needed.
    """
    import yaml as _yaml

    with tarfile.open(nemo_model_path, "r") as tar:
        members = tar.getnames()
        yaml_name = next((m for m in members if m.endswith(".yaml")), None)
        if not yaml_name:
            return nemo_model_path
        f = tar.extractfile(yaml_name)
        if f is None:
            return nemo_model_path
        raw = f.read().decode("utf-8")

    if "measure_cfg" not in raw:
        return nemo_model_path

    cfg = _yaml.unsafe_load(raw)

    def _strip_deprecated(d):
        if isinstance(d, dict):
            d.pop("measure_cfg", None)
            for k in [k for k, v in d.items() if v == "DEPRECATED"]:
                d.pop(k)
            for v in d.values():
                _strip_deprecated(v)
        elif isinstance(d, list):
            for v in d:
                _strip_deprecated(v)

    _strip_deprecated(cfg)

    patched_dir = tempfile.mkdtemp()
    patched_path = os.path.join(patched_dir, "model_patched.nemo")

    with tarfile.open(nemo_model_path, "r") as src, \
         tarfile.open(patched_path, "w") as dst:
        for member in src.getmembers():
            if member.name == yaml_name:
                patched_yaml = _yaml.dump(
                    cfg, default_flow_style=False, allow_unicode=True
                )
                data = patched_yaml.encode("utf-8")
                member.size = len(data)
                dst.addfile(member, io.BytesIO(data))
            else:
                dst.addfile(member, src.extractfile(member))

    return patched_path


def main():
    """CLI entry point for ONNX export."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export NeMo ASR model to ONNX (Whissle API compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    # Full export
    python -m promptingnemo.export.to_onnx --nemo-model model.nemo --output-dir ./exported

    # Metadata only (no GPU needed)
    python -m promptingnemo.export.to_onnx --nemo-model model.nemo --output-dir ./exported --metadata-only

    # CPU export (lower memory)
    python -m promptingnemo.export.to_onnx --nemo-model model.nemo --output-dir ./exported --cpu
""",
    )
    parser.add_argument(
        "--nemo-model", "--nemo_model", required=True, help="Path to .nemo model file"
    )
    parser.add_argument(
        "--output-dir", "--output_dir", default="./exported", help="Output directory"
    )
    parser.add_argument(
        "--opset-version", type=int, default=17, help="ONNX opset version (default: 17)"
    )
    parser.add_argument("--cpu", action="store_true", help="Export on CPU")
    parser.add_argument(
        "--fp16", action="store_true", help="Export in FP16 (requires GPU)"
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only extract tokenizer and config (no ONNX export)",
    )

    args = parser.parse_args()
    export_nemo_to_onnx(
        nemo_model_path=args.nemo_model,
        output_dir=args.output_dir,
        opset_version=args.opset_version,
        use_cpu=args.cpu,
        half_precision=args.fp16,
        metadata_only=args.metadata_only,
    )


if __name__ == "__main__":
    main()
