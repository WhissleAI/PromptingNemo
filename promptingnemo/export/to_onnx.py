"""Export a NeMo ASR model to ONNX format with tokenizer and config.

Extracted from scripts/utils/nemo2onnx.py.
"""

import glob
import os
import shutil
import tarfile
from pathlib import Path

import torch
import yaml
from nemo.collections.asr.models import ASRModel


def export_nemo_to_onnx(
    nemo_model_path: str,
    save_directory: str,
    device: str = None,
) -> Path:
    """Export a .nemo checkpoint to ONNX with tokenizer and magic config files.

    Args:
        nemo_model_path: Path to the .nemo model checkpoint.
        save_directory: Directory where ONNX model, tokenizer, and config will be saved.
        device: 'cuda' or 'cpu'. If None, auto-detected.

    Returns:
        Path to the save directory.
    """
    save_directory = Path(save_directory)
    os.makedirs(save_directory, exist_ok=True)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    nemo_model = ASRModel.restore_from(nemo_model_path, map_location=device)

    config = {
        "task": "speech_recognition",
        "sample_rate": 16000,
        "encoder.onnx": "model.onnx",
        "tokenizer.model": "tokenizer/tokenizer.model",
        "onnx.intra_op_num_threads": 1,
        "preprocessor": dict(nemo_model.cfg['preprocessor']),
    }

    nemo_model.export(save_directory / "model.onnx")

    nemo_archive = tarfile.open(nemo_model_path)
    nemo_archive.extractall(save_directory / "extract")

    os.makedirs(save_directory / "tokenizer", exist_ok=True)
    tokenizer_model_path = glob.glob(str(save_directory) + "/extract/*tokenizer.model")[0]
    shutil.copy(tokenizer_model_path, save_directory / "tokenizer/tokenizer.model")

    shutil.rmtree(save_directory / "extract")

    # Write plain text config
    config_text = "\n".join(f"{key}={value}" for key, value in config.items()) + "\n"
    magic_file = open(save_directory / "magic.txt", 'w')
    magic_file.write(config_text)
    magic_file.close()

    # Write YAML config
    with open(save_directory / "magic.yaml", 'w+') as f:
        yaml.dump(config, f)

    return save_directory
