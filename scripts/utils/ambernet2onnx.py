from nemo.collections.asr.models import ASRModel
import nemo
import torch
from pathlib import Path
import os
import tarfile 
import glob
import shutil


nemo_model_path = '/projects/svanga/asr_models/nemo/langid_ambernet_v1.12.0/ambernet.nemo'

save_directory = Path("ambernet_onnx")

os.makedirs(save_directory, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

nemo_model = ASRModel.restore_from(nemo_model_path, map_location=device)

nemo_model.export(save_directory/"model.onnx")
