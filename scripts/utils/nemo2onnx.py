from nemo.collections.asr.models import ASRModel
import torch
from pathlib import Path
import os
import tarfile 
import glob
import shutil
import yaml


nemo_model_path = '/external/ksingla/models/hf/stt/stt_pb_conformer_ctc_large.nemo'
model_shelf =  "/external2/artifacts/whissle/model_shelf"
save_directory = Path(model_shelf) / "PB_conformer_ctc_large"

os.makedirs(save_directory)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

nemo_model = ASRModel.restore_from(nemo_model_path, map_location=device)

config = {
    "task": "speech_recognition",
    "sample_rate": 16000,
    "encoder.onnx": "model.onnx",
    "tokenizer.model": "tokenizer/tokenizer.model",
    "onnx.intra_op_num_threads": 1,
    "preprocessor": dict(nemo_model.cfg['preprocessor'])
}

nemo_model.export(save_directory/"model.onnx")
nemo_archive = tarfile.open(nemo_model_path) 
nemo_archive.extractall(save_directory/"extract") 

os.makedirs(save_directory/"tokenizer")
tokenizer_model_path = glob.glob(str(save_directory) + "/extract/*tokenizer.model")[0]
shutil.copy(tokenizer_model_path, save_directory/"tokenizer/tokenizer.model")

shutil.rmtree(save_directory/"extract")

# Convert dictionary to plain text format
config_text = "\n".join(f"{key}={value}" for key, value in config.items()) + "\n"

# Write the plain text to magic.txt
magic_file = open(save_directory / "magic.txt",'w')
magic_file.write(config_text)
magic_file.close()

with open(save_directory/"magic.yaml", 'w+') as f:
    yaml.dump(config,f)