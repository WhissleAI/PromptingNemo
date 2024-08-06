import os
import shutil
from huggingface_hub import HfApi, upload_file, create_repo

# Define your repository ID and local directory
repo_id = "WhissleAI/stt_en_conformer_ctc_slurp_iot"
local_dir = "/external2/karan_exp/experiments/slurp-3adapter"
nemo_model_path = "/external2/karan_exp/experiments/slurp-3adapter-newtokenizer/2024-08-01_17-47-22/ckpt/slurp-3adapter.nemo"

# Set your Hugging Face access token
hf_token = "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"
os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

# Create the local directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

# Copy the .nemo file to the local directory
shutil.copy(nemo_model_path, os.path.join(local_dir, "slurp-3adapter.nemo"))

# Ensure the repository exists on Hugging Face
api = HfApi()
try:
    api.repo_info(repo_id)
except:
    create_repo(repo_id, token=hf_token)

# Create a README file with metadata
readme_text = """
# 1step ASR-NL for Slurp dataset

This is a NeMo ASR model fine-tuned for ASR tasks. It was trained on [dataset name] and achieves [performance metrics]. This model is suitable for [use cases].

## Model Details

- **Model type**: NeMo ASR
- **Architecture**: Conformer CTC
- **Language**: English
- **Training data**: Slurp dataset
- **Performance metrics**: [Metrics]

## Usage

To use this model, you need to install the NeMo library:

```bash
pip install nemo_toolkit
"""

with open(os.path.join(local_dir, "README.md"), "w") as f:
    f.write(readme_text)
    
upload_file(
path_or_fileobj=os.path.join(local_dir, "slurp-3adapter.nemo"),
path_in_repo="slurp-3adapter.nemo",
repo_id=repo_id,
token=hf_token
)

upload_file(
path_or_fileobj=os.path.join(local_dir, "README.md"),
path_in_repo="README.md",
repo_id=repo_id,
token=hf_token
)

print("Files uploaded successfully")