import os
import shutil
import sys

from huggingface_hub import HfApi, upload_file, create_repo

# Define your repository ID and local directory
repo_id = "WhissleAI/speech-tagger_hi_ctc_meta"
local_dir = "/projects/whissle/experiments/punjabi-hf"
nemo_model_path = "/projects/whissle/experiments/punjabi_adapter-ai4bharat/2024-10-16_17-51-06/checkpoints/punjabi_adapter-ai4bharat.nemo"

# Set your Hugging Face access token
hf_token = sys.argv[1]
os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

# Create the local directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

# Copy the .nemo file to the local directory
shutil.copy(nemo_model_path, os.path.join(local_dir, "punjabi_adapter-ai4bharat.nemo"))

# Ensure the repository exists on Hugging Face
api = HfApi()
try:
    api.repo_info(repo_id)
except:
    create_repo(repo_id, token=hf_token)

# Create a README file with metadata
readme_text = """
# This speech tagger performs transcription for Punjabi, annotates key entities, predict speaker age, dialiect and intent.

Model is suitable for voiceAI applications, real-time and offline.

## Model Details

- **Model type**: NeMo ASR
- **Architecture**: Conformer CTC
- **Language**: Punjabi
- **Training data**: AI4Bharat IndicVoices Punjabi V1 and V2 dataset
- **Performance metrics**: [Metrics]

## Usage

To use this model, you need to install the NeMo library:

```bash
pip install nemo_toolkit
```

### How to run

```python
import nemo.collections.asr as nemo_asr

# Step 1: Load the ASR model from Hugging Face
model_name = 'WhissleAI/stt_pa_conformer_ctc_entities_age_dialiect_intent'
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name)

# Step 2: Provide the path to your audio file
audio_file_path = '/path/to/your/audio_file.wav'

# Step 3: Transcribe the audio
transcription = asr_model.transcribe(paths2audio_files=[audio_file_path])
print(f'Transcription: {transcription[0]}')
```

Dataset is from AI4Bharat IndicVoices Hindi V1 and V2 dataset.

https://indicvoices.ai4bharat.org/

"""

with open(os.path.join(local_dir, "README.md"), "w") as f:
    f.write(readme_text)
    
upload_file(
path_or_fileobj=os.path.join(local_dir, "punjabi_adapter-ai4bharat.nemo"),
path_in_repo="punjabi_adapter-ai4bharat.nemo",
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