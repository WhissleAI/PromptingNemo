from nemo.utils.exp_manager import exp_manager
from nemo.collections import nlp as nemo_nlp
from nemo.collections import asr as nemo_asr

import os
import wget
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf


DATA_DIR = '/n/disk1/audio_datasets/manifests/'
WORK_DIR = '/n/disk1/1SPI'
MODEL_CONFIG = "punctuation_capitalization_lexical_audio_config.yaml"

# model parameters
TOKENS_IN_BATCH = 1024
MAX_SEQ_LENGTH = 64
LEARNING_RATE = 0.00002


# download the model's configuration file
config_dir = WORK_DIR + '/configs/'
os.makedirs(config_dir, exist_ok=True)
if not os.path.exists(config_dir + MODEL_CONFIG):
    print('Downloading config file...')
    wget.download(f'https://raw.githubusercontent.com/NVIDIA/NeMo/{BRANCH}/examples/nlp/token_classification/conf/' + MODEL_CONFIG, config_dir)
else:
    print ('config file already exists')

# this line will print the entire config of the model
config_path = f'{WORK_DIR}/configs/{MODEL_CONFIG}'
print(config_path)
config = OmegaConf.load(config_path)
print(OmegaConf.to_yaml(config))


print("Trainer config - \n")
print(OmegaConf.to_yaml(config.trainer))

# lets modify some trainer configs
# checks if we have GPU available and uses it
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
config.trainer.devices = 1
config.trainer.accelerator = accelerator
config.trainer.precision = 16 if torch.cuda.is_available() else 32

# For mixed precision training, use precision=16 and amp_level=O1

# Reduces maximum number of epochs to 1 for a quick training
config.trainer.max_epochs = 1

# Remove distributed training flags
config.trainer.strategy = "auto"
config.exp_manager.use_datetime_version=False
config.exp_manager.explicit_log_dir='OneStepSpeechInstructor'

trainer = pl.Trainer(**config.trainer)
print("Trainer", trainer)

exp_dir = exp_manager(trainer, config.get("exp_manager", None))

# the exp_dir provides a path to the current experiment for easy access
exp_dir = str(exp_dir)
exp_dir

# complete list of supported BERT-like models
print(nemo_nlp.modules.get_pretrained_lm_models_list())

PRETRAINED_BERT_MODEL = "bert-base-uncased"

# complete list of supported ASR models
#print(nemo_asr.models.ASRModel.list_available_models())

PRETRAINED_ASR_MODEL = "stt_en_conformer_ctc_large"

# add the specified above model parameters to the config
config.model.language_model.pretrained_model_name = PRETRAINED_BERT_MODEL
config.model.train_ds.tokens_in_batch = TOKENS_IN_BATCH
config.model.train_ds.train_manifest = "/n/disk1/audio_datasets/manifests/train-slurp-tagged.json"
config.model.validation_ds.tokens_in_batch = TOKENS_IN_BATCH
config.model.validation_ds.validation_manifest = "/n/disk1/audio_datasets/manifests/devel-slurp-tagged.json"
config.model.optim.lr = LEARNING_RATE
config.model.audio_encoder.pretrained_model = PRETRAINED_ASR_MODEL
config.model.train_ds.preload_audios = True
config.model.validation_ds.preload_audios = True

# initialize the model
# during this stage, the dataset and data loaders we'll be prepared for training and evaluation
model = nemo_nlp.models.CrossStitchedCTCModel(cfg=config.model, trainer=trainer)

trainer.fit(model)