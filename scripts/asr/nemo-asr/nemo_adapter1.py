import os

# Set the GPUs to be used
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # Specify the GPUs you want to use

import logging
import subprocess
import sys
from pathlib import Path
import yaml
import torch
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.utils import model_utils
from nemo.core import adapter_mixins
import sentencepiece as spm
from nemo.utils import exp_manager
import json
import re

def train_sentencepiece_tokenizer(manifest_file, tokenizer_folder, special_tokens=None, vocab_size=5000):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting the tokenizer training process")        

    def read_manifest(manifest_path):
        with open(manifest_path, 'r' , encoding='utf-8') as f:
            lines = f.readlines()
        return [json.loads(line)['text'] for line in lines]
    
    text_data = read_manifest(manifest_file)
    logging.info(f"Extracted {len(text_data)} sentences from the manifest file")

    if not os.path.exists(tokenizer_folder):
        os.makedirs(tokenizer_folder)
    
    temp_text_file = os.path.join(tokenizer_folder, 'text_data.txt')
    with open(temp_text_file, 'w', encoding='utf-8') as f:
        for sentence in text_data:
            f.write(sentence + '\n')

    model_prefix = os.path.join(tokenizer_folder, 'tokenizer')

    if special_tokens:
        user_defined_symbols = ','.join(special_tokens)
        logging.info(f"Special tokens provided: {special_tokens}")
        spm.SentencePieceTrainer.train(
            input=temp_text_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            user_defined_symbols=user_defined_symbols
        )
    else:
        spm.SentencePieceTrainer.train(
            input=temp_text_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size
        )

    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"

    vocab_txt_file = os.path.join(tokenizer_folder, 'vocab.txt')
    with open(vocab_file, 'r', encoding='utf-8') as vf, open(vocab_txt_file, 'w', encoding='utf-8') as vtf:
        for line in vf:
            token = line.split('\t')[0]
            vtf.write(token + '\n')

    return model_file, vocab_file, vocab_txt_file


class ASRModelTrainer:
    def __init__(self, config_path):
        self.load_config(config_path)
        self.cfg = None
        self.model = None
        self.trainer = None

    def load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.model_root = Path(self.config['model']['model_root'])
        self.model_path = self.model_root / self.config['model']['model_name']
        self.tokenizer_dir = self.model_root / self.config['model']['tokenizer_folder']
        self.extended_tokenizer_dir = self.model_root / self.config['model']['new_tokenizer_folder']
        self.data_dir = Path(self.config['training']['data_dir'])
        self.train_manifest = self.data_dir / self.config['training']['train_manifest']
        self.test_manifest = self.data_dir / self.config['training']['test_manifest']
        self.batch_size = self.config['training']['batch_size']
        self.max_steps = self.config['training']['max_steps']
        self.exp_config = exp_manager.ExpManagerConfig(
            #resume_if_exists=True,
            #version = "2025-03-28_14-50-34",
            exp_dir=self.config['experiment']['exp_dir'],
            name=self.config['experiment']['exp_name'],
            checkpoint_callback_params=exp_manager.CallbackParams(
                monitor=self.config['experiment']['monitor'],
                mode=self.config['experiment']['mode'],
                always_save_nemo=self.config['experiment']['always_save_nemo'],
                save_best_model=self.config['experiment']['save_best_model'],
            ),
        )

    def load_and_update_model_config(self):
        self.cfg = ASRModel.restore_from(restore_path=self.model_path, return_config=True)
        self.cfg = self.update_model_config_to_support_adapter(self.cfg)

    def restore_model_with_updated_config(self):
        self.model = ASRModel.restore_from(self.model_path, override_config_path=self.cfg)

    def prepare_data_and_tokens(self, tags_type="auto", tokenizer_state="new", vocab_size=5000):
        taglist = []

        all_tags_path = self.data_dir / "keywords.txt"

        if tags_type == "auto":
            taglist = self.extract_special_tokens_from_manifest()

            with open(all_tags_path, 'w' , encoding='utf-8') as f:
                for tag in sorted(taglist):
                    f.write(tag + '\n')

            logging.info(f"Extracted {len(taglist)} special tokens and saved to {all_tags_path}")

        elif tags_type == "unmapped":
            with open(all_tags_path, 'r', encoding='utf-8') as f:
                taglist = [line.strip() for line in f.readlines()]

        #punctuations = ['.', ',', '?', '!', ';', ':', '-', '(', ')', '[', ']', '{', '}', '<', '>', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '+', '=', '~', '`', '_', '"', "'"]
        tokens = list(taglist) #+ [str(i) for i in range(10)] + punctuations


        if tokenizer_state == "new":
            _ = train_sentencepiece_tokenizer(self.train_manifest, self.extended_tokenizer_dir, special_tokens=taglist, vocab_size=vocab_size)
            self.model.change_vocabulary(self.extended_tokenizer_dir, "bpe")

    def extract_special_tokens_from_manifest(self):
        special_tokens = set()

        with open(self.train_manifest, 'r',  encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text = data['text']

                matches = re.findall(r'\b[A-Z_0-9]+\b', text)

                for match in matches:
                    parts = re.split(r'[_]', match)
                    for part in parts:
                        if part.isalnum() and len(part) > 1:
                            special_tokens.add(part)

        return special_tokens

    def configure_trainer(self):
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

        self.trainer = Trainer(
            devices=[0, 1],  # These will correspond to GPUs 2 and 3
            accelerator=accelerator,
            max_steps=self.max_steps,
            enable_checkpointing=False,
            logger=False,
            log_every_n_steps=50,
            check_val_every_n_epoch=1,
            accumulate_grad_batches=4
        )

        self.model.set_trainer(self.trainer)

    def configure_model_for_training(self):
        with open_dict(self.model.cfg):
            self.model.cfg.train_ds.manifest_filepath = str(self.train_manifest)
            self.model.cfg.train_ds.batch_size = self.batch_size
            self.model.cfg.train_ds.is_tarred = False
            self.model.cfg.train_ds.tarred_audio_filepaths = None
            self.model.cfg.validation_ds.manifest_filepath = str(self.test_manifest)
            self.model.cfg.validation_ds.batch_size = self.batch_size
            self.model.cfg.train_ds.num_workers = 8

        self.model.setup_training_data(self.model.cfg.train_ds)
        self.model.setup_multiple_validation_data(self.model.cfg.validation_ds)

    def prepare_experiment_manager(self):
        os.environ.pop('NEMO_EXPM_VERSION', None)

        exp_config = OmegaConf.structured(self.exp_config)
        exp_manager.exp_manager(self.trainer, exp_config)

    def train(self):
        self.trainer.fit(self.model)

    @staticmethod
    def update_model_config_to_support_adapter(model_cfg):
        with open_dict(model_cfg):
            adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder._target_)
            if adapter_metadata is not None:
                model_cfg.encoder._target_ = adapter_metadata.adapter_class_path
        return model_cfg


# Usage
model_trainer = ASRModelTrainer(config_path='/external1/hkoduri/PromptingNemo/scripts/nemo/asr/config/config.yml')
model_trainer.load_and_update_model_config()
model_trainer.restore_model_with_updated_config()
model_trainer.prepare_data_and_tokens(tags_type="auto", tokenizer_state="new", vocab_size=1704)
model_trainer.configure_trainer()
model_trainer.configure_model_for_training()
model_trainer.prepare_experiment_manager()
model_trainer.train()