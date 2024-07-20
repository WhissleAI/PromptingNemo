import os
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

import os
import json
import logging

def train_sentencepiece_tokenizer(manifest_file, tokenizer_folder, special_tokens=None, vocab_size=5000):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Starting the tokenizer training process")

    # Step 1: Read the manifest file and extract text data
    def read_manifest(manifest_path):
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
        return [json.loads(line)['text'] for line in lines]
    
    logging.info("Reading manifest file")
    text_data = read_manifest(manifest_file)
    logging.info(f"Extracted {len(text_data)} sentences from the manifest file")
    
    # Step 2: Save the extracted text to a temporary file
    if not os.path.exists(tokenizer_folder):
        os.makedirs(tokenizer_folder)
    
    temp_text_file = os.path.join(tokenizer_folder, 'text_data.txt')
    logging.info(f"Saving extracted text to {temp_text_file}")
    with open(temp_text_file, 'w') as f:
        for sentence in text_data:
            f.write(sentence + '\n')
    
    # Step 3: Train the SentencePiece tokenizer with special tokens if provided
    model_prefix = os.path.join(tokenizer_folder, 'tokenizer')
    
    # Prepare special tokens string
    if special_tokens:
        user_defined_symbols = ','.join(special_tokens)
        logging.info(f"Special tokens provided: {special_tokens}")
        logging.info("Starting SentencePiece training with special tokens")
        spm.SentencePieceTrainer.train(
            input=temp_text_file, 
            model_prefix=model_prefix, 
            vocab_size=vocab_size,
            user_defined_symbols=user_defined_symbols
        )
    else:
        logging.info("Starting SentencePiece training without special tokens")
        spm.SentencePieceTrainer.train(
            input=temp_text_file, 
            model_prefix=model_prefix, 
            vocab_size=vocab_size
        )
    
    # Step 4: Return the paths to the tokenizer model and vocab files
    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"

    logging.info(f"Tokenizer training completed")
    logging.info(f"Model file: {model_file}")
    logging.info(f"Vocab file: {vocab_file}")

    # Step 5: Create a vocab.txt file
    vocab_txt_file = os.path.join(tokenizer_folder, 'vocab.txt')
    logging.info(f"Creating vocab.txt file at {vocab_txt_file}")
    with open(vocab_file, 'r') as vf, open(vocab_txt_file, 'w') as vtf:
        for line in vf:
            token = line.split('\t')[0]
            vtf.write(token + '\n')
    
    logging.info(f"vocab.txt file created at {vocab_txt_file}")
    
    return model_file, vocab_file, vocab_txt_file

class ASRModelTrainer:
    def __init__(self, config_path):
        self.load_config(config_path)
        self.cfg = None
        self.model = None
        self.trainer = None

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_root = Path(self.config['model']['model_root'])
        self.model_path = self.model_root / self.config['model']['model_name']
        self.tokenizer_dir = self.model_root / self.config['model']['tokenizer_folder']
        self.extended_tokenizer_dir = self.model_root / self.config['model']['new_tokenizer_folder']
        self.tokenizer_model_file = self.tokenizer_dir / 'tokenizer.model'
        self.vocab_file = self.tokenizer_dir / 'tokenizer.vocab'
        self.vocab_txt_file = self.tokenizer_dir / 'vocab.txt'
        self.proto_file = self.config['model']['proto_file']
        self.proto_dir = self.config['model']['proto_dir']
        self.data_dir = Path(self.config['training']['data_dir'])
        self.train_manifest = self.data_dir / self.config['training']['train_manifest']
        self.test_manifest = self.data_dir / self.config['training']['test_manifest']
        self.batch_size = self.config['training']['batch_size']
        self.max_steps = self.config['training']['max_steps']
        self.adapter_cfg = LinearAdapterConfig(
            in_features=None,  # This will be set later based on the model configuration
            dim=self.config['adapter']['dim'],
            activation=self.config['adapter']['activation'],
            norm_position=self.config['adapter']['norm_position'],
        )
        self.exp_config = exp_manager.ExpManagerConfig(
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
        print(self.cfg)
    
    def restore_model_with_updated_config(self):
        self.model = ASRModel.restore_from(self.model_path, override_config_path=self.cfg)
    
    def prepare_data_and_tokens(self, tokenizer_state="extended", vocab_size=2000):
        taglist = []
        all_tags_path = os.path.join(self.data_dir, "alltags.txt")
        with open(all_tags_path, 'r') as f:
            for line in f:
                word, tag = line.split()
                taglist.append(tag)

        punctuations = ['.', ',', '?', '!', ';', ':', '-', '(', ')', '[', ']', '{', '}', '<', '>', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '+', '=', '~', '`', '_', '"', "'"]
        tokens = taglist + [str(i) for i in range(10)] + punctuations
        is_userdefined = True

        if tokenizer_state == "extended":
            self.edit_spt_model(self.tokenizer_model_file, self.extended_tokenizer_dir, tokens, self.vocab_file, self.vocab_txt_file, is_userdefined)
        else:
            _ = train_sentencepiece_tokenizer(self.train_manifest, self.extended_tokenizer_dir, special_tokens=taglist, vocab_size=vocab_size)
        
        self.model.change_vocabulary(self.extended_tokenizer_dir, "bpe")
    
    def configure_trainer(self):
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

        self.trainer = Trainer(
            devices=1, 
            accelerator=accelerator, 
            max_steps=self.max_steps,
            enable_checkpointing=False, 
            logger=False,
            log_every_n_steps=50, 
            check_val_every_n_epoch=1
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
            self.model.cfg.train_ds.num_workers = 8  # Adding num_workers for training dataloader

        self.model.setup_training_data(self.model.cfg.train_ds)
        self.model.setup_multiple_validation_data(self.model.cfg.validation_ds)
        self.model.setup_multiple_test_data(self.model.cfg.validation_ds)

    def configure_spec_augmentation(self):
        with open_dict(self.model.cfg):
            self.model.cfg.spec_augment.freq_masks = self.model.cfg.spec_augment.freq_masks
            self.model.cfg.spec_augment.freq_width = self.model.cfg.spec_augment.freq_width
            self.model.cfg.spec_augment.time_masks = self.model.cfg.spec_augment.time_masks
            self.model.cfg.spec_augment.time_width = self.model.cfg.spec_augment.time_width

        self.model.spec_augmentation = self.model.from_config_dict(self.model.cfg.spec_augment)

    def configure_optimization(self):
        if 'optim' in self.model.cfg:
            print(OmegaConf.to_yaml(self.model.cfg.optim))

        with open_dict(self.model.cfg):
            self.model.cfg.optim.lr = 0.1
            self.model.cfg.optim.weight_decay = 0.001
            self.model.cfg.optim.sched.warmup_steps = 1000

        self.model.setup_optimization(self.model.cfg.optim)
    
    def setup_adapters(self):
        if hasattr(self.model, 'adapter_module_names'):
            print(self.model.adapter_module_names)

        for module in self.model.children():
            if hasattr(module, 'get_accepted_adapter_types'):
                types = module.get_accepted_adapter_types()
                print("Module:", module.__class__.__name__)
                for tp in types:
                    print(tp)
                print()

        self.adapter_cfg.in_features = self.model.cfg.encoder.d_model  # Set in_features based on model configuration

        print(self.adapter_cfg)

        self.model.add_adapter(name=self.config['adapter']['name'], cfg=self.adapter_cfg)
        self.model.set_enabled_adapters(enabled=False)  # Disable all adapters
        self.model.set_enabled_adapters(name=self.config['adapter']['name'], enabled=True)  # Enable only the current adapter we want to train
        self.model.freeze()
        self.model.unfreeze_enabled_adapters()
        self.model.decoder.unfreeze()
        self.model.summarize()

    def prepare_experiment_manager(self):
        # Environment variable generally used for multi-node multi-gpu training.
        # In notebook environments, this flag is unnecessary and can cause logs of multiple training runs to overwrite each other.
        os.environ.pop('NEMO_EXPM_VERSION', None)

        exp_config = OmegaConf.structured(self.exp_config)

        logdir = exp_manager.exp_manager(self.trainer, exp_config)
        print(f"Experiment log directory: {logdir}")

    def train(self):
        self.trainer.fit(self.model)

    def summarize_model(self):
        self.model.summarize()

    @staticmethod
    def update_model_config_to_support_adapter(model_cfg):
        with open_dict(model_cfg):
            adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder._target_)
            if adapter_metadata is not None:
                model_cfg.encoder._target_ = adapter_metadata.adapter_class_path
        print("Updated encoder _target_ model:", model_cfg.encoder._target_)
        return model_cfg

    @staticmethod
    def generate_sentencepiece_model_pb2(script_dir, proto_file_path):
        command = ['protoc', f'--python_out={script_dir}', proto_file_path]
        try:
            subprocess.run(command, check=True)
            print("Successfully generated sentencepiece_model_pb2.py")
        except subprocess.CalledProcessError as e:
            print(f"Error generating sentencepiece_model_pb2.py: {e}")
            sys.exit(1)

    @staticmethod
    def edit_spt_model(tokenizer_model_file, extended_tokenizer_dir, tokens, vocab_file, vocab_txt_file, is_userdefined=False):
        from sentencepiece_model_pb2 import ModelProto  # Ensure this import is after the proto generation
        if not os.path.exists(extended_tokenizer_dir):
            os.makedirs(extended_tokenizer_dir)
        
        output_model_file = os.path.join(extended_tokenizer_dir, 'tokenizer.model')
        output_vocab_file = os.path.join(extended_tokenizer_dir, 'tokenizer.vocab')
        output_vocab_txt_file = os.path.join(extended_tokenizer_dir, 'vocab.txt')
        token_type = 3 if not is_userdefined else 4

        model = ModelProto()
        model.ParseFromString(open(tokenizer_model_file, 'rb').read())
        existing_tokens = {piece.piece for piece in model.pieces}
        new_tokens = []

        for token in tokens:
            if token in existing_tokens:
                logging.warning(f"Special Token '{token}' already exists in the input model, skipping.")
                continue
            piece = model.SentencePiece(piece=token, score=0.0, type=token_type)
            model.pieces.append(piece)
            new_tokens.append(token)

        sp = spm.SentencePieceProcessor()
        try:
            sp.LoadFromSerializedProto(model.SerializeToString())
            for token in new_tokens:
                id = sp.piece_to_id(token)
                logging.info(f"Created token '{token}' at ID {id}")
            logging.info(f"New tokenizer vocab size: {sp.get_piece_size()}")
        except:
            logging.error("Could not appropriately configure new tokenizer. Verify if the special tokens already exist.")
            sys.exit(1)

        with open(output_model_file, 'wb') as outf:
            outf.write(model.SerializeToString())
        logging.info(f"Created new tokenizer at: {output_model_file}")

        # Read the original vocab file and append the new tokens
        with open(vocab_file, 'r') as original_vocab_file:
            original_vocab = original_vocab_file.readlines()
        with open(output_vocab_file, 'w') as updated_vocab_file:
            updated_vocab_file.writelines(original_vocab)
            for token in new_tokens:
                updated_vocab_file.write(f"{token}\n")

        # Update vocab.txt
        with open(vocab_txt_file, 'r') as original_vocab_txt_file:
            original_vocab_txt = original_vocab_txt_file.readlines()
        with open(output_vocab_txt_file, 'w') as updated_vocab_txt_file:
            updated_vocab_txt_file.writelines(original_vocab_txt)
            for token in new_tokens:
                updated_vocab_txt_file.write(f"{token}\n")

        logging.info(f"Updated vocab files: {output_vocab_file}, {output_vocab_txt_file}")

# Usage
model_trainer = ASRModelTrainer(config_path='config.yml')
model_trainer.load_and_update_model_config()
model_trainer.restore_model_with_updated_config()
model_trainer.prepare_data_and_tokens(tokenizer_state="new", vocab_size=2500)
model_trainer.configure_trainer()
model_trainer.configure_model_for_training()
model_trainer.configure_spec_augmentation()
model_trainer.configure_optimization()
#model_trainer.setup_adapters()
model_trainer.prepare_experiment_manager()
model_trainer.summarize_model()
model_trainer.train()
