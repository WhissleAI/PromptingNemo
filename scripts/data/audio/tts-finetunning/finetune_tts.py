from huggingface_hub import hf_hub_download
import zipfile
import os

repo_id = "WhissleAI/emotion-tagged-audio-data"
filename = "emotion_data.zip"

zip_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset"
)

extract_dir = "/content/data"
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Data extracted to:", extract_dir)

import os, tempfile

os.makedirs('tmp_exec', exist_ok=True)
tempfile.tempdir = os.path.join(os.getcwd(), 'tmp_exec')

# !pip install torch torchvision torchaudio transformers datasets librosa soundfile accelerate wandb phonemizer TTS -q
# !apt-get install -y -q espeak espeak-data libespeak1 libespeak-dev

# CELL 1: Installations
# We will NOT specify versions for torch, torchvision, torchaudio, numpy, pandas, scipy initially.
# Let TTS and transformers guide these.
# !pip install transformers datasets librosa soundfile accelerate wandb phonemizer TTS -q

# # Install espeak for phonemizer backend
# !apt-get update -q
# !apt-get install -y -q espeak-data libespeak-dev # espeak itself should be a dependency

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import json
from pathlib import Path
import random
from typing import List, Dict, Tuple, Optional

# Set device
device = torch.device('cuda')
print(f"Using device: {device}")

# !apt-get update -q
# !apt-get install -y festival festvox-kallpc16k
# !apt-get install -y espeak espeak-data libespeak1 libespeak-dev

# Set environment variable to help with espeak memory issues
import os
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = '/usr/lib/x86_64-linux-gnu/libespeak.so.1'

print("Phonemizer backends installed successfully!")

# Phonemizer specific imports
from phonemizer import phonemize
from phonemizer.separator import Separator

class EmotionTTSDataset(Dataset):
    def __init__(self, data_file: str, audio_dir: str, sample_rate: int = 22050,
                 mel_channels: int = 80, max_audio_length: int = 8*22050):
        """
        Dataset for emotion-controlled TTS training

        Args:
            data_file: Path to your processed data file (CSV/TSV format)
            audio_dir: Directory containing audio files
            sample_rate: Target sample rate
            mel_channels: Number of mel channels
            max_audio_length: Maximum audio length in samples
        """
        # Read data and handle the header issue
        # na_filter=False to treat empty strings as "" not NaN, important for text column
        try:
            self.data = pd.read_csv(data_file, sep='|', header=None, na_filter=False,
                                    names=['audio_file', 'text', 'speaker', 'emotion'], # Directly assign column names
                                    dtype={'audio_file': str, 'text': str, 'speaker': str, 'emotion': str}) # Ensure all are strings
        except pd.errors.EmptyDataError:
            print(f"Error: The file {data_file} is empty or not found.")
            self.data = pd.DataFrame(columns=['audio_file', 'text', 'speaker', 'emotion']) # Create empty dataframe
            # Consider raising an error or exiting if data is critical
            # raise FileNotFoundError(f"Data file {data_file} is empty or not found.")


        # Remove header row if it exists (check if first row contains 'audio_file' or typical header names)
        if not self.data.empty and 'audio_file' in str(self.data.iloc[0]['audio_file']).lower(): # Check first row robustly
            print(f"Detected header row in data: {self.data.iloc[0].to_dict()}. Removing it.")
            self.data = self.data.drop(0).reset_index(drop=True)

        # Filter out any rows with invalid emotions or speakers (like 'emotion_name', 'speaker_name')
        # These should be your actual valid emotion and speaker codes from the dataset
        valid_emotions = ['ANG', 'HAP', 'NEU', 'SAD']  # Adjust with your actual emotions
        # Example speaker IDs, adjust as needed. Assumes speakers like 'm0001', 'f0001'
        known_speakers = self.data['speaker'].unique().tolist() # Get all speakers from data first
        # You might want to define valid_speakers explicitly if you know them beforehand
        # valid_speakers = [f'm{i:04d}' for i in range(1, 11)] + [f'f{i:04d}' for i in range(1, 11)] # Example

        if not self.data.empty:
            self.data = self.data[
                (self.data['emotion'].isin(valid_emotions)) &
                (self.data['speaker'].isin(known_speakers)) # Or use your predefined valid_speakers
            ].reset_index(drop=True)
        else:
            print("Warning: Dataframe was empty before filtering.")


        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.mel_channels = mel_channels
        self.max_audio_length = max_audio_length

        # Encode emotions and speakers
        self.emotion_encoder = LabelEncoder()
        self.speaker_encoder = LabelEncoder()

        if not self.data.empty:
            self.data['emotion_id'] = self.emotion_encoder.fit_transform(self.data['emotion'])
            self.data['speaker_id'] = self.speaker_encoder.fit_transform(self.data['speaker'])
            print(f"Dataset loaded: {len(self.data)} valid samples after filtering.")
            print(f"Emotions found and encoded: {list(self.emotion_encoder.classes_)}")
            print(f"Speakers found and encoded: {list(self.speaker_encoder.classes_)}")
        else:
            print("Warning: Dataset is empty after filtering. No samples to process.")
            # Initialize encoders with dummy data to prevent errors if classes_ is accessed later
            self.emotion_encoder.fit(['NEU']) # Dummy
            self.speaker_encoder.fit(['p000']) # Dummy
            self.data['emotion_id'] = []
            self.data['speaker_id'] = []


        # Phonemizer for text processing
        self.phonemizer_fn = phonemize  # Store the function
        self.phonemizer_separator = Separator(phone='|', word=' ', syllable='')

        if not self.data.empty:
            sample_audio_filename = self.data.iloc[0]['audio_file']
            if isinstance(sample_audio_filename, str) and sample_audio_filename: # Check if filename is valid
                 sample_audio = self.audio_dir / sample_audio_filename
                 print(f"Sample audio path for first valid entry: {sample_audio}")
                 print(f"Sample audio exists: {sample_audio.exists()}")
                 if not sample_audio.exists():
                     print(f"WARNING: Sample audio file {sample_audio} does NOT exist. Check audio_dir and filenames.")
            else:
                print(f"Warning: Invalid audio filename for first data row: {sample_audio_filename}")
        else:
            print("Warning: Dataset is empty. Cannot check sample audio path.")

    def text_to_phonemes(self, text: str) -> str:
      original_input_text_for_error_log = text  # For better error logging
      try:
          # Ensure text is a string and strip it
          if not isinstance(text, str):
              text = str(text)  # Convert potential NaN or other types to string
          text = text.strip()

          if not text:  # Check for empty string after stripping
              return ""  # Return empty string for empty input

          # Process word by word to avoid mismatch warnings
          words = text.split()
          phoneme_words = []

          for word in words:
              try:
                  # First try espeak
                  phoneme = self.phonemizer_fn(
                      word,
                      language='en-us',
                      backend='espeak',
                      strip=True,
                      preserve_punctuation=False,
                      with_stress=False,
                  )
                  if phoneme:  # Only add non-empty phonemes
                      phoneme_words.append(phoneme.strip())

              except Exception as espeak_error:
                  # If espeak fails due to memory issues, try festival
                  try:
                      phoneme = self.phonemizer_fn(
                          word,
                          language='en-us',
                          backend='festival',
                          strip=True,
                          preserve_punctuation=False,
                          with_stress=False,
                      )
                      if phoneme:
                          phoneme_words.append(phoneme.strip())
                      else:
                          # If festival returns empty, keep original word
                          phoneme_words.append(word)

                  except Exception as festival_error:
                      # If both backends fail, keep the original word
                      print(f"Warning: Both espeak and festival failed for word '{word}'. Espeak: {espeak_error}, Festival: {festival_error}. Keeping original word.")
                      phoneme_words.append(word)

          result = ' '.join(phoneme_words)
          return result if result else ""

      except Exception as e:
          # More detailed error logging for phonemizer issues
          if "cannot apply additional memory protection after relocation" in str(e) or "Cannot allocate memory" in str(e):
              print(f"CRITICAL Phonemizer Memory Error for text '{original_input_text_for_error_log}': {e}. This is a Colab/espeak memory issue. Falling back to text.")
          elif "EspeakError: espeak.is_exe()" in str(e) or "No such file or directory" in str(e) and 'espeak' in str(e).lower():
              print(f"CRITICAL Phonemizer/Espeak Error: Espeak command not found or not executable. For text '{original_input_text_for_error_log}': {e}. Ensure espeak is installed and in PATH. Falling back to text.")
          elif "EspeakError" in str(e):
              print(f"Phonemizer/Espeak backend error for text '{original_input_text_for_error_log}': {e}. This might be due to unsupported characters. Falling back to text.")
          else:
              print(f"General Phonemizer error for text '{original_input_text_for_error_log}': {e}. Falling back to text.")
          return str(original_input_text_for_error_log).strip()  # Fallback to original text
    def load_audio(self, audio_path: str) -> torch.Tensor:
        try:
            if not isinstance(audio_path, str): # Ensure path is string
                audio_path = str(audio_path)

            if not Path(audio_path).exists():
                print(f"Error: Audio file not found at {audio_path}")
                return torch.zeros(self.max_audio_length)
            if not Path(audio_path).is_file():
                print(f"Error: Path {audio_path} is not a file.")
                return torch.zeros(self.max_audio_length)


            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        except Exception as e1:
            # print(f"Librosa failed for {audio_path}: {e1}. Trying soundfile.")
            try:
                audio_data, sr_sf = sf.read(audio_path, dtype='float32') # Read as float32
                if sr_sf != self.sample_rate:
                    # print(f"Resampling from {sr_sf} to {self.sample_rate} for {audio_path}")
                    # Ensure audio_data is 1D for resample if it was stereo
                    if audio_data.ndim > 1:
                        audio_data = audio_data.mean(axis=1) # Or take first channel: audio_data[:, 0]
                    audio = librosa.resample(y=audio_data, orig_sr=sr_sf, target_sr=self.sample_rate)
                else:
                    audio = audio_data
                # If soundfile read stereo, convert to mono (e.g., by averaging or taking one channel)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1) # Example: average channels
            except Exception as e2:
                print(f"Soundfile also failed for {audio_path}: {e2}. Librosa error was: {e1}")
                return torch.zeros(self.max_audio_length)

        if len(audio) < 100:
            # print(f"Warning: Very short or empty audio file {audio_path}, length {len(audio)}")
            return torch.zeros(self.max_audio_length)

        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
        if len(audio_trimmed) >= 100: # Use trimmed if still valid
            audio = audio_trimmed
        # else:
            # print(f"Warning: Audio became too short after trimming for {audio_path}. Using original (pre-trim).")


        max_val = np.max(np.abs(audio))
        if max_val > 1e-6:
            audio = audio / max_val
        else:
            # print(f"Warning: Silent or near-silent audio file {audio_path}")
            return torch.zeros(self.max_audio_length) # Avoid division by zero for silent audio

        if len(audio) > self.max_audio_length:
            audio = audio[:self.max_audio_length]
        else:
            audio = np.pad(audio, (0, self.max_audio_length - len(audio)), mode='constant')

        return torch.FloatTensor(audio)

    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        # Ensure audio is numpy array for librosa
        audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio

        mel = librosa.feature.melspectrogram(
            y=audio_np,
            sr=self.sample_rate,
            n_mels=self.mel_channels,
            n_fft=1024,
            hop_length=256,
            win_length=1024
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        return torch.FloatTensor(mel)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError("Index out of bounds")
        row = self.data.iloc[idx]

        audio_filename = row['audio_file']
        if not isinstance(audio_filename, str): # Should be handled by dtype in read_csv but good practice
            audio_filename = str(audio_filename)

        audio_path = self.audio_dir / audio_filename.strip() # Strip any whitespace from filename
        audio = self.load_audio(str(audio_path))
        mel = self.audio_to_mel(audio)

        text_input = row['text']
        if not isinstance(text_input, str): # Should be handled by dtype in read_csv
            text_input = str(text_input)

        phonemes = self.text_to_phonemes(text_input)

        return {
            'audio': audio,
            'mel': mel,
            'text': text_input,
            'phonemes': phonemes,
            'emotion_id': row['emotion_id'],
            'speaker_id': row['speaker_id'],
            'emotion': row['emotion'],
            'speaker': row['speaker']
        }

# ========================================
# STEP 4: Emotion-Conditioned TTS Model
# ========================================

class EmotionEmbedding(nn.Module):
    def __init__(self, num_emotions: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_emotions, embedding_dim)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, emotion_ids):
        emb = self.embedding(emotion_ids)
        return self.projection(emb)

class SpeakerEmbedding(nn.Module):
    def __init__(self, num_speakers: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_speakers, embedding_dim)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, speaker_ids):
        emb = self.embedding(speaker_ids)
        return self.projection(emb)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, text_ids):
        embedded = self.embedding(text_ids)
        output, _ = self.lstm(embedded)
        return self.projection(output)

class EmotionTTSModel(nn.Module):
    def __init__(self, vocab_size: int, num_emotions: int, num_speakers: int,
                 text_embedding_dim: int = 256, emotion_embedding_dim: int = 64,
                 speaker_embedding_dim: int = 64, hidden_dim: int = 512,
                 mel_channels: int = 80):
        super().__init__()

        self.text_encoder = TextEncoder(vocab_size, text_embedding_dim, hidden_dim)
        self.emotion_embedding = EmotionEmbedding(num_emotions, emotion_embedding_dim)
        self.speaker_embedding = SpeakerEmbedding(num_speakers, speaker_embedding_dim)

        # Fusion layer
        total_dim = hidden_dim + emotion_embedding_dim + speaker_embedding_dim
        self.fusion = nn.Linear(total_dim, hidden_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, mel_channels)
        )

        # Duration predictor (simple version)
        self.duration_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )

    def forward(self, text_ids, emotion_ids, speaker_ids, mel_targets=None):
        # Encode text
        text_encoded = self.text_encoder(text_ids)  # [B, T, hidden_dim]

        # Get emotion and speaker embeddings
        emotion_emb = self.emotion_embedding(emotion_ids)  # [B, emotion_dim]
        speaker_emb = self.speaker_embedding(speaker_ids)  # [B, speaker_dim]

        # Expand emotion and speaker embeddings to match text sequence length
        batch_size, seq_len, _ = text_encoded.shape
        emotion_emb = emotion_emb.unsqueeze(1).expand(-1, seq_len, -1)
        speaker_emb = speaker_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # Fuse all embeddings
        fused = torch.cat([text_encoded, emotion_emb, speaker_emb], dim=-1)
        fused = self.fusion(fused)

        # Predict duration
        durations = self.duration_predictor(fused).squeeze(-1)

        # Generate mel spectrogram
        mel_output = self.decoder(fused)

        return {
            'mel_output': mel_output,
            'durations': durations,
            'encoded_features': fused
        }

# ========================================
# STEP 5: Training Configuration and Setup
# ========================================

class TTSTrainingConfig:
    def __init__(self):
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.num_epochs = 100
        self.warmup_steps = 4000
        self.save_every = 10
        self.validate_every = 5
        self.grad_clip = 1.0

        # Model parameters
        self.vocab_size = 1000  # Will be updated based on tokenizer
        self.text_embedding_dim = 256
        self.emotion_embedding_dim = 64
        self.speaker_embedding_dim = 64
        self.hidden_dim = 512
        self.mel_channels = 80

# Simple tokenizer for phonemes/characters
class SimpleTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0

    def build_vocab(self, texts: List[str]):
        chars = set()
        for text in texts:
            chars.update(text.lower())

        # Add special tokens
        chars = ['<pad>', '<unk>', '<sos>', '<eos>'] + sorted(list(chars))

        self.char_to_id = {char: i for i, char in enumerate(chars)}
        self.id_to_char = {i: char for i, char in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str, max_length: int = 200) -> torch.Tensor:
        ids = [self.char_to_id.get(char, self.char_to_id['<unk>'])
               for char in text.lower()]

        # Pad or truncate
        if len(ids) < max_length:
            ids.extend([self.char_to_id['<pad>']] * (max_length - len(ids)))
        else:
            ids = ids[:max_length]

        return torch.LongTensor(ids)

# ========================================
# STEP 6: Training Loop
# ========================================

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    audio = torch.stack([item['audio'] for item in batch])
    mel = torch.stack([item['mel'] for item in batch])
    emotion_ids = torch.LongTensor([item['emotion_id'] for item in batch])
    speaker_ids = torch.LongTensor([item['speaker_id'] for item in batch])

    return {
        'audio': audio,
        'mel': mel,
        'emotion_ids': emotion_ids,
        'speaker_ids': speaker_ids,
        'texts': [item['text'] for item in batch],
        'phonemes': [item['phonemes'] for item in batch]
    }

def train_emotion_tts():
    # Configuration
    config = TTSTrainingConfig()
    config.batch_size = 4  # Reduce batch size to avoid memory issues

    # Load your data - Use the fixed data file
    data_file = "/content/data/metadata_tts.csv"  # Use the fixed data file
    audio_dir = "/content/data"  # Directory with audio files

    print(f"Loading data from: {data_file}")
    print(f"Audio directory: {audio_dir}")

    # # Check if fixed file exists, if not create it
    # if not os.path.exists(data_file):
    #     print("Fixed data file not found. Creating it...")
    #     # Run the quick fix
    #     exec(open('/content/quick_fix_script.py').read()) if os.path.exists('/content/quick_fix_script.py') else print("Please run the quick fix script first")
    #     return

    # Create dataset
    dataset = EmotionTTSDataset(data_file, audio_dir)

    # Build tokenizer
    tokenizer = SimpleTokenizer()
    all_texts = [item['phonemes'] for item in dataset]
    tokenizer.build_vocab(all_texts)
    config.vocab_size = tokenizer.vocab_size

    # Create model
    model = EmotionTTSModel(
        vocab_size=config.vocab_size,
        num_emotions=len(dataset.emotion_encoder.classes_),
        num_speakers=len(dataset.speaker_encoder.classes_),
        text_embedding_dim=config.text_embedding_dim,
        emotion_embedding_dim=config.emotion_embedding_dim,
        speaker_embedding_dim=config.speaker_embedding_dim,
        hidden_dim=config.hidden_dim,
        mel_channels=config.mel_channels
    ).to(device) # Move model to device here

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs)

    # Training loop
    model.train()
    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # Prepare inputs and move to device
            mel_targets = batch['mel'].to(device)
            emotion_ids = batch['emotion_ids'].to(device)
            speaker_ids = batch['speaker_ids'].to(device)

            # Tokenize texts and move to device
            text_ids = torch.stack([
                tokenizer.encode(text) for text in batch['phonemes']
            ]).to(device)

            # Forward pass
            outputs = model(text_ids, emotion_ids, speaker_ids, mel_targets)

            # Calculate loss (MSE for mel spectrogram)
            mel_loss = F.mse_loss(outputs['mel_output'], mel_targets)

            # Duration loss (simple version)
            duration_loss = F.mse_loss(
                outputs['durations'],
                torch.ones_like(outputs['durations'])
            )

            total_loss_batch = mel_loss + 0.1 * duration_loss

            # Backward pass
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            total_loss += total_loss_batch.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss_batch.item():.4f}')

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')

        # Validation
        if epoch % config.validate_every == 0:
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    # Prepare inputs and move to device
                    mel_targets = batch['mel'].to(device)
                    emotion_ids = batch['emotion_ids'].to(device)
                    speaker_ids = batch['speaker_ids'].to(device)

                    text_ids = torch.stack([
                        tokenizer.encode(text) for text in batch['phonemes']
                    ]).to(device)

                    outputs = model(text_ids, emotion_ids, speaker_ids, mel_targets)
                    mel_loss = F.mse_loss(outputs['mel_output'], mel_targets)
                    val_loss += mel_loss.item()

            val_loss /= len(val_loader)
            print(f'Validation Loss: {val_loss:.4f}')

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'config': config,
                    'tokenizer': tokenizer,
                    'emotion_encoder': dataset.emotion_encoder,
                    'speaker_encoder': dataset.speaker_encoder
                }, 'best_emotion_tts_model.pth')
                print('Best model saved!')

            model.train()

        # Save checkpoint
        if epoch % config.save_every == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
            }, f'emotion_tts_checkpoint_epoch_{epoch}.pth')

# ========================================
# STEP 7: Inference Function
# ========================================

def generate_speech_with_emotion(model, tokenizer, emotion_encoder, speaker_encoder,
                                text: str, emotion: str, speaker: str, device):
    """Generate speech with specified emotion and speaker"""
    model.eval()

    with torch.no_grad():
        # Prepare inputs
        text_ids = tokenizer.encode(text).unsqueeze(0).to(device)
        emotion_id = torch.LongTensor([emotion_encoder.transform([emotion])[0]]).to(device)
        speaker_id = torch.LongTensor([speaker_encoder.transform([speaker])[0]]).to(device)

        # Generate
        outputs = model(text_ids, emotion_id, speaker_id)
        mel_output = outputs['mel_output'].squeeze(0).cpu().numpy()

        return mel_output

# ========================================
# STEP 8: Usage Example
# ========================================

def main():
    """Main training function"""
    print("Starting Emotion-Controlled TTS Training...")
    print("=" * 50)

    # Check data format first
    print("Checking your data format...")
    data_file = "/content/data/metadata_tts.csv"

    # Read first few lines to check format
    with open(data_file, 'r') as f:
        lines = f.readlines()[:5]
        print("First 5 lines of your data:")
        for i, line in enumerate(lines):
            print(f"{i+1}: {line.strip()}")

    print("\nStarting training...")
    train_emotion_tts()

    print("Training setup complete!")
    print("Remember to:")
    print("- Monitor GPU memory usage")
    print("- Adjust batch_size if you get OOM errors")
    print("- Use wandb for experiment tracking (optional)")
    print("- Save intermediate checkpoints regularly")

# Run the setup
if __name__ == "__main__":
    main()



# Test the text_to_phonemes function
sample_text = "This is a test sentence."
dataset_instance = EmotionTTSDataset(data_file="/content/data/metadata_tts.csv", audio_dir="/content/data")

try:
    phonemes = dataset_instance.text_to_phonemes(sample_text)
    print(f"Original text: {sample_text}")
    print(f"Phonemes: {phonemes}")
except Exception as e:
    print(f"Error during phonemization test: {e}")