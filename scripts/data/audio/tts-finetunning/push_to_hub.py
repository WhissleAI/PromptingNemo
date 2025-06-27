import os
import time
import json
from pathlib import Path
from datasets import Dataset, DatasetDict, Audio
from huggingface_hub import login
import pandas as pd
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioDatasetUploader:
    def __init__(self, dataset_name: str, audio_dir: str, data_file: str, batch_size: int = 50):
        """
        Initialize the uploader
        
        Args:
            dataset_name: Name for your dataset on HF Hub (e.g., "username/dataset-name")
            audio_dir: Directory containing audio files
            data_file: Path to your data file (JSON/JSONL/CSV)
            batch_size: Number of samples to process in each batch
        """
        self.dataset_name = dataset_name
        self.audio_dir = Path(audio_dir)
        self.data_file = data_file
        self.batch_size = batch_size
        self.delay_between_batches = 60  # seconds
        
    def load_data(self) -> List[Dict[str, Any]]:
        """Load data from file"""
        if self.data_file.endswith('.json'):
            with open(self.data_file, 'r') as f:
                data = json.load(f)
        elif self.data_file.endswith('.jsonl'):
            data = []
            with open(self.data_file, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        elif self.data_file.endswith('.csv'):
            df = pd.read_csv(self.data_file)
            data = df.to_dict('records')
        else:
            raise ValueError("Supported formats: .json, .jsonl, .csv")
        
        logger.info(f"Loaded {len(data)} samples")
        return data
    
    def prepare_dataset_batch(self, data_batch: List[Dict[str, Any]]) -> Dataset:
        """Prepare a batch of data for upload"""
        processed_data = {
            'audio': [],
            'transcription': [],
            'emotion': []
        }
        
        for item in data_batch:
            # Extract just the filename from audio_filepath
            audio_filename = os.path.basename(item['audio_filepath'])
            audio_path = self.audio_dir / audio_filename
            
            if audio_path.exists():
                processed_data['audio'].append(str(audio_path))
                processed_data['transcription'].append(item['transcription'])
                processed_data['emotion'].append(item['emotion'])
            else:
                logger.warning(f"Audio file not found: {audio_path}")
        
        # Create dataset
        dataset = Dataset.from_dict(processed_data)
        
        # Cast audio column to Audio feature
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        return dataset
    
    def upload_in_batches(self, private: bool = False):
        """Upload dataset in batches to avoid rate limits"""
        # Load all data
        all_data = self.load_data()
        
        # For simplicity, let's process all data at once for the first upload
        # You can modify this to use smaller batches if needed
        logger.info(f"Processing {len(all_data)} samples")
        
        try:
            # Prepare entire dataset
            dataset = self.prepare_dataset_batch(all_data)
            
            if len(dataset) == 0:
                logger.warning("No valid audio files found, aborting upload...")
                return
            
            # Create DatasetDict with only train split
            dataset_dict = DatasetDict({
                "train": dataset
            })
            
            # Upload dataset
            logger.info("Uploading dataset to Hugging Face Hub...")
            dataset_dict.push_to_hub(
                self.dataset_name,
                private=private,
                token=True
            )
            logger.info("Dataset upload completed successfully!")
                    
        except Exception as e:
            logger.error(f"Error uploading dataset: {e}")
            if "429" in str(e) or "Too Many Requests" in str(e):
                logger.info("Rate limit hit, please try again later...")
            raise e

def main():
    # Configuration
    DATASET_NAME = "WhissleAI/emotion-tagged-small-v1"  # Change this
    AUDIO_DIR = "E:/Meta_asr/datasets/data/wavs"  # Change this
    DATA_FILE = "E:/Meta_asr/datasets/data/output_hub.jsonl"  # Change this - your JSON/JSONL/CSV file
    BATCH_SIZE = 50  # Adjust based on your needs
    
    # Login to Hugging Face (make sure you have your token set)
    # You can set your token with: huggingface-cli login
    # Or set HF_TOKEN environment variable
    
    
    # Create uploader instance
    uploader = AudioDatasetUploader(
        dataset_name=DATASET_NAME,
        audio_dir=AUDIO_DIR,
        data_file=DATA_FILE,
        batch_size=BATCH_SIZE
    )
    
    # Upload dataset
    uploader.upload_in_batches(private=False)  # Set to True for private dataset

if __name__ == "__main__":
    main()
