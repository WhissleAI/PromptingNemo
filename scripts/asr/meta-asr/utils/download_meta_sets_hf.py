import os
import json
import io
import multiprocessing
from functools import partial
from datasets import load_dataset, Audio
import soundfile as sf
from tqdm import tqdm
import librosa
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_entry(item_tuple, audio_dir, target_sr):
    """
    Processes a single dataset entry.
    """
    i, item = item_tuple
    audio_data = item.get('audio')
    text = item.get('text')

    if audio_data is None or text is None:
        logging.warning(f"Skipping item {i} due to missing audio or text.")
        return None
    
    audio_bytes = audio_data.get('bytes')

    if audio_bytes is None:
        logging.warning(f"Skipping item {i} due to missing audio bytes.")
        return None

    try:
        waveform, sample_rate = sf.read(io.BytesIO(audio_bytes))
    except Exception as e:
        logging.error(f"Could not read audio for item {i}: {e}")
        return None

    # Ensure waveform is a 1D array
    if waveform.ndim > 1:
        waveform = waveform.squeeze()

    # Resample if necessary
    if sample_rate != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr
    
    duration = len(waveform) / sample_rate
    cleaned_text = text

    # Save audio file
    audio_filename = f"wellness_podcast_{i}.wav"
    audio_filepath = os.path.join(audio_dir, audio_filename)
    
    try:
        sf.write(audio_filepath, waveform, sample_rate)
    except Exception as e:
        logging.error(f"Could not write audio file for item {i}: {e}")
        return None

    # Create manifest entry
    return {
        "audio_filepath": os.path.abspath(audio_filepath),
        "text": cleaned_text,
        "duration": duration,
        "lang_id": "EN",
    }

def create_manifest(dataset_name, output_dir, manifest_filename, target_sr=16000):
    """
    Downloads a Hugging Face dataset, saves audio files, and creates a NeMo manifest.

    Args:
        dataset_name (str): The name of the dataset on Hugging Face.
        output_dir (str): The directory to save the audio files and manifest.
        manifest_filename (str): The name of the manifest file.
        target_sr (int): The target sample rate for the audio.
    """
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, manifest_filename)

    logging.info(f"Loading dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name, split="train")
    logging.info("Casting audio column to not decode to avoid torchcodec dependency.")
    dataset = dataset.cast_column("audio", Audio(decode=False))
    logging.info("Dataset loaded successfully.")

    logging.info(f"Processing dataset and writing to manifest: {manifest_path}")
    
    process_func = partial(process_entry, audio_dir=audio_dir, target_sr=target_sr)
    num_processes = multiprocessing.cpu_count()
    logging.info(f"Using {num_processes} processes for parallel processing.")

    with open(manifest_path, "w", encoding='utf-8') as manifest_file:
        with multiprocessing.Pool(processes=num_processes) as pool:
            for manifest_entry in tqdm(
                pool.imap_unordered(process_func, enumerate(dataset)),
                total=len(dataset),
                desc="Processing samples"
            ):
                if manifest_entry:
                    manifest_file.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    DATASET_NAME = "WhissleAI/Meta_STT_EN_Welness_podcast"
    # Per user request, base directory for audio
    BASE_OUTPUT_DIR = "/external/hf/"
    DATASET_SPECIFIC_DIR = os.path.join(BASE_OUTPUT_DIR, "Meta_STT_EN_Welness_podcast")
    MANIFEST_FILENAME = "wellness_podcast_manifest.json"

    logging.info("Starting dataset processing.")
    create_manifest(DATASET_NAME, DATASET_SPECIFIC_DIR, MANIFEST_FILENAME)
    logging.info(f"Manifest created at {os.path.join(DATASET_SPECIFIC_DIR, MANIFEST_FILENAME)}")
