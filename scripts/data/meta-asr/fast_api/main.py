import os
import gc
import re
import json
import time
import uuid
import base64
import torch
import librosa
import asyncio
import logging
import numpy as np
import soundfile as sf
import sys
import resampy
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum

from fastapi import (
    FastAPI, HTTPException, BackgroundTasks, Request,
    File, UploadFile, Form, Query
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field as PydanticField # Alias to avoid conflict
import uvicorn

from dotenv import load_dotenv

# --- Model Imports ---
import torch.nn as nn
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model # Added for AgeGenderModel definition
)

import google.generativeai as genai

# Try importing WhissleClient - handle potential import error
try:
    from whissle import WhissleClient
    WHISSLE_AVAILABLE = True
except ImportError:
    print("Warning: WhissleClient SDK not found or failed to import. Whissle model will be unavailable.")
    WHISSLE_AVAILABLE = False
    class WhissleClient: pass # Dummy class

# --- Configuration & Setup ---
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants and Globals ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WHISSLE_AUTH_TOKEN = os.getenv("WHISSLE_AUTH_TOKEN") # Assuming token via env var now
GEMINI_CONFIGURED = False
WHISSLE_CONFIGURED = False
AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
TARGET_SAMPLE_RATE = 16000

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# --- Initialize FastAPI ---
app = FastAPI(
    title="Audio Processing API",
    description="Transcribes audio, optionally predicts Age/Gender/Emotion, annotates Intent/Entities, "
                "and saves results to a JSONL manifest file.",
    version="1.4.0" # Incremented version for new features
)

# --- Model Loading ---
age_gender_model = None
age_gender_processor = None
emotion_model = None
emotion_feature_extractor = None

# Resolve the path relative to the current file
current_file_path = Path(__file__).parent
static_dir = current_file_path / "static"

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", include_in_schema=False) # Route to serve the HTML file
async def serve_index():
    index_html_path = current_file_path / "static" / "index.html"
    if not index_html_path.is_file():
        logger.error(f"HTML file not found at: {index_html_path}")
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_html_path)

# --- Age/Gender Model Definition (Copied from provided code) ---
class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1) # Predicts a single value for age
        self.gender = ModelHead(config, 3) # Predicts 3 classes for gender
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1) # Pool features
        logits_age = self.age(hidden_states)
        logits_gender = self.gender(hidden_states)
        return hidden_states, logits_age, logits_gender

try:
    logger.info("Loading Age/Gender model...")
    age_gender_model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    age_gender_processor = Wav2Vec2Processor.from_pretrained(age_gender_model_name)
    age_gender_model = AgeGenderModel.from_pretrained(age_gender_model_name).to(device)
    age_gender_model.eval() # Set to evaluation mode
    logger.info("Age/Gender model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Age/Gender model: {e}", exc_info=True)

try:
    logger.info("Loading Emotion model...")
    emotion_model_name = "superb/hubert-large-superb-er"
    emotion_feature_extractor = AutoFeatureExtractor.from_pretrained(emotion_model_name)
    emotion_model = AutoModelForAudioClassification.from_pretrained(emotion_model_name).to(device)
    emotion_model.eval() # Set to evaluation mode
    logger.info("Emotion model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Emotion model: {e}", exc_info=True)

# --- Configure Gemini ---
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("Gemini API configured successfully.")
        GEMINI_CONFIGURED = True
    except Exception as e:
        logger.error(f"Error configuring Gemini API: {e}. Gemini features will be unavailable.")
else:
    logger.warning("Warning: GOOGLE_API_KEY environment variable not set. Gemini features will be unavailable.")

# --- Configure Whissle ---
if WHISSLE_AVAILABLE and WHISSLE_AUTH_TOKEN:
    logger.info("Whissle Auth Token found.")
    WHISSLE_CONFIGURED = True
elif WHISSLE_AVAILABLE:
    logger.warning("Warning: WHISSLE_AUTH_TOKEN environment variable not set. Whissle model will be unavailable.")
else:
    pass # Message already printed

# --- Pydantic Models ---
class ModelChoice(str, Enum):
    gemini = "gemini"
    whissle = "whissle"

# Request model for both endpoints (can be reused)
class ProcessRequest(BaseModel):
    directory_path: str = PydanticField(..., description="Absolute path to the directory containing audio files.", example="/path/to/audio")
    model_choice: ModelChoice = PydanticField(..., description="The transcription model to use.")
    output_jsonl_path: str = PydanticField(..., description="Absolute path for the output JSONL file.", example="/path/to/output/results.jsonl")

# Simplified record for transcription-only manifest
class TranscriptionJsonlRecord(BaseModel):
    audio_filepath: str
    text: Optional[str] = None # Just the transcription
    duration: Optional[float] = None
    model_used_for_transcription: str
    error: Optional[str] = None

# Record for annotated manifest (matches previous definition)
class AnnotatedJsonlRecord(BaseModel):
    audio_filepath: str
    text: Optional[str] = None # Combined annotated text
    duration: Optional[float] = None
    model_used_for_transcription: str
    error: Optional[str] = None

# Response model for both endpoints (can be reused)
class ProcessResponse(BaseModel):
    message: str
    output_file: str
    processed_files: int
    saved_records: int
    errors: int

# --- Helper Functions (Keep all existing helpers) ---
def load_audio(audio_path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    """Loads audio file, resamples if necessary. Returns (audio_array, sample_rate, error_message)."""
    try:
        audio, sr = sf.read(str(audio_path), dtype='float32')
        if audio.ndim > 1:
            logger.warning(f"Audio file {audio_path.name} is not mono ({audio.ndim} channels). Converting to mono by averaging.")
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            logger.debug(f"Resampling {audio_path.name} from {sr}Hz to {target_sr}Hz.") # Debug level for resampling
            audio = resampy.resample(audio, sr, target_sr)
            sr = target_sr
        return audio, sr, None
    except Exception as e:
        logger.error(f"Error loading audio file {audio_path}: {e}", exc_info=True)
        return None, None, f"Failed to load audio: {e}"

def get_audio_duration(audio_path: Path) -> Optional[float]:
    """Gets the duration of an audio file in seconds."""
    try:
        info = sf.info(str(audio_path))
        return info.duration
    except Exception as e:
        logger.warning(f"Could not get duration for {audio_path.name} using soundfile info: {e}. Trying librosa load.")
        try:
            # Use librosa with a specific duration limit to avoid loading huge files entirely
            duration = librosa.get_duration(path=str(audio_path))
            return duration
        except Exception as le:
             logger.error(f"Failed to get duration for {audio_path.name}: {le}", exc_info=True)
             return None

def predict_age_gender(audio_data: np.ndarray, sampling_rate: int) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    """Predicts age and gender. Returns (predicted_age, predicted_gender_idx, error_message)."""
    if age_gender_model is None or age_gender_processor is None:
        return None, None, "Age/Gender model not loaded."
    if audio_data is None or len(audio_data) == 0:
        return None, None, "Empty audio data provided for Age/Gender."

    try:
        inputs = age_gender_processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        # attention_mask = inputs.attention_mask.to(device) if 'attention_mask' in inputs else None # Attention mask not needed for this model's forward

        with torch.no_grad():
            # Pass only input_values as required by the custom AgeGenderModel forward
            outputs = age_gender_model(input_values)

        age_pred = outputs[1].detach().cpu().numpy().flatten()[0]
        gender_logits = outputs[2].detach().cpu().numpy()
        gender_pred_idx = np.argmax(gender_logits, axis=1)[0]

        return float(age_pred), int(gender_pred_idx), None

    except Exception as e:
        logger.error(f"Error during Age/Gender prediction: {e}", exc_info=True)
        return None, None, f"Age/Gender prediction failed: {e}"

def predict_emotion(audio_data: np.ndarray, sampling_rate: int) -> Tuple[Optional[str], Optional[str]]:
    """Predicts emotion. Returns (emotion_label, error_message)."""
    if emotion_model is None or emotion_feature_extractor is None:
        return None, "Emotion model not loaded."
    if audio_data is None or len(audio_data) == 0:
        return None, "Empty audio data provided for Emotion."
    # Add check for minimum length required by the model (e.g., 0.1 seconds)
    min_length = int(sampling_rate * 0.1) # Example: 100ms minimum
    if len(audio_data) < min_length:
        logger.warning(f"Audio data for {audio_data.shape} too short for reliable emotion prediction (min {min_length} samples).")
        return "SHORT_AUDIO", None # Return specific label

    try:
        inputs = emotion_feature_extractor(
            audio_data,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True # Pad shorter sequences
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = emotion_model(**inputs)

        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=-1).item()
        emotion_label = emotion_model.config.id2label.get(predicted_class_idx, "UNKNOWN_EMOTION")

        return emotion_label, None

    except Exception as e:
        logger.error(f"Error during Emotion prediction: {e}", exc_info=True)
        return None, f"Emotion prediction failed: {e}"

def format_age_gender_emotion_tags(age: Optional[float], gender_idx: Optional[int], emotion_label: Optional[str]) -> str:
    """Formats the predicted labels into a tag string."""
    tags = []
    if age is not None:
        try:
            actual_age = round(age, 1) # Use the direct prediction
            # Define age brackets (adjust as needed)
            age_brackets: List[Tuple[float, str]] = [
                (18, "0_17"), (25, "18_24"), (35, "25_34"),
                (45, "35_44"), (55, "45_54"), (65, "55_64"),
                (float('inf'), "65PLUS")
            ]
            age_tag = "UNKNOWN"
            for threshold, bracket in age_brackets:
                if actual_age < threshold:
                    age_tag = bracket
                    break
            tags.append(f"AGE_{age_tag}")
        except Exception as age_e:
             logger.error(f"Error formatting age tag for age value {age}: {age_e}")
             tags.append("AGE_ERROR")
    else:
        tags.append("AGE_UNKNOWN")

    if gender_idx is not None:
        # Mapping based on audeering model's typical output: 0: female, 1: male
        # Verify this mapping if using a different model
        if gender_idx == 1: tags.append("GENDER_MALE")
        elif gender_idx == 0: tags.append("GENDER_FEMALE")
        # Assuming the model doesn't output a specific 'other' index. Adjust if it does.
        else: tags.append("GENDER_UNKNOWN") # Or GENDER_OTHER if model supports
    else:
        tags.append("GENDER_UNKNOWN")

    if emotion_label and emotion_label != "SHORT_AUDIO":
        emotion_tag = emotion_label.upper().replace(" ", "_") # Ensure uppercase and underscores
        tags.append(f"EMOTION_{emotion_tag}")
    elif emotion_label == "SHORT_AUDIO":
         tags.append("EMOTION_SHORT_AUDIO") # Specific tag for short audio
    else:
        tags.append("EMOTION_UNKNOWN")

    return " ".join(tags)

def combine_tags(gemini_annotated_text: str, age_gender_emotion_tags: str) -> str:
    """Inserts age/gender/emotion tags before the INTENT tag or at the end."""
    if not gemini_annotated_text:
        return age_gender_emotion_tags # Return only A/G/E tags if annotation is empty

    # Ensure A/G/E tags are present before trying to insert
    if not age_gender_emotion_tags:
        return gemini_annotated_text # Return original if no A/G/E tags

    # Find the last occurrence of INTENT_ at the end of the string
    intent_match = re.search(r'\s(INTENT_[A-Z_]+)$', gemini_annotated_text)
    if intent_match:
        insertion_point = intent_match.start()
        # Insert A/G/E tags before the space preceding the INTENT tag
        return f"{gemini_annotated_text[:insertion_point]} {age_gender_emotion_tags}{gemini_annotated_text[insertion_point:]}".strip()
    else:
        # If no INTENT tag found at the end, append A/G/E tags with a space
        return f"{gemini_annotated_text} {age_gender_emotion_tags}".strip()

# --- Transcription/Annotation Functions (Keep all existing) ---

async def transcribe_with_whissle_single(audio_path: Path, model_name="en-US-0.6b") -> tuple[Optional[str], Optional[str]]:
    """Transcribe using Whissle. Returns (text, error_message)."""
    if not WHISSLE_CONFIGURED:
        return None, "Whissle is not configured (token missing or SDK unavailable)."
    try:
        # Consider creating the client once if performance is critical, but re-creating ensures fresh state
        whissle_client = WhissleClient(auth_token=WHISSLE_AUTH_TOKEN)
        logger.info(f"Transcribing {audio_path.name} with Whissle...")

        # --- THIS IS THE CORRECTED LINE ---
        # Directly await the async function from the Whissle client SDK
        response = await whissle_client.speech_to_text(str(audio_path), model_name=model_name)
        # --- END OF CORRECTION ---


        logger.debug(f"Whissle response type: {type(response)}")
        logger.debug(f"Whissle response content (first 100): {str(response)[:100]}...")

        # Handle different possible response types/structures from Whissle SDK
        if isinstance(response, dict):
             text = response.get('text')
             if text is not None:
                 logger.info(f"Whissle transcription success for {audio_path.name}.")
                 return text.strip(), None
             else:
                 error_detail = response.get('error') or response.get('message', 'Unknown Whissle API error structure')
                 logger.error(f"Whissle API error for {audio_path.name}: {error_detail} | Full response: {response}")
                 return None, f"Whissle API error: {error_detail}"
        # Check if it's an object with a 'transcript' attribute (common pattern)
        elif hasattr(response, 'transcript') and isinstance(response.transcript, str):
            logger.info(f"Received Whissle response object with transcript for {audio_path.name}.")
            return response.transcript.strip(), None
        # Add more checks based on actual SDK behavior if necessary
        else:
             logger.error(f"Unexpected Whissle response format for {audio_path.name}: {type(response)} | Content: {response}")
             # This case should be less likely now, but kept as a safeguard
             return None, f"Unexpected Whissle response format after await: {type(response)}"

    except Exception as e:
        logger.error(f"Error transcribing {audio_path.name} with Whissle: {e}", exc_info=True)
        # Check if the error object has specific details (e.g., status code, message)
        error_msg = f"Whissle SDK error: {type(e).__name__}: {e}"
        return None, error_msg

def get_mime_type(audio_file_path: Path) -> str:
    """Determines the MIME type of an audio file."""
    ext = audio_file_path.suffix.lower()
    mime_map = { ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg", ".m4a": "audio/mp4" }
    # Fallback for unknown types, Gemini might still handle it
    return mime_map.get(ext, "application/octet-stream")

async def transcribe_with_gemini_single(audio_path: Path) -> tuple[Optional[str], Optional[str]]:
    """Transcribe using Gemini. Returns (text, error_message)."""
    if not GEMINI_CONFIGURED:
         return None, "Gemini API is not configured (key missing or invalid)."
    # Use the latest appropriate model, 1.5 Flash is good for speed/cost
    model_name = "models/gemini-1.5-flash"
    try:
        # Initialize model here - potentially cache this if making many calls rapidly
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        logger.error(f"Error initializing Gemini model {model_name}: {e}", exc_info=True)
        return None, f"Error initializing Gemini model: {e}"

    mime_type = get_mime_type(audio_path)
    logger.info(f"Uploading {audio_path.name} ({mime_type}) to Gemini for transcription...")
    uploaded_file = None
    try:
        # Upload file - use asyncio.to_thread for blocking I/O
        uploaded_file = await asyncio.to_thread(genai.upload_file, path=str(audio_path), mime_type=mime_type)
        logger.info(f"File {audio_path.name} uploaded. Gemini Name: {uploaded_file.name}. Waiting for processing...")

        # Poll for processing completion
        while uploaded_file.state.name == "PROCESSING":
            await asyncio.sleep(2) # Wait 2 seconds before checking again
            # Use get_file to update the status - run in thread
            uploaded_file = await asyncio.to_thread(genai.get_file, name=uploaded_file.name)
            logger.debug(f"Checked Gemini file status for {uploaded_file.name}: {uploaded_file.state.name}")

        if uploaded_file.state.name != "ACTIVE":
            error_msg = f"Gemini file processing failed for {audio_path.name}. State: {uploaded_file.state.name}"
            logger.error(error_msg)
            # Attempt to delete the failed file resource
            try: await asyncio.to_thread(genai.delete_file, name=uploaded_file.name); logger.debug(f"Deleted failed Gemini file resource: {uploaded_file.name}")
            except Exception as del_e: logger.warning(f"Could not delete failed Gemini file resource {uploaded_file.name}: {del_e}")
            return None, error_msg # Return the specific error

        # Generate content (transcription)
        logger.info(f"File {uploaded_file.name} active. Generating transcription...")
        # Simple transcription prompt
        prompt = "Transcribe the audio accurately. Provide only the spoken text. If no speech is detected, return an empty string."
        # Use asyncio.to_thread for the blocking API call
        response = await asyncio.to_thread(
            model.generate_content,
            [prompt, uploaded_file],
            request_options={'timeout': 300} # 5-minute timeout for transcription
        )

        # Clean up the uploaded file resource ASAP
        try: await asyncio.to_thread(genai.delete_file, name=uploaded_file.name); logger.debug(f"Deleted Gemini file resource after transcription: {uploaded_file.name}"); uploaded_file = None # Clear variable too
        except Exception as del_e: logger.warning(f"Could not delete Gemini file resource {uploaded_file.name} after transcription: {del_e}")

        # Process response
        if response.candidates:
            try:
                # Access text safely
                if hasattr(response, 'text') and response.text is not None:
                    transcription = response.text.strip()
                elif response.candidates[0].content.parts:
                     transcription = "".join(part.text for part in response.candidates[0].content.parts).strip()
                else:
                     logger.warning(f"Gemini transcription candidate exists but no text found for {audio_path.name}. Response: {response}")
                     return None, "Gemini response candidate found, but no text content."

                # Handle potentially empty transcription (no speech)
                if not transcription:
                    logger.info(f"No speech detected by Gemini in {audio_path.name}.")
                    return "", None # Return empty string, no error
                else:
                    logger.info(f"Gemini transcription success for {audio_path.name}: {transcription[:100]}...")
                    return transcription, None

            except (AttributeError, IndexError, ValueError, TypeError) as resp_e:
                logger.warning(f"Could not extract text from Gemini transcription response for {audio_path.name}: {resp_e}. Response: {response}")
                return None, f"Error parsing Gemini transcription response: {resp_e}"
        else:
            # Handle blocked responses or empty candidate list
            error_message = f"No candidates returned from Gemini transcription for {audio_path.name}."
            try:
                # Check for feedback (e.g., safety blocks)
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                     feedback = getattr(response.prompt_feedback, 'block_reason_message', str(response.prompt_feedback))
                     error_message += f" Feedback: {feedback}"
                # Check for explicit error attribute (less common for generate_content)
                elif hasattr(response, 'error'):
                     error_message += f" API Error: {response.error}"
            except Exception as fb_e: error_message += f" (Feedback/Error access error: {fb_e})"
            logger.error(error_message)
            return None, error_message

    except Exception as e:
        # Catch broader errors (network, API issues, etc.)
        logger.error(f"Error during Gemini transcription process for {audio_path.name}: {e}", exc_info=True)
        # Ensure cleanup if file was uploaded but error occurred before deletion
        if uploaded_file and hasattr(uploaded_file, 'name'):
            try: await asyncio.to_thread(genai.delete_file, name=uploaded_file.name); logger.info(f"Deleted temp Gemini file after error: {uploaded_file.name}")
            except Exception as del_e: logger.warning(f"Could not delete temp Gemini file {uploaded_file.name} after error: {del_e}")
        return None, f"Gemini API/SDK error during transcription: {type(e).__name__}: {e}"
    finally:
        # Final check for cleanup, although the explicit deletion points are usually sufficient
        if uploaded_file and hasattr(uploaded_file, 'name'):
             try: await asyncio.to_thread(genai.delete_file, name=uploaded_file.name); logger.debug(f"Ensured Gemini file deletion in finally block: {uploaded_file.name}")
             except Exception as del_e: logger.warning(f"Could not delete Gemini file {uploaded_file.name} in finally block: {del_e}")


def remove_existing_tags(text):
    """Removes only legacy NER_/ENTITY_ tags, preserving specific known tags."""
    if not isinstance(text, str): return text

    # Tags to preserve (case-insensitive matching, but preserved with original case)
    preserve_patterns = [
        r'\bAGE_[A-Z0-9_]+\b',
        r'\bGENDER_(MALE|FEMALE|OTHER|UNKNOWN)\b', # Added UNKNOWN
        r'\bEMOTION_[A-Z_]+\b',
        r'\bSPEAKER_CHANGE\b',
        # Add other specific tags if needed, e.g., r'\bMY_CUSTOM_TAG\b'
        r'\bINTENT_[A-Z_]+\b', # Preserve existing intent tags if they might be present
    ]

    preserved_tags_map = {}
    placeholder_count = 0

    def replace_preserve(match):
        nonlocal placeholder_count
        tag = match.group(0)
        # Store with original case
        placeholder = f"__PRESERVED_{placeholder_count}__"
        preserved_tags_map[placeholder] = tag
        placeholder_count += 1
        return f" {placeholder} " # Add spaces

    temp_text = text
    for pattern in preserve_patterns:
        temp_text = re.sub(pattern, replace_preserve, temp_text, flags=re.IGNORECASE)

    # Remove legacy/unwanted tags (adjust patterns as needed)
    # Example: Remove NER_ followed by word characters
    cleaned_text = re.sub(r'\bNER_\w+\s?', '', temp_text, flags=re.IGNORECASE)
    # Example: Remove specific old entity tags if necessary
    # cleaned_text = re.sub(r'\bENTITY_(OLDTAG1|OLDTAG2)\s?', '', cleaned_text, flags=re.IGNORECASE)
    # Avoid removing generic 'END' unless context is certain. `fix_end_tags` is safer.

    # Restore preserved tags with original casing
    for placeholder, original_tag in preserved_tags_map.items():
        # Replace placeholder surrounded by spaces with original tag surrounded by spaces
        cleaned_text = cleaned_text.replace(f" {placeholder} ", f" {original_tag} ")

    # Final cleanup
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()
    return cleaned_text


def fix_end_tags(text):
    """Cleans up spacing around tags and fixes split entity types."""
    if not isinstance(text, str): return text

    # Fix split entity types like 'ENTITY_TYPE X END' -> 'ENTITY_TYPEX END'
    # This regex looks for ENTITY_, captures the type part, checks for a space,
    # captures a single alphanumeric char, then spaces, then END.
    text = re.sub(r'(ENTITY_[A-Z0-9_]+)\s([A-Z0-9])\s+END\b', r'\1\2 END', text, flags=re.IGNORECASE)

    # Ensure space before END if attached to a word: wordEND -> word END
    text = re.sub(r'(\S)END\b', r'\1 END', text, flags=re.IGNORECASE)
    # Ensure space after ENTITY tag if attached to next word: ENTITY_TYPEword -> ENTITY_TYPE word
    text = re.sub(r'(ENTITY_[A-Z0-9_]+)(\S)', r'\1 \2', text, flags=re.IGNORECASE)
    # Ensure space before INTENT tag if attached to a word: wordINTENT_ -> word INTENT_
    text = re.sub(r'(\S)(INTENT_[A-Z_]+\b)', r'\1 \2', text, flags=re.IGNORECASE) # Added \b to INTENT_

    # Ensure space *after* END if followed immediately by a non-space char: ENDword -> END word
    text = re.sub(r'\bEND(\S)', r'END \1', text, flags=re.IGNORECASE)

    # Clean up multiple spaces
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

async def annotate_text_with_gemini(text_to_annotate: str) -> tuple[Optional[str], Optional[str]]:
    """Annotates text for intent/entities using Gemini, preserving existing specific tags."""
    if not GEMINI_CONFIGURED: return None, "Gemini API is not configured."
    if not text_to_annotate or text_to_annotate.isspace():
        logger.warning("Skipping Gemini annotation: Input text is empty.")
        return "", None # Return empty string for empty input, consistent with transcription

    # Clean potential legacy tags but preserve A/G/E/Speaker/Intent tags passed in
    cleaned_text_for_prompt = remove_existing_tags(text_to_annotate)
    if not cleaned_text_for_prompt or cleaned_text_for_prompt.isspace():
        logger.warning(f"Skipping Gemini annotation: Text empty after cleaning. Original: '{text_to_annotate[:50]}...'")
        # If cleaning removed everything, return the original text (which likely only had tags)
        return text_to_annotate, None

    # Load the extensive entity list from the prompt's JSON string
    entity_list_json_str = """
    ["PERSON_NAME", "ORGANIZATION", "LOCATION", "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP_CODE", "CURRENCY", "PRICE",
    "DATE", "TIME", "DURATION", "APPOINTMENT_DATE", "APPOINTMENT_TIME", "NUMBER", "DEADLINE", "DELIVERY_DATE", "DELIVERY_TIME",
    "EVENT", "MEETING", "TASK", "PROJECT_NAME", "ACTION_ITEM", "PRIORITY", "FEEDBACK", "REVIEW", "RATING", "COMPLAINT",
    "QUESTION", "RESPONSE", "NOTIFICATION_TYPE", "AGENDA", "REMINDER", "NOTE", "RECORD", "ANNOUNCEMENT", "UPDATE",
    "SCHEDULE", "BOOKING_REFERENCE", "APPOINTMENT_NUMBER", "ORDER_NUMBER", "INVOICE_NUMBER", "PAYMENT_METHOD",
    "PAYMENT_AMOUNT", "BANK_NAME", "ACCOUNT_NUMBER", "CREDIT_CARD_NUMBER", "TAX_ID", "SOCIAL_SECURITY_NUMBER",
    "DRIVER'S_LICENSE", "PASSPORT_NUMBER", "INSURANCE_PROVIDER", "POLICY_NUMBER", "INSURANCE_PLAN", "CLAIM_NUMBER",
    "POLICY_HOLDER", "BENEFICIARY", "RELATIONSHIP", "EMERGENCY_CONTACT", "PROJECT_PHASE", "VERSION", "DEVELOPMENT_STAGE",
    "DEVICE_NAME", "OPERATING_SYSTEM", "SOFTWARE_VERSION", "BRAND", "MODEL_NUMBER", "LICENSE_PLATE", "VEHICLE_MAKE",
    "VEHICLE_MODEL", "VEHICLE_TYPE", "FLIGHT_NUMBER", "HOTEL_NAME", "ROOM_NUMBER", "TRANSACTION_ID", "TICKET_NUMBER",
    "SEAT_NUMBER", "GATE", "TERMINAL", "TRANSACTION_TYPE", "PAYMENT_STATUS", "PAYMENT_REFERENCE", "INVOICE_STATUS",
    "SYMPTOM", "DIAGNOSIS", "MEDICATION", "DOSAGE", "ALLERGY", "PRESCRIPTION", "TEST_NAME", "TEST_RESULT", "MEDICAL_RECORD",
    "HEALTH_STATUS", "HEALTH_METRIC", "VITAL_SIGN", "DOCTOR_NAME", "HOSPITAL_NAME", "DEPARTMENT", "WARD", "CLINIC_NAME",
    "WEBSITE", "URL", "IP_ADDRESS", "MAC_ADDRESS", "USERNAME", "PASSWORD", "LANGUAGE", "CODE_SNIPPET", "DATABASE_NAME",
    "API_KEY", "WEB_TOKEN", "URL_PARAMETER", "SERVER_NAME", "ENDPOINT", "DOMAIN",
    "PRODUCT", "SERVICE", "CATEGORY", "BRAND_NAME", "ORDER_STATUS", "DELIVERY_METHOD", "RETURN_STATUS", "WARRANTY_PERIOD",
    "CANCELLATION_REASON", "REFUND_AMOUNT", "EXCHANGE_ITEM", "GIFT_OPTION", "GIFT_MESSAGE",
    "FOOD_ITEM", "DRINK_ITEM", "CUISINE", "MENU_ITEM", "RECIPE", "INGREDIENT",
    "DISH_NAME", "PORTION_SIZE", "COOKING_TIME", "PREPARATION_METHOD",
    "NATIONALITY", "RELIGION", "MARITAL_STATUS", "OCCUPATION", "EDUCATION_LEVEL", "DEGREE",
    "SKILL", "EXPERIENCE", "YEARS_OF_EXPERIENCE", "CERTIFICATION",
    "MEASUREMENT", "DISTANCE", "WEIGHT", "HEIGHT", "VOLUME", "TEMPERATURE", "SPEED", "CAPACITY", "DIMENSION", "AREA",
    "SHAPE", "COLOR", "MATERIAL", "TEXTURE", "PATTERN", "STYLE",
    "WEATHER_CONDITION", "TEMPERATURE_SETTING", "HUMIDITY_LEVEL", "WIND_SPEED", "RAIN_INTENSITY", "AIR_QUALITY",
    "POLLUTION_LEVEL", "UV_INDEX",
    "QUESTION_TYPE", "REQUEST_TYPE", "SUGGESTION_TYPE", "ALERT_TYPE", "REMINDER_TYPE", "STATUS", "ACTION", "COMMAND",
    "USER_HANDLE", "EMAIL_ADDRESS", "PHONE_NUMBER", "IPV4_ADDRESS", "IPV6_ADDRESS", "GPS_COORDINATE", "LATITUDE", "LONGITUDE", "GEOLOCATION",
    "STREET_NAME", "BUILDING_NUMBER", "FLOOR_NUMBER", "BUSINESS_NAME", "MODEL", "SERIAL_NUMBER", "IMEI", "IMSI", "DEVICE_ID",
    "OS_VERSION", "FILE_PATH", "FILE_NAME", "FILE_EXTENSION", "DOCUMENT_TITLE", "DOCUMENT_ID", "LEGAL_ENTITY", "TAX_DOCUMENT",
    "BILLING_ADDRESS", "SHIPPING_ADDRESS", "COUPON_CODE", "LOYALTY_CARD_NUMBER", "PRODUCT_ID", "SKU", "BARCODE", "QR_CODE",
    "TRANSACTION_CODE", "EVENT_ID", "SESSION_ID", "ACTION_ID", "CLICK_POSITION", "SCROLL_DEPTH",
    "VIDEO_ID", "AUDIO_TRACK", "SUBTITLE_LANGUAGE", "CHAPTER_TITLE", "CHAPTER_NUMBER", "EPISODE_NUMBER",
    "MOVIE_TITLE", "DIRECTOR_NAME", "ACTOR_NAME", "BOOK_TITLE", "AUTHOR_NAME", "PUBLISHER", "ISBN", "ISSN",
    "COURSE_NAME", "INSTRUCTOR_NAME", "STUDENT_ID", "GRADE", "CLASSROOM_NUMBER", "SCHOOL_NAME", "DEGREE_PROGRAM",
    "MAJOR", "MINOR", "CERTIFICATE_NAME", "EXAM_SCORE", "CERTIFICATION_ID", "TRAINING_PROGRAM",
    "PLATFORM", "APPLICATION", "SOFTWARE_PACKAGE", "API_ENDPOINT", "SERVICE_NAME", "SERVER_IP", "DATABASE_TABLE",
    "QUERY", "ERROR_CODE", "LOG_LEVEL", "SESSION_DURATION", "BROWSER_TYPE", "DEVICE_TYPE"
    ]
    """
    entity_list_for_prompt = json.dumps(json.loads(entity_list_json_str))


    prompt = f'''Analyze the following sentence. It might already contain tags like AGE_*, GENDER_*, EMOTION_*, and SPEAKER_CHANGE.

Your tasks are:
1.  **Preserve Existing Tags:** Keep any AGE_*, GENDER_*, EMOTION_*, or SPEAKER_CHANGE tags exactly where they are in the input sentence. Do not modify or move them.
2.  **Annotate Entities:** Identify and tag entities **only** from this specific list: {entity_list_for_prompt}. Use the format `ENTITY_<TYPE> identified text END`. Ensure `<TYPE>` exactly matches an entry in the list (e.g., `ENTITY_PERSON_NAME`, `ENTITY_DATE`). Do not tag anything not on the list. Place the start tag immediately before the entity and the ` END` tag (with a preceding space) immediately after it.
3.  **Classify Intent:** Determine the primary intent of the sentence. Choose a concise, descriptive intent label (e.g., `REQUEST_INFO`, `BOOK_APPOINTMENT`, `PROVIDE_FEEDBACK`, `MAKE_PURCHASE`, `SOCIAL_CHITCHAT`, `HEALTH_UPDATE`, `TECHNICAL_SUPPORT`).
4.  **Add Intent Tag:** Append exactly ONE `INTENT_<ChosenIntentLabel>` tag at the VERY END of the entire processed sentence, after all text, entity tags, preserved tags, and ` END` markers.
5.  **Output:** Return the fully processed sentence as a single string.

**CRITICAL FORMATTING RULES:**
*   Do NOT add extra spaces around the `ENTITY_<TYPE>` start tag.
*   Ensure there is exactly one space between the identified entity text and its ` END` tag. Example: `ENTITY_CITY london END` (Correct), `ENTITY_CITYlondon END` (Incorrect), `ENTITY_CITY londonEND` (Incorrect).
*   Ensure the full entity type name is used and kept together. Example: `ENTITY_PERSON_NAME david END` (Correct), `ENTITY_PERSON_NAM E david END` (Incorrect).
*   The final `INTENT_<TYPE>` tag must be the absolute last part of the output string.

**Example Input:** "can you book a flight for ENTITY_PERSON_NAME alice END to ENTITY_CITY paris END on ENTITY_DATE next tuesday END GENDER_FEMALE AGE_25_34 EMOTION_NEUTRAL"
**Example Output:** "can you book a flight for ENTITY_PERSON_NAME alice END to ENTITY_CITY paris END on ENTITY_DATE next tuesday END GENDER_FEMALE AGE_25_34 EMOTION_NEUTRAL INTENT_BOOK_FLIGHT"

**Example Input:** "the meeting is scheduled for ENTITY_TIME 3 pm END in the ENTITY_LOCATION main conference room END AGE_45_54 GENDER_MALE EMOTION_CALM SPEAKER_CHANGE"
**Example Output:** "the meeting is scheduled for ENTITY_TIME 3 pm END in the ENTITY_LOCATION main conference room END AGE_45_54 GENDER_MALE EMOTION_CALM SPEAKER_CHANGE INTENT_PROVIDE_INFO"

**Text to Annotate:** "{cleaned_text_for_prompt}"'''
 

    logger.info(f"Annotating with Gemini (input after cleaning: '{cleaned_text_for_prompt[:100]}...')")
    try:
        # Consider caching the model instance if making frequent calls
        model = genai.GenerativeModel("gemini-1.5-flash") # Or "gemini-1.5-pro" for potentially higher quality

        # Blocking call in thread
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            request_options={'timeout': 180}, 
        )

        if response.candidates:
             annotated_text = ""
             try:
                
                 if hasattr(response, 'text') and response.text is not None:
                     annotated_text = response.text.strip()
                 elif response.candidates[0].content.parts:
                     annotated_text = "".join(part.text for part in response.candidates[0].content.parts).strip()
                 else:
                    logger.warning(f"Gemini annotation: No text found in response candidate for input '{cleaned_text_for_prompt[:50]}...'. Response: {response}")
                    # Fallback: Return the input text + error message
                    return text_to_annotate, "Gemini response candidate found, but no text content."

                 # Clean potential markdown code blocks if Gemini wraps the output
                 if annotated_text.startswith("```") and annotated_text.endswith("```"):
                     annotated_text = re.sub(r'^```[a-z]*\n?|\n?```$', '', annotated_text).strip()

                 # Apply tag spacing fixes (critical after LLM generation)
                 final_text = fix_end_tags(annotated_text)

                 # Validation: Check if an INTENT tag exists at the end
                 if not re.search(r'\sINTENT_[A-Z_]+$', final_text):
                     logger.warning(f"Gemini annotation result missing expected INTENT tag at the end: '{final_text[-60:]}'")
                     # Decide fallback: Append a generic intent? Or return as is?
                     # Let's append a generic one for consistency in the manifest.
                     final_text = f"{final_text.strip()} INTENT_UNKNOWN".strip()
                     logger.info("Appended INTENT_UNKNOWN due to missing intent tag.")


                 logger.info(f"Gemini annotation successful (result: '{final_text[:100]}...')")
                 return final_text, None

             except (AttributeError, IndexError, ValueError, TypeError) as resp_e:
                 logger.warning(f"Could not extract text from Gemini annotation response for input '{cleaned_text_for_prompt[:50]}...': {resp_e}. Response: {response}")
                 # Fallback: Return the input text + error message
                 return text_to_annotate, f"Error parsing Gemini annotation response: {resp_e}"

        else: # Handle blocked responses etc.
            error_message = f"No candidates returned from Gemini annotation for input '{cleaned_text_for_prompt[:50]}...'."
            try:
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                     feedback = getattr(response.prompt_feedback, 'block_reason_message', str(response.prompt_feedback))
                     error_message += f" Feedback: {feedback}"
                elif hasattr(response, 'error') : # Check for explicit error
                     error_message += f" API Error: {response.error}"
            except Exception as fb_e: error_message += f" (Feedback/Error access error: {fb_e})"
            logger.error(error_message)
            # Fallback: Return the input text + error message
            return text_to_annotate, error_message

    except Exception as e:
        logger.error(f"Error during Gemini annotation API call for input '{cleaned_text_for_prompt[:50]}...': {e}", exc_info=True)
        # Fallback: Return the input text + error message
        return text_to_annotate, f"Gemini API annotation error: {type(e).__name__}: {e}"


# --- Shared Validation and File Handling Logic ---

def validate_paths(directory_path: str, output_jsonl_path: str) -> Tuple[Path, Path]:
    """Validates input directory and output file paths."""
    try:
        dir_path = Path(directory_path).resolve(strict=True)
        if not dir_path.is_dir():
            raise HTTPException(status_code=404, detail=f"Directory not found or is not a directory: {dir_path}")
    except (FileNotFoundError, ValueError, TypeError, OSError) as path_e: # Added OSError
         raise HTTPException(status_code=400, detail=f"Invalid directory path: {directory_path}. Error: {path_e}")

    try:
        output_path = Path(output_jsonl_path).resolve()
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Output path is a directory, must be a file path: {output_path}")
        # Check write permissions for the parent directory
        if not os.access(output_path.parent, os.W_OK):
             raise HTTPException(status_code=403, detail=f"No write permissions for output directory: {output_path.parent}")
        # Log if overwriting
        if output_path.exists():
             logger.warning(f"Output file {output_path} exists and will be overwritten.")
    except HTTPException: # Re-raise validation errors
        raise
    except Exception as out_path_e:
        raise HTTPException(status_code=400, detail=f"Invalid or inaccessible output JSONL path: {output_jsonl_path}. Error: {out_path_e}")

    return dir_path, output_path

def discover_audio_files(dir_path: Path) -> List[Path]:
    """Finds supported audio files in the directory."""
    try:
        audio_files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]
        logger.info(f"Found {len(audio_files)} supported audio files in {dir_path}")
        return audio_files
    except Exception as e:
         logger.error(f"Error reading directory {dir_path}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Error scanning directory for audio files: {e}")

# --- Endpoint 1: Create Transcription-Only Manifest ---

@app.post("/create_transcription_manifest/",
          response_model=ProcessResponse,
          summary="Create Transcription-Only Manifest",
          description="Loads audio files from a directory, transcribes using the selected model (Gemini/Whissle), "
                      "and saves results (audio_filepath, text, duration) to a JSONL file.")
async def create_transcription_manifest_endpoint(process_request: ProcessRequest):
    """
    API endpoint to handle transcription-only manifest creation.
    """
    model_choice = process_request.model_choice
    if model_choice == ModelChoice.whissle and not WHISSLE_CONFIGURED:
         raise HTTPException(status_code=400, detail="Whissle model selected, but not configured (token/SDK).")
    if model_choice == ModelChoice.gemini and not GEMINI_CONFIGURED:
        raise HTTPException(status_code=400, detail="Gemini model selected, but not configured (API key).")

    dir_path, output_jsonl_path = validate_paths(process_request.directory_path, process_request.output_jsonl_path)
    audio_files = discover_audio_files(dir_path)

    if not audio_files:
        # Create an empty file as requested by validate_paths implicitly allowing it
        try:
            with open(output_jsonl_path, "w", encoding="utf-8") as _:
                pass # Create empty file
            logger.info(f"No audio files found in {dir_path}. Created empty output file: {output_jsonl_path}")
        except IOError as e:
             raise HTTPException(status_code=500, detail=f"Failed to create empty output file: {e}")

        return ProcessResponse(
            message=f"No supported audio files found in {dir_path}. Empty manifest file created.",
            output_file=str(output_jsonl_path),
            processed_files=0, saved_records=0, errors=0
        )

    processed_files_count = 0
    saved_records_count = 0
    error_count = 0

    # --- Processing Loop (Transcription Only) ---
    try:
        with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
            for audio_file in audio_files:
                file_error: Optional[str] = None
                transcription_text: Optional[str] = None
                duration: Optional[float] = None

                logger.info(f"--- Processing {audio_file.name} (Transcription Only) ---")
                processed_files_count += 1

                try:
                    # 1. Get Duration
                    duration = get_audio_duration(audio_file)
                    if duration is None:
                        file_error = "Failed to get audio duration."
                        # Continue to attempt transcription? Maybe duration isn't critical for Nemo sometimes.
                        # Let's allow it to continue but log the duration error.

                    # 2. Transcription (Async)
                    transcription_error = None
                    if model_choice == ModelChoice.whissle:
                        transcription_text, transcription_error = await transcribe_with_whissle_single(audio_file)
                    else: # Gemini
                        transcription_text, transcription_error = await transcribe_with_gemini_single(audio_file)

                    if transcription_error:
                        if file_error: file_error += f"; Transcription failed: {transcription_error}" # Append error
                        else: file_error = f"Transcription failed: {transcription_error}"
                    elif transcription_text is None:
                        # Safeguard against transcription returning None without error
                        if file_error: file_error += "; Transcription returned None unexpectedly."
                        else: file_error = "Transcription returned None unexpectedly."
                    else:
                        transcription_text = transcription_text.strip() # Clean whitespace

                except Exception as e:
                    logger.error(f"Unexpected error processing {audio_file.name} (Transcription Only): {e}", exc_info=True)
                    file_error = f"Unexpected processing error: {type(e).__name__}: {e}"
                    # Don't increment error_count here, it's handled below based on file_error

                # Create JSONL record
                record = TranscriptionJsonlRecord(
                    audio_filepath=str(audio_file.resolve()), # Use absolute path
                    text=transcription_text, # Just the raw transcription
                    duration=duration,
                    model_used_for_transcription=model_choice.value,
                    error=file_error # Log specific step errors for this file
                )

                # Write record to file immediately
                try:
                    outfile.write(record.model_dump_json(exclude_none=True) + "\n")
                    # outfile.flush() # Optional: force write to disk, might impact performance
                except Exception as write_e:
                    # Log error for this specific write, but try to continue with others
                    logger.error(f"Failed to write record for {audio_file.name} to {output_jsonl_path}: {write_e}", exc_info=True)
                    # Store the error on the record itself if possible? No, the record is already defined.
                    # Mark this file as having an error
                    if not file_error: file_error = f"Failed to write record to JSONL: {write_e}"
                    else: file_error += f"; Failed to write record to JSONL: {write_e}"


                if file_error:
                    error_count += 1
                else:
                    saved_records_count += 1

                # Clean up (less critical here as no large tensors involved)
                gc.collect()

        logger.info(f"Finished transcription-only processing. Wrote {processed_files_count} records attempt to {output_jsonl_path}")

    except IOError as e:
        logger.error(f"Fatal I/O error writing to {output_jsonl_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to write output file: {e}")

    # --- Final Response ---
    final_message = (
        f"Processed {processed_files_count}/{len(audio_files)} files for transcription-only manifest. "
        f"Attempted to save {processed_files_count} records. "
        f"{saved_records_count} records saved successfully. "
        f"{error_count} files encountered processing or writing errors (check logs and 'error' field in JSONL)."
    )

    return ProcessResponse(
        message=final_message,
        output_file=str(output_jsonl_path),
        processed_files=processed_files_count,
        saved_records=saved_records_count,
        errors=error_count
    )


# --- Endpoint 2: Create Annotated Manifest (Refactored Original Logic) ---

@app.post("/create_annotated_manifest/",
          response_model=ProcessResponse,
          summary="Create Annotated Manifest",
          description="Loads audio, transcribes (Gemini/Whissle), predicts Age/Gender/Emotion, "
                      "annotates Intent/Entities (Gemini), combines results into the 'text' field, "
                      "and saves to a JSONL file.")
async def create_annotated_manifest_endpoint(process_request: ProcessRequest):
    """
    API endpoint to handle full processing and annotation, saving to JSONL.
    (This is the refactored logic from the original /process_audio_directory/ endpoint)
    """
    model_choice = process_request.model_choice
    # --- Input Validation ---
    if model_choice == ModelChoice.whissle and not WHISSLE_CONFIGURED:
         raise HTTPException(status_code=400, detail="Whissle model selected, but not configured (token/SDK).")
    # Gemini is required for both transcription (if selected) AND annotation
    if not GEMINI_CONFIGURED:
        raise HTTPException(status_code=400, detail="Gemini model required for annotation (and potentially transcription) is not configured (API key).")

    # Check if annotation models are loaded
    models_missing = []
    if age_gender_model is None: models_missing.append("Age/Gender")
    if emotion_model is None: models_missing.append("Emotion")
    # Gemini check already done above

    if models_missing:
         # Warn the user but proceed, annotations will be partial
         warning_msg = f"Warning: Annotation models not loaded: {', '.join(models_missing)}. Proceeding without these predictions."
         logger.warning(warning_msg)
         # If these are absolutely essential, you could raise an HTTPException here instead:
         # raise HTTPException(status_code=503, detail=f"Required annotation models not loaded: {', '.join(models_missing)}")

    dir_path, output_jsonl_path = validate_paths(process_request.directory_path, process_request.output_jsonl_path)
    audio_files = discover_audio_files(dir_path)

    if not audio_files:
        try:
            with open(output_jsonl_path, "w", encoding="utf-8") as _: pass
            logger.info(f"No audio files found in {dir_path}. Created empty output file: {output_jsonl_path}")
        except IOError as e:
             raise HTTPException(status_code=500, detail=f"Failed to create empty output file: {e}")

        return ProcessResponse(
            message=f"No supported audio files found in {dir_path}. Empty manifest file created.",
            output_file=str(output_jsonl_path),
            processed_files=0, saved_records=0, errors=0
        )

    processed_files_count = 0
    saved_records_count = 0
    error_count = 0

    # --- Processing Loop (Full Annotation) ---
    try:
        with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
            for audio_file in audio_files:
                file_error_details = [] # Collect multiple error messages for a file
                combined_text: Optional[str] = None
                duration: Optional[float] = None
                audio_data: Optional[np.ndarray] = None
                sample_rate: Optional[int] = None
                transcription_text: Optional[str] = None
                age_gender_emotion_tags: str = "" # Initialize

                logger.info(f"--- Processing {audio_file.name} (Full Annotation) ---")
                processed_files_count += 1

                try:
                    # 1. Get Duration & Load Audio (Needed for A/G/E)
                    duration = get_audio_duration(audio_file)
                    if duration is None:
                        file_error_details.append("Failed to get audio duration.")
                        # Attempt to load audio anyway, A/G/E might still work partially
                    audio_data, sample_rate, load_err = load_audio(audio_file, TARGET_SAMPLE_RATE)
                    if load_err:
                        file_error_details.append(load_err)
                    elif audio_data is None or sample_rate != TARGET_SAMPLE_RATE:
                        # This case might indicate an issue post-load or during resampling
                        file_error_details.append("Audio loading failed or Sample Rate mismatch after loading.")
                    # Don't halt processing yet, A/G/E might still be attempted if audio_data exists partially


                    # 2. Transcription (Async)
                    transcription_error = None
                    if model_choice == ModelChoice.whissle:
                        transcription_text, transcription_error = await transcribe_with_whissle_single(audio_file)
                    else: # Gemini
                        transcription_text, transcription_error = await transcribe_with_gemini_single(audio_file)

                    if transcription_error:
                        file_error_details.append(f"Transcription failed: {transcription_error}")
                        # Transcription failed, but we might still have audio_data for A/G/E
                    elif transcription_text is None:
                        file_error_details.append("Transcription returned None unexpectedly.")
                    else:
                        transcription_text = transcription_text.strip() # Clean whitespace

                    # 3. Predict Age/Gender/Emotion (Run IF audio loaded successfully)
                    age_pred, gender_idx, emotion_label = None, None, None # Initialize
                    if audio_data is not None and sample_rate == TARGET_SAMPLE_RATE:
                        try:
                            # Run predictions in parallel using asyncio.gather for efficiency
                            age_gender_task = asyncio.to_thread(predict_age_gender, audio_data, TARGET_SAMPLE_RATE)
                            emotion_task = asyncio.to_thread(predict_emotion, audio_data, TARGET_SAMPLE_RATE)

                            age_gender_result, emotion_result = await asyncio.gather(age_gender_task, emotion_task)

                            age_pred, gender_idx, age_gender_err = age_gender_result
                            if age_gender_err: logger.warning(f"Age/Gender prediction issue for {audio_file.name}: {age_gender_err}"); file_error_details.append(f"AGE_GENDER_WARN: {age_gender_err}")

                            emotion_label, emotion_err = emotion_result
                            if emotion_err: logger.warning(f"Emotion prediction issue for {audio_file.name}: {emotion_err}"); file_error_details.append(f"EMOTION_WARN: {emotion_err}")

                        except Exception as pred_e:
                             logger.error(f"Error during parallel A/G/E prediction for {audio_file.name}: {pred_e}", exc_info=True)
                             file_error_details.append(f"A/G/E Prediction Error: {pred_e}")
                    elif not file_error_details or "Audio loading failed" not in " ".join(file_error_details):
                        # Log if audio loading failed *before* this step
                        logger.warning(f"Skipping A/G/E prediction for {audio_file.name} due to audio loading issue.")
                        file_error_details.append("Skipped A/G/E prediction (audio load issue).")


                    # Format Age/Gender/Emotion Tags (always attempt, handles None values)
                    age_gender_emotion_tags = format_age_gender_emotion_tags(age_pred, gender_idx, emotion_label)


                    # 4. Annotate with Gemini IF transcription was successful
                    if transcription_text is not None and transcription_text != "":
                         # Prepare text for Gemini: Transcription + A/G/E tags (Gemini will preserve them)
                         text_with_pre_tags = f"{transcription_text} {age_gender_emotion_tags}".strip()

                         gemini_annotated_text, gemini_err = await annotate_text_with_gemini(text_with_pre_tags)

                         if gemini_err:
                             logger.warning(f"Gemini Intent/Entity annotation failed for {audio_file.name}: {gemini_err}. Using transcription + A/G/E tags as fallback.")
                             file_error_details.append(f"ANNOTATION_FAIL: {gemini_err}")
                             # Fallback: Use the text + AGE/GENDER/EMOTION tags
                             combined_text = text_with_pre_tags
                             # Ensure a fallback intent tag if annotation failed
                             if not re.search(r'\sINTENT_[A-Z_]+$', combined_text):
                                 combined_text = f"{combined_text.strip()} INTENT_ANNOTATION_FAILED".strip()
                         elif gemini_annotated_text is None or gemini_annotated_text.strip() == "":
                              logger.warning(f"Gemini annotation returned empty/None for {audio_file.name}. Using transcription + A/G/E tags.")
                              file_error_details.append("ANNOTATION_EMPTY: Gemini returned empty result.")
                              combined_text = text_with_pre_tags # Fallback
                              if not re.search(r'\sINTENT_[A-Z_]+$', combined_text):
                                  combined_text = f"{combined_text.strip()} INTENT_ANNOTATION_EMPTY".strip()
                         else:
                             # Success: Use the fully annotated text from Gemini
                             # Note: The A/G/E tags were already included in the input,
                             # and the prompt asked Gemini to preserve them.
                             combined_text = gemini_annotated_text
                    elif transcription_text == "":
                         # Handle case of empty transcription (no speech detected)
                         logger.info(f"No speech detected in {audio_file.name}. Using A/G/E tags only.")
                         combined_text = f"[NO_SPEECH] {age_gender_emotion_tags}".strip()
                         # Add a specific intent for no speech
                         combined_text = f"{combined_text.strip()} INTENT_NO_SPEECH".strip()
                    else:
                         # Handle transcription failure (transcription_text is None or error occurred)
                         logger.warning(f"Transcription failed for {audio_file.name}. Using A/G/E tags only.")
                         # We already added transcription error to file_error_details
                         combined_text = f"[TRANSCRIPTION_FAILED] {age_gender_emotion_tags}".strip()
                         combined_text = f"{combined_text.strip()} INTENT_TRANSCRIPTION_FAILED".strip()


                except Exception as e:
                    logger.error(f"Unexpected error processing {audio_file.name} (Full Annotation): {e}", exc_info=True)
                    file_error_details.append(f"Unexpected processing error: {type(e).__name__}: {e}")
                    # Ensure some text output even in unexpected error
                    if not combined_text:
                         combined_text = f"[PROCESSING_ERROR] {age_gender_emotion_tags}".strip()
                         combined_text = f"{combined_text.strip()} INTENT_PROCESSING_ERROR".strip()

                # Final error string
                final_error_msg = "; ".join(file_error_details) if file_error_details else None

                # Create JSONL record
                record = AnnotatedJsonlRecord(
                    audio_filepath=str(audio_file.resolve()),
                    text=combined_text, # Use the final combined text
                    duration=duration,
                    model_used_for_transcription=model_choice.value,
                    error=final_error_msg # Store combined errors/warnings
                )

                # Write record to file immediately
                try:
                    outfile.write(record.model_dump_json(exclude_none=True) + "\n")
                    # outfile.flush() # Optional
                except Exception as write_e:
                     logger.error(f"Failed to write annotated record for {audio_file.name} to {output_jsonl_path}: {write_e}", exc_info=True)
                     # Mark error if not already marked by other steps
                     if not final_error_msg: final_error_msg = f"Failed to write record to JSONL: {write_e}"
                     else: final_error_msg += f"; Failed to write record to JSONL: {write_e}"
                     # Update the error count based on the final status
                     error_count += 1 # Increment here as this is a definite failure for this record


                # Increment error count ONLY IF there was an error string generated
                if final_error_msg and "Failed to write record" not in final_error_msg:
                     # Avoid double counting if write failed above
                     error_count += 1
                elif not final_error_msg:
                    saved_records_count += 1

                # Clean up GPU memory aggressively
                del audio_data
                # Force Python garbage collection
                gc.collect()
                # If using PyTorch with CUDA, empty cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.info(f"Finished full annotation processing. Wrote {processed_files_count} records attempt to {output_jsonl_path}")

    except IOError as e:
        logger.error(f"Fatal I/O error writing to {output_jsonl_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to write output file: {e}")

    # --- Final Response ---
    final_message = (
        f"Processed {processed_files_count}/{len(audio_files)} files for full annotation manifest. "
        f"Attempted to save {processed_files_count} records. "
        f"{saved_records_count} records saved representing successful processing. "
        f"{error_count} files encountered errors or warnings during processing/writing (check logs and 'error' field in JSONL)."
    )

    return ProcessResponse(
        message=final_message,
        output_file=str(output_jsonl_path),
        processed_files=processed_files_count,
        saved_records=saved_records_count,
        errors=error_count
    )


# --- Root Endpoint (Keep as is) ---
@app.get("/status", summary="API Status", description="Provides basic information about the API and model status.")
async def get_status():
    return {
        "message": "Welcome to the Audio Processing API v1.4.0",
        "docs_url": "/docs",
        "html_interface": "/",
        "endpoints": {
            "transcription_only": "/create_transcription_manifest/",
            "full_annotation": "/create_annotated_manifest/"
        },
        "gemini_configured": GEMINI_CONFIGURED,
        "whissle_available": WHISSLE_AVAILABLE,
        "whissle_configured": WHISSLE_CONFIGURED,
        "age_gender_model_loaded": age_gender_model is not None,
        "emotion_model_loaded": emotion_model is not None,
        "device": str(device)
        }

# --- Run the app (Keep as is) ---
if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    fastapi_script_name = Path(__file__).stem
    app_module_string = f"{fastapi_script_name}:app"

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    # Default reload to false for production/stability, true for development
    reload_env = os.getenv("RELOAD", "true").lower() # Changed default to true for easier dev
    reload = reload_env == "true"

    log_level = "info" # Or "debug" for more detailed logs

    logger.info(f"Starting FastAPI server for '{app_module_string}' on {host}:{port}...")
    logger.info(f"Log Level: {log_level.upper()}")
    logger.info(f"Reload Mode: {'Enabled' if reload else 'Disabled'}")
    logger.info(f"API documentation available at http://{host}:{port}/docs")
    logger.info(f"HTML interface available at http://{host}:{port}/")


    uvicorn.run(
        app_module_string,
        host=host,
        port=port,
        reload=reload,
        reload_dirs=[str(script_dir)] if reload else None,
        log_level=log_level # Set log level for Uvicorn
    )
