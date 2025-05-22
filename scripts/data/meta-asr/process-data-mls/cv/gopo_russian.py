# -*- coding: utf-8 -*- # Ensure editor recognizes UTF-8 for Russian script comments/examples
import os
import gc
import json
import re
import torch
import librosa
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2Model,          # Keep Wav2Vec2Model for the class definition
    Wav2Vec2PreTrainedModel # Keep for the class definition
)
import torch.nn as nn
import google.generativeai as genai
import time # Added for potential retries/delays
import soundfile as sf # Added for broader audio format support

# --- Initial Setup ---
load_dotenv()
torch.cuda.empty_cache()
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    device = torch.device("cuda:0")
else:
    print("CUDA not available, using CPU.")
    device = torch.device("cpu")

# Configure Google Generative AI
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not found or is empty.")
    genai.configure(api_key=api_key)
    print("Google Generative AI configured successfully.")
except Exception as e:
    print(f"Error configuring or testing Google Generative AI: {e}. Annotation step will likely fail.")
    genai = None

# --- Age/Gender Model Definition (<<< CHANGE: Using the definition from the user's first block) ---
class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        # Linear layers for age (1 output) and gender (2 outputs: male/female)
        self.age = torch.nn.Linear(config.hidden_size, 1)
        self.gender = torch.nn.Linear(config.hidden_size, 2) # 2 classes (male/female)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        # Mean pool the hidden states across the time dimension
        hidden_states_pooled = torch.mean(hidden_states, dim=1)
        # Pass pooled states through respective linear layers
        logits_age = self.age(hidden_states_pooled)
        # Apply softmax to gender logits AFTER the linear layer
        logits_gender = self.gender(hidden_states_pooled)
        # Return raw age logits and gender logits (softmax will be applied later if needed for probabilities)
        # Or apply softmax here if the loss function expects probabilities. For inference, argmax on logits is fine.
        # Let's return logits for age and softmaxed probabilities for gender for clarity in prediction.
        probabilities_gender = torch.softmax(logits_gender, dim=-1)
        return logits_age, probabilities_gender # Return raw age logit and gender probabilities


# --- Emotion Extraction Function (Unchanged) ---
def extract_emotion(audio_data: np.ndarray, sampling_rate: int = 16000, model_info: dict = None) -> str:
    if model_info is None or 'model' not in model_info or 'feature_extractor' not in model_info:
        print("Error: Emotion model not loaded correctly.")
        return "ErrorLoadingModel"
    if audio_data is None or len(audio_data) == 0:
        return "No_Audio"

    try:
        inputs = model_info['feature_extractor'](
            audio_data,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model_info['model'](**inputs)
        predicted_class_idx = outputs.logits.argmax(-1).item()
        emotion_label = model_info['model'].config.id2label.get(predicted_class_idx, "Unknown")
        return emotion_label.upper()
    except Exception as e:
        print(f"Error during emotion extraction: {e}")
        return "Extraction_Error"

# --- Data Structures ---
@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    speaker: str # Can be extracted if needed, or set to default
    age: float # <<< CHANGE: Will store the scaled 0-100 age
    gender: str
    transcription: str # This will hold Russian text
    emotion: str
    chunk_filename: str
    duration: float

@dataclass
class ChunkData:
    segments: List[AudioSegment] = field(default_factory=list)
    filepath: str = ""

    def get_formatted_text(self) -> str:
        if not self.segments: return ""
        segment = self.segments[0]
        # <<< CHANGE: Pass the already scaled age (0-100) to get_age_bucket
        age_bucket = self.get_age_bucket(segment.age)
        gender_text = segment.gender # Already "MALE" or "FEMALE"
        emotion_text = segment.emotion.upper()
        transcription = segment.transcription.strip() # Keep original case for AI
        metadata = f"AGE_{age_bucket} GENDER_{gender_text} EMOTION_{emotion_text}"
        return f"{transcription.strip()} {metadata.strip()}"

    @staticmethod
    def get_age_bucket(scaled_age: float) -> str: # <<< CHANGE: Input is scaled_age (0-100)
        """ Creates age bucket string from a scaled age value (0-100). """
        # Handle potential error case where age is negative or invalid
        if scaled_age < 0: return "UNKNOWN"
        # Age is already scaled 0-100, just round it
        actual_age = round(scaled_age)
        age_brackets = [
            (18, "0_18"), (30, "18_30"), (45, "30_45"),
            (60, "45_60"), (float('inf'), "60PLUS")
        ]
        for threshold, bracket in age_brackets:
            if actual_age < threshold: return bracket
        return "60PLUS" # Default

# --- Manifest Reading Function (Unchanged) ---
def read_manifest(manifest_path: str, base_data_dir: str) -> List[Dict[str, Any]]:
    """Reads a JSON Lines manifest file and constructs full audio paths."""
    records = []
    try:
        print(f"Reading manifest file: {manifest_path}")
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    record = json.loads(line.strip())
                    if "audio_filepath" not in record or "text" not in record:
                         print(f"Warning: Skipping line {line_num+1} in manifest. Missing 'audio_filepath' or 'text'. Content: {line.strip()}")
                         continue
                    record['full_audio_filepath'] = os.path.join(base_data_dir, record['audio_filepath'])
                    record['transcription'] = record.get('text_no_preprocessing', record.get('text', '')).strip()
                    if not record['transcription']:
                        print(f"Warning: Skipping line {line_num+1} due to empty transcription. Audio: {record['audio_filepath']}")
                        continue
                    if 'duration' not in record or not isinstance(record['duration'], (int, float)) or record['duration'] <= 0:
                        record['duration'] = -1.0
                    records.append(record)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line {line_num+1} in manifest: {line.strip()}")
                except Exception as e:
                    print(f"Warning: Error processing line {line_num+1} in manifest: {e}. Line: {line.strip()}")
        print(f"Successfully read {len(records)} valid records from manifest.")
        return records
    except FileNotFoundError:
        print(f"Error: Manifest file not found at {manifest_path}")
        return []
    except Exception as e:
        print(f"Error reading manifest file {manifest_path}: {e}")
        return []


# --- AI Annotation Functions (Unchanged - Russian prompt maintained) ---
def correct_entity_tag_spaces(text: str) -> str:
    """Removes spaces within ENTITY_TYPE names like 'CIT Y' -> 'CITY'."""
    if not isinstance(text, str): return text
    def replace_spaces(match):
        tag_part = match.group(1)
        type_part = tag_part[len("ENTITY_"):]
        corrected_type = type_part.replace(' ', '')
        return f"ENTITY_{corrected_type}"
    pattern = r'\b(ENTITY_[A-Z0-9_ ]*?[A-Z0-9_])(?=\s+\S)'
    corrected_text = re.sub(pattern, replace_spaces, text)
    return corrected_text

def fix_end_tags_and_spacing(text: str) -> str:
    """Cleans up END tags and fixes spacing around tags and standard Russian punctuation."""
    if not isinstance(text, str): return text
    text = correct_entity_tag_spaces(text)
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+END\s+(\b(?:AGE_|GENDER_|EMOTION_|INTENT_)|$)', r' \1', text)
    text = re.sub(r'\s+END\s+END\b', ' END', text)
    text = re.sub(r'\s+END\s+END\b', ' END', text)
    pattern_add_end = r'(ENTITY_[A-Z0-9_]+\s+\S.*?)(?<!\sEND)(?=\s+(\bAGE_|\bGENDER_|\bEMOTION_|\bINTENT_|\bENTITY_)|$)'
    text = re.sub(pattern_add_end, r'\1 END', text)
    text = re.sub(r'(ENTITY_[A-Z0-9_]+)(\S)', r'\1 \2', text)
    text = re.sub(r'(\S)(END\b)', r'\1 \2', text)
    text = re.sub(r'\s+([?!:;,.])', r'\1', text)
    text = re.sub(r'([?!:;,.])(\w)', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Russian Prompt (Unchanged)
def annotate_batch_texts(texts_to_annotate: List[str]):
    """Sends a batch of texts to Gemini for annotation (Prompt adapted for Russian)."""
    if not genai:
        print("Error: Google Generative AI not configured. Skipping annotation.")
        return texts_to_annotate
    if not texts_to_annotate:
        return []

    prompt = f'''You are an expert linguistic annotator for **Russian** text written in the **Cyrillic** script.
You will receive a list of Russian sentences. Each sentence already includes metadata tags (AGE_*, GENDER_*, EMOTION_*) at the end.

Your task is crucial and requires precision:
1.  **PRESERVE EXISTING TAGS:** Keep the `AGE_`, `GENDER_`, and `EMOTION_` tags exactly as they appear at the end of each sentence. DO NOT modify or move them.
2.  **ENTITY ANNOTATION (Russian Text Only):** Identify entities ONLY within the Russian transcription part of the sentence. Use ONLY the entity types from the provided list.
3.  **ENTITY TAG FORMAT (VERY IMPORTANT):**
    *   Insert the tag **BEFORE** the entity: `ENTITY_<TYPE>` (e.g., `ENTITY_CITY`, `ENTITY_PERSON_NAME`).
    *   **NO SPACES** are allowed within the `<TYPE>` part (e.g., use `PERSON_NAME`, NOT `PERSON_ NAM E`).
    *   Immediately **AFTER** the Russian entity text, add a single space followed by `END`.
    *   Example: `... ENTITY_CITY Москве END ...` (using Cyrillic script for Moscow)
    *   **DO NOT** add an `END` tag just before the `AGE_`, `GENDER_`, `EMOTION_` tags unless it belongs to a preceding entity.
4.  **INTENT TAG:** Determine the single primary intent of the Russian transcription (e.g., INFORM, QUESTION, REQUEST, COMMAND, GREETING, etc.). Add ONE `INTENT_<INTENT_TYPE>` tag at the absolute end of the entire string, AFTER all other tags.
5.  **OUTPUT FORMAT:** Return a JSON array of strings, where each string is a fully annotated sentence adhering to all rules.
6.  **RUSSIAN SPECIFICS:** Handle **Cyrillic script**, standard Russian punctuation (periods, commas, question marks, exclamation points, etc.), and spacing correctly according to standard Russian rules. Ensure proper spacing around the inserted tags.

**ENTITY TYPES LIST (USE ONLY THESE):**
[
    "PERSON_NAME", "ORGANIZATION", "LOCATION", "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP_CODE", "CURRENCY", "PRICE",
    "DATE", "TIME", "DURATION", "APPOINTMENT_DATE", "APPOINTMENT_TIME", "DEADLINE", "DELIVERY_DATE", "DELIVERY_TIME",
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
    "API_KEY", "WEB_TOKEN", "URL_PARAMETER", "SERVER_NAME", "ENDPOINT", "DOMAIN"
]

**Example Input String (Russian):**
"Я видел Марию в Москве вчера. AGE_30_45 GENDER_FEMALE EMOTION_NEUTRAL"

**CORRECT Example Output String (Russian):**
"Я видел ENTITY_PERSON_NAME Марию END в ENTITY_CITY Москве END вчера. AGE_30_45 GENDER_FEMALE EMOTION_NEUTRAL INTENT_INFORM"

**INCORRECT Example Output String (Spaces in Tag):**
"Я видел ENTITY_PERSON_ NAM E Марию END в ENTITY_CIT Y Москве END вчера. AGE_30_45 GENDER_FEMALE EMOTION_NEUTRAL INTENT_INFORM"

**INCORRECT Example Output String (Extra END before metadata):**
"Я видел ENTITY_PERSON_NAME Марию END в ENTITY_CITY Москве END вчера. END AGE_30_45 GENDER_FEMALE EMOTION_NEUTRAL INTENT_INFORM"


**Sentences to Annotate Now:**
{json.dumps(texts_to_annotate, ensure_ascii=False, indent=2)}
'''

    max_retries = 3
    retry_delay = 5 # seconds
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel("gemini-2.0-flash") # Or "gemini-1.5-pro-latest"
            response = model.generate_content(prompt)
            assistant_reply = response.text.strip()

            if assistant_reply.startswith("```json"):
                assistant_reply = assistant_reply[len("```json"):].strip()
            elif assistant_reply.startswith("```"):
                 assistant_reply = assistant_reply[len("```"):].strip()
            if assistant_reply.endswith("```"):
                assistant_reply = assistant_reply[:-len("```")].strip()

            if not (assistant_reply.startswith('[') and assistant_reply.endswith(']')):
                 match = re.search(r'\[.*\]', assistant_reply, re.DOTALL)
                 if match:
                     print("Warning: Extracted JSON content from potentially malformed response.")
                     assistant_reply = match.group(0)
                 else:
                    raise json.JSONDecodeError("Response does not appear to contain a JSON list.", assistant_reply, 0)

            annotated_sentences_raw = json.loads(assistant_reply)

            if isinstance(annotated_sentences_raw, list) and len(annotated_sentences_raw) == len(texts_to_annotate):
                processed_sentences = []
                original_indices = {text: i for i, text in enumerate(texts_to_annotate)}

                for idx, sentence in enumerate(annotated_sentences_raw):
                     if isinstance(sentence, str):
                          corrected_sentence = correct_entity_tag_spaces(sentence)
                          final_sentence = fix_end_tags_and_spacing(corrected_sentence)
                          processed_sentences.append(final_sentence)
                     else:
                          print(f"Warning: Non-string item received in annotation list at index {idx}: {sentence}")
                          try:
                              original_text = texts_to_annotate[idx]
                              processed_sentences.append(original_text + " ANNOTATION_ERROR_NON_STRING")
                          except IndexError:
                              print(f"Error: Could not map non-string item at index {idx} back to original text.")
                              processed_sentences.append("ANNOTATION_ERROR_UNKNOWN_ORIGINAL")

                if len(processed_sentences) == len(texts_to_annotate):
                    return processed_sentences
                else:
                    print(f"Error: Mismatch after processing non-string elements. Expected {len(texts_to_annotate)}, Got {len(processed_sentences)}")
                    if attempt == max_retries - 1: return texts_to_annotate
                    else: raise ValueError("Processing error lead to length mismatch.")

            else:
                print(f"Error: API did not return a valid list or mismatched length. Expected {len(texts_to_annotate)}, Got {len(annotated_sentences_raw) if isinstance(annotated_sentences_raw, list) else 'Invalid Type'}")
                if attempt == max_retries - 1: return texts_to_annotate
                else:
                    print(f"Retrying annotation (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(retry_delay * (attempt + 1))

        except json.JSONDecodeError as e:
            print(f"JSON decoding failed (Attempt {attempt+1}/{max_retries}): {e}")
            print("Problematic Raw response snippet:", assistant_reply[:500] if 'assistant_reply' in locals() else "N/A")
            if attempt == max_retries - 1: return texts_to_annotate
            print(f"Retrying annotation...")
            time.sleep(retry_delay * (attempt + 1))
        except Exception as e:
            print(f"Error calling/processing Generative AI (Attempt {attempt+1}/{max_retries}): {e}")
            if "rate limit" in str(e).lower():
                 print("Rate limit likely hit.")
                 time.sleep(retry_delay * (attempt + 1) * 5)
            elif "API key not valid" in str(e):
                 print("FATAL: Invalid Google API Key.")
                 return texts_to_annotate
            else:
                 time.sleep(retry_delay * (attempt + 1))
            if attempt == max_retries - 1: return texts_to_annotate

    print("Error: Max retries reached for annotation.")
    return texts_to_annotate


# --- Main Processing Function (MODIFIED for New Age/Gender Model) ---
def process_audio_and_annotate(manifest_path: str, base_data_dir: str, output_jsonl_path: str, batch_size: int = 10) -> None:
    """
    Processes Russian audio files listed in a manifest, extracts metadata, gets transcriptions,
    formats text, annotates with AI (using Russian prompt), and saves final JSONL output.
    """
    output_dir = os.path.dirname(output_jsonl_path)
    os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"Attempting to clear output file: {output_jsonl_path}")
        with open(output_jsonl_path, 'w', encoding='utf-8') as f_clear:
            f_clear.write("")
        print(f"Output file {output_jsonl_path} cleared successfully.")
    except IOError as e:
        print(f"Error clearing output file {output_jsonl_path}: {e}. Please check permissions.")
        return

    # --- Load Models ---
    print("Loading models...")
    # <<< CHANGE: Update Age/Gender Model Name
    age_gender_model_name = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    try:
        # Use the processor associated with the new model if needed, but standard Wav2Vec2Processor often works
        # Sticking with the original processor loading unless issues arise
        processor = Wav2Vec2Processor.from_pretrained(age_gender_model_name)
        # <<< CHANGE: Use the updated AgeGenderModel class definition
        age_gender_model = AgeGenderModel.from_pretrained(age_gender_model_name).to(device)
        age_gender_model.eval()
        print("Age/Gender model loaded:", age_gender_model_name)
    except Exception as e:
        print(f"FATAL Error loading Age/Gender model ({age_gender_model_name}): {e}. Exiting.")
        return

    # Emotion Model (Unchanged loading)
    emotion_model_name = "superb/hubert-large-superb-er"
    emotion_model_info = {}
    try:
        emotion_model_info['model'] = AutoModelForAudioClassification.from_pretrained(emotion_model_name).to(device)
        emotion_model_info['model'].eval()
        emotion_model_info['feature_extractor'] = AutoFeatureExtractor.from_pretrained(emotion_model_name)
        print("Emotion model loaded:", emotion_model_name)
    except Exception as e:
        print(f"Error loading Emotion model: {e}. Emotion extraction will use 'ErrorLoadingModel'.")

    print("-" * 30)

    # --- Read Manifest File ---
    manifest_records = read_manifest(manifest_path, base_data_dir)
    if not manifest_records:
        print("No valid records found in the manifest file. Exiting.")
        return

    total_files = len(manifest_records)
    print(f"Found {total_files} records in manifest to process.")
    print(f"Output will be saved to: {output_jsonl_path}")
    print("-" * 30)

    processed_records_buffer = []
    batch_num = 0
    files_processed_count = 0
    total_records_saved = 0

    # --- Process Records from Manifest ---
    for i, manifest_entry in enumerate(manifest_records):
        audio_path = manifest_entry['full_audio_filepath']
        relative_audio_path = manifest_entry['audio_filepath']
        transcription = manifest_entry['transcription']
        original_duration = manifest_entry.get('duration', -1.0)

        print(f"\nProcessing record {i+1}/{total_files}: {relative_audio_path}")

        if not os.path.exists(audio_path):
            print(f"  Skipping: Audio file not found at {audio_path}")
            continue

        try:
            # 1. Load Audio (Unchanged)
            try:
                target_sr = 16000
                try:
                    signal, sr = sf.read(audio_path, dtype='float32')
                except Exception as sf_err:
                    print(f"  Soundfile failed ({sf_err}), trying librosa...")
                    signal, sr = librosa.load(audio_path, sr=None, mono=False)

                if sr != target_sr:
                    print(f"  Info: Resampling {os.path.basename(audio_path)} from {sr} Hz to {target_sr} Hz.")
                    signal = signal.astype(np.float32) if not np.issubdtype(signal.dtype, np.floating) else signal
                    if signal.ndim > 1: signal = np.mean(signal, axis=1)
                    signal = librosa.resample(y=signal, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr
                if signal.ndim > 1: signal = np.mean(signal, axis=1)
                if signal is None or len(signal) == 0:
                    print(f"  Skipping: Failed to load or empty audio in {audio_path}")
                    continue

                duration = round(len(signal) / sr, 2)
                if original_duration <= 0:
                    print(f"  Info: Calculated duration: {duration:.2f}s")
                else:
                    if abs(original_duration - duration) > 0.5:
                         print(f"  Warning: Manifest duration ({original_duration:.2f}s) differs significantly from calculated ({duration:.2f}s). Using calculated duration.")
                    else:
                         duration = original_duration

                if duration < 0.1:
                     print(f"  Skipping: Audio too short ({duration:.2f}s) in {audio_path}")
                     continue
            except Exception as load_err:
                 print(f"  Skipping: Error loading/processing audio {audio_path}: {load_err}")
                 continue

            # 2. Extract Age/Gender (<<< CHANGE: Using new model and logic)
            try:
                # Process audio with the Wav2Vec2 Processor
                inputs = processor(signal, sampling_rate=sr, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device)

                # Truncate long inputs if needed (adjust max_len as required)
                max_len_inference = 30 * sr # 30 seconds
                if input_values.shape[1] > max_len_inference:
                    print(f"  Info: Audio longer than 30s ({duration:.2f}s), truncating for Age/Gender/Emotion extraction.")
                    input_values_truncated = input_values[:, :max_len_inference]
                else:
                    input_values_truncated = input_values

                with torch.no_grad():
                    # Get raw age logit and gender probabilities from the updated model
                    logits_age, probabilities_gender = age_gender_model(input_values_truncated)

                # Process Age: Apply scaling and clipping (0-100)
                age_raw_logit = logits_age.cpu().item()
                # <<< CHANGE: Apply the specific scaling from predict_age_gender
                predicted_age = min(max(age_raw_logit * 100 + 30, 0), 100)
                print(f"  Raw Age Logit: {age_raw_logit:.4f}, Scaled Age: {predicted_age:.1f}")

                # Process Gender: Argmax on probabilities (or logits) for 2 classes
                gender_idx = torch.argmax(probabilities_gender, dim=1).cpu().item()
                # <<< CHANGE: Map index 0/1 to FEMALE/MALE
                gender = "MALE" if gender_idx == 1 else "FEMALE"
                print(f"  Gender Probabilities: {probabilities_gender.cpu().numpy()}, Predicted Index: {gender_idx}, Gender: {gender}")

            except RuntimeError as e:
                 if "CUDA out of memory" in str(e):
                     print(f"  CUDA OOM Error during Age/Gender extraction for {os.path.basename(audio_path)}. Skipping file.")
                     torch.cuda.empty_cache()
                     gc.collect()
                     continue
                 else:
                     print(f"  Runtime Error during Age/Gender extraction: {e}")
                     predicted_age = -1.0 # Indicate error
                     gender = "ERROR"
            except Exception as age_gender_err:
                print(f"  Error during Age/Gender extraction: {age_gender_err}")
                predicted_age = -1.0 # Indicate error
                gender = "ERROR"

            # 3. Extract Emotion (Unchanged logic, uses truncated signal if necessary)
            emotion_signal = signal[:max_len_inference] if 'input_values' in locals() and input_values.shape[1] > max_len_inference else signal
            emotion = extract_emotion(emotion_signal, sr, emotion_model_info)

            # 4. Create Initial Record Structure
            try:
                 path_parts = Path(relative_audio_path).parts
                 speaker = path_parts[1] if len(path_parts) > 2 else "UNKNOWN_SPEAKER"
            except Exception:
                 speaker = "UNKNOWN_SPEAKER"

            segment_data = AudioSegment(
                start_time=0, end_time=duration, speaker=speaker,
                # <<< CHANGE: Store the final scaled age
                age=predicted_age,
                gender=gender, # Already "MALE" or "FEMALE"
                transcription=transcription, emotion=emotion,
                chunk_filename=os.path.basename(audio_path), duration=duration
            )
            chunk = ChunkData(segments=[segment_data], filepath=relative_audio_path)
            initial_formatted_text = chunk.get_formatted_text() # This now uses scaled age via get_age_bucket

            # Store essential info needed for the final JSONL
            record = {
                "audio_filepath": relative_audio_path,
                "duration": duration,
                "initial_text": initial_formatted_text, # Includes AGE_BUCKET GENDER_MALE/FEMALE etc.
                "raw_age_output": age_raw_logit if 'age_raw_logit' in locals() else None, # Store the raw logit for reference
                "scaled_age_prediction": predicted_age, # Store the 0-100 age
                "raw_gender_prediction": gender,
                "raw_emotion_prediction": emotion,
                "speaker_id": speaker,
            }
            processed_records_buffer.append(record)
            files_processed_count += 1

            # 5. Annotate and Save in Batches (Unchanged logic)
            if len(processed_records_buffer) >= batch_size or (i + 1) == total_files:
                batch_num += 1
                current_batch_size = len(processed_records_buffer)
                print(f"\n--- Annotating Batch {batch_num} ({current_batch_size} records) ---")

                texts_to_annotate = [rec["initial_text"] for rec in processed_records_buffer]
                annotated_texts = annotate_batch_texts(texts_to_annotate)

                if len(annotated_texts) != current_batch_size:
                    print(f"  FATAL BATCH ERROR: Annotation count mismatch! Expected {current_batch_size}, Got {len(annotated_texts)}. Skipping save for this batch.")
                else:
                    try:
                        lines_written_in_batch = 0
                        with open(output_jsonl_path, 'a', encoding='utf-8') as f_out:
                            for record_data, annotated_text in zip(processed_records_buffer, annotated_texts):
                                final_record = {
                                    "audio_filepath": record_data["audio_filepath"],
                                    "duration": record_data["duration"],
                                    "text": annotated_text, # AI annotated text
                                    # Optional: Add back other metadata if needed in final output
                                    # "speaker_id": record_data["speaker_id"],
                                    # "scaled_age": record_data["scaled_age_prediction"],
                                    # "gender": record_data["raw_gender_prediction"],
                                    # "emotion": record_data["raw_emotion_prediction"],
                                }
                                json_str = json.dumps(final_record, ensure_ascii=False)
                                f_out.write(json_str + '\n')
                                lines_written_in_batch += 1

                        total_records_saved += lines_written_in_batch
                        print(f"--- Batch {batch_num} annotated and saved ({lines_written_in_batch} records). Total saved: {total_records_saved} ---")

                    except IOError as io_err:
                         print(f"  Error writing batch {batch_num} to {output_jsonl_path}: {io_err}")
                    except Exception as write_err:
                         print(f"  Unexpected error writing batch {batch_num}: {write_err}")

                processed_records_buffer = []
                del texts_to_annotate, annotated_texts
                if 'input_values' in locals(): del input_values
                if 'input_values_truncated' in locals(): del input_values_truncated
                if 'signal' in locals(): del signal
                if 'logits_age' in locals(): del logits_age, probabilities_gender
                torch.cuda.empty_cache()
                gc.collect()

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")
            break
        except Exception as e:
            print(f"  FATAL ERROR processing record {relative_audio_path}: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()
            continue

    print("\n" + "="*30)
    print(f"Processing Finished.")
    print(f"Total manifest records processed attempt: {files_processed_count}/{total_files}")
    print(f"Total records saved to {output_jsonl_path}: {total_records_saved}")
    print("="*30)

# --- Main Execution (Unchanged) ---
if __name__ == "__main__":
    # --- *** Configuration for Russian Data *** ---
    BASE_DATA_DIR = "/external4/datasets/golos-data/test_wav"
    MANIFEST_FILENAME = "manifest_processed_annotated.jsonl"
    FINAL_OUTPUT_JSONL = os.path.join(BASE_DATA_DIR, "russian_annoted_manifest.jsonl") # Changed output name slightly
    PROCESSING_BATCH_SIZE = 10
    MANIFEST_FILE_PATH = os.path.join(BASE_DATA_DIR, MANIFEST_FILENAME)

    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not found in environment or .env file.")
        exit(1)
    if genai is None:
        try:
            api_key_recheck = os.getenv("GOOGLE_API_KEY")
            if not api_key_recheck: raise ValueError("GOOGLE_API_KEY still not found.")
            genai.configure(api_key=api_key_recheck)
            print("Google Generative AI re-configured successfully after .env load.")
        except Exception as e:
             print(f"ERROR: Failed to configure Google Generative AI: {e}")
             exit(1)

    print("Starting Russian Audio Processing and Annotation Workflow...")
    print(f"Input Base Directory: {BASE_DATA_DIR}")
    print(f"Manifest File: {MANIFEST_FILE_PATH}")
    print(f"Final Output File: {FINAL_OUTPUT_JSONL}")
    print(f"Processing Batch Size: {PROCESSING_BATCH_SIZE}")
    print("-" * 30)

    if not os.path.isdir(BASE_DATA_DIR):
        print(f"ERROR: Base data directory not found: {BASE_DATA_DIR}")
        exit(1)
    if not os.path.isfile(MANIFEST_FILE_PATH):
         print(f"ERROR: Manifest file not found: {MANIFEST_FILE_PATH}")
         exit(1)

    process_audio_and_annotate(
        manifest_path=MANIFEST_FILE_PATH,
        base_data_dir=BASE_DATA_DIR,
        output_jsonl_path=FINAL_OUTPUT_JSONL,
        batch_size=PROCESSING_BATCH_SIZE
    )

    print("Workflow complete.")