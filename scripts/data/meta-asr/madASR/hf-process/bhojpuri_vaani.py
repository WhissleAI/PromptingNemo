# -*- coding: utf-8 -*- # Ensure editor recognizes UTF-8 for Devanagari script (Bhojpuri)
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
from typing import Dict, List, Tuple, Any, Optional
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel
)
import torch.nn as nn
import google.generativeai as genai
import time
import soundfile as sf

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

# --- Age/Gender Model Definition (Unchanged) ---
class ModelHead(nn.Module):
    # ... (same as before)
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features; x = self.dropout(x); x = self.dense(x); x = torch.tanh(x); x = self.dropout(x); x = self.out_proj(x)
        return x

class AgeGenderModel(Wav2Vec2PreTrainedModel):
    # ... (same as before)
    def __init__(self, config):
        super().__init__(config); self.config = config; self.wav2vec2 = Wav2Vec2Model(config); self.age = ModelHead(config, 1); self.gender = ModelHead(config, 3); self.init_weights()
    def forward(self, input_values):
        outputs = self.wav2vec2(input_values); hidden_states = outputs[0]; hidden_states_pooled = torch.mean(hidden_states, dim=1); logits_age = self.age(hidden_states_pooled); logits_gender = torch.softmax(self.gender(hidden_states_pooled), dim=-1)
        return hidden_states_pooled, logits_age, logits_gender

# --- Emotion Extraction Function (Unchanged) ---
def extract_emotion(audio_data: np.ndarray, sampling_rate: int = 16000, model_info: dict = None) -> str:
    # ... (same as before)
    if model_info is None or 'model' not in model_info or 'feature_extractor' not in model_info: return "ErrorLoadingModel"
    if audio_data is None or len(audio_data) == 0: return "No_Audio"
    try:
        inputs = model_info['feature_extractor'](audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad(): outputs = model_info['model'](**inputs)
        predicted_class_idx = outputs.logits.argmax(-1).item()
        emotion_label = model_info['model'].config.id2label.get(predicted_class_idx, "Unknown")
        return emotion_label.upper()
    except Exception as e: print(f"Error during emotion extraction: {e}"); return "Extraction_Error"

# --- Data Structures (Unchanged) ---
@dataclass
class AudioSegment:
    start_time: float; end_time: float; speaker: str
    age: float # From ML model
    gender: str # From JSON metadata
    transcription: str # From JSON metadata (Bhojpuri text in Devanagari)
    emotion: str # From ML model
    chunk_filename: str; duration: float

@dataclass
class ChunkData:
    segments: List[AudioSegment] = field(default_factory=list); filepath: str = ""
    def get_formatted_text(self) -> str:
        # ... (same logic as before, uses data from AudioSegment)
        if not self.segments: return ""
        segment = self.segments[0]; age_bucket = self.get_age_bucket(segment.age)
        gender_text = segment.gender.upper() if segment.gender else "UNKNOWN"
        emotion_text = segment.emotion.upper()
        transcription = segment.transcription.strip() # Bhojpuri text
        metadata = f"AGE_{age_bucket} GENDER_{gender_text} EMOTION_{emotion_text}"
        return f"{transcription.strip()} {metadata.strip()}"
    @staticmethod
    def get_age_bucket(age: float) -> str:
        # ... (same logic as before)
        if age < 0: return "UNKNOWN"
        actual_age = round(age * 100)
        age_brackets = [(18, "0_18"), (30, "18_30"), (45, "30_45"), (60, "45_60"), (float('inf'), "60PLUS")]
        for threshold, bracket in age_brackets:
            if actual_age < threshold: return bracket
        return "60PLUS"

# --- File Handling (Unchanged logic) ---
def get_file_pairs(audio_dir: str, metadata_dir: str) -> List[Tuple[str, str]]:
    # ... (same logic as before, finds audio/json pairs)
    try:
        audio_files_list = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.flac', '.wav', '.mp3'))]
        metadata_files_list = [f for f in os.listdir(metadata_dir) if f.lower().endswith('.json')]
        audio_files = {os.path.splitext(f)[0]: os.path.join(audio_dir, f) for f in audio_files_list}
        metadata_files = {os.path.splitext(f)[0]: os.path.join(metadata_dir, f) for f in metadata_files_list}
        pairs = []
        missing_metadata_count = 0
        for base_name, audio_path in audio_files.items():
            if base_name in metadata_files: pairs.append((audio_path, metadata_files[base_name]))
            else: missing_metadata_count += 1
        if missing_metadata_count > 0: print(f"Warning: Found {missing_metadata_count} audio files without matching JSON metadata files.")
        print(f"Found {len(pairs)} matching audio-metadata pairs.")
        if not pairs:
             print(f"Searched in:\n Audio: {audio_dir}\n Metadata: {metadata_dir}")
             print(f"Audio files found: {len(audio_files_list)}")
             print(f"Metadata files found: {len(metadata_files_list)}")
        return pairs
    except FileNotFoundError as e: print(f"Error finding files: {e}. Check paths."); return []
    except Exception as e: print(f"An unexpected error in get_file_pairs: {e}"); return []

# --- Get Info from Metadata JSON (Unchanged logic) ---
def get_metadata_info(metadata_path: str) -> Tuple[Optional[str], Optional[str]]:
    # ... (same logic as before, uses utf-8 for Devanagari)
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f: data = json.load(f)
        transcript = data.get("transcript")
        gender = data.get("gender")
        if not transcript: print(f"Warning: 'transcript' missing/empty in {metadata_path}"); transcript = None
        if not gender: print(f"Warning: 'gender' missing/empty in {metadata_path}"); gender = None
        else:
             gender = gender.upper()
             if gender not in ["FEMALE", "MALE", "OTHER"]: # Added OTHER
                  print(f"Warning: Unexpected gender '{gender}' in {metadata_path}. Mapping to UNKNOWN.")
                  gender = "UNKNOWN"
        return transcript, gender
    except FileNotFoundError: print(f"Error: Metadata file not found: {metadata_path}"); return None, None
    except json.JSONDecodeError: print(f"Error: Could not decode JSON from {metadata_path}"); return None, None
    except Exception as e: print(f"Error reading metadata {metadata_path}: {str(e)}"); return None, None

# --- AI Annotation Functions (MODIFIED PROMPT for Bhojpuri) ---

# [Functions: correct_entity_tag_spaces, fix_end_tags_and_spacing remain the same]
# ... (Keep the existing correct_entity_tag_spaces function) ...
def correct_entity_tag_spaces(text: str) -> str:
    if not isinstance(text, str): return text
    def replace_spaces(match):
        tag_part = match.group(1); type_part = tag_part[len("ENTITY_"):]; corrected_type = type_part.replace(' ', '')
        return f"ENTITY_{corrected_type}"
    pattern = r'\b(ENTITY_[A-Z0-9_ ]*?[A-Z0-9_])(?=\s+\S)'; corrected_text = re.sub(pattern, replace_spaces, text)
    return corrected_text

# ... (Keep the existing fix_end_tags_and_spacing function - handles । ) ...
def fix_end_tags_and_spacing(text: str) -> str:
    if not isinstance(text, str): return text
    text = re.sub(r'\s+', ' ', text).strip()
    def remove_internal_spaces(match):
        tag_prefix = match.group(1); tag_body = match.group(2); corrected_body = re.sub(r'\s+', '', tag_body)
        return f"{tag_prefix}{corrected_body}"
    text = re.sub(r'\b(ENTITY_)([A-Z0-9_]+(?: +[A-Z0-9_]+)+)\b', remove_internal_spaces, text)
    text = re.sub(r'(\bENTITY_[A-Z0-9_]+)(\S)', r'\1 \2', text)
    text = re.sub(r'(\S)(END\b)', r'\1 \2', text)
    text = re.sub(r'(\bEND)(\S)', r'\1 \2', text)
    text = re.sub(r'\bEND\s+END\b', 'END', text); text = re.sub(r'\bEND\s+END\b', 'END', text)
    text = re.sub(r'\bEND\s*(\b(?:AGE_|GENDER_|EMOTION_|INTENT_)|$)', r'\1', text)
    text = re.sub(r'\bEND\s*(\b(?:AGE_|GENDER_|EMOTION_|INTENT_)|$)', r'\1', text)
    text = re.sub(r'\s+([?!:;,.।])', r'\1', text) # Includes Devanagari danda ।
    text = re.sub(r'([?!:;,.।])(\w)', r'\1 \2', text) # Includes Devanagari danda ।
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- MODIFIED AND MORE DETAILED PROMPT FOR BHOJPURI ---
def annotate_batch_texts(texts_to_annotate: List[str]):
    """Sends a batch of texts to Gemini for annotation (Prompt refined for clarity and stricter rules for Bhojpuri)."""
    if not genai:
        print("Error: Google Generative AI not configured. Skipping annotation.")
        return texts_to_annotate
    if not texts_to_annotate:
        return []

    # Input texts already have metadata like: "bhojpuri text AGE_X GENDER_Y EMOTION_Z"

    # *** REFINED BHOJPURI PROMPT STARTS HERE ***
    prompt = f'''You are an expert linguistic annotator specifically for **Bhojpuri** text written in the **Devanagari script**. Your task is to process a list of Bhojpuri sentences, each already containing `AGE_*`, `GENDER_*`, and `EMOTION_*` metadata tags at the end. Follow these instructions with extreme precision:

1.  **Preserve Existing Metadata Tags:**
    *   The `AGE_*`, `GENDER_*`, and `EMOTION_*` tags at the end of each sentence **must** remain exactly as they are.
    *   **Do not** modify, move, delete, or change these existing metadata tags.

2.  **Entity Annotation (Bhojpuri Text Only):**
    *   Identify entities **only** within the main Bhojpuri transcription part of the sentence (before the metadata).
    *   Use **only** the entity types provided in the `Allowed ENTITY TYPES` list below. Do not invent new types.

3.  **Strict Entity Tagging Format:**
    *   **Tag Insertion:** Place the entity tag **immediately before** the identified Bhojpuri entity text. The tag format is `ENTITY_<TYPE>` (e.g., `ENTITY_CITY`, `ENTITY_PERSON_NAME`).
    *   **NO SPACES IN TYPE:** Ensure the `<TYPE>` part of the tag contains **no spaces** (e.g., `PERSON_NAME` is correct, `PERSON_ NAM E` is **incorrect**).
    *   **`END` Tag Placement:** Place the literal string `END` **immediately after** the *complete* Bhojpuri entity text.
    *   **Spacing:** There must be exactly **one space** between the `ENTITY_<TYPE>` tag and the start of the entity text. There must be exactly **one space** between the end of the entity text and the `END` tag. There must be exactly **one space** after the `END` tag before the next word begins (unless followed by punctuation).
    *   **Example:** For the entity "पटना", the correct annotation is `ENTITY_CITY पटना END`.
    *   **Crucial `END` Rule:** Only add **one** `END` tag right after the full entity phrase. Do **not** add `END` tags after individual words within a multi-word entity or after words that simply follow an entity.
    *   **Avoid Extra `END` Before Metadata:** Do **not** place an `END` tag just before the `AGE_`, `GENDER_`, `EMOTION_`, or `INTENT_` tags unless that `END` tag correctly marks the end of an immediately preceding entity.

4.  **Intent Tag:**
    *   Determine the single primary intent of the Bhojpuri sentence (e.g., `INFORM`, `QUESTION`, `REQUEST`, `COMMAND`, `GREETING`).
    *   Append **one** `INTENT_<INTENT_TYPE>` tag at the **absolute end** of the entire string, after all other tags (`AGE_`, `GENDER_`, `EMOTION_`).

5.  **Output Format:**
    *   Return a **JSON array** of strings. Each string must be a fully annotated sentence following all the rules above.

6.  **Bhojpuri Language Specifics:**
    *   Handle Devanagari script correctly.
    *   Ensure correct spacing around punctuation (like `.`, `?`, `!`, `।`). Remove space before punctuation, ensure space after punctuation if followed by a word.
    *   The final output string must be clean, with single spaces separating words and tags according to the rules.

**Allowed ENTITY TYPES (Use Only These):**
    [ "PERSON_NAME", "ORGANIZATION", "LOCATION", "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP_CODE", "CURRENCY", "PRICE", "DATE",
    "TIME", "DURATION", "APPOINTMENT_DATE", "APPOINTMENT_TIME", "DEADLINE", "DELIVERY_DATE", "DELIVERY_TIME", "EVENT", "MEETING",
    "TASK", "PROJECT_NAME", "ACTION_ITEM", "PRIORITY", "FEEDBACK", "REVIEW", "RATING", "COMPLAINT", "QUESTION", "RESPONSE", "NOTIFICATION_TYPE", "AGENDA",
    "REMINDER", "NOTE", "RECORD", "ANNOUNCEMENT", "UPDATE", "SCHEDULE", "BOOKING_REFERENCE", "APPOINTMENT_NUMBER", "ORDER_NUMBER", "INVOICE_NUMBER", "PAYMENT_METHOD", "PAYMENT_AMOUNT", "BANK_NAME", "ACCOUNT_NUMBER", "CREDIT_CARD_NUMBER", "TAX_ID", "SOCIAL_SECURITY_NUMBER", "DRIVER'S_LICENSE", "PASSPORT_NUMBER", "INSURANCE_PROVIDER", "POLICY_NUMBER", "INSURANCE_PLAN", "CLAIM_NUMBER", "POLICY_HOLDER", "BENEFICIARY",
    "RELATIONSHIP", "EMERGENCY_CONTACT", "PROJECT_PHASE", "VERSION", "DEVELOPMENT_STAGE", "DEVICE_NAME", "OPERATING_SYSTEM", "SOFTWARE_VERSION", "BRAND", "MODEL_NUMBER", "LICENSE_PLATE", "VEHICLE_MAKE", "VEHICLE_MODEL", "VEHICLE_TYPE", "FLIGHT_NUMBER", "HOTEL_NAME", "ROOM_NUMBER", "TRANSACTION_ID", "TICKET_NUMBER", "SEAT_NUMBER", "GATE", "TERMINAL", "TRANSACTION_TYPE", "PAYMENT_STATUS", "PAYMENT_REFERENCE", "INVOICE_STATUS",
    "SYMPTOM", "DIAGNOSIS", "MEDICATION", "DOSAGE", "ALLERGY", "PRESCRIPTION", "TEST_NAME", "TEST_RESULT", "MEDICAL_RECORD", "HEALTH_STATUS", "HEALTH_METRIC", "VITAL_SIGN", "DOCTOR_NAME", "HOSPITAL_NAME", "DEPARTMENT", "WARD", "CLINIC_NAME", "WEBSITE", "URL", "IP_ADDRESS", "MAC_ADDRESS", "USERNAME", "PASSWORD", "LANGUAGE", "CODE_SNIPPET", "DATABASE_NAME", "API_KEY", "WEB_TOKEN", "URL_PARAMETER", "SERVER_NAME", "ENDPOINT", "DOMAIN" ]

**Examples Demonstrating Correct Formatting (Bhojpuri):**

*   **Input:** `"हम काल्ह मारिया से पटना में मिलब। AGE_30_45 GENDER_MALE EMOTION_NEUTRAL"`
*   **Correct Output:** `"हम काल्ह ENTITY_PERSON_NAME मारिया END से ENTITY_CITY पटना END में मिलब। AGE_30_45 GENDER_MALE EMOTION_NEUTRAL INTENT_INFORM"`

*   **Input:** `"सीबीआई के इ सब दलील मद्रास कोर्ट के जज प्रकाश खारिज कर दिहलन। AGE_30_45 GENDER_MALE EMOTION_NEUTRAL"`
*   **Correct Output:** `"ENTITY_ORGANIZATION सीबीआई END के इ सब दलील ENTITY_LOCATION मद्रास END कोर्ट के जज ENTITY_PERSON_NAME प्रकाश END खारिज कर दिहलन। AGE_30_45 GENDER_MALE EMOTION_NEUTRAL INTENT_INFORM"`
    *(Note: Assuming "मद्रास" is location, "प्रकाश" is name. Note space after each `END` unless followed by punctuation like `।`)*

*   **Input:** `"हमर जन्मदिन दिसंबर १२ के बा। AGE_18_30 GENDER_MALE EMOTION_HAPPY"`
*   **Correct Output:** `"हमर जन्मदिन ENTITY_DATE दिसंबर १२ END के बा। AGE_18_30 GENDER_MALE EMOTION_HAPPY INTENT_INFORM"`

*   **Input:** `"रउआ किरपा क के चुप रहब? AGE_45_60 GENDER_FEMALE EMOTION_ANNOYED"`
*   **Correct Output:** `"रउआ किरपा क के चुप रहब? AGE_45_60 GENDER_FEMALE EMOTION_ANNOYED INTENT_REQUEST"`

**Incorrect Examples (Common Mistakes to Avoid):**
*   `... ENTITY_PERSON_ NAM E मारिया END ...` (Space in TYPE)
*   `... ENTITY_CITY पटनाEND ...` (Missing space before END)
*   `... ENTITY_CITY पटना ENDमें ...` (Missing space after END)
*   `... दिहलन END AGE_...` (Unnecessary END before metadata)
*   `... जज END प्रकाश END ...` (Incorrectly adding END after non-entity words)

Provide only the JSON array containing the correctly annotated sentences based precisely on these instructions.

**Sentences to Annotate Now:**
{json.dumps(texts_to_annotate, ensure_ascii=False, indent=2)}
'''
    # *** REFINED BHOJPURI PROMPT ENDS HERE ***

    # --- API Call and Response Handling (Unchanged Logic) ---
    max_retries = 3; retry_delay = 5
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash-latest")
            response = model.generate_content(prompt)
            assistant_reply = response.text.strip()

            # Robust JSON Extraction (same as before)
            json_match = re.search(r'\[.*\]', assistant_reply, re.DOTALL)
            if json_match: json_str = json_match.group(0)
            else:
                 if assistant_reply.startswith("```json"): assistant_reply = assistant_reply[len("```json"):].strip()
                 elif assistant_reply.startswith("```"): assistant_reply = assistant_reply[len("```"):].strip()
                 if assistant_reply.endswith("```"): assistant_reply = assistant_reply[:-len("```")].strip()
                 if assistant_reply.startswith('[') and assistant_reply.endswith(']'): json_str = assistant_reply
                 else: raise json.JSONDecodeError("Response does not appear to contain a JSON list.", assistant_reply, 0)

            # JSON Parsing and Post-Processing (same as before)
            try: annotated_sentences_raw = json.loads(json_str)
            except json.JSONDecodeError as json_e:
                print(f"JSON decoding failed on extracted string (Attempt {attempt+1}): {json_e}"); print("Extracted snippet:", json_str[:500]); raise json_e

            if isinstance(annotated_sentences_raw, list) and len(annotated_sentences_raw) == len(texts_to_annotate):
                processed_sentences = []
                for idx, sentence in enumerate(annotated_sentences_raw):
                     if isinstance(sentence, str): processed_sentences.append(fix_end_tags_and_spacing(sentence))
                     else:
                          print(f"Warning: Non-string item at index {idx}: {sentence}")
                          try: original_text = texts_to_annotate[idx]; processed_sentences.append(fix_end_tags_and_spacing(original_text) + " ANNOTATION_ERROR_NON_STRING")
                          except IndexError: processed_sentences.append("ANNOTATION_ERROR_UNKNOWN_ORIGINAL")

                if len(processed_sentences) == len(texts_to_annotate): return processed_sentences
                else:
                     print(f"Error: Mismatch after processing non-strings. Expected {len(texts_to_annotate)}, Got {len(processed_sentences)}")
                     if attempt == max_retries - 1: return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_PROCESSING" for t in texts_to_annotate]
                     else: raise ValueError("Processing error lead to length mismatch.")
            else:
                print(f"Error: API invalid list/length (Attempt {attempt+1}). Expected {len(texts_to_annotate)}, Got {len(annotated_sentences_raw) if isinstance(annotated_sentences_raw, list) else type(annotated_sentences_raw)}")
                if attempt == max_retries - 1: return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_LENGTH" for t in texts_to_annotate]
                else: print(f"Retrying annotation (attempt {attempt + 2}/{max_retries})..."); time.sleep(retry_delay * (attempt + 1))

        except json.JSONDecodeError as e:
            print(f"JSON decoding failed (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1: return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_JSON_DECODE" for t in texts_to_annotate]
            print(f"Retrying annotation..."); time.sleep(retry_delay * (attempt + 1))
        except Exception as e:
            print(f"Error calling/processing Generative AI (Attempt {attempt+1}/{max_retries}): {e}")
            import traceback; traceback.print_exc()
            if "rate limit" in str(e).lower(): time.sleep(retry_delay * (attempt + 1) * 5)
            elif "API key not valid" in str(e) or "permission" in str(e).lower():
                 print("FATAL: Invalid Google API Key or permission issue."); return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_KEY" for t in texts_to_annotate]
            else: time.sleep(retry_delay * (attempt + 1))
            if attempt == max_retries - 1: return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_CALL" for t in texts_to_annotate]

    print("Error: Max retries reached for annotation batch.")
    return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_MAX_RETRIES" for t in texts_to_annotate]

# --- Main Processing Function (Updated Comments for Bhojpuri) ---
def process_audio_and_annotate(base_dir: str, output_jsonl_path: str, batch_size: int = 10) -> None:
    """
    Processes Bhojpuri audio files, extracts metadata (transcript, gender) from JSON,
    predicts age/emotion, formats text, annotates with AI (using Bhojpuri prompt),
    and saves final JSONL output.
    """
    output_dir = os.path.dirname(output_jsonl_path); os.makedirs(output_dir, exist_ok=True)
    try: # Clear output file
        print(f"Attempting to clear output file: {output_jsonl_path}")
        with open(output_jsonl_path, 'w', encoding='utf-8') as f_clear: f_clear.write("")
        print(f"Output file {output_jsonl_path} cleared successfully.")
    except IOError as e: print(f"Error clearing output file {output_jsonl_path}: {e}. Check permissions."); return

    # Load Models (same models)
    print("Loading models...")
    age_gender_model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    processor = None; age_gender_model = None
    try:
        processor = Wav2Vec2Processor.from_pretrained(age_gender_model_name)
        age_gender_model = AgeGenderModel.from_pretrained(age_gender_model_name).to(device); age_gender_model.eval()
        print("Age/Gender model loaded.")
    except Exception as e: print(f"FATAL Error loading Age/Gender model: {e}. Exiting."); return

    emotion_model_name = "superb/hubert-large-superb-er"; emotion_model_info = {}
    try:
        emotion_model_info['model'] = AutoModelForAudioClassification.from_pretrained(emotion_model_name).to(device); emotion_model_info['model'].eval()
        emotion_model_info['feature_extractor'] = AutoFeatureExtractor.from_pretrained(emotion_model_name)
        print("Emotion model loaded.")
    except Exception as e: print(f"Error loading Emotion model: {e}. Emotion extraction will use 'ErrorLoadingModel'.")
    print("-" * 30)

    # Prepare File Paths (Using 'audio' and 'metadata' subdirs)
    audio_dir = os.path.join(base_dir, "audio"); metadata_dir = os.path.join(base_dir, "metadata")
    if not os.path.exists(audio_dir) or not os.path.exists(metadata_dir):
        print(f"Error: Audio ({audio_dir}) or Metadata ({metadata_dir}) directory not found in {base_dir}")
        print("Please ensure your Bhojpuri data follows the structure:")
        print(f"{base_dir}/\n  ├── audio/\n  └── metadata/")
        return

    print(f"Processing Bhojpuri files from:\n  Audio: {audio_dir}\n  Metadata: {metadata_dir}") # Updated print
    print(f"Output will be saved to: {output_jsonl_path}")
    print("-" * 30)

    file_pairs = get_file_pairs(audio_dir, metadata_dir)
    if not file_pairs: print("No matching audio-metadata files found. Exiting."); return
    total_files = len(file_pairs); print(f"Found {total_files} audio-metadata pairs to process.")

    processed_records_buffer = []; batch_num = 0; files_processed_count = 0; total_records_saved = 0

    # --- Process Files (Main loop - logic unchanged, handles Bhojpuri data) ---
    for i, (audio_path, metadata_path) in enumerate(file_pairs):
        print(f"\nProcessing file {i+1}/{total_files}: {os.path.basename(audio_path)}")
        try:
            # 1. Get Transcription and Gender from Metadata (Handles Devanagari via UTF-8)
            transcription, metadata_gender = get_metadata_info(metadata_path)
            if transcription is None or metadata_gender is None: print(f"  Skipping: Missing 'transcript' or 'gender' in {metadata_path}"); continue

            # 2. Load Audio (Unchanged)
            signal, sr, duration = None, 16000, 0
            try:
                try: signal_raw, sr_orig = sf.read(audio_path, dtype='float32')
                except Exception: signal_raw, sr_orig = librosa.load(audio_path, sr=None, mono=False)
                target_sr = 16000
                signal_float = signal_raw.astype(np.float32) if not np.issubdtype(signal_raw.dtype, np.floating) else signal_raw
                signal_mono = np.mean(signal_float, axis=1) if signal_float.ndim > 1 else signal_float
                if sr_orig != target_sr: signal = librosa.resample(y=signal_mono, orig_sr=sr_orig, target_sr=target_sr); sr = target_sr
                else: signal = signal_mono; sr = sr_orig
                if signal is None or len(signal) == 0: print(f"  Skipping: Empty audio {audio_path}"); continue
                duration = round(len(signal) / sr, 2)
                if duration < 0.1: print(f"  Skipping: Audio too short ({duration:.2f}s) {audio_path}"); continue
            except Exception as load_err: print(f"  Skipping: Error loading audio {audio_path}: {load_err}"); continue

            # 3. Extract Age/Emotion (Unchanged Models)
            predicted_age, predicted_emotion = -1.0, "ERROR"; ml_predicted_gender = "ERROR"
            try:
                inputs = processor(signal, sampling_rate=sr, return_tensors="pt", padding=True)
                input_values = inputs.input_values; max_len_inference = 30 * sr
                if input_values.shape[1] > max_len_inference: input_values_truncated = input_values[:, :max_len_inference].to(device); emotion_signal = signal[:max_len_inference]
                else: input_values_truncated = input_values.to(device); emotion_signal = signal
                with torch.no_grad(): _, logits_age, logits_gender_ml = age_gender_model(input_values_truncated)
                predicted_age = logits_age.cpu().numpy().item()
                gender_idx_ml = torch.argmax(logits_gender_ml, dim=-1).cpu().numpy().item()
                gender_map_ml = {0: "FEMALE", 1: "MALE", 2: "OTHER"}; ml_predicted_gender = gender_map_ml.get(gender_idx_ml, "UNKNOWN")
                predicted_emotion = extract_emotion(emotion_signal, sr, emotion_model_info)
            except RuntimeError as e:
                 if "CUDA out of memory" in str(e): print(f"  CUDA OOM Error. Skipping."); torch.cuda.empty_cache(); gc.collect(); continue
                 else: print(f"  Runtime Error Age/Emotion: {e}"); predicted_age = -1.0; predicted_emotion = "ERROR"; ml_predicted_gender = "RUNTIME_ERROR"
            except Exception as predict_err: print(f"  Error Age/Emotion: {predict_err}"); predicted_age = -1.0; predicted_emotion = "ERROR"; ml_predicted_gender = "PREDICT_ERROR"

            # 4. Create Initial Record (Uses Bhojpuri transcript/gender from metadata)
            try: speaker = os.path.splitext(os.path.basename(audio_path))[0].split('_')[0]
            except IndexError: speaker = "UNKNOWN_SPEAKER"
            segment_data = AudioSegment(start_time=0, end_time=duration, speaker=speaker, age=predicted_age, gender=metadata_gender, transcription=transcription, emotion=predicted_emotion, chunk_filename=os.path.basename(audio_path), duration=duration)
            chunk = ChunkData(segments=[segment_data], filepath=os.path.abspath(audio_path))
            initial_formatted_text = chunk.get_formatted_text() # Formats with Bhojpuri text
            record = { "audio_filepath": chunk.filepath, "duration": duration, "initial_text": initial_formatted_text, "raw_age_output": predicted_age, "metadata_gender": metadata_gender, "ml_predicted_gender": ml_predicted_gender, "raw_emotion_prediction": predicted_emotion, "speaker_id": speaker }
            processed_records_buffer.append(record); files_processed_count += 1

            # 5. Annotate and Save in Batches (Unchanged Logic)
            if len(processed_records_buffer) >= batch_size or (i + 1) == total_files:
                batch_num += 1; current_batch_size = len(processed_records_buffer)
                print(f"\n--- Annotating Batch {batch_num} ({current_batch_size} records) ---")
                texts_to_annotate = [rec["initial_text"] for rec in processed_records_buffer]
                annotated_texts = annotate_batch_texts(texts_to_annotate) # Calls Bhojpuri-specific prompt

                if len(annotated_texts) != current_batch_size: print(f"  FATAL BATCH ERROR: Annotation count mismatch! Expected {current_batch_size}, Got {len(annotated_texts)}. Skipping save.")
                else:
                    try:
                        lines_written_in_batch = 0
                        with open(output_jsonl_path, 'a', encoding='utf-8') as f_out:
                            for record_data, annotated_text in zip(processed_records_buffer, annotated_texts):
                                final_record = { "audio_filepath": record_data["audio_filepath"], "duration": record_data["duration"], "text": annotated_text }
                                if "ANNOTATION_ERROR" in annotated_text: print(f"  Warning: Saving {os.path.basename(record_data['audio_filepath'])} with error: {annotated_text.split()[-1]}")
                                json_str = json.dumps(final_record, ensure_ascii=False) # ensure_ascii=False for Devanagari
                                f_out.write(json_str + '\n'); lines_written_in_batch += 1
                        total_records_saved += lines_written_in_batch
                        print(f"--- Batch {batch_num} processed and saved ({lines_written_in_batch} records). Total saved: {total_records_saved} ---")
                    except IOError as io_err: print(f"  Error writing batch {batch_num} to {output_jsonl_path}: {io_err}")
                    except Exception as write_err: print(f"  Unexpected error writing batch {batch_num}: {write_err}")

                # Cleanup (Unchanged)
                processed_records_buffer = []; del texts_to_annotate, annotated_texts
                if 'inputs' in locals(): del inputs
                if 'input_values' in locals(): del input_values, input_values_truncated
                if 'signal' in locals(): del signal, signal_raw, signal_mono, emotion_signal
                if 'logits_age' in locals(): del logits_age, logits_gender_ml
                torch.cuda.empty_cache(); gc.collect()

        except KeyboardInterrupt: print("\nProcessing interrupted by user."); break
        except Exception as e: print(f"  FATAL ERROR processing file {os.path.basename(audio_path)}: {e}"); import traceback; traceback.print_exc(); torch.cuda.empty_cache(); gc.collect(); continue

    print("\n" + "="*30 + f"\nProcessing Finished.\nTotal files processed attempt: {files_processed_count}/{total_files}\nTotal records saved to {output_jsonl_path}: {total_records_saved}\n" + "="*30)

# --- Main Execution ---
if __name__ == "__main__":
    # --- *** Configuration for Bhojpuri Data (Vaani-like Structure) *** ---
    # ---> !! IMPORTANT: UPDATE THESE PATHS TO YOUR BHOJPURI DATASET !! <---
    #      It should contain the 'audio' and 'metadata' subdirectories.
    BASE_AUDIO_META_DIR = "/external4/datasets/madasr/Vaani-transcription-part/Bhojpuri/bj" # UPDATE THIS PATH (use 'bh' or 'bho' or similar)
    FINAL_OUTPUT_JSONL = "/external4/datasets/madasr/Vaani-transcription-part/Bhojpuri/bj/bh_vaani_annotated.jsonl" # UPDATE THIS PATH
    PROCESSING_BATCH_SIZE = 10 # Adjust as needed

    # --- API Key and Setup (Unchanged) ---
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"): print("ERROR: GOOGLE_API_KEY not found."); exit(1)
    if genai is None: # Re-check genai config
        try:
            api_key_recheck = os.getenv("GOOGLE_API_KEY")
            if not api_key_recheck: raise ValueError("GOOGLE_API_KEY still not found.")
            genai.configure(api_key=api_key_recheck)
            print("Google Generative AI re-configured successfully.")
        except Exception as e: print(f"ERROR: Failed to configure Google Generative AI: {e}"); exit(1)

    print("Starting Bhojpuri Audio Processing and Annotation Workflow...") # Updated print
    print(f"Input Base Directory: {BASE_AUDIO_META_DIR}")
    print(f" (Expecting subdirs: {os.path.join(BASE_AUDIO_META_DIR, 'audio')} and {os.path.join(BASE_AUDIO_META_DIR, 'metadata')})")
    print(f"Final Output File: {FINAL_OUTPUT_JSONL}")
    print(f"Processing Batch Size: {PROCESSING_BATCH_SIZE}")
    print("-" * 30)

    if not os.path.isdir(BASE_AUDIO_META_DIR): print(f"ERROR: Base directory not found: {BASE_AUDIO_META_DIR}"); exit(1)
    output_dir = os.path.dirname(FINAL_OUTPUT_JSONL)
    os.makedirs(output_dir, exist_ok=True) # Ensure output dir exists

    process_audio_and_annotate(
        base_dir=BASE_AUDIO_META_DIR,
        output_jsonl_path=FINAL_OUTPUT_JSONL,
        batch_size=PROCESSING_BATCH_SIZE
    )

    print("Workflow complete.")