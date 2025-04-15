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
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel
)
import torch.nn as nn
import google.generativeai as genai
import time # Added for potential retries/delays

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
    # *** Ensure your GOOGLE_API_KEY is set in your .env file or environment ***
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not found or is empty.")
    genai.configure(api_key=api_key)
    print("Google Generative AI configured successfully.")
    # Optional: Test connectivity (uncomment if needed)
    # test_model = genai.GenerativeModel("gemini-1.5-flash-latest")
    # test_model.generate_content("Test")
    # print("Generative AI connectivity test successful.")
except Exception as e:
    print(f"Error configuring or testing Google Generative AI: {e}. Annotation step will likely fail.")
    genai = None # Set genai to None to prevent further errors if configuration failed

# --- Age/Gender Model Definition (Unchanged - Assumed Language Agnostic) ---
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
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states_pooled = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states_pooled)
        logits_gender = torch.softmax(self.gender(hidden_states_pooled), dim=-1)
        return hidden_states_pooled, logits_age, logits_gender

# --- Emotion Extraction Function (Unchanged - Assumed Mostly Language Agnostic) ---
# Note: Emotion model performance might vary across languages.
# Consider fine-tuning or using a Punjabi-specific model if results are poor.
def extract_emotion(audio_data: np.ndarray, sampling_rate: int = 16000, model_info: dict = None) -> str:
    if model_info is None or 'model' not in model_info or 'feature_extractor' not in model_info:
        print("Error: Emotion model not loaded correctly.")
        return "ErrorLoadingModel"
    if audio_data is None or len(audio_data) == 0:
        return "No_Audio"
    if len(audio_data) < sampling_rate * 0.1:
        # Keep processing short audio for now, maybe add logging
        pass

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


# --- Data Structures (Unchanged - Structure is Language Agnostic) ---
@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    speaker: str
    age: float
    gender: str
    transcription: str # This will hold Punjabi text
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
        age_bucket = self.get_age_bucket(segment.age)
        gender_text = segment.gender
        emotion_text = segment.emotion.upper()
        transcription = segment.transcription.strip() # Keep original case for AI, convert later if needed
        metadata = f"AGE_{age_bucket} GENDER_{gender_text} EMOTION_{emotion_text}"
        # Ensure single space between transcription and metadata
        return f"{transcription.strip()} {metadata.strip()}"

    @staticmethod
    def get_age_bucket(age: float) -> str:
        actual_age = round(age * 100) # Model outputs 0-1 range
        age_brackets = [
            (18, "0_18"), (30, "18_30"), (45, "30_45"),
            (60, "45_60"), (float('inf'), "60PLUS")
        ]
        for threshold, bracket in age_brackets:
            if actual_age < threshold: return bracket
        return "60PLUS" # Default if age is very high or calculation is off

# --- File Handling (Unchanged - Logic is Language Agnostic, Paths Configurable) ---
def get_file_pairs(audio_dir: str, text_dir: str) -> List[Tuple[str, str]]:
    try:
        audio_files_list = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.flac', '.wav', '.mp3'))] # Allow common audio types
        text_files_list = [f for f in os.listdir(text_dir) if f.lower().endswith('.txt')]
        audio_files = {os.path.splitext(f)[0]: os.path.join(audio_dir, f) for f in audio_files_list}
        text_files = {os.path.splitext(f)[0]: os.path.join(text_dir, f) for f in text_files_list}
        pairs = []
        for base_name, audio_path in audio_files.items():
            if base_name in text_files:
                pairs.append((audio_path, text_files[base_name]))
            # else: # Optional: log missing pairs
            #     print(f"Debug: No matching text file found for {os.path.basename(audio_path)}")
        print(f"Found {len(pairs)} matching audio-text pairs.")
        if not pairs:
             print(f"Searched in:\n Audio: {audio_dir}\n Text: {text_dir}")
             print(f"Audio files found: {len(audio_files_list)}")
             print(f"Text files found: {len(text_files_list)}")
        return pairs
    except FileNotFoundError as e:
        print(f"Error finding files: {e}. Check directory paths.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred in get_file_pairs: {e}")
        return []

def get_transcription(text_file: str) -> str:
    try:
        # *** CRITICAL: Ensure UTF-8 for Punjabi (Gurmukhi script) ***
        with open(text_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading transcription file {text_file}: {str(e)}")
        return ""

# --- AI Annotation Functions (MODIFIED for Punjabi) ---

def correct_entity_tag_spaces(text: str) -> str:
    """Removes spaces within ENTITY_TYPE names like 'CIT Y' -> 'CITY'."""
    # (Unchanged - Logic is based on tag format, not language)
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
    """
    Cleans up END tags and fixes spacing around tags and Punjabi punctuation.
    """
    if not isinstance(text, str):
        return text

    # 1. Remove spaces within Entity Type names
    text = correct_entity_tag_spaces(text)

    # 2. Remove potential leading/trailing whitespace
    text = text.strip()

    # 3. Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # 4. Remove END tags immediately preceding metadata tags or end of string
    text = re.sub(r'\s+END\s+(\b(?:AGE_|GENDER_|EMOTION_|INTENT_)|$)', r' \1', text)

    # 5. Remove duplicate END tags
    text = re.sub(r'\s+END\s+END\b', ' END', text)
    text = re.sub(r'\s+END\s+END\b', ' END', text) # Run twice

    # 6. Ensure ENTITY_TAG<space>text is followed by <space>END
    pattern_add_end = r'(ENTITY_[A-Z0-9_]+\s+\S.*?)(?<!\sEND)(?=\s+(\bAGE_|\bGENDER_|\bEMOTION_|\bINTENT_|\bENTITY_)|$)'
    text = re.sub(pattern_add_end, r'\1 END', text)

    # 7. Ensure space between tag and text, and text and END
    text = re.sub(r'(ENTITY_[A-Z0-9_]+)(\S)', r'\1 \2', text) # Space after tag if missing
    text = re.sub(r'(\S)(END\b)', r'\1 \2', text)        # Space before END if missing

    # 8. Punjabi/Common punctuation spacing rules (apply last)
    # Remove space BEFORE specific punctuation (including Punjabi Danda '।')
    # This regex should work for Punjabi '।' as it did for Hindi '।'
    text = re.sub(r'\s+([।?!:;,.])', r'\1', text)
    # Ensure space AFTER punctuation if followed by a word character (Gurmukhi or Latin)
    # \w includes Unicode letters, covering Gurmukhi script characters.
    text = re.sub(r'([।?!:;,.])(\w)', r'\1 \2', text)

    # 9. Final trim and space normalization
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# *** MODIFIED PROMPT FOR PUNJABI ***
def annotate_batch_texts(texts_to_annotate: List[str]):
    """Sends a batch of texts to Gemini for annotation (Prompt adapted for Punjabi)."""
    if not genai:
        print("Error: Google Generative AI not configured. Skipping annotation.")
        return texts_to_annotate # Return original texts
    if not texts_to_annotate:
        return []

    # Input texts already have metadata like: "punjabi text AGE_X GENDER_Y EMOTION_Z"

    # *** PUNJABI PROMPT STARTS HERE ***
    prompt = f'''You are an expert linguistic annotator for **Punjabi** text written in the **Gurmukhi** script.
You will receive a list of Punjabi sentences. Each sentence already includes metadata tags (AGE_*, GENDER_*, EMOTION_*) at the end.

Your task is crucial and requires precision:
1.  **PRESERVE EXISTING TAGS:** Keep the `AGE_`, `GENDER_`, and `EMOTION_` tags exactly as they appear at the end of each sentence. DO NOT modify or move them.
2.  **ENTITY ANNOTATION (Punjabi Text Only):** Identify entities ONLY within the Punjabi transcription part of the sentence. Use ONLY the entity types from the provided list.
3.  **ENTITY TAG FORMAT (VERY IMPORTANT):**
    *   Insert the tag **BEFORE** the entity: `ENTITY_<TYPE>` (e.g., `ENTITY_CITY`, `ENTITY_PERSON_NAME`).
    *   **NO SPACES** are allowed within the `<TYPE>` part (e.g., use `PERSON_NAME`, NOT `PERSON_ NAM E`).
    *   Immediately **AFTER** the Punjabi entity text, add a single space followed by `END`.
    *   Example: `... ENTITY_CITY ਦਿੱਲੀ END ...`
    *   **DO NOT** add an `END` tag just before the `AGE_`, `GENDER_`, `EMOTION_` tags unless it belongs to a preceding entity.
4.  **INTENT TAG:** Determine the single primary intent of the Punjabi transcription (e.g., INFORM, QUESTION, REQUEST, COMMAND, GREETING, etc.). Add ONE `INTENT_<INTENT_TYPE>` tag at the absolute end of the entire string, AFTER all other tags.
5.  **OUTPUT FORMAT:** Return a JSON array of strings, where each string is a fully annotated sentence adhering to all rules.
6.  **PUNJABI SPECIFICS:** Handle **Punjabi script (Gurmukhi)**, punctuation (like the Danda `।`), and spacing correctly according to standard Punjabi rules. Ensure proper spacing around the inserted tags.

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

**Example Input String (Punjabi):**
"ਮੈਂ ਕੱਲ੍ਹ ਮਾਰੀਆ ਨੂੰ ਦਿੱਲੀ ਵਿੱਚ ਵੇਖਿਆ। AGE_30_45 GENDER_FEMALE EMOTION_NEUTRAL"

**CORRECT Example Output String (Punjabi):**
"ਮੈਂ ਕੱਲ੍ਹ ENTITY_PERSON_NAME ਮਾਰੀਆ END ਨੂੰ ENTITY_CITY ਦਿੱਲੀ END ਵਿੱਚ ਵੇਖਿਆ। AGE_30_45 GENDER_FEMALE EMOTION_NEUTRAL INTENT_INFORM"

**INCORRECT Example Output String (Spaces in Tag):**
"ਮੈਂ ਕੱਲ੍ਹ ENTITY_PERSON_ NAM E ਮਾਰੀਆ END ਨੂੰ ENTITY_CIT Y ਦਿੱਲੀ END ਵਿੱਚ ਵੇਖਿਆ। AGE_30_45 GENDER_FEMALE EMOTION_NEUTRAL INTENT_INFORM"

**INCORRECT Example Output String (Extra END before metadata):**
"ਮੈਂ ਕੱਲ੍ਹ ENTITY_PERSON_NAME ਮਾਰੀਆ END ਨੂੰ ENTITY_CITY ਦਿੱਲੀ END ਵਿੱਚ ਵੇਖਿਆ। END AGE_30_45 GENDER_FEMALE EMOTION_NEUTRAL INTENT_INFORM"


**Sentences to Annotate Now:**
{json.dumps(texts_to_annotate, ensure_ascii=False, indent=2)}
'''
    # *** PUNJABI PROMPT ENDS HERE ***

    max_retries = 3
    retry_delay = 5 # seconds
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash-latest") # Or "gemini-1.5-pro-latest" for potentially better quality
            # Consider adding safety settings if needed (same as before)
            # safety_settings = [...]
            # response = model.generate_content(prompt, safety_settings=safety_settings)
            response = model.generate_content(prompt)

            # Debug: Print raw response if needed
            # print(f"--- Raw Gemini Response (Attempt {attempt+1}) ---")
            # print(response.text)
            # print("-------------------------")

            assistant_reply = response.text.strip()

            # Handle potential markdown code block formatting
            if assistant_reply.startswith("```json"):
                assistant_reply = assistant_reply[len("```json"):].strip()
            elif assistant_reply.startswith("```"):
                 assistant_reply = assistant_reply[len("```"):].strip()
            if assistant_reply.endswith("```"):
                assistant_reply = assistant_reply[:-len("```")].strip()

            # Basic check if response looks like JSON list
            if not (assistant_reply.startswith('[') and assistant_reply.endswith(']')):
                 # Try to extract JSON even if it's not perfectly formatted at start/end
                 match = re.search(r'\[.*\]', assistant_reply, re.DOTALL)
                 if match:
                     print("Warning: Extracted JSON content from potentially malformed response.")
                     assistant_reply = match.group(0)
                 else:
                    raise json.JSONDecodeError("Response does not appear to contain a JSON list.", assistant_reply, 0)


            annotated_sentences_raw = json.loads(assistant_reply)

            if isinstance(annotated_sentences_raw, list) and len(annotated_sentences_raw) == len(texts_to_annotate):
                # Apply post-processing fixes robustly
                processed_sentences = []
                for sentence in annotated_sentences_raw:
                     if isinstance(sentence, str):
                          # 1. Correct spaces in tags FIRST
                          corrected_sentence = correct_entity_tag_spaces(sentence)
                          # 2. Fix END tags and spacing (using Punjabi-aware function)
                          final_sentence = fix_end_tags_and_spacing(corrected_sentence)
                          processed_sentences.append(final_sentence)
                     else:
                          # Handle non-string elements if necessary (e.g., log error, keep original)
                          print(f"Warning: Non-string item received in annotation list: {sentence}")
                          # Try to convert to string or use original if possible
                          original_index = annotated_sentences_raw.index(sentence) # Requires unique items or careful handling
                          processed_sentences.append(texts_to_annotate[original_index] + " ANNOTATION_ERROR_NON_STRING")

                # Extra check: Ensure final list length still matches
                if len(processed_sentences) == len(texts_to_annotate):
                    return processed_sentences
                else:
                    print(f"Error: Mismatch after processing non-string elements. Expected {len(texts_to_annotate)}, Got {len(processed_sentences)}")
                    # Fallback to original texts only on the last attempt
                    if attempt == max_retries - 1:
                        return texts_to_annotate
                    else:
                        raise ValueError("Processing error lead to length mismatch.") # Force retry

            else:
                print(f"Error: API did not return a valid list or mismatched length. Expected {len(texts_to_annotate)}, Got {len(annotated_sentences_raw) if isinstance(annotated_sentences_raw, list) else 'Invalid Type'}")
                # Fallback to original texts only on the last attempt
                if attempt == max_retries - 1:
                    return texts_to_annotate
                else:
                    print(f"Retrying annotation (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(retry_delay * (attempt + 1)) # Exponential backoff

        except json.JSONDecodeError as e:
            print(f"JSON decoding failed (Attempt {attempt+1}/{max_retries}): {e}")
            print("Problematic Raw response snippet:", assistant_reply[:500] if 'assistant_reply' in locals() else "N/A")
            if attempt == max_retries - 1: return texts_to_annotate
            print(f"Retrying annotation...")
            time.sleep(retry_delay * (attempt + 1))
        except Exception as e:
            # Catch potential API errors (rate limits, connection issues, etc.)
            print(f"Error calling/processing Generative AI (Attempt {attempt+1}/{max_retries}): {e}")
            # Check for specific API error types if library provides them
            if "rate limit" in str(e).lower():
                 print("Rate limit likely hit.")
                 # Increase delay significantly for rate limits
                 time.sleep(retry_delay * (attempt + 1) * 5)
            elif "API key not valid" in str(e):
                 print("FATAL: Invalid Google API Key. Please check your GOOGLE_API_KEY environment variable.")
                 # No point retrying if key is bad
                 return texts_to_annotate
            else:
                 time.sleep(retry_delay * (attempt + 1))
            if attempt == max_retries - 1: return texts_to_annotate # Fallback on final attempt

    # Should not be reached if loop completes, but as a safeguard
    print("Error: Max retries reached for annotation.")
    return texts_to_annotate


# --- Main Processing Function (Logic Unchanged, Paths Configurable) ---
def process_audio_and_annotate(base_dir: str, output_jsonl_path: str, batch_size: int = 10) -> None:
    """
    Processes audio files, extracts metadata, gets transcriptions,
    formats text, annotates with AI, and saves final JSONL output.
    (Now configured for Punjabi via prompt and path settings)
    """
    output_dir = os.path.dirname(output_jsonl_path)
    os.makedirs(output_dir, exist_ok=True)

    # --- Clear output file at the very beginning ---
    try:
        print(f"Attempting to clear output file: {output_jsonl_path}")
        # *** CRITICAL: Ensure UTF-8 for writing Punjabi output ***
        with open(output_jsonl_path, 'w', encoding='utf-8') as f_clear:
            f_clear.write("") # Write empty string to ensure file is cleared/created
        print(f"Output file {output_jsonl_path} cleared successfully.")
    except IOError as e:
        print(f"Error clearing output file {output_jsonl_path}: {e}. Please check permissions.")
        return # Exit if we can't write to the output file

    # --- Load Models ---
    print("Loading models...")
    # Age/Gender Model (Language Agnostic)
    age_gender_model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    try:
        processor = Wav2Vec2Processor.from_pretrained(age_gender_model_name)
        age_gender_model = AgeGenderModel.from_pretrained(age_gender_model_name).to(device)
        age_gender_model.eval()
        print("Age/Gender model loaded.")
    except Exception as e:
        print(f"FATAL Error loading Age/Gender model: {e}. Exiting.")
        return

    # Emotion Model (Largely Language Agnostic, but monitor performance)
    emotion_model_name = "superb/hubert-large-superb-er"
    emotion_model_info = {}
    try:
        emotion_model_info['model'] = AutoModelForAudioClassification.from_pretrained(emotion_model_name).to(device)
        emotion_model_info['model'].eval()
        emotion_model_info['feature_extractor'] = AutoFeatureExtractor.from_pretrained(emotion_model_name)
        print("Emotion model loaded.")
    except Exception as e:
        print(f"Error loading Emotion model: {e}. Emotion extraction will use 'ErrorLoadingModel'.")
        # Allow continuing without emotion model

    print("-" * 30)

    # --- Prepare File Paths (Using base_dir for Punjabi data) ---
    audio_dir = os.path.join(base_dir, "audio") # Standard subfolder names
    text_dir = os.path.join(base_dir, "text")   # Standard subfolder names

    if not os.path.exists(audio_dir) or not os.path.exists(text_dir):
        print(f"Error: Audio ({audio_dir}) or text ({text_dir}) directory not found in {base_dir}")
        print("Please ensure your Punjabi data follows the structure:")
        print(f"{base_dir}/")
        print(f"  ├── audio/  (contains .flac/.wav files)")
        print(f"  └── text/   (contains .txt files with matching basenames)")
        return

    print(f"Processing Punjabi files from:\n  Audio: {audio_dir}\n  Text:  {text_dir}")
    print(f"Output will be saved to: {output_jsonl_path}")
    print("-" * 30)

    # Get matching audio-text pairs
    file_pairs = get_file_pairs(audio_dir, text_dir)
    if not file_pairs:
        print("No matching audio-text files found. Exiting.")
        return

    total_files = len(file_pairs)
    print(f"Found {total_files} audio-text pairs to process.")

    processed_records_buffer = [] # Holds records before AI annotation batch
    batch_num = 0
    files_processed_count = 0
    total_records_saved = 0

    # --- Process Files ---
    for i, (audio_path, text_path) in enumerate(file_pairs):
        print(f"\nProcessing file {i+1}/{total_files}: {os.path.basename(audio_path)}")
        try:
            # 1. Get Punjabi Transcription (Ensures UTF-8)
            transcription = get_transcription(text_path)
            if not transcription:
                print(f"  Skipping: Empty or unreadable transcription in {text_path}")
                continue

            # 2. Load Audio
            try:
                # Use soundfile for potentially broader codec support if librosa fails
                try:
                    signal, sr = librosa.load(audio_path, sr=16000, mono=True)
                except Exception as librosa_err:
                    print(f"  Librosa failed ({librosa_err}), trying soundfile...")
                    import soundfile as sf
                    signal, sr = sf.read(audio_path, dtype='float32')
                    if sr != 16000:
                        # Resample if necessary (should ideally be pre-processed)
                        print(f"  Warning: Resampling {audio_path} from {sr} Hz to 16000 Hz.")
                        signal = librosa.resample(y=signal, orig_sr=sr, target_sr=16000)
                    # Ensure mono
                    if signal.ndim > 1 and signal.shape[1] > 1:
                         signal = librosa.to_mono(signal.T) # Transpose if shape is (samples, channels)
                    elif signal.ndim > 1 and signal.shape[0] > 1:
                         signal = librosa.to_mono(signal)   # Assume shape is (channels, samples)

                if signal is None or len(signal) == 0:
                    print(f"  Skipping: Failed to load or empty audio in {audio_path}")
                    continue
                duration = round(len(signal) / 16000, 2) # Use target SR 16000
                if duration < 0.1:
                     print(f"  Skipping: Audio too short ({duration:.2f}s) in {audio_path}")
                     continue
            except Exception as load_err:
                 print(f"  Skipping: Error loading audio {audio_path}: {load_err}")
                 continue

            # 3. Extract Age/Gender
            try:
                inputs = processor(signal, sampling_rate=16000, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device)
                # Handle potential long audio causing memory issues
                max_len = 30 * 16000 # Process max 30 seconds for age/gender/emotion if needed
                if input_values.shape[1] > max_len:
                    print(f"  Warning: Audio longer than 30s ({duration}s), truncating for Age/Gender/Emotion extraction.")
                    input_values = input_values[:, :max_len]
                    # Note: This affects emotion extraction too if done here

                with torch.no_grad():
                    _, logits_age, logits_gender = age_gender_model(input_values)
                age = logits_age.cpu().numpy().item()
                gender_idx = torch.argmax(logits_gender, dim=-1).cpu().numpy().item()
                gender_map = {0: "FEMALE", 1: "MALE", 2: "OTHER"}
                gender = gender_map.get(gender_idx, "UNKNOWN")
            except RuntimeError as e:
                 if "CUDA out of memory" in str(e):
                     print(f"  CUDA OOM Error during Age/Gender extraction for {audio_path}. Skipping file.")
                     torch.cuda.empty_cache()
                     gc.collect()
                     continue # Skip this file
                 else:
                     print(f"  Runtime Error during Age/Gender extraction: {e}")
                     age = -1.0
                     gender = "ERROR"
            except Exception as age_gender_err:
                print(f"  Error during Age/Gender extraction: {age_gender_err}")
                age = -1.0
                gender = "ERROR"

            # 4. Extract Emotion (Using potentially truncated signal if long)
            emotion_signal = signal[:max_len] if 'max_len' in locals() and len(signal) > max_len else signal
            emotion = extract_emotion(emotion_signal, 16000, emotion_model_info)

            # 5. Create Initial Record Structure
            # Try to get speaker ID from filename (e.g., speakerid_...)
            try:
                speaker = os.path.splitext(os.path.basename(audio_path))[0].split('_')[0]
            except:
                speaker = "UNKNOWN_SPEAKER"

            segment_data = AudioSegment(
                start_time=0, end_time=duration, speaker=speaker, age=age,
                gender=gender, transcription=transcription, emotion=emotion,
                chunk_filename=os.path.basename(audio_path), duration=duration
            )
            chunk = ChunkData(segments=[segment_data], filepath=os.path.abspath(audio_path))
            initial_formatted_text = chunk.get_formatted_text()

            # Store essential info needed for the final JSONL
            record = {
                "audio_filepath": chunk.filepath,
                "duration": duration,
                "initial_text": initial_formatted_text, # Temp field for AI input
                 # Keep raw values for potential analysis, can be removed if not needed
                 "raw_age_output": age,
                 "raw_gender_prediction": gender,
                 "raw_emotion_prediction": emotion,
                 "speaker_id": speaker,
            }
            processed_records_buffer.append(record)
            files_processed_count += 1

            # 6. Annotate and Save in Batches
            if len(processed_records_buffer) >= batch_size or (i + 1) == total_files:
                batch_num += 1
                current_batch_size = len(processed_records_buffer)
                print(f"\n--- Annotating Batch {batch_num} ({current_batch_size} records) ---")

                texts_to_annotate = [rec["initial_text"] for rec in processed_records_buffer]

                # Call Gemini for annotation (with retries and Punjabi prompt)
                annotated_texts = annotate_batch_texts(texts_to_annotate)

                # Verify annotation results length BEFORE attempting to save
                if len(annotated_texts) != current_batch_size:
                    print(f"  FATAL BATCH ERROR: Annotation count mismatch! Expected {current_batch_size}, Got {len(annotated_texts)}. Skipping save for this batch to prevent data corruption.")
                    # Log the failed batch for later inspection if needed
                    # error_log_path = output_jsonl_path + f".error_batch_{batch_num}.log"
                    # print(f"  Logging error details to: {error_log_path}")
                    # try:
                    #     with open(error_log_path, 'w', encoding='utf-8') as f_err:
                    #         json.dump({"original_records": processed_records_buffer,
                    #                   "texts_sent_to_ai": texts_to_annotate,
                    #                   "texts_received_from_ai": annotated_texts},
                    #                   f_err, indent=2, ensure_ascii=False)
                    # except Exception as log_e:
                    #     print(f"    Failed to write error log: {log_e}")

                else:
                     # Save the annotated batch to the final JSONL
                    try:
                        lines_written_in_batch = 0
                        # *** CRITICAL: Ensure UTF-8 for appending Punjabi output ***
                        with open(output_jsonl_path, 'a', encoding='utf-8') as f_out:
                            for record_data, annotated_text in zip(processed_records_buffer, annotated_texts):
                                # Prepare final record: use 'text' for the final annotated string
                                final_record = {
                                    "audio_filepath": record_data["audio_filepath"],
                                    "duration": record_data["duration"],
                                    "text": annotated_text, # This is the AI annotated + processed text
                                    # Optionally include other fields if needed
                                    # "speaker_id": record_data["speaker_id"],
                                }
                                # *** CRITICAL: ensure_ascii=False for Punjabi ***
                                json_str = json.dumps(final_record, ensure_ascii=False)
                                f_out.write(json_str + '\n')
                                lines_written_in_batch += 1

                        total_records_saved += lines_written_in_batch
                        print(f"--- Batch {batch_num} annotated and saved ({lines_written_in_batch} records). Total saved: {total_records_saved} ---")

                    except IOError as io_err:
                         print(f"  Error writing batch {batch_num} to {output_jsonl_path}: {io_err}")
                    except Exception as write_err:
                         print(f"  Unexpected error writing batch {batch_num}: {write_err}")

                # Clear buffer and clean up memory regardless of save success/failure
                processed_records_buffer = []
                torch.cuda.empty_cache()
                gc.collect()
                # Optional small delay between batches
                # time.sleep(1)

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")
            break
        except Exception as e:
            print(f"  FATAL ERROR processing file {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            continue # Skip to the next file

    print("\n" + "="*30)
    print(f"Processing Finished.")
    print(f"Total files processed attempt: {files_processed_count}/{total_files}")
    print(f"Total records saved to {output_jsonl_path}: {total_records_saved}")
    print("="*30)


# --- Main Execution ---
if __name__ == "__main__":
    # --- *** Configuration for Punjabi Data *** ---
    # 1. SET THE BASE DIRECTORY CONTAINING YOUR PUNJABI 'audio' and 'text' subfolders
    BASE_AUDIO_TEXT_DIR = "/external4/datasets/Shrutilipi/punjabi/pn"  # <--- CHANGE THIS

    # 2. SET THE DESIRED OUTPUT FILENAME FOR THE ANNOTATED PUNJABI DATA
    FINAL_OUTPUT_JSONL = "/external4/datasets/Shrutilipi/punjabi/pn/pa_annotated_data.jsonl" # <--- CHANGE THIS

    # 3. SET THE BATCH SIZE for AI Annotation (start small, e.g., 5 or 10)
    PROCESSING_BATCH_SIZE = 50# Adjust based on API limits and system memory

    # --- Ensure API key is loaded before starting ---
    load_dotenv() # Load .env file if it exists
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not found in environment or .env file.")
        print("Please set the GOOGLE_API_KEY environment variable or create a .env file.")
        exit(1)
    # Re-check genai configuration after loading .env
    if genai is None:
        try:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            print("Google Generative AI re-configured successfully after .env load.")
        except Exception as e:
             print(f"ERROR: Failed to configure Google Generative AI even after loading .env: {e}")
             exit(1)


    print("Starting Punjabi Audio Processing and Annotation Workflow...")
    print(f"Input Base Directory: {BASE_AUDIO_TEXT_DIR}")
    print(f"Final Output File: {FINAL_OUTPUT_JSONL}")
    print(f"Processing Batch Size: {PROCESSING_BATCH_SIZE}")
    print("-" * 30)

    # Check if base directory exists
    if not os.path.isdir(BASE_AUDIO_TEXT_DIR):
        print(f"ERROR: Base directory not found: {BASE_AUDIO_TEXT_DIR}")
        exit(1)

    # Check if output directory exists, create if not
    output_dir = os.path.dirname(FINAL_OUTPUT_JSONL)
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"ERROR: Could not create output directory: {e}")
            exit(1)

    process_audio_and_annotate(
        base_dir=BASE_AUDIO_TEXT_DIR,
        output_jsonl_path=FINAL_OUTPUT_JSONL,
        batch_size=PROCESSING_BATCH_SIZE
    )

    print("Workflow complete.")