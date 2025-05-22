# -*- coding: utf-8 -*- # Ensure editor recognizes UTF-8 for Bengali script comments/examples
import os
import gc
import json
import re
import torch
import librosa
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    # Added back for age/gender prediction fallback
    Wav2Vec2Processor,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel
)
import torch.nn as nn # Added back for age/gender prediction fallback
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
    device = torch.device("cuda:1")
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
    print(f"Error configuring Google Generative AI: {e}. Annotation step will likely fail.")
    genai = None

# --- Age/Gender Model Definition (Reintroduced for Fallback) ---
class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features; x = self.dropout(x); x = self.dense(x)
        x = torch.tanh(x); x = self.dropout(x); x = self.out_proj(x)
        return x

class AgeGenderModel(Wav2Vec2PreTrainedModel):
    # Uses Wav2Vec2 model to predict age (float) and gender (3 classes)
    def __init__(self, config):
        super().__init__(config)
        self.config = config; self.wav2vec2 = Wav2Vec2Model(config)
        # We only need the age head for the fallback prediction
        self.age = ModelHead(config, 1)
        # self.gender = ModelHead(config, 3) # Gender head not needed if only predicting age
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values); hidden_states = outputs[0]
        hidden_states_pooled = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states_pooled)
        # logits_gender = torch.softmax(self.gender(hidden_states_pooled), dim=-1) # Not needed
        # Return only age logit
        return logits_age

# --- Emotion Extraction Function (Unchanged) ---
def extract_emotion(audio_data: np.ndarray, sampling_rate: int = 16000, model_info: dict = None) -> str:
    """Extracts emotion from audio using a preloaded model."""
    if model_info is None or 'model' not in model_info or 'feature_extractor' not in model_info:
        print("Error: Emotion model not loaded correctly."); return "ERRORLOADINGMODEL" # Consistent uppercase
    if audio_data is None or len(audio_data) == 0: return "NO_AUDIO"
    try:
        inputs = model_info['feature_extractor'](audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        max_len_inference_emotion = 30 * sampling_rate
        if inputs['input_values'].shape[1] > max_len_inference_emotion:
            inputs['input_values'] = inputs['input_values'][:, :max_len_inference_emotion]
            if 'attention_mask' in inputs:
                 inputs['attention_mask'] = inputs['attention_mask'][:, :max_len_inference_emotion]

        with torch.no_grad(): outputs = model_info['model'](**inputs)
        predicted_class_idx = outputs.logits.argmax(-1).item()
        emotion_label = model_info['model'].config.id2label.get(predicted_class_idx, "UNKNOWN")
        return emotion_label.upper()
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"CUDA OOM Error during emotion extraction. Returning 'OOM_ERROR'.");
            torch.cuda.empty_cache(); gc.collect(); return "OOM_ERROR"
        else: print(f"Runtime Error during emotion extraction: {e}"); return "RUNTIME_ERROR"
    except Exception as e: print(f"Error during emotion extraction: {e}"); return "EXTRACTION_ERROR"

# --- Age Prediction Function (New Fallback Function) ---
def predict_age_from_audio(signal: np.ndarray, sampling_rate: int, processor: Wav2Vec2Processor, model: AgeGenderModel) -> Optional[float]:
    """Predicts age (raw float) from audio signal using the provided model."""
    if signal is None or len(signal) == 0:
        print("  Age Prediction Error: No audio signal provided.")
        return None
    try:
        max_len_inference = 30 * sampling_rate # Limit input length for stability
        inputs = processor(signal, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        input_values = inputs.input_values
        input_values_truncated = input_values[:, :max_len_inference].to(device) if input_values.shape[1] > max_len_inference else input_values.to(device)

        with torch.no_grad():
            logits_age = model(input_values_truncated)
        age_prediction_float = logits_age.cpu().numpy().item()
        return age_prediction_float
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"  CUDA OOM Error during age prediction. Cannot predict age.");
            torch.cuda.empty_cache(); gc.collect()
            return None
        else:
            print(f"  Runtime Error during age prediction: {e}"); return None
    except Exception as e:
        print(f"  Error during age prediction: {e}"); return None

# --- Age Bucketing Function (New - Converts predicted float to category) ---
def get_age_bucket_from_prediction(age_float: Optional[float]) -> str:
    """Converts predicted age float (0.0 to 1.0 range approx) to age bucket string."""
    if age_float is None or age_float < 0: # Handle prediction failure or invalid values
        return "UNKNOWN"
    try:
        # Model output is often scaled differently, assume roughly maps 0->0yrs, 1->100yrs for bucketing
        # Adjust multiplier if model has different known output scale
        actual_age = round(age_float * 100)
        if actual_age < 18: return "0_17" # Using slightly different buckets for clarity
        elif actual_age < 30: return "18_29"
        elif actual_age < 45: return "30_44"
        elif actual_age < 60: return "45_59"
        else: return "60PLUS"
    except Exception as e:
        print(f"  Error bucketing predicted age {age_float}: {e}")
        return "UNKNOWN" # Fallback if conversion fails


# --- AI Annotation Helper Functions (fix_end_tags_and_spacing - Unchanged) ---
def fix_end_tags_and_spacing(text: str) -> str:
    """
    Cleans up AI-generated annotations for Bengali text, focusing on END tag placement,
    spacing around tags, internal spaces in tag types, and common punctuation.
    Handles Bengali script characters via \w in regex. Uses Bengali Dari '।'
    """
    if not isinstance(text, str): return text
    text = re.sub(r'\s+', ' ', text).strip()
    def remove_internal_spaces(match):
        tag_prefix = match.group(1); tag_body = match.group(2)
        corrected_body = re.sub(r'\s+', '', tag_body); return f"{tag_prefix}{corrected_body}"
    text = re.sub(r'\b(ENTITY_)([A-Z0-9_]+(?: +[A-Z0-9_]+)+)\b', remove_internal_spaces, text)
    text = re.sub(r'(\bENTITY_[A-Z0-9_]+)(\S)', r'\1 \2', text)
    text = re.sub(r'(\S)(END\b)', r'\1 \2', text)
    text = re.sub(r'(\bEND)(\S)', r'\1 \2', text)
    text = re.sub(r'\bEND\s+END\b', 'END', text)
    text = re.sub(r'\bEND\s+END\b', 'END', text)
    text = re.sub(r'\bEND\s*(\b(?:AGE_|GENDER_|EMOTION_|DOMAIN_|INTENT_)|$)', r'\1', text)
    text = re.sub(r'\bEND\s*(\b(?:AGE_|GENDER_|EMOTION_|DOMAIN_|INTENT_)|$)', r'\1', text)
    text = re.sub(r'\s+([?!:;,.।])', r'\1', text)
    text = re.sub(r'([?!:;,.।])(\w)', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- MODIFIED PROMPT FOR BENGALI (annotate_batch_texts - Unchanged from previous Bengali version) ---
def annotate_batch_texts(texts_to_annotate: List[str]):
    """Sends a batch of texts to Gemini for annotation (Prompt refined for Bengali)."""
    if not genai:
        print("Error: Google Generative AI not configured. Skipping annotation.")
        return [t + " ANNOTATION_ERROR_NO_GENAI" for t in texts_to_annotate]
    if not texts_to_annotate:
        return []

    # Input texts have metadata like: "bengali text AGE_X GENDER_Y EMOTION_Z DOMAIN_W" (DOMAIN_W is already capitalized)

    # *** REFINED BENGALI PROMPT STARTS HERE ***
    prompt = f'''You are an expert linguistic annotator specifically for **Bengali** text written in the **Bengali script**. Your task is to process a list of Bengali sentences, each already containing `AGE_*`, `GENDER_*`, `EMOTION_*`, and `DOMAIN_*` metadata tags at the end. Follow these instructions with extreme precision:

1.  **Preserve Existing Metadata Tags:**
    *   The `AGE_*`, `GENDER_*`, `EMOTION_*`, and `DOMAIN_*` tags at the end of each sentence **must** remain exactly as they are, including the case (e.g., `DOMAIN_BANKING`).
    *   **Do not** modify, move, delete, or change these existing metadata tags.

2.  **Entity Annotation (Bengali Text Only):**
    *   Identify entities **only** within the main Bengali transcription part of the sentence (before the metadata).
    *   Use **only** the entity types provided in the `Allowed ENTITY TYPES` list below. Do not invent new types.

3.  **Strict Entity Tagging Format:**
    *   **Tag Insertion:** Place the entity tag **immediately before** the identified Bengali entity text. The tag format is `ENTITY_<TYPE>` (e.g., `ENTITY_CITY`, `ENTITY_PERSON_NAME`).
    *   **NO SPACES IN TYPE:** Ensure the `<TYPE>` part of the tag contains **no spaces** (e.g., `PERSON_NAME` is correct, `PERSON_ NAM E` is **incorrect**).
    *   **`END` Tag Placement:** Place the literal string `END` **immediately after** the *complete* Bengali entity text.
    *   **Spacing:** There must be exactly **one space** between the `ENTITY_<TYPE>` tag and the start of the entity text. There must be exactly **one space** between the end of the entity text and the `END` tag. There must be exactly **one space** after the `END` tag before the next word begins (unless followed by punctuation).
    *   **Example:** For the entity "কলকাতা", the correct annotation is `ENTITY_CITY কলকাতা END`.
    *   **Crucial `END` Rule:** Only add **one** `END` tag right after the full entity phrase. Do **not** add `END` tags after individual words within a multi-word entity or after words that simply follow an entity.
    *   **Avoid Extra `END` Before Metadata:** Do **not** place an `END` tag just before the `AGE_`, `GENDER_`, `EMOTION_`, `DOMAIN_`, or `INTENT_` tags unless that `END` tag correctly marks the end of an immediately preceding entity.

4.  **Intent Tag:**
    *   Determine the single primary intent of the Bengali sentence (e.g., `INFORM`, `QUESTION`, `REQUEST`, `COMMAND`, `GREETING`, `OTHER`).
    *   Append **one** `INTENT_<INTENT_TYPE>` tag at the **absolute end** of the entire string, after all other tags (`AGE_`, `GENDER_`, `EMOTION_`, `DOMAIN_`).

5.  **Output Format:**
    *   Return a **JSON array** of strings. Each string must be a fully annotated sentence following all the rules above.

6.  **Bengali Language Specifics:**
    *   Handle Bengali script correctly.
    *   Ensure correct spacing around Bengali punctuation (like `।`, `.`, `?`, `!`). Remove space before punctuation, ensure space after punctuation if followed by a word.
    *   The final output string must be clean, with single spaces separating words and tags according to the rules.

**Allowed ENTITY TYPES (Use Only These):**
    [ "PERSON_NAME", "ORGANIZATION", "LOCATION", "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP_CODE", "CURRENCY", "PRICE", "DATE",
    "TIME", "DURATION", "APPOINTMENT_DATE", "APPOINTMENT_TIME", "DEADLINE", "DELIVERY_DATE", "DELIVERY_TIME", "EVENT", "MEETING",
    "TASK", "PROJECT_NAME", "ACTION_ITEM", "PRIORITY", "FEEDBACK", "REVIEW", "RATING", "COMPLAINT", "QUESTION", "RESPONSE", "NOTIFICATION_TYPE", "AGENDA",
    "REMINDER", "NOTE", "RECORD", "ANNOUNCEMENT", "UPDATE", "SCHEDULE", "BOOKING_REFERENCE", "APPOINTMENT_NUMBER", "ORDER_NUMBER", "INVOICE_NUMBER", "PAYMENT_METHOD", "PAYMENT_AMOUNT", "BANK_NAME", "ACCOUNT_NUMBER", "CREDIT_CARD_NUMBER", "TAX_ID", "SOCIAL_SECURITY_NUMBER", "DRIVER'S_LICENSE", "PASSPORT_NUMBER", "INSURANCE_PROVIDER", "POLICY_NUMBER", "INSURANCE_PLAN", "CLAIM_NUMBER", "POLICY_HOLDER", "BENEFICIARY",
    "RELATIONSHIP", "EMERGENCY_CONTACT", "PROJECT_PHASE", "VERSION", "DEVELOPMENT_STAGE", "DEVICE_NAME", "OPERATING_SYSTEM", "SOFTWARE_VERSION", "BRAND", "MODEL_NUMBER", "LICENSE_PLATE", "VEHICLE_MAKE", "VEHICLE_MODEL", "VEHICLE_TYPE", "FLIGHT_NUMBER", "HOTEL_NAME", "ROOM_NUMBER", "TRANSACTION_ID", "TICKET_NUMBER", "SEAT_NUMBER", "GATE", "TERMINAL", "TRANSACTION_TYPE", "PAYMENT_STATUS", "PAYMENT_REFERENCE", "INVOICE_STATUS",
    "SYMPTOM", "DIAGNOSIS", "MEDICATION", "DOSAGE", "ALLERGY", "PRESCRIPTION", "TEST_NAME", "TEST_RESULT", "MEDICAL_RECORD", "HEALTH_STATUS", "HEALTH_METRIC", "VITAL_SIGN", "DOCTOR_NAME", "HOSPITAL_NAME", "DEPARTMENT", "WARD", "CLINIC_NAME", "WEBSITE", "URL", "IP_ADDRESS", "MAC_ADDRESS", "USERNAME", "PASSWORD", "LANGUAGE", "CODE_SNIPPET", "DATABASE_NAME", "API_KEY", "WEB_TOKEN", "URL_PARAMETER", "SERVER_NAME", "ENDPOINT", "DOMAIN" ]

**Examples Demonstrating Correct Formatting (Bengali - Using structure examples):**

*   **Input:** `"আমার নাম রমেশ। AGE_30-45 GENDER_MALE EMOTION_NEUTRAL DOMAIN_GENERAL"`
*   **Correct Output:** `"আমার নাম ENTITY_PERSON_NAME রমেশ END । AGE_30-45 GENDER_MALE EMOTION_NEUTRAL DOMAIN_GENERAL INTENT_INFORM"`

*   **Input:** `"সে কলকাতা তে থাকে। AGE_45-60 GENDER_FEMALE EMOTION_NEUTRAL DOMAIN_LOCATION"`
*   **Correct Output:** `"সে ENTITY_CITY কলকাতা END তে থাকে। AGE_45-60 GENDER_FEMALE EMOTION_NEUTRAL DOMAIN_LOCATION INTENT_INFORM"`

*   **Input:** `"মিটিং ১০ই মে হবে। AGE_18-24 GENDER_MALE EMOTION_NEUTRAL DOMAIN_MEETING"`
*   **Correct Output:** `"মিটিং ENTITY_DATE ১০ই মে END হবে। AGE_18-24 GENDER_MALE EMOTION_NEUTRAL DOMAIN_MEETING INTENT_INFORM"`

*   **Input:** `"দয়া করে চুপ করুন? AGE_60+ GENDER_MALE EMOTION_ANGRY DOMAIN_INSTRUCTION"`
*   **Correct Output:** `"দয়া করে চুপ করুন? AGE_60+ GENDER_MALE EMOTION_ANGRY DOMAIN_INSTRUCTION INTENT_REQUEST"`

**Incorrect Examples (Common Mistakes to Avoid):**
*   `... ENTITY_PERSON_ NAM E রমেশ END ...` (Space in TYPE)
*   `... ENTITY_CITY কলকাতাEND ...` (Missing space before END)
*   `... ENTITY_CITY কলকাতা ENDতে ...` (Missing space after END)
*   `... থাকে END AGE_...` (Unnecessary END before metadata)
*   `... ১০ই END মে END ...` (Incorrectly adding END after parts of a multi-word entity)

Provide only the JSON array containing the correctly annotated sentences based precisely on these instructions.

**Sentences to Annotate Now:**
{json.dumps(texts_to_annotate, ensure_ascii=False, indent=2)}
'''
    # *** REFINED BENGALI PROMPT ENDS HERE ***

    max_retries = 3; retry_delay = 5 # seconds
    for attempt in range(max_retries):
        try:
            if not genai:
                 print("Error: genai object not available in annotate_batch_texts. Skipping.");
                 raise ConnectionError("Google Generative AI not configured.")

            model = genai.GenerativeModel("gemini-1.5-flash-latest") # Use a suitable model
            response = model.generate_content(prompt)
            assistant_reply = response.text.strip()

            # --- Robust JSON Extraction ---
            json_match = re.search(r'\[.*\]', assistant_reply, re.DOTALL)
            if json_match: json_str = json_match.group(0)
            else:
                # Handle potential markdown ```json ... ```
                if assistant_reply.startswith("```json"): assistant_reply = assistant_reply[len("```json"):].strip()
                elif assistant_reply.startswith("```"): assistant_reply = assistant_reply[len("```"):].strip()
                if assistant_reply.endswith("```"): assistant_reply = assistant_reply[:-len("```")].strip()

                # Check if the cleaned string looks like a JSON list
                if assistant_reply.startswith('[') and assistant_reply.endswith(']'): json_str = assistant_reply
                else:
                    print(f"Error: Could not extract valid JSON list from response (Attempt {attempt+1}). Response snippet:\n---\n{assistant_reply[:500]}\n---")
                    # Fallback: Try parsing the whole response if it vaguely looks like JSON
                    try:
                        potential_json = json.loads(assistant_reply)
                        if isinstance(potential_json, list):
                            print("Warning: Extracted JSON by parsing the whole response.")
                            json_str = assistant_reply
                        else:
                            raise json.JSONDecodeError("Response does not appear to contain a JSON list.", assistant_reply, 0)
                    except json.JSONDecodeError:
                         raise json.JSONDecodeError("Response does not appear to contain a JSON list.", assistant_reply, 0)


            # --- JSON Parsing and Post-Processing ---
            try: annotated_sentences_raw = json.loads(json_str)
            except json.JSONDecodeError as json_e:
                print(f"JSON decoding failed specifically on extracted string (Attempt {attempt+1}): {json_e}"); print("Extracted string snippet:", json_str[:500]); raise json_e

            if isinstance(annotated_sentences_raw, list) and len(annotated_sentences_raw) == len(texts_to_annotate):
                processed_sentences = []
                for idx, sentence in enumerate(annotated_sentences_raw):
                     if isinstance(sentence, str): processed_sentences.append(fix_end_tags_and_spacing(sentence)) # Apply Bengali fixing rules
                     else:
                          print(f"Warning: Non-string item received in annotation list at index {idx}: {sentence}")
                          try: processed_sentences.append(fix_end_tags_and_spacing(texts_to_annotate[idx]) + " ANNOTATION_ERROR_NON_STRING")
                          except IndexError: print(f"Error: Could not map non-string item at index {idx} back to original text."); processed_sentences.append("ANNOTATION_ERROR_UNKNOWN_ORIGINAL")

                if len(processed_sentences) == len(texts_to_annotate): return processed_sentences # Success
                else:
                    print(f"Error: Mismatch after processing non-string elements. Expected {len(texts_to_annotate)}, Got {len(processed_sentences)}")
                    if attempt == max_retries - 1: return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_PROCESSING" for t in texts_to_annotate]
                    else: raise ValueError("Processing error lead to length mismatch.")
            else:
                print(f"Error: API returned invalid list or mismatched length (Attempt {attempt+1}). Expected {len(texts_to_annotate)}, Got {len(annotated_sentences_raw) if isinstance(annotated_sentences_raw, list) else type(annotated_sentences_raw)}")
                if attempt == max_retries - 1: return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_LENGTH" for t in texts_to_annotate]
                else: print(f"Retrying annotation (attempt {attempt + 2}/{max_retries})..."); time.sleep(retry_delay * (attempt + 1))

        except json.JSONDecodeError as e:
            print(f"JSON decoding failed (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1: return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_JSON_DECODE" for t in texts_to_annotate]
            print(f"Retrying annotation..."); time.sleep(retry_delay * (attempt + 1))
        except Exception as e:
            print(f"Error calling/processing Generative AI (Attempt {attempt+1}/{max_retries}): {e}")
            import traceback; traceback.print_exc()
            if "rate limit" in str(e).lower(): print("Rate limit likely hit."); time.sleep(retry_delay * (attempt + 1) * 5)
            elif "API key not valid" in str(e) or "permission" in str(e).lower(): print("FATAL: Invalid Google API Key or permission issue."); return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_KEY" for t in texts_to_annotate]
            elif "ConnectionError" in str(e): print("FATAL: GenAI not configured properly."); return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_NO_GENAI" for t in texts_to_annotate]
            else: time.sleep(retry_delay * (attempt + 1))
            if attempt == max_retries - 1: return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_CALL" for t in texts_to_annotate]

    print("Error: Max retries reached for annotation batch.")
    return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_MAX_RETRIES" for t in texts_to_annotate]


# --- Main Processing Function for Bengali ---
def process_bengali_json(input_json_path: str, output_jsonl_path: str, base_audio_path: str, batch_size: int = 10) -> None:
    """
    Processes a Bengali JSON file, determines age group via fallback logic
    (reassigned -> original -> predicted), predicts emotion, constructs absolute audio paths,
    annotates text with Gemini, capitalizes domain tag, and saves final JSONL output.

    Args:
        input_json_path: Path to the input JSON file.
        output_jsonl_path: Path to save the output JSONL file.
        base_audio_path: The absolute base directory where relative wav_paths start. REQUIRED.
        batch_size: Number of records to process before calling Gemini API.
    """
    if not base_audio_path or not os.path.isdir(base_audio_path):
        print(f"FATAL ERROR: base_audio_path '{base_audio_path}' is required and must be a valid directory.")
        return

    output_parent_dir = os.path.dirname(output_jsonl_path)
    os.makedirs(output_parent_dir, exist_ok=True)

    # --- Clear output file ---
    try:
        print(f"Attempting to clear output file: {output_jsonl_path}")
        with open(output_jsonl_path, 'w', encoding='utf-8') as f_clear: f_clear.write("")
        print(f"Output file {output_jsonl_path} cleared successfully.")
    except IOError as e: print(f"Error clearing output file {output_jsonl_path}: {e}. Please check permissions."); return

    # --- Load Models ---
    print("Loading models...")
    # Emotion Model
    emotion_model_name = "superb/hubert-large-superb-er"; emotion_model_info = {}
    try:
        emotion_model_info['model'] = AutoModelForAudioClassification.from_pretrained(emotion_model_name).to(device); emotion_model_info['model'].eval()
        emotion_model_info['feature_extractor'] = AutoFeatureExtractor.from_pretrained(emotion_model_name)
        print("Emotion model loaded.")
    except Exception as e:
        print(f"FATAL Error loading Emotion model: {e}. Cannot proceed without emotion model.")
        return

    # Age Model (for fallback)
    age_model = None; age_processor = None
    age_model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    try:
        age_processor = Wav2Vec2Processor.from_pretrained(age_model_name)
        age_model = AgeGenderModel.from_pretrained(age_model_name).to(device); age_model.eval()
        print("Age prediction model loaded (for fallback).")
    except Exception as e:
        print(f"Warning: Failed to load age prediction model ({e}). Age prediction fallback will not be available.")
        age_model = None; age_processor = None # Ensure they are None if loading failed

    print("-" * 30)

    # --- Load Input JSON Data ---
    input_data = None
    if not os.path.exists(input_json_path):
        print(f"Error: Input JSON file not found: {input_json_path}"); return
    try:
        print(f"Loading Bengali data from {input_json_path}...")
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        if not isinstance(input_data, dict) or not input_data:
            print(f"Error: Input JSON does not contain a valid dictionary or is empty."); return
        num_records = len(input_data)
        print(f"Loaded {num_records} records from {input_json_path}.")
    except json.JSONDecodeError as e: print(f"Error reading or parsing input JSON: {e}"); return
    except Exception as e: print(f"An unexpected error occurred loading input JSON: {e}"); return

    # --- Process Records ---
    records_to_process = list(input_data.items())
    total_records = len(records_to_process)
    print(f"Starting processing of {total_records} Bengali records...")
    print(f"Using Base Audio Directory: {base_audio_path}")
    print(f"Output will be saved to: {output_jsonl_path}"); print("-" * 30)

    processed_records_buffer = []; batch_num = 0
    files_processed_count = 0; total_records_saved = 0; skipped_records_count = 0

    for i, (record_key, record_data) in enumerate(records_to_process):
        print(f"\nProcessing record {i+1}/{total_records}: {record_key}")
        final_age_group = "UNKNOWN" # Default age group
        signal, sr = None, 16000 # Initialize audio variables

        # --- 1. Validate and Extract Required Data ---
        required_keys = ["wav_path", "text", "gender_reassigned", "domain", "duration"] # age_group/reassigned checked separately
        missing_keys = [key for key in required_keys if key not in record_data]
        if missing_keys:
            print(f"  Skipping: Record missing required keys: {', '.join(missing_keys)}"); skipped_records_count += 1; continue

        wav_path_relative = record_data["wav_path"]
        text = record_data["text"]
        gender = record_data["gender_reassigned"] # Using reassigned gender directly
        domain = record_data["domain"]
        duration = record_data["duration"]

        if not text or not text.strip():
             print(f"  Skipping: Record has empty text field."); skipped_records_count += 1; continue
        if not isinstance(duration, (int, float)) or duration <= 0:
             print(f"  Skipping: Invalid duration value ({duration})."); skipped_records_count += 1; continue

        # --- 2. Determine Age Group using Fallback Logic ---
        age_group_reassigned = record_data.get("age_group_reassigned")
        age_group_original = record_data.get("age_group") # Get original age group as well

        # Priority 1: Use age_group_reassigned if valid
        if age_group_reassigned and str(age_group_reassigned).strip().upper() != 'NA':
            final_age_group = str(age_group_reassigned).strip()
            print(f"  Using age_group_reassigned: {final_age_group}")
        # Priority 2: Use age_group if reassigned is NA and original is valid
        elif age_group_original and str(age_group_original).strip().upper() != 'NA':
            final_age_group = str(age_group_original).strip()
            print(f"  Using original age_group (reassigned was NA): {final_age_group}")
        # Priority 3: Predict from audio if both are NA (or missing)
        else:
            print("  Age from JSON is 'NA' or missing. Attempting prediction from audio...")
            # Need to load audio first for prediction
            audio_loaded_for_age = False
            try:
                # Construct Absolute Audio Path
                if wav_path_relative.startswith('/'): wav_path_relative = wav_path_relative[1:]
                audio_path_abs = os.path.abspath(os.path.join(base_audio_path, wav_path_relative))
                if not os.path.isfile(audio_path_abs):
                    print(f"  Skipping: Audio file not found for age prediction: {audio_path_abs}")
                    skipped_records_count += 1; continue # Skip if audio missing for prediction

                # Load Audio Signal
                try: signal_raw, sr_orig = sf.read(audio_path_abs, dtype='float32')
                except Exception: signal_raw, sr_orig = librosa.load(audio_path_abs, sr=None, mono=False)
                target_sr = 16000
                if sr_orig != target_sr:
                    signal_float = signal_raw.astype(np.float32) if not np.issubdtype(signal_raw.dtype, np.floating) else signal_raw
                    signal_mono = np.mean(signal_float, axis=1) if signal_float.ndim > 1 else signal_float
                    signal = librosa.resample(y=signal_mono, orig_sr=sr_orig, target_sr=target_sr); sr = target_sr
                else:
                    signal = signal_raw.astype(np.float32) if not np.issubdtype(signal_raw.dtype, np.floating) else signal_raw
                    if signal.ndim > 1: signal = np.mean(signal, axis=1)
                    sr = sr_orig
                if signal is None or len(signal) == 0:
                    print(f"  Skipping: Failed to load or empty audio for age prediction: {audio_path_abs}")
                    skipped_records_count += 1; continue # Skip if audio loading fails
                audio_loaded_for_age = True

                # Predict Age if Model Loaded
                if age_model and age_processor:
                    raw_age_prediction = predict_age_from_audio(signal, sr, age_processor, age_model)
                    if raw_age_prediction is not None:
                        final_age_group = get_age_bucket_from_prediction(raw_age_prediction)
                        print(f"  Predicted age group from audio: {final_age_group} (Raw: {raw_age_prediction:.4f})")
                    else:
                        print("  Age prediction from audio failed.")
                        final_age_group = "UNKNOWN" # Keep default if prediction fails
                else:
                    print("  Age prediction model not loaded. Cannot predict age. Using UNKNOWN.")
                    final_age_group = "UNKNOWN"

            except Exception as audio_err:
                print(f"  Skipping: Error during audio loading/processing for age prediction: {audio_err}")
                skipped_records_count += 1; continue # Skip record if audio handling fails here

        # --- 3. Construct Absolute Audio Path (if not already done for age prediction) ---
        # This step is now potentially redundant if age prediction happened, but ensures path exists for emotion.
        try:
            if 'audio_path_abs' not in locals() or not audio_path_abs: # Check if already defined
                 if wav_path_relative.startswith('/'): wav_path_relative = wav_path_relative[1:]
                 audio_path_abs = os.path.abspath(os.path.join(base_audio_path, wav_path_relative))
                 if not os.path.isfile(audio_path_abs):
                     print(f"  Skipping: Audio file path check failed: {audio_path_abs}")
                     skipped_records_count += 1; continue
        except Exception as path_err:
            print(f"  Skipping: Error constructing absolute path: {path_err}")
            skipped_records_count += 1; continue


        # --- 4. Load Audio (if not already loaded for age prediction) ---
        try:
            if signal is None: # Check if audio needs loading (i.e., age came from JSON)
                try: signal_raw, sr_orig = sf.read(audio_path_abs, dtype='float32')
                except Exception: signal_raw, sr_orig = librosa.load(audio_path_abs, sr=None, mono=False)
                target_sr = 16000
                if sr_orig != target_sr:
                    signal_float = signal_raw.astype(np.float32) if not np.issubdtype(signal_raw.dtype, np.floating) else signal_raw
                    signal_mono = np.mean(signal_float, axis=1) if signal_float.ndim > 1 else signal_float
                    signal = librosa.resample(y=signal_mono, orig_sr=sr_orig, target_sr=target_sr); sr = target_sr
                else:
                    signal = signal_raw.astype(np.float32) if not np.issubdtype(signal_raw.dtype, np.floating) else signal_raw
                    if signal.ndim > 1: signal = np.mean(signal, axis=1)
                    sr = sr_orig
                if signal is None or len(signal) == 0:
                    print(f"  Skipping: Failed to load or empty audio {audio_path_abs}"); skipped_records_count += 1; continue
        except Exception as load_err:
            print(f"  Skipping: Error loading audio {audio_path_abs}: {load_err}"); skipped_records_count += 1; continue

        # --- 5. Extract Emotion ---
        emotion = "ERROR"
        try:
            emotion = extract_emotion(signal, sr, emotion_model_info)
            emotion = emotion.upper() # Ensure uppercase
            if emotion in ["ERRORLOADINGMODEL", "NO_AUDIO", "OOM_ERROR", "RUNTIME_ERROR", "EXTRACTION_ERROR"]:
                 print(f"  Warning: Emotion extraction failed with status: {emotion}")
        except Exception as emotion_err:
            print(f"  Error during emotion extraction call: {emotion_err}"); emotion = "ERROR"


        # --- 6. Prepare Text for Gemini Annotation ---
        # Use the determined final_age_group
        safe_age = re.sub(r'\s+', '_', str(final_age_group))
        safe_gender = re.sub(r'\s+', '_', str(gender).upper()) # Use gender from JSON
        safe_emotion = re.sub(r'\s+', '_', str(emotion).upper())
        safe_domain = re.sub(r'\s+', '_', str(domain).upper()) # Capitalize domain

        initial_text_for_gemini = f"{text.strip()} AGE_{safe_age} GENDER_{safe_gender} EMOTION_{safe_emotion} DOMAIN_{safe_domain}"
        # print(f"  Text prepared for Gemini: {initial_text_for_gemini}") # Debug print

        # --- 7. Add to Batch Buffer ---
        buffer_item = {
            "audio_filepath": audio_path_abs, # Store the absolute path
            "duration": duration,
            "text_for_gemini": initial_text_for_gemini,
            "original_key": record_key,
            "determined_age_source": "reassigned" if (age_group_reassigned and str(age_group_reassigned).strip().upper() != 'NA') else \
                                     "original" if (age_group_original and str(age_group_original).strip().upper() != 'NA') else \
                                     "predicted" if final_age_group != "UNKNOWN" else "unknown" # Track age source
        }
        processed_records_buffer.append(buffer_item)
        files_processed_count += 1

        # --- 8. Annotate and Save in Batches ---
        if len(processed_records_buffer) >= batch_size or (i + 1) == total_records:
            batch_num += 1; current_batch_size = len(processed_records_buffer)
            print(f"\n--- Annotating Batch {batch_num} ({current_batch_size} records) ---")
            texts_to_annotate = [rec["text_for_gemini"] for rec in processed_records_buffer]
            annotated_texts = annotate_batch_texts(texts_to_annotate) # Uses Bengali-specific func

            if len(annotated_texts) != current_batch_size:
                 print(f"  FATAL BATCH ERROR: Annotation count mismatch! Expected {current_batch_size}, Got {len(annotated_texts)}. Skipping save for this batch.")
                 skipped_records_count += current_batch_size
            else:
                try:
                    lines_written_in_batch = 0
                    with open(output_jsonl_path, 'a', encoding='utf-8') as f_out:
                        for record_data, annotated_text in zip(processed_records_buffer, annotated_texts):
                            final_record = {
                                "audio_filepath": record_data["audio_filepath"],
                                "text": annotated_text,
                                "duration": record_data["duration"]
                                # Optional: Add age source if needed: "age_source": record_data["determined_age_source"]
                            }
                            if "ANNOTATION_ERROR" in annotated_text:
                                print(f"  Warning: Saving record for key '{record_data['original_key']}' with error flag: {annotated_text.split()[-1]}")
                            json_str = json.dumps(final_record, ensure_ascii=False)
                            f_out.write(json_str + '\n'); lines_written_in_batch += 1
                    total_records_saved += lines_written_in_batch
                    print(f"--- Batch {batch_num} processed and saved ({lines_written_in_batch} records). Total saved: {total_records_saved} ---")
                except IOError as io_err: print(f"  Error writing batch {batch_num} to {output_jsonl_path}: {io_err}")
                except Exception as write_err: print(f"  Unexpected error writing batch {batch_num}: {write_err}")

            # --- Clean up Batch Data and Memory ---
            processed_records_buffer = []
            del texts_to_annotate, annotated_texts
            # Explicitly delete signal here as it's potentially large
            if 'signal' in locals(): del signal
            if 'signal_raw' in locals(): del signal_raw
            if 'signal_float' in locals(): del signal_float
            if 'signal_mono' in locals(): del signal_mono
            torch.cuda.empty_cache(); gc.collect()

        # --- Clean up per-record audio data ---
        if 'signal' in locals(): del signal
        if 'signal_raw' in locals(): del signal_raw
        if 'signal_float' in locals(): del signal_float
        if 'signal_mono' in locals(): del signal_mono
        # Remove reference to absolute path variable for the next loop iteration
        if 'audio_path_abs' in locals(): del audio_path_abs
        gc.collect()


    # --- Final Summary ---
    print("\n" + "="*30); print("Processing Finished.")
    print(f"Total records found in input JSON: {total_records}")
    print(f"Records successfully processed (before annotation): {files_processed_count}")
    print(f"Records skipped (missing data, audio error, path error, etc.): {skipped_records_count}")
    print(f"Total records saved to {output_jsonl_path}: {total_records_saved}")
    print("Note: Age group determined via fallback (reassigned -> original -> predicted).")
    print("Note: Annotation errors within saved records are flagged in the 'text' field.")
    print("="*30)


# --- Main Execution ---
if __name__ == "__main__":
    # --- *** Configuration for Bengali *** ---
    INPUT_JSON_FILE = "/external4/datasets/MADASR/IISc_RESPIN_train_small/data/bn_meta_data.json"
    FINAL_OUTPUT_JSONL = "/external4/datasets/MADASR/process_train_small/bengali_output.jsonl"
    BASE_AUDIO_DIRECTORY = "/external4/datasets/MADASR/IISc_RESPIN_train_small"
    PROCESSING_BATCH_SIZE = 10

    # --- Verifications ---
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"): print("ERROR: GOOGLE_API_KEY not set."); exit(1)
    if genai is None: print("ERROR: Google Generative AI failed to configure."); exit(1)
    if not BASE_AUDIO_DIRECTORY or not os.path.isdir(BASE_AUDIO_DIRECTORY):
         print(f"ERROR: BASE_AUDIO_DIRECTORY ('{BASE_AUDIO_DIRECTORY}') is not set or is not a valid directory."); exit(1)
    if not os.path.isfile(INPUT_JSON_FILE): print(f"ERROR: Input JSON file not found: {INPUT_JSON_FILE}"); exit(1)
    output_dir = os.path.dirname(FINAL_OUTPUT_JSONL)
    try: os.makedirs(output_dir, exist_ok=True); print(f"Output directory created/exists: {output_dir}")
    except OSError as e: print(f"Error creating output directory {output_dir}: {e}"); exit(1)

    print("Starting Bengali JSON Processing with Age Fallback Logic...")
    print(f"Input JSON File: {INPUT_JSON_FILE}")
    print(f"Final Output File: {FINAL_OUTPUT_JSONL}")
    print(f"Base Audio Directory: {BASE_AUDIO_DIRECTORY}")
    print(f"Processing Batch Size: {PROCESSING_BATCH_SIZE}")
    print("-" * 30)

    process_bengali_json(
        input_json_path=INPUT_JSON_FILE,
        output_jsonl_path=FINAL_OUTPUT_JSONL,
        base_audio_path=BASE_AUDIO_DIRECTORY,
        batch_size=PROCESSING_BATCH_SIZE
    )

    print("Bengali Workflow with Age Fallback complete.")