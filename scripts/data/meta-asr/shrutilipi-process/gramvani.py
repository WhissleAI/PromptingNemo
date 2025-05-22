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
# Added for potential retries/dela
# --- Initial Setup (Unchanged) ---
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

# Configure Google Generative AI (Unchanged)
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable not found.")
    print("Google Generative AI configured successfully.")
except Exception as e:
    print(f"Error configuring or testing Google Generative AI: {e}. Annotation step will likely fail.")
    genai = None

# --- Age/Gender Model Definition (Unchanged) ---
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

# --- Emotion Extraction Function (Unchanged) ---
def extract_emotion(audio_data: np.ndarray, sampling_rate: int = 16000, model_info: dict = None) -> str:
    if model_info is None or 'model' not in model_info or 'feature_extractor' not in model_info:
        print("Error: Emotion model not loaded correctly.")
        return "ErrorLoadingModel"
    if audio_data is None or len(audio_data) == 0:
        return "No_Audio"
    # Keep short audio check if desired, or remove if Graamvaani has very short valid clips
    # if len(audio_data) < sampling_rate * 0.1:
    #     return "Audio_Too_Short"

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
        return emotion_label.upper() # Ensure uppercase consistency
    except Exception as e:
        print(f"Error during emotion extraction: {e}")
        return "Extraction_Error"

# --- Data Structures (Simplified/Modified) ---
# We might not need the full dataclass structure if we construct the output directly

# Helper to get age bucket from model's continuous output
def get_age_bucket_from_model(age_model_output: float) -> str:
    """Converts continuous age prediction (scaled 0-1) to age bucket."""
    # The model output is typically a value between 0 and 1, representing age range.
    # Assuming the model was trained similarly to the audeering one, this value needs scaling.
    # The original code scaled by 100. Let's stick to that.
    # If the model output is different, this scaling needs adjustment.
    actual_age = round(age_model_output * 100)
    age_brackets = [
        (18, "0_18"), (30, "18_30"), (45, "30_45"),
        (60, "45_60"), (float('inf'), "60PLUS")
    ]
    for threshold, bracket in age_brackets:
        if actual_age < threshold: return bracket
    return "60PLUS" # Default for > 60 or unexpected values

# Helper to check if manifest value is valid (not None, NA, empty)
def is_valid_manifest_value(value: Optional[str]) -> bool:
    """Checks if a value from the manifest is usable."""
    if value is None:
        return False
    val_strip = value.strip()
    if not val_strip or val_strip.upper() == "NA":
        return False
    return True

# --- File Handling (REPLACED) ---
# Removed get_file_pairs and get_transcription

def read_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """Reads a JSONL manifest file and returns a list of dictionaries."""
    data = []
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found at {manifest_path}")
        return []
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {i+1} in {manifest_path}: {e}")
    except Exception as e:
        print(f"Error reading manifest file {manifest_path}: {e}")
        return []
    print(f"Read {len(data)} entries from {manifest_path}")
    return data


# --- AI Annotation Functions (Unchanged) ---
# Includes: correct_entity_tag_spaces, fix_end_tags_and_spacing, annotate_batch_texts
# (Keeping them exactly as they were in the previous version)
def correct_entity_tag_spaces(text: str) -> str:
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
    if not isinstance(text, str):
        return text
    text = correct_entity_tag_spaces(text)
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+END\s+(\b(?:AGE_|GENDER_|EMOTION_|INTENT_)|$)', r' \1', text)
    text = re.sub(r'\s+END\s+END\b', ' END', text)
    text = re.sub(r'\s+END\s+END\b', ' END', text) # Run twice for triple cases
    pattern_add_end = r'(ENTITY_[A-Z0-9_]+\s+\S.*?)(?<!\sEND)(?=\s+(\bAGE_|\bGENDER_|\bEMOTION_|\bINTENT_|\bENTITY_)|$)'
    text = re.sub(pattern_add_end, r'\1 END', text)
    text = re.sub(r'(ENTITY_[A-Z0-9_]+)(\S)', r'\1 \2', text) # Space after tag if missing
    text = re.sub(r'(\S)(END\b)', r'\1 \2', text)        # Space before END if missing
    text = re.sub(r'\s+([।?!:;,.])', r'\1', text)
    text = re.sub(r'([।?!:;,.])(\w)', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def annotate_batch_texts(texts_to_annotate: List[str]):
    """Sends a batch of texts to Gemini for annotation."""
    if not genai:
        print("Error: Google Generative AI not configured. Skipping annotation.")
        return texts_to_annotate # Return original texts
    if not texts_to_annotate:
        return []

    # Prompt remains the same as it expects "TEXT METADATA" input format
    prompt = f'''You are an expert linguistic annotator for Hindi text.
You will receive a list of Hindi sentences. Each sentence already includes metadata tags (AGE_*, GENDER_*, EMOTION_*) at the end.

Your task is crucial and requires precision:
1.  **PRESERVE EXISTING TAGS:** Keep the `AGE_`, `GENDER_`, and `EMOTION_` tags exactly as they appear at the end of each sentence. DO NOT modify or move them.
2.  **ENTITY ANNOTATION (Hindi Text Only):** Identify entities ONLY within the Hindi transcription part of the sentence. Use ONLY the entity types from the provided list.
3.  **ENTITY TAG FORMAT (VERY IMPORTANT):**
    *   Insert the tag **BEFORE** the entity: `ENTITY_<TYPE>` (e.g., `ENTITY_CITY`, `ENTITY_PERSON_NAME`).
    *   **NO SPACES** are allowed within the `<TYPE>` part (e.g., use `PERSON_NAME`, NOT `PERSON_ NAM E`).
    *   Immediately **AFTER** the Hindi entity text, add a single space followed by `END`.
    *   Example: `... ENTITY_CITY दिल्ली END ...`
    *   **DO NOT** add an `END` tag just before the `AGE_`, `GENDER_`, `EMOTION_` tags unless it belongs to a preceding entity.
4.  **INTENT TAG:** Determine the single primary intent of the Hindi transcription (e.g., INFORM, QUESTION, REQUEST, COMMAND, GREETING, etc.). Add ONE `INTENT_<INTENT_TYPE>` tag at the absolute end of the entire string, AFTER all other tags.
5.  **OUTPUT FORMAT:** Return a JSON array of strings, where each string is a fully annotated sentence adhering to all rules.
6.  **HINDI SPECIFICS:** Handle Hindi script, punctuation (like ।), and spacing correctly according to standard Hindi rules. Ensure proper spacing around the inserted tags.

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

**Example Input String:**
"मैंने कल मारिया को दिल्ली में देखा। AGE_30_45 GENDER_FEMALE EMOTION_NEUTRAL"

**CORRECT Example Output String:**
"मैंने कल ENTITY_PERSON_NAME मारिया END को ENTITY_CITY दिल्ली END में देखा। AGE_30_45 GENDER_FEMALE EMOTION_NEUTRAL INTENT_INFORM"

**INCORRECT Example Output String (Spaces in Tag):**
"मैंने कल ENTITY_PERSON_ NAM E मारिया END को ENTITY_CIT Y दिल्ली END में देखा। AGE_30_45 GENDER_FEMALE EMOTION_NEUTRAL INTENT_INFORM"

**INCORRECT Example Output String (Extra END before metadata):**
"मैंने कल ENTITY_PERSON_NAME मारिया END को ENTITY_CITY दिल्ली END में देखा। END AGE_30_45 GENDER_FEMALE EMOTION_NEUTRAL INTENT_INFORM"


**Sentences to Annotate Now:**
{json.dumps(texts_to_annotate, ensure_ascii=False, indent=2)}
'''

    max_retries = 3
    retry_delay = 5 # seconds
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash-latest") # Or "gemini-1.5-pro-latest"
            # Add safety settings if needed
            # safety_settings = [...]
            # response = model.generate_content(prompt, safety_settings=safety_settings)
            response = model.generate_content(prompt)

            # Debug: Print raw response
            # print(f"--- Raw Gemini Response (Attempt {attempt+1}) ---")
            # print(response.text)
            # print("-------------------------")

            assistant_reply = response.text.strip()
            if assistant_reply.startswith("```json"):
                assistant_reply = assistant_reply[len("```json"):].strip()
            elif assistant_reply.startswith("```"):
                 assistant_reply = assistant_reply[len("```"):].strip()
            if assistant_reply.endswith("```"):
                assistant_reply = assistant_reply[:-len("```")].strip()

            if not (assistant_reply.startswith('[') and assistant_reply.endswith(']')):
                 raise json.JSONDecodeError("Response does not look like a JSON list.", assistant_reply, 0)

            annotated_sentences_raw = json.loads(assistant_reply)

            if isinstance(annotated_sentences_raw, list) and len(annotated_sentences_raw) == len(texts_to_annotate):
                processed_sentences = []
                for sentence in annotated_sentences_raw:
                     if isinstance(sentence, str):
                          corrected_sentence = correct_entity_tag_spaces(sentence)
                          final_sentence = fix_end_tags_and_spacing(corrected_sentence)
                          processed_sentences.append(final_sentence)
                     else:
                          print(f"Warning: Non-string item received in annotation list: {sentence}")
                          processed_sentences.append(sentence) # Keep as is or handle differently
                return processed_sentences
            else:
                print(f"Error: API did not return a valid list or mismatched length. Expected {len(texts_to_annotate)}, Got {len(annotated_sentences_raw) if isinstance(annotated_sentences_raw, list) else 'Invalid Type'}")
                if attempt == max_retries - 1: return texts_to_annotate
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
            else:
                 time.sleep(retry_delay * (attempt + 1))
            if attempt == max_retries - 1: return texts_to_annotate # Fallback on final attempt

    print("Error: Max retries reached for annotation.")
    return texts_to_annotate


# --- Main Processing Function (MODIFIED FOR MANIFEST INPUT) ---
def process_manifest_and_annotate(dataset_dir: str, output_jsonl_path: str, batch_size: int = 10) -> None:
    """
    Processes audio files listed in a manifest.jsonl, extracts/selects metadata,
    gets transcriptions from manifest, formats text, annotates with AI,
    and saves final JSONL output.
    """
    manifest_path = os.path.join(dataset_dir, "manifest.jsonl")
    output_dir = os.path.dirname(output_jsonl_path)
    os.makedirs(output_dir, exist_ok=True)

    # --- Clear output file at the beginning ---
    try:
        print(f"Attempting to clear output file: {output_jsonl_path}")
        with open(output_jsonl_path, 'w') as f_clear:
            f_clear.write("")
        print(f"Output file {output_jsonl_path} cleared successfully.")
    except IOError as e:
        print(f"Error clearing output file {output_jsonl_path}: {e}. Please check permissions.")
        return

    # --- Load Models ---
    print("Loading models...")
    # Age/Gender Model
    age_gender_model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    processor = None
    age_gender_model = None
    try:
        processor = Wav2Vec2Processor.from_pretrained(age_gender_model_name)
        age_gender_model = AgeGenderModel.from_pretrained(age_gender_model_name).to(device)
        age_gender_model.eval()
        print("Age/Gender model loaded.")
    except Exception as e:
        print(f"FATAL Error loading Age/Gender model: {e}. Cannot proceed without it for age/gender prediction. Exiting.")
        return

    # Emotion Model
    emotion_model_name = "superb/hubert-large-superb-er"
    emotion_model_info = {}
    try:
        emotion_model_info['model'] = AutoModelForAudioClassification.from_pretrained(emotion_model_name).to(device)
        emotion_model_info['model'].eval()
        emotion_model_info['feature_extractor'] = AutoFeatureExtractor.from_pretrained(emotion_model_name)
        print("Emotion model loaded.")
    except Exception as e:
        print(f"Error loading Emotion model: {e}. Emotion extraction will use 'ErrorLoadingModel'.")
        # Continue, but emotion will be marked as error

    print("-" * 30)

    # --- Read Manifest ---
    manifest_data = read_manifest(manifest_path)
    if not manifest_data:
        print("No data read from manifest. Exiting.")
        return

    total_files = len(manifest_data)
    print(f"Found {total_files} entries in manifest to process.")
    print(f"Output will be saved to: {output_jsonl_path}")
    print("-" * 30)

    processed_records_buffer = [] # Holds records before AI annotation batch
    batch_num = 0
    files_processed_count = 0
    total_records_saved = 0

    # --- Process Manifest Entries ---
    for i, entry in enumerate(manifest_data):
        print(f"\nProcessing entry {i+1}/{total_files}")
        try:
            # 1. Extract Data from Manifest Entry
            audio_path = entry.get("path")
            transcription = entry.get("text", "")
            duration_manifest = entry.get("duration") # Keep for reference/check
            gender_manifest = entry.get("gender")     # Use if valid
            # age_group_manifest = entry.get("age_group") # We ignore this per requirement (always use model for age)
            # sentiment_manifest = entry.get("sentiment") # Ignore
            # dialect_manifest = entry.get("dialect")     # Ignore

            # Basic Validation
            if not audio_path:
                print("  Skipping: Manifest entry missing 'path'.")
                continue
            if not transcription:
                print("  Skipping: Manifest entry missing 'text'.")
                continue
            # Ensure the audio path from manifest is absolute or resolve it relative to dataset_dir
            if not os.path.isabs(audio_path):
                 audio_path = os.path.abspath(os.path.join(dataset_dir, audio_path)) # Make absolute

            if not os.path.exists(audio_path):
                print(f"  Skipping: Audio file not found at resolved path: {audio_path}")
                continue

            print(f"  Audio: {os.path.basename(audio_path)}")
            print(f"  Text: {transcription[:50]}...") # Print snippet

            # 2. Load Audio
            try:
                signal, sr = librosa.load(audio_path, sr=16000)
                if signal is None or len(signal) == 0:
                    print(f"  Skipping: Failed to load or empty audio in {audio_path}")
                    continue
                duration_librosa = round(len(signal) / sr, 3) # Use librosa duration, higher precision
                # Optional: Check duration difference
                if duration_manifest and abs(duration_librosa - float(duration_manifest)) > 0.5:
                    print(f"  Warning: Duration mismatch - Manifest: {duration_manifest}, Librosa: {duration_librosa:.3f}")
                if duration_librosa < 0.1:
                     print(f"  Skipping: Audio too short ({duration_librosa:.3f}s) according to Librosa.")
                     continue
            except Exception as load_err:
                 print(f"  Skipping: Error loading audio {audio_path}: {load_err}")
                 continue

            # 3. Extract Age/Gender (Always run model)
            gender_model = "MODEL_ERROR"
            age_model_output = -1.0
            try:
                inputs = processor(signal, sampling_rate=sr, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device)
                with torch.no_grad():
                    _, logits_age, logits_gender = age_gender_model(input_values)
                age_model_output = logits_age.cpu().numpy().item()
                gender_idx = torch.argmax(logits_gender, dim=-1).cpu().numpy().item()
                gender_map = {0: "FEMALE", 1: "MALE", 2: "OTHER"} # Keep uppercase
                gender_model = gender_map.get(gender_idx, "UNKNOWN")
            except Exception as age_gender_err:
                print(f"  Error during Age/Gender extraction: {age_gender_err}")
                # Keep default error values

            # 4. Determine Final Gender (Manifest if valid, else Model)
            if is_valid_manifest_value(gender_manifest):
                final_gender = gender_manifest.strip().upper() # Use manifest value, ensure uppercase
                print(f"  Using Gender from Manifest: {final_gender}")
            else:
                final_gender = gender_model # Use model prediction
                print(f"  Using Gender from Model: {final_gender}")

            # 5. Determine Final Age Bucket (Always use Model)
            final_age_bucket = get_age_bucket_from_model(age_model_output)
            print(f"  Using Age Bucket from Model: {final_age_bucket} (Raw output: {age_model_output:.4f})")

            # 6. Extract Emotion (Always use Model)
            emotion = extract_emotion(signal, sr, emotion_model_info)
            print(f"  Using Emotion from Model: {emotion}")

            # 7. Format Text for AI Annotation
            # Format: "text<space>AGE_X<space>GENDER_Y<space>EMOTION_Z"
            metadata_str = f"AGE_{final_age_bucket} GENDER_{final_gender} EMOTION_{emotion}"
            initial_formatted_text = f"{transcription.strip()} {metadata_str.strip()}"
            initial_formatted_text = re.sub(r'\s+', ' ', initial_formatted_text).strip() # Normalize spaces

            # 8. Store Record for Batching
            record = {
                "audio_filepath": audio_path,
                "duration": duration_librosa,
                "initial_text": initial_formatted_text, # Temp field for AI input
                 # Optionally keep raw values if needed for analysis
                 "raw_age_output": age_model_output,
                 "raw_gender_prediction": gender_model,
                 "raw_emotion_prediction": emotion,
                 "manifest_gender": gender_manifest, # Store for reference
            }
            processed_records_buffer.append(record)
            files_processed_count += 1

            # 9. Annotate and Save in Batches
            if len(processed_records_buffer) >= batch_size or (i + 1) == total_files:
                batch_num += 1
                current_batch_size = len(processed_records_buffer)
                print(f"\n--- Annotating Batch {batch_num} ({current_batch_size} records) ---")

                texts_to_annotate = [rec["initial_text"] for rec in processed_records_buffer]

                # Call Gemini for annotation
                annotated_texts = annotate_batch_texts(texts_to_annotate)

                if len(annotated_texts) != current_batch_size:
                    print(f"  FATAL BATCH ERROR: Annotation count mismatch! Expected {current_batch_size}, Got {len(annotated_texts)}. Skipping save for this batch.")
                    # Optional: Log failed batch details
                else:
                     # Save the annotated batch
                    try:
                        lines_written_in_batch = 0
                        with open(output_jsonl_path, 'a', encoding='utf-8') as f_out:
                            for record_data, annotated_text in zip(processed_records_buffer, annotated_texts):
                                # Prepare final record
                                final_record = {
                                    "audio_filepath": record_data["audio_filepath"],
                                    "duration": record_data["duration"],
                                    "text": annotated_text, # AI annotated + processed text
                                    # Add back optional fields if desired
                                    # "manifest_gender": record_data["manifest_gender"],
                                    # "model_gender": record_data["raw_gender_prediction"] if record_data["manifest_gender"] != record_data["raw_gender_prediction"] else None # Example
                                }
                                # Remove None values if added optional fields
                                # final_record = {k: v for k, v in final_record.items() if v is not None}

                                json_str = json.dumps(final_record, ensure_ascii=False)
                                f_out.write(json_str + '\n')
                                lines_written_in_batch += 1

                        total_records_saved += lines_written_in_batch
                        print(f"--- Batch {batch_num} annotated and saved ({lines_written_in_batch} records). Total saved: {total_records_saved} ---")

                    except IOError as io_err:
                         print(f"  Error writing batch {batch_num} to {output_jsonl_path}: {io_err}")
                    except Exception as write_err:
                         print(f"  Unexpected error writing batch {batch_num}: {write_err}")

                # Clear buffer and memory
                processed_records_buffer = []
                del texts_to_annotate # Explicitly delete large list
                del annotated_texts   # Explicitly delete large list
                torch.cuda.empty_cache()
                gc.collect()
                # time.sleep(1) # Optional delay

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")
            break
        except Exception as e:
            print(f"  FATAL ERROR processing entry {i+1} (Audio: {audio_path if 'audio_path' in locals() else 'N/A'}): {e}")
            import traceback
            traceback.print_exc()
            # Clear buffer before continuing to prevent reprocessing failed entry in next batch
            processed_records_buffer = []
            torch.cuda.empty_cache()
            gc.collect()
            continue # Skip to the next entry

    print("\n" + "="*30)
    print(f"Processing Finished.")
    print(f"Total manifest entries processed attempt: {files_processed_count}/{total_files}")
    print(f"Total records saved to {output_jsonl_path}: {total_records_saved}")
    print("="*30)


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    # ===> SET THIS TO THE DIRECTORY CONTAINING manifest.jsonl and wavs/ <===
    GV_DATASET_DIR = "/external4/datasets/Graamvaani_hindi/GV_Train_100h"

    # ===> SET THE DESIRED OUTPUT PATH <===
    FINAL_OUTPUT_JSONL = "/external4/datasets/Graamvaani_hindi/GV_Train_100h_processed/updated_fix_gramvaani.jsonl"

    # ===> ADJUST BATCH SIZE (Start small, e.g., 5 or 10) <===
    PROCESSING_BATCH_SIZE = 10 # Gemini might have input token limits per request

    print("Starting Graamvaani Audio Processing and Annotation Workflow...")
    print(f"Input Dataset Directory: {GV_DATASET_DIR}")
    print(f"Final Output File: {FINAL_OUTPUT_JSONL}")
    print(f"Processing Batch Size: {PROCESSING_BATCH_SIZE}")
    print("-" * 30)

    if not os.path.exists(GV_DATASET_DIR):
         print(f"ERROR: Input directory not found: {GV_DATASET_DIR}")
         exit(1)
    if not os.path.exists(os.path.join(GV_DATASET_DIR, "manifest.jsonl")):
         print(f"ERROR: manifest.jsonl not found in {GV_DATASET_DIR}")
         exit(1)

    if not genai:
         print("ERROR: Google Generative AI failed to configure. Exiting.")
         exit(1)

    process_manifest_and_annotate(
        dataset_dir=GV_DATASET_DIR,
        output_jsonl_path=FINAL_OUTPUT_JSONL,
        batch_size=PROCESSING_BATCH_SIZE
    )

    print("Workflow complete.")