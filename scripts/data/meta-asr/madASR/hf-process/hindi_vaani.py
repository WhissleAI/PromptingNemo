# -*- coding: utf-8 -*- # Ensure editor recognizes UTF-8 for Devanagari script (Hindi)
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
# Using the same pre-trained multilingual/cross-lingual models for age/gender/emotion
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
        self.gender = ModelHead(config, 3) # Assuming 3 labels: Female, Male, Other/Unknown
        self.init_weights()
    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states_pooled = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states_pooled)
        logits_gender = torch.softmax(self.gender(hidden_states_pooled), dim=-1)
        return hidden_states_pooled, logits_age, logits_gender

# --- Emotion Extraction Function (Unchanged) ---
# Using the same pre-trained model
def extract_emotion(audio_data: np.ndarray, sampling_rate: int = 16000, model_info: dict = None) -> str:
    if model_info is None or 'model' not in model_info or 'feature_extractor' not in model_info:
        return "ErrorLoadingModel"
    if audio_data is None or len(audio_data) == 0:
        return "No_Audio"
    try:
        inputs = model_info['feature_extractor'](audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model_info['model'](**inputs)
        predicted_class_idx = outputs.logits.argmax(-1).item()
        emotion_label = model_info['model'].config.id2label.get(predicted_class_idx, "Unknown")
        return emotion_label.upper()
    except Exception as e:
        print(f"Error during emotion extraction: {e}")
        return "Extraction_Error"

# --- Data Structures (Unchanged Logic) ---
@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    speaker: str
    age: float # From ML model
    gender: str # From JSON metadata
    transcription: str # From JSON metadata (Hindi text in Devanagari)
    emotion: str # From ML model
    chunk_filename: str
    duration: float

@dataclass
class ChunkData:
    segments: List[AudioSegment] = field(default_factory=list)
    filepath: str = ""

    def get_formatted_text(self) -> str:
        """Formats the text including Hindi transcription and metadata tags."""
        if not self.segments:
            return ""
        # Assuming one segment per chunk for this structure
        segment = self.segments[0]
        age_bucket = self.get_age_bucket(segment.age)
        gender_text = segment.gender.upper() if segment.gender else "UNKNOWN"
        emotion_text = segment.emotion.upper() if segment.emotion else "UNKNOWN"
        transcription = segment.transcription.strip() if segment.transcription else "" # Hindi text

        # Format: "Hindi Transcription AGE_BUCKET GENDER_X EMOTION_Y"
        metadata = f"AGE_{age_bucket} GENDER_{gender_text} EMOTION_{emotion_text}"
        return f"{transcription} {metadata}".strip() # Ensure no leading/trailing spaces if transcription is empty

    @staticmethod
    def get_age_bucket(age: float) -> str:
        """Categorizes predicted age into predefined buckets."""
        if age < 0: return "UNKNOWN" # Handle error case
        actual_age = round(age * 100) # Model output is 0-1 range
        age_brackets = [
            (18, "0_18"),
            (30, "18_30"),
            (45, "30_45"),
            (60, "45_60"),
            (float('inf'), "60PLUS")
        ]
        for threshold, bracket in age_brackets:
            if actual_age < threshold:
                return bracket
        return "60PLUS" # Should not be reached if age >= 0

# --- File Handling (Unchanged logic) ---
def get_file_pairs(audio_dir: str, metadata_dir: str) -> List[Tuple[str, str]]:
    """Finds matching audio (.flac, .wav, .mp3) and metadata (.json) files."""
    try:
        audio_files_list = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.flac', '.wav', '.mp3'))]
        metadata_files_list = [f for f in os.listdir(metadata_dir) if f.lower().endswith('.json')]

        audio_files = {os.path.splitext(f)[0]: os.path.join(audio_dir, f) for f in audio_files_list}
        metadata_files = {os.path.splitext(f)[0]: os.path.join(metadata_dir, f) for f in metadata_files_list}

        pairs = []
        missing_metadata_count = 0
        for base_name, audio_path in audio_files.items():
            if base_name in metadata_files:
                pairs.append((audio_path, metadata_files[base_name]))
            else:
                missing_metadata_count += 1

        if missing_metadata_count > 0:
            print(f"Warning: Found {missing_metadata_count} audio files without matching JSON metadata files.")
        print(f"Found {len(pairs)} matching audio-metadata pairs.")
        if not pairs:
             print(f"Searched in:\n Audio: {audio_dir}\n Metadata: {metadata_dir}")
             print(f"Audio files found: {len(audio_files_list)}")
             print(f"Metadata files found: {len(metadata_files_list)}")
        return pairs
    except FileNotFoundError as e:
        print(f"Error finding files: {e}. Check if audio/metadata directories exist at specified paths.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred in get_file_pairs: {e}")
        return []

# --- Get Info from Metadata JSON (Handles Hindi via UTF-8) ---
def get_metadata_info(metadata_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Extracts Hindi transcript and gender from a JSON metadata file using UTF-8."""
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f: # Ensure UTF-8 for Hindi
            data = json.load(f)

        transcript = data.get("transcript")
        gender = data.get("gender")

        if not transcript:
            print(f"Warning: 'transcript' key missing or empty in {metadata_path}")
            transcript = None # Return None if missing
        if not gender:
            print(f"Warning: 'gender' key missing or empty in {metadata_path}")
            gender = None # Return None if missing
        else:
             # Normalize gender
             gender = gender.upper()
             if gender not in ["FEMALE", "MALE", "OTHER"]:
                  print(f"Warning: Unexpected gender value '{gender}' found in {metadata_path}. Mapping to UNKNOWN.")
                  gender = "UNKNOWN"

        return transcript, gender
    except FileNotFoundError:
        print(f"Error: Metadata file not found: {metadata_path}")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {metadata_path}. Check file format.")
        return None, None
    except Exception as e:
        print(f"Error reading metadata file {metadata_path}: {str(e)}")
        return None, None

# --- AI Annotation Functions (MODIFIED PROMPT for Hindi) ---

# [Functions: correct_entity_tag_spaces, fix_end_tags_and_spacing remain the same]
# These utility functions handle tag formatting and spacing around punctuation,
# including the Devanagari danda '।', which is used in Hindi.
def correct_entity_tag_spaces(text: str) -> str:
    """Removes spaces within entity type names like ENTITY_PERSON NAME -> ENTITY_PERSON_NAME."""
    if not isinstance(text, str): return text
    # Correct spaces within the entity type name itself
    def replace_spaces(match):
        tag_part = match.group(1) # The full ENTITY_... part
        type_part = tag_part[len("ENTITY_"):] # Get the type part after ENTITY_
        corrected_type = type_part.replace(' ', '_') # Replace spaces with underscores
        return f"ENTITY_{corrected_type}"

    # Regex to find ENTITY_ followed by potential multi-word types with spaces,
    # ensuring it's followed by a space and then non-space (start of entity text)
    pattern = r'\b(ENTITY_[A-Z0-9_ ]*[A-Z0-9_])(?=\s+\S)'
    corrected_text = re.sub(pattern, replace_spaces, text)
    return corrected_text

def fix_end_tags_and_spacing(text: str) -> str:
    """Cleans up spacing around ENTITY/END tags and punctuation (including ।)."""
    if not isinstance(text, str): return text
    # Collapse multiple spaces to one
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove potential spaces WITHIN entity types that correct_entity_tag_spaces might miss (e.g., if no text follows)
    def remove_internal_spaces(match):
        tag_prefix = match.group(1) # e.g., "ENTITY_"
        tag_body = match.group(2) # e.g., "PERSON NAME"
        corrected_body = re.sub(r'\s+', '_', tag_body) # Replace internal spaces with underscores
        return f"{tag_prefix}{corrected_body}"
    text = re.sub(r'\b(ENTITY_)([A-Z0-9_]+(?: +[A-Z0-9_]+)+)\b', remove_internal_spaces, text)

    # Ensure space *after* ENTITY tag and *before* entity text
    text = re.sub(r'(\bENTITY_[A-Z0-9_]+)(\S)', r'\1 \2', text)
    # Ensure space *before* END tag and *after* entity text
    text = re.sub(r'(\S)(END\b)', r'\1 \2', text)
    # Ensure space *after* END tag
    text = re.sub(r'(\bEND)(\S)', r'\1 \2', text)
    # Remove redundant END tags (e.g., "END END")
    text = re.sub(r'\bEND\s+END\b', 'END', text); text = re.sub(r'\bEND\s+END\b', 'END', text) # Run twice for overlaps
    # Remove END tags immediately before metadata tags or end of string (unless validly ending an entity)
    # This pattern looks for END followed by optional space and then a metadata tag start or end of string ($)
    text = re.sub(r'\bEND\s*(\b(?:AGE_|GENDER_|EMOTION_|INTENT_)|$)', r'\1', text)
    text = re.sub(r'\bEND\s*(\b(?:AGE_|GENDER_|EMOTION_|INTENT_)|$)', r'\1', text) # Run twice

    # Fix spacing around punctuation: remove space *before*, ensure space *after* (if not end of string)
    # Includes common English punctuation and Devanagari danda '।'
    text = re.sub(r'\s+([?!:;,.।])', r'\1', text) # Remove space before punctuation
    text = re.sub(r'([?!:;,.।])(\w)', r'\1 \2', text) # Add space after punctuation if followed by a word character

    # Final cleanup of multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- *** MODIFIED AND MORE DETAILED PROMPT FOR HINDI *** ---
def annotate_batch_texts(texts_to_annotate: List[str]):
    """Sends a batch of texts to Gemini for annotation (Prompt refined for Hindi)."""
    if not genai:
        print("Error: Google Generative AI not configured. Skipping annotation.")
        # Return original texts with an error marker, applying basic cleanup
        return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_NO_GENAI" for t in texts_to_annotate]
    if not texts_to_annotate:
        return []

    # Input texts have format: "hindi text AGE_X GENDER_Y EMOTION_Z"

    # *** REFINED HINDI PROMPT STARTS HERE ***
    prompt = f'''You are an expert linguistic annotator specifically for **Hindi** text written in the **Devanagari script**. Your task is to process a list of Hindi sentences, each already containing `AGE_*`, `GENDER_*`, and `EMOTION_*` metadata tags at the end. Follow these instructions with extreme precision:

1.  **Preserve Existing Metadata Tags:**
    *   The `AGE_*`, `GENDER_*`, and `EMOTION_*` tags at the end **must** remain exactly as they are. Do not modify, move, or delete them.

2.  **Entity Annotation (Hindi Text Only):**
    *   Identify entities **only** within the main Hindi transcription part (before the metadata).
    *   Use **only** the entity types provided in the `Allowed ENTITY TYPES` list below. Use underscores for spaces in types (e.g., `PERSON_NAME`).

3.  **Strict Entity Tagging Format:**
    *   **Tag Insertion:** `ENTITY_<TYPE>` tag immediately before the entity text.
    *   **NO SPACES IN TYPE:** Ensure `<TYPE>` uses underscores instead of spaces (e.g., `PERSON_NAME`, `PROJECT_NAME`).
    *   **`END` Tag Placement:** Literal string `END` immediately after the *complete* entity text.
    *   **Spacing:** Exactly one space between `ENTITY_<TYPE>` and entity start. Exactly one space between entity end and `END`. Exactly one space after `END` before the next word (unless punctuation like `।` follows).
    *   **Example:** For "पुणे", correct annotation is `ENTITY_CITY पुणे END`. For "राम कुमार", correct is `ENTITY_PERSON_NAME राम कुमार END`.
    *   **Crucial `END` Rule:** Only **one** `END` tag right after the full entity. No extra `END` tags.
    *   **Avoid Extra `END` Before Metadata:** Do not place `END` just before `AGE_`, `GENDER_`, `EMOTION_`, or `INTENT_` tags unless it correctly marks the end of an immediately preceding entity.

4.  **Intent Tag:**
    *   Determine the single primary intent (e.g., `INFORM`, `QUESTION`, `REQUEST`, `COMMAND`, `GREETING`).
    *   Append **one** `INTENT_<INTENT_TYPE>` tag at the **absolute end**, after all other tags.

5.  **Output Format:**
    *   Return a **JSON array** of strings, each a fully annotated sentence. Ensure valid JSON format.

6.  **Hindi Language Specifics:**
    *   Handle Devanagari script correctly.
    *   Ensure correct spacing around punctuation (like `.`, `?`, `!`, `।`). Remove space before, ensure space after if followed by a word. Use `।` (danda) as the standard full stop in Hindi.
    *   Clean output with single spaces between words and tags as per rules.

**Allowed ENTITY TYPES (Use Only These, with underscores):**
    [ "PERSON_NAME", "ORGANIZATION", "LOCATION", "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP_CODE", "CURRENCY", "PRICE", "DATE",
    "TIME", "DURATION", "APPOINTMENT_DATE", "APPOINTMENT_TIME", "DEADLINE", "DELIVERY_DATE", "DELIVERY_TIME", "EVENT", "MEETING",
    "TASK", "PROJECT_NAME", "ACTION_ITEM", "PRIORITY", "FEEDBACK", "REVIEW", "RATING", "COMPLAINT", "QUESTION", "RESPONSE", "NOTIFICATION_TYPE", "AGENDA",
    "REMINDER", "NOTE", "RECORD", "ANNOUNCEMENT", "UPDATE", "SCHEDULE", "BOOKING_REFERENCE", "APPOINTMENT_NUMBER", "ORDER_NUMBER", "INVOICE_NUMBER", "PAYMENT_METHOD", "PAYMENT_AMOUNT", "BANK_NAME", "ACCOUNT_NUMBER", "CREDIT_CARD_NUMBER", "TAX_ID", "SOCIAL_SECURITY_NUMBER", "DRIVER'S_LICENSE", "PASSPORT_NUMBER", "INSURANCE_PROVIDER", "POLICY_NUMBER", "INSURANCE_PLAN", "CLAIM_NUMBER", "POLICY_HOLDER", "BENEFICIARY",
    "RELATIONSHIP", "EMERGENCY_CONTACT", "PROJECT_PHASE", "VERSION", "DEVELOPMENT_STAGE", "DEVICE_NAME", "OPERATING_SYSTEM", "SOFTWARE_VERSION", "BRAND", "MODEL_NUMBER", "LICENSE_PLATE", "VEHICLE_MAKE", "VEHICLE_MODEL", "VEHICLE_TYPE", "FLIGHT_NUMBER", "HOTEL_NAME", "ROOM_NUMBER", "TRANSACTION_ID", "TICKET_NUMBER", "SEAT_NUMBER", "GATE", "TERMINAL", "TRANSACTION_TYPE", "PAYMENT_STATUS", "PAYMENT_REFERENCE", "INVOICE_STATUS",
    "SYMPTOM", "DIAGNOSIS", "MEDICATION", "DOSAGE", "ALLERGY", "PRESCRIPTION", "TEST_NAME", "TEST_RESULT", "MEDICAL_RECORD", "HEALTH_STATUS", "HEALTH_METRIC", "VITAL_SIGN", "DOCTOR_NAME", "HOSPITAL_NAME", "DEPARTMENT", "WARD", "CLINIC_NAME", "WEBSITE", "URL", "IP_ADDRESS", "MAC_ADDRESS", "USERNAME", "PASSWORD", "LANGUAGE", "CODE_SNIPPET", "DATABASE_NAME", "API_KEY", "WEB_TOKEN", "URL_PARAMETER", "SERVER_NAME", "ENDPOINT", "DOMAIN" ]

**Examples Demonstrating Correct Formatting (Hindi):**

*   **Input:** `"मैं कल मारिया से पुणे में मिलूँगा। AGE_30_45 GENDER_MALE EMOTION_NEUTRAL"`
*   **Correct Output:** `"मैं कल ENTITY_PERSON_NAME मारिया END से ENTITY_CITY पुणे END में मिलूँगा। AGE_30_45 GENDER_MALE EMOTION_NEUTRAL INTENT_INFORM"`

*   **Input:** `"सीबीआई के ये सभी तर्क मद्रास न्यायालय के न्यायाधीश प्रकाश ने खारिज कर दिए। AGE_30_45 GENDER_MALE EMOTION_NEUTRAL"`
*   **Correct Output:** `"ENTITY_ORGANIZATION सीबीआई END के ये सभी तर्क ENTITY_LOCATION मद्रास END न्यायालय के न्यायाधीश ENTITY_PERSON_NAME प्रकाश END ने खारिज कर दिए। AGE_30_45 GENDER_MALE EMOTION_NEUTRAL INTENT_INFORM"`
    *(Note: Assuming "मद्रास" is location, "प्रकाश" is name. Note space after each `END` unless followed by punctuation like `।`)*

*   **Input:** `"मेरा जन्मदिन १२ दिसंबर को है। AGE_18_30 GENDER_MALE EMOTION_HAPPY"`
*   **Correct Output:** `"मेरा जन्मदिन ENTITY_DATE १२ दिसंबर END को है। AGE_18_30 GENDER_MALE EMOTION_HAPPY INTENT_INFORM"`

*   **Input:** `"क्या आप कृपया शांत रहेंगे? AGE_45_60 GENDER_FEMALE EMOTION_ANNOYED"`
*   **Correct Output:** `"क्या आप कृपया शांत रहेंगे? AGE_45_60 GENDER_FEMALE EMOTION_ANNOYED INTENT_QUESTION"`

*   **Input:** `"यह प्रोजेक्ट 'अल्फा' अगले महीने पूरा होगा। AGE_30_45 GENDER_MALE EMOTION_NEUTRAL"`
*   **Correct Output:** `"यह प्रोजेक्ट ENTITY_PROJECT_NAME 'अल्फा' END अगले महीने पूरा होगा। AGE_30_45 GENDER_MALE EMOTION_NEUTRAL INTENT_INFORM"`

**Incorrect Examples (Common Mistakes to Avoid):**
*   `... ENTITY_PERSON NAME मारिया END ...` (Space in TYPE, use underscore: `PERSON_NAME`)
*   `... ENTITY_CITY पुणेEND ...` (Missing space before END)
*   `... ENTITY_CITY पुणे ENDमें ...` (Missing space after END before next word)
*   `... है END AGE_...` (Unnecessary END before metadata tag)
*   `... न्यायाधीश END प्रकाश END ...` (Adding END after non-entity words)

Provide only the JSON array containing the correctly annotated Hindi sentences based precisely on these instructions. Do not include any explanatory text before or after the JSON array.

**Sentences to Annotate Now:**
{json.dumps(texts_to_annotate, ensure_ascii=False, indent=2)}
'''
    # *** REFINED HINDI PROMPT ENDS HERE ***

    # --- API Call and Response Handling (Unchanged Logic) ---
    max_retries = 3
    retry_delay = 5 # seconds
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash-latest") # Or your preferred Gemini model
            response = model.generate_content(prompt)
            assistant_reply = response.text.strip()

            # Robust JSON Extraction (Handles potential markdown ```json ... ```)
            json_match = re.search(r'\[.*\]', assistant_reply, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Attempt cleanup if markdown markers are present but regex failed
                if assistant_reply.startswith("```json"):
                    assistant_reply = assistant_reply[len("```json"):].strip()
                elif assistant_reply.startswith("```"):
                     assistant_reply = assistant_reply[len("```"):].strip()
                if assistant_reply.endswith("```"):
                    assistant_reply = assistant_reply[:-len("```")].strip()

                # Check if the cleaned string looks like a JSON list
                if assistant_reply.startswith('[') and assistant_reply.endswith(']'):
                    json_str = assistant_reply
                else:
                    # If it still doesn't look like JSON, raise error
                    print(f"Warning: Could not extract JSON list from response (Attempt {attempt+1}). Response snippet:\n{assistant_reply[:500]}...")
                    raise json.JSONDecodeError("Response does not appear to contain a JSON list.", assistant_reply, 0)

            # JSON Parsing and Post-Processing
            try:
                annotated_sentences_raw = json.loads(json_str)
            except json.JSONDecodeError as json_e:
                print(f"JSON decoding failed on extracted string (Attempt {attempt+1}): {json_e}")
                print("Extracted snippet causing error:", json_str[:500])
                raise json_e # Re-raise to trigger retry or final failure

            if isinstance(annotated_sentences_raw, list) and len(annotated_sentences_raw) == len(texts_to_annotate):
                processed_sentences = []
                # Apply final formatting fixes to each annotated sentence
                for idx, sentence in enumerate(annotated_sentences_raw):
                     if isinstance(sentence, str):
                         # Apply both space correction functions here
                         corrected_sentence = correct_entity_tag_spaces(sentence)
                         final_sentence = fix_end_tags_and_spacing(corrected_sentence)
                         processed_sentences.append(final_sentence)
                     else:
                          print(f"Warning: API returned non-string item at index {idx} in batch: {sentence}. Replacing with original text + error tag.")
                          try:
                              # Fallback to original text + error marker + basic cleanup
                              original_text = texts_to_annotate[idx]
                              processed_sentences.append(fix_end_tags_and_spacing(original_text) + " ANNOTATION_ERROR_NON_STRING")
                          except IndexError:
                              # Should not happen if lengths match, but safety check
                              processed_sentences.append("ANNOTATION_ERROR_UNKNOWN_ORIGINAL")

                # Final check on length after handling potential non-strings
                if len(processed_sentences) == len(texts_to_annotate):
                    print(f"Annotation successful for batch (Attempt {attempt+1}).")
                    return processed_sentences
                else:
                     # This indicates a problem during the non-string handling fallback
                     print(f"Error: Mismatch after processing non-strings. Expected {len(texts_to_annotate)}, Got {len(processed_sentences)}. Raising error.")
                     if attempt == max_retries - 1:
                         # On final attempt, return originals with error markers
                         return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_PROCESSING" for t in texts_to_annotate]
                     else:
                         # Force a retry if possible
                         raise ValueError("Processing error led to length mismatch.")

            else:
                print(f"Error: API returned invalid list or incorrect length (Attempt {attempt+1}). Expected {len(texts_to_annotate)} items, Got {len(annotated_sentences_raw) if isinstance(annotated_sentences_raw, list) else type(annotated_sentences_raw)}. Retrying...")
                # Raise ValueError to trigger retry logic, unless it's the last attempt
                if attempt == max_retries - 1:
                    print("Max retries reached for API length mismatch.")
                    return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_LENGTH" for t in texts_to_annotate]
                else:
                    print(f"Retrying annotation (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(retry_delay * (attempt + 1)) # Exponential backoff

        except json.JSONDecodeError as e:
            print(f"JSON decoding failed during annotation (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print("Max retries reached due to JSON decoding errors.")
                return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_JSON_DECODE" for t in texts_to_annotate]
            print(f"Retrying annotation...")
            time.sleep(retry_delay * (attempt + 1))
        except Exception as e:
            print(f"Error calling/processing Google Generative AI (Attempt {attempt+1}/{max_retries}): {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            # Handle specific API errors if needed (e.g., rate limits, auth)
            if "rate limit" in str(e).lower():
                print("Rate limit likely hit. Waiting longer before retry...")
                time.sleep(retry_delay * (attempt + 1) * 5) # Wait longer for rate limits
            elif "API key not valid" in str(e) or "permission" in str(e).lower():
                 print("FATAL: Invalid Google API Key or permission issue. Cannot annotate.")
                 return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_KEY" for t in texts_to_annotate]
            else:
                time.sleep(retry_delay * (attempt + 1)) # Standard retry delay

            if attempt == max_retries - 1:
                print("Max retries reached due to API call errors.")
                return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_CALL" for t in texts_to_annotate]

    # Fallback if all retries fail
    print("Error: Max retries reached for annotation batch.")
    return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_MAX_RETRIES" for t in texts_to_annotate]


# --- Main Processing Function (Updated Comments/Paths for Hindi) ---
def process_audio_and_annotate(base_dir: str, output_jsonl_path: str, batch_size: int = 10) -> None:
    """
    Processes Hindi audio files, extracts metadata (transcript, gender) from JSON,
    predicts age/emotion using ML models, formats text, annotates with AI (using Hindi prompt),
    and saves the final structured data to a JSONL file.
    """
    output_dir = os.path.dirname(output_jsonl_path)
    os.makedirs(output_dir, exist_ok=True)

    # Clear output file at the start
    try:
        print(f"Attempting to clear/create output file: {output_jsonl_path}")
        with open(output_jsonl_path, 'w', encoding='utf-8') as f_clear:
            f_clear.write("") # Write empty string to clear or create the file
        print(f"Output file {output_jsonl_path} ready.")
    except IOError as e:
        print(f"Error clearing/creating output file {output_jsonl_path}: {e}. Check permissions or path.")
        return

    # Load Models (Using same models as before)
    print("Loading ML models...")
    age_gender_model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    processor = None
    age_gender_model = None
    try:
        processor = Wav2Vec2Processor.from_pretrained(age_gender_model_name)
        age_gender_model = AgeGenderModel.from_pretrained(age_gender_model_name).to(device)
        age_gender_model.eval() # Set to evaluation mode
        print("Age/Gender model loaded successfully.")
    except Exception as e:
        print(f"FATAL Error loading Age/Gender model '{age_gender_model_name}': {e}. Exiting.")
        return

    emotion_model_name = "superb/hubert-large-superb-er"
    emotion_model_info = {}
    try:
        emotion_model_info['model'] = AutoModelForAudioClassification.from_pretrained(emotion_model_name).to(device)
        emotion_model_info['model'].eval() # Set to evaluation mode
        emotion_model_info['feature_extractor'] = AutoFeatureExtractor.from_pretrained(emotion_model_name)
        print("Emotion model loaded successfully.")
    except Exception as e:
        print(f"Warning: Error loading Emotion model '{emotion_model_name}': {e}. Emotion extraction will result in 'ErrorLoadingModel'.")
        # Allow continuation, but emotion will be marked as error
    print("-" * 30)

    # Prepare File Paths (Expecting 'audio' and 'metadata' subdirs for Hindi data)
    audio_dir = os.path.join(base_dir, "audio")
    metadata_dir = os.path.join(base_dir, "metadata")

    if not os.path.exists(audio_dir) or not os.path.exists(metadata_dir):
        print(f"Error: Required subdirectories not found in '{base_dir}'.")
        print(f"Please ensure your Hindi data follows the structure:")
        print(f"{base_dir}/")
        print(f"  ├── audio/       # Contains .flac, .wav, .mp3 files")
        print(f"  └── metadata/    # Contains corresponding .json files")
        return

    print(f"Processing Hindi files from:")
    print(f"  Audio Directory:    {audio_dir}")
    print(f"  Metadata Directory: {metadata_dir}")
    print(f"Output will be saved to: {output_jsonl_path}")
    print("-" * 30)

    file_pairs = get_file_pairs(audio_dir, metadata_dir)
    if not file_pairs:
        print("No matching audio-metadata file pairs found in the specified directories. Exiting.")
        return

    total_files = len(file_pairs)
    print(f"Found {total_files} audio-metadata pairs to process.")

    processed_records_buffer = [] # Holds records for the current batch
    batch_num = 0
    files_processed_count = 0
    total_records_saved = 0

    # --- Process Files (Main loop - logic unchanged, handles Hindi data via UTF-8 and Hindi prompt) ---
    for i, (audio_path, metadata_path) in enumerate(file_pairs):
        print(f"\nProcessing file {i+1}/{total_files}: {os.path.basename(audio_path)}")
        try:
            # 1. Get Hindi Transcription and Gender from Metadata (Handles Devanagari via UTF-8)
            transcription, metadata_gender = get_metadata_info(metadata_path)
            if transcription is None or metadata_gender is None:
                print(f"  Skipping: Missing required 'transcript' or 'gender' in metadata file {os.path.basename(metadata_path)}")
                continue # Skip this file pair

            # 2. Load Audio (Unchanged logic, handles various formats)
            signal, sr, duration = None, 16000, 0.0
            try:
                # Try soundfile first (often faster, handles FLAC well)
                try:
                    signal_raw, sr_orig = sf.read(audio_path, dtype='float32')
                except Exception as sf_err:
                    # Fallback to librosa if soundfile fails or format not supported
                    print(f"  Soundfile failed ({sf_err}), trying librosa for {os.path.basename(audio_path)}")
                    signal_raw, sr_orig = librosa.load(audio_path, sr=None, mono=False) # Load original SR, potentially stereo

                target_sr = 16000 # Target sample rate for models

                # Ensure float32
                signal_float = signal_raw.astype(np.float32) if not np.issubdtype(signal_raw.dtype, np.floating) else signal_raw

                # Ensure mono
                signal_mono = np.mean(signal_float, axis=1) if signal_float.ndim > 1 else signal_float

                # Resample if necessary
                if sr_orig != target_sr:
                    signal = librosa.resample(y=signal_mono, orig_sr=sr_orig, target_sr=target_sr)
                    sr = target_sr
                    print(f"  Resampled audio from {sr_orig}Hz to {target_sr}Hz.")
                else:
                    signal = signal_mono
                    sr = sr_orig

                if signal is None or len(signal) == 0:
                    print(f"  Skipping: Audio signal is empty after loading/processing {os.path.basename(audio_path)}")
                    continue
                duration = round(len(signal) / sr, 2)

                # Add a check for minimum duration
                min_duration_sec = 0.1
                if duration < min_duration_sec:
                    print(f"  Skipping: Audio duration ({duration:.2f}s) is less than minimum ({min_duration_sec}s) for {os.path.basename(audio_path)}")
                    continue

                print(f"  Audio loaded successfully. Duration: {duration:.2f}s")

            except Exception as load_err:
                print(f"  Skipping: Critical error loading audio file {os.path.basename(audio_path)}: {load_err}")
                continue # Skip this file if audio loading fails critically

            # 3. Extract Age/Gender/Emotion (Using unchanged pre-trained models)
            predicted_age = -1.0 # Default error value
            predicted_emotion = "ERROR"
            ml_predicted_gender = "ERROR"
            try:
                # Prepare input for Wav2Vec2 based models
                inputs = processor(signal, sampling_rate=sr, return_tensors="pt", padding=True)
                input_values = inputs.input_values

                # Truncate long inputs for age/gender model to prevent OOM, based on model limits
                max_len_inference = 30 * sr # e.g., 30 seconds * 16000 Hz
                if input_values.shape[1] > max_len_inference:
                    input_values_truncated = input_values[:, :max_len_inference].to(device)
                    # Use truncated signal for emotion too if needed, or handle separately
                    emotion_signal = signal[:max_len_inference]
                    print(f"  Truncated input for Age/Gender/Emotion prediction to {max_len_inference/sr:.1f}s.")
                else:
                    input_values_truncated = input_values.to(device)
                    emotion_signal = signal # Use full signal for emotion if short enough

                # Run Age/Gender model
                with torch.no_grad():
                    _, logits_age, logits_gender_ml = age_gender_model(input_values_truncated)

                predicted_age = logits_age.cpu().numpy().item() # Age is single regression value
                gender_idx_ml = torch.argmax(logits_gender_ml, dim=-1).cpu().numpy().item()
                gender_map_ml = {0: "FEMALE", 1: "MALE", 2: "OTHER"} # Based on audeering model card
                ml_predicted_gender = gender_map_ml.get(gender_idx_ml, "UNKNOWN")

                # Run Emotion model
                predicted_emotion = extract_emotion(emotion_signal, sr, emotion_model_info)

                print(f"  ML Predictions: Age ~{round(predicted_age * 100)}, Gender={ml_predicted_gender}, Emotion={predicted_emotion}")

            except RuntimeError as e:
                 if "CUDA out of memory" in str(e):
                     print(f"  CUDA OOM Error during ML prediction for {os.path.basename(audio_path)}. Skipping predictions for this file.")
                     torch.cuda.empty_cache()
                     gc.collect()
                     # Assign error values but continue processing the file if possible
                     predicted_age = -1.0
                     predicted_emotion = "OOM_ERROR"
                     ml_predicted_gender = "OOM_ERROR"
                 else:
                     print(f"  Runtime Error during ML prediction for {os.path.basename(audio_path)}: {e}")
                     predicted_age = -1.0
                     predicted_emotion = "RUNTIME_ERROR"
                     ml_predicted_gender = "RUNTIME_ERROR"
            except Exception as predict_err:
                print(f"  Unexpected Error during ML prediction for {os.path.basename(audio_path)}: {predict_err}")
                predicted_age = -1.0
                predicted_emotion = "PREDICT_ERROR"
                ml_predicted_gender = "PREDICT_ERROR"

            # 4. Create Initial Record (Uses Hindi transcript/gender from metadata)
            try:
                # Try to extract speaker ID from filename (e.g., speakerid_...)
                speaker = os.path.splitext(os.path.basename(audio_path))[0].split('_')[0]
            except IndexError:
                speaker = "UNKNOWN_SPEAKER" # Fallback if filename format doesn't match

            segment_data = AudioSegment(
                start_time=0,
                end_time=duration,
                speaker=speaker,
                age=predicted_age,
                gender=metadata_gender, # Use gender from JSON metadata
                transcription=transcription, # Use Hindi transcription from JSON metadata
                emotion=predicted_emotion, # Use predicted emotion
                chunk_filename=os.path.basename(audio_path),
                duration=duration
            )
            chunk = ChunkData(segments=[segment_data], filepath=os.path.abspath(audio_path))

            # Get the formatted text string ready for annotation (Hindi text + ML metadata)
            initial_formatted_text = chunk.get_formatted_text()

            # Store intermediate data needed for the batch
            record = {
                "audio_filepath": chunk.filepath,
                "duration": duration,
                "initial_text": initial_formatted_text, # Text sent to Gemini
                # Keep raw predictions for potential analysis later if needed
                "raw_age_output": predicted_age,
                "metadata_gender": metadata_gender,
                "ml_predicted_gender": ml_predicted_gender,
                "raw_emotion_prediction": predicted_emotion,
                "speaker_id": speaker
            }
            processed_records_buffer.append(record)
            files_processed_count += 1

            # 5. Annotate and Save in Batches (Uses Hindi-specific prompt via annotate_batch_texts)
            # Process batch if full or if it's the last file
            if len(processed_records_buffer) >= batch_size or (i + 1) == total_files:
                batch_num += 1
                current_batch_size = len(processed_records_buffer)
                print(f"\n--- Annotating Batch {batch_num} ({current_batch_size} records) ---")

                texts_to_annotate = [rec["initial_text"] for rec in processed_records_buffer]
                annotated_texts = annotate_batch_texts(texts_to_annotate) # Calls function with Hindi prompt

                # Check if annotation returned expected number of results
                if len(annotated_texts) != current_batch_size:
                    print(f"  FATAL BATCH ERROR: Annotation count mismatch!")
                    print(f"  Expected {current_batch_size}, Got {len(annotated_texts)}. Skipping save for this batch.")
                    # Potentially log problematic texts or attempt partial save with errors?
                    # For now, just skip saving this batch to avoid data corruption.
                else:
                    # Save the processed batch to JSONL
                    try:
                        lines_written_in_batch = 0
                        with open(output_jsonl_path, 'a', encoding='utf-8') as f_out:
                            for record_data, annotated_text in zip(processed_records_buffer, annotated_texts):
                                # Final output structure: filepath, duration, annotated text
                                final_record = {
                                    "audio_filepath": record_data["audio_filepath"],
                                    "duration": record_data["duration"],
                                    "text": annotated_text # The fully annotated text from Gemini
                                }
                                # Log if annotation resulted in an error tag added by our functions
                                if "ANNOTATION_ERROR" in annotated_text:
                                    print(f"  Warning: Saving {os.path.basename(record_data['audio_filepath'])} with annotation error tag: {annotated_text.split()[-1]}")

                                # Write as JSON line, ensuring UTF-8 for Hindi
                                json_str = json.dumps(final_record, ensure_ascii=False)
                                f_out.write(json_str + '\n')
                                lines_written_in_batch += 1

                        total_records_saved += lines_written_in_batch
                        print(f"--- Batch {batch_num} processed and saved ({lines_written_in_batch} records). Total records saved: {total_records_saved} ---")
                    except IOError as io_err:
                        print(f"  Error writing batch {batch_num} to output file {output_jsonl_path}: {io_err}")
                    except Exception as write_err:
                        print(f"  Unexpected error writing batch {batch_num} data: {write_err}")

                # Cleanup batch buffer and potentially free memory
                processed_records_buffer = [] # Clear buffer for next batch
                del texts_to_annotate, annotated_texts # Explicitly delete large lists
                if 'inputs' in locals(): del inputs # Delete tensor data
                if 'input_values' in locals(): del input_values, input_values_truncated
                if 'signal' in locals(): del signal, signal_raw, signal_mono, emotion_signal # Delete audio data
                if 'logits_age' in locals(): del logits_age, logits_gender_ml # Delete model outputs
                if torch.cuda.is_available(): torch.cuda.empty_cache() # Clear GPU cache
                gc.collect() # Trigger garbage collection

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user (Ctrl+C).")
            break # Exit the loop gracefully
        except Exception as e:
            # Catch unexpected errors during the processing of a single file
            print(f"  FATAL ERROR processing file {os.path.basename(audio_path)}: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging
            # Attempt to clean up GPU memory if an error occurred
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()
            continue # Continue to the next file

    # --- Final Summary ---
    print("\n" + "="*30)
    print("Processing Finished.")
    print(f"Total audio-metadata pairs found: {total_files}")
    print(f"Total files attempted processing: {files_processed_count}")
    print(f"Total records successfully saved to {output_jsonl_path}: {total_records_saved}")
    print("="*30)

# --- Main Execution ---
if __name__ == "__main__":
    # --- *** Configuration for Hindi Data (e.g., Vaani-like Structure) *** ---
    # ---> !! IMPORTANT: UPDATE THESE PATHS TO YOUR HINDI DATASET !! <---
    #      The base directory should contain 'audio' and 'metadata' subdirectories.
    BASE_AUDIO_META_DIR = "/external4/datasets/madasr/Vaani-transcription-part/Hindi"  # UPDATE THIS PATH (use 'hi' or appropriate code)
    FINAL_OUTPUT_JSONL = "/external4/datasets/madasr/Vaani-transcription-part/Hindi/hi_corpus_annotated.jsonl" # UPDATE THIS PATH
    PROCESSING_BATCH_SIZE = 10 # Adjust based on API limits and memory (10 is usually safe)

    # --- API Key and Setup (Unchanged) ---
    load_dotenv() # Load GOOGLE_API_KEY from .env file if present
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not found.")
        print("Please set the GOOGLE_API_KEY environment variable or place it in a .env file.")
        exit(1)
    # Re-check genai configuration status after initial setup attempt
    if genai is None:
        print("ERROR: Google Generative AI could not be configured at startup.")
        print("Ensure the API key is valid and there are no network issues.")
        exit(1)

    print("Starting Hindi Audio Processing and Annotation Workflow...") # Updated print
    print(f"Input Base Directory: {BASE_AUDIO_META_DIR}")
    print(f" (Expecting subdirs: {os.path.join(BASE_AUDIO_META_DIR, 'audio')} and {os.path.join(BASE_AUDIO_META_DIR, 'metadata')})")
    print(f"Final Output File: {FINAL_OUTPUT_JSONL}")
    print(f"Processing Batch Size: {PROCESSING_BATCH_SIZE}")
    print("-" * 30)

    # Basic validation of input directory
    if not os.path.isdir(BASE_AUDIO_META_DIR):
        print(f"ERROR: Base input directory not found: {BASE_AUDIO_META_DIR}")
        exit(1)

    # Ensure output directory exists
    output_dir = os.path.dirname(FINAL_OUTPUT_JSONL)
    if output_dir: # Only create if it's not the current directory
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir}")

    # Run the main processing function
    process_audio_and_annotate(
        base_dir=BASE_AUDIO_META_DIR,
        output_jsonl_path=FINAL_OUTPUT_JSONL,
        batch_size=PROCESSING_BATCH_SIZE
    )

    print("\nWorkflow complete.")