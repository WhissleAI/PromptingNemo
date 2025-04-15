# -*- coding: utf-8 -*- # Ensure editor recognizes UTF-8 for Sanskrit script comments/examples
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
# Note: Emotion model performance might vary across languages like Sanskrit.
# Consider fine-tuning or using a language-specific model if results are poor.
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

# --- Data Structures (Unchanged - Structure is Language Agnostic) ---
@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    speaker: str
    age: float
    gender: str
    transcription: str # This will hold Sanskrit text
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
        transcription = segment.transcription.strip() # Keep original case for AI
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
        return "60PLUS" # Default

# --- File Handling (Unchanged - Logic is Language Agnostic, Paths Configurable) ---
def get_file_pairs(audio_dir: str, text_dir: str) -> List[Tuple[str, str]]:
    try:
        # Allow common audio types
        audio_files_list = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.flac', '.wav', '.mp3'))]
        text_files_list = [f for f in os.listdir(text_dir) if f.lower().endswith('.txt')]
        audio_files = {os.path.splitext(f)[0]: os.path.join(audio_dir, f) for f in audio_files_list}
        text_files = {os.path.splitext(f)[0]: os.path.join(text_dir, f) for f in text_files_list}
        pairs = []
        missing_text_count = 0
        for base_name, audio_path in audio_files.items():
            if base_name in text_files:
                pairs.append((audio_path, text_files[base_name]))
            else:
                missing_text_count += 1
        if missing_text_count > 0:
            print(f"Warning: Found {missing_text_count} audio files without matching text files.")

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
    """Reads Sanskrit transcription from a text file."""
    try:
        # *** CRITICAL: Ensure UTF-8 for Sanskrit script (Devanagari) ***
        with open(text_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading transcription file {text_file}: {str(e)}")
        return ""

# --- AI Annotation Functions (MODIFIED for Sanskrit) ---

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
    Cleans up AI-generated annotations, focusing on END tag placement,
    spacing around tags, internal spaces in tag types, and general punctuation.
    Handles Sanskrit script characters (Devanagari) via \w in regex.
    """
    if not isinstance(text, str):
        return text

    # --- Pre-processing ---
    # 1. Normalize whitespace to single spaces and strip ends
    text = re.sub(r'\s+', ' ', text).strip()

    # --- Tag Structure Correction ---
    # 2. Fix spaces *within* ENTITY_TYPE names (e.g., "ENTITY_PERSON_ NAM E" -> "ENTITY_PERSON_NAME")
    def remove_internal_spaces(match):
        tag_prefix = match.group(1) # e.g., "ENTITY_"
        tag_body = match.group(2)   # e.g., "PERSON_ NAM E"
        corrected_body = re.sub(r'\s+', '', tag_body) # Remove all spaces
        return f"{tag_prefix}{corrected_body}"
    text = re.sub(r'\b(ENTITY_)([A-Z0-9_]+(?: +[A-Z0-9_]+)+)\b', remove_internal_spaces, text)

    # 3. Ensure space *after* ENTITY_TAG (if followed by non-space)
    text = re.sub(r'(\bENTITY_[A-Z0-9_]+)(\S)', r'\1 \2', text)

    # 4. Ensure space *before* END tag (if preceded by non-space)
    text = re.sub(r'(\S)(END\b)', r'\1 \2', text)

    # 5. Ensure space *after* END tag (if followed by non-space)
    text = re.sub(r'(\bEND)(\S)', r'\1 \2', text)

    # --- Redundant/Misplaced END Tag Removal ---
    # 6. Remove duplicate END tags (run twice for overlapping cases like END END END)
    text = re.sub(r'\bEND\s+END\b', 'END', text)
    text = re.sub(r'\bEND\s+END\b', 'END', text)

    # 7. Remove END tag if it immediately precedes metadata or end-of-string
    text = re.sub(r'\bEND\s*(\b(?:AGE_|GENDER_|EMOTION_|INTENT_)|$)', r'\1', text)
    text = re.sub(r'\bEND\s*(\b(?:AGE_|GENDER_|EMOTION_|INTENT_)|$)', r'\1', text) # Run again

    # --- Final Spacing and Punctuation ---
    # 8. General/Common punctuation spacing rules
    #    Remove space BEFORE specific punctuation (., ?, !, |, etc. - । is Devanagari Danda)
    text = re.sub(r'\s+([?!:;,।\.])', r'\1', text) # Includes Devanagari Danda
    #    Ensure space AFTER punctuation if followed by a word character (\w handles Devanagari script)
    text = re.sub(r'([?!:;,।\.])(\w)', r'\1 \2', text) # Includes Devanagari Danda

    # 9. Final whitespace normalization and stripping
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# --- MODIFIED AND MORE DETAILED PROMPT FOR SANSKRIT (Devanagari Script Focus) ---
def annotate_batch_texts(texts_to_annotate: List[str]):
    """Sends a batch of texts to Gemini for annotation (Prompt refined for clarity and stricter rules for Sanskrit in Devanagari script)."""
    if not genai:
        print("Error: Google Generative AI not configured. Skipping annotation.")
        return texts_to_annotate # Return original texts
    if not texts_to_annotate:
        return []

    # Input texts already have metadata like: "sanskrit text AGE_X GENDER_Y EMOTION_Z"

    # *** REFINED SANSKRIT PROMPT STARTS HERE ***
    prompt = f'''You are an expert linguistic annotator specifically for **Sanskrit** text written in the **Devanagari** script. Your task is to process a list of Sanskrit sentences, each already containing `AGE_*`, `GENDER_*`, and `EMOTION_*` metadata tags at the end. Follow these instructions with extreme precision:

1.  **Preserve Existing Metadata Tags:**
    *   The `AGE_*`, `GENDER_*`, and `EMOTION_*` tags at the end of each sentence **must** remain exactly as they are.
    *   **Do not** modify, move, delete, or change these existing metadata tags.

2.  **Entity Annotation (Sanskrit Text Only):**
    *   Identify entities **only** within the main Sanskrit transcription part of the sentence (before the metadata).
    *   Use **only** the entity types provided in the `Allowed ENTITY TYPES` list below. Do not invent new types.

3.  **Strict Entity Tagging Format:**
    *   **Tag Insertion:** Place the entity tag **immediately before** the identified Sanskrit entity text. The tag format is `ENTITY_<TYPE>` (e.g., `ENTITY_CITY`, `ENTITY_PERSON_NAME`).
    *   **NO SPACES IN TYPE:** Ensure the `<TYPE>` part of the tag contains **no spaces** (e.g., `PERSON_NAME` is correct, `PERSON_ NAM E` is **incorrect**).
    *   **`END` Tag Placement:** Place the literal string `END` **immediately after** the *complete* Sanskrit entity text.
    *   **Spacing:** There must be exactly **one space** between the `ENTITY_<TYPE>` tag and the start of the entity text. There must be exactly **one space** between the end of the entity text and the `END` tag. There must be exactly **one space** after the `END` tag before the next word begins (unless followed by punctuation).
    *   **Example:** For the entity "काशी", the correct annotation is `ENTITY_CITY काशी END`.
    *   **Crucial `END` Rule:** Only add **one** `END` tag right after the full entity phrase. Do **not** add `END` tags after individual words within a multi-word entity or after words that simply follow an entity.
    *   **Avoid Extra `END` Before Metadata:** Do **not** place an `END` tag just before the `AGE_`, `GENDER_`, `EMOTION_`, or `INTENT_` tags unless that `END` tag correctly marks the end of an immediately preceding entity.

4.  **Intent Tag:**
    *   Determine the single primary intent of the Sanskrit sentence (e.g., `INFORM`, `QUESTION`, `REQUEST`, `COMMAND`, `GREETING`, `BLESSING`, `STATEMENT`).
    *   Append **one** `INTENT_<INTENT_TYPE>` tag at the **absolute end** of the entire string, after all other tags (`AGE_`, `GENDER_`, `EMOTION_`).

5.  **Output Format:**
    *   Return a **JSON array** of strings. Each string must be a fully annotated sentence following all the rules above.

6.  **Sanskrit Language Specifics:**
    *   Handle Sanskrit script (Devanagari) correctly.
    *   Ensure correct spacing around common punctuation (like `।`, `?`, `!`, `.`). Remove space before punctuation, ensure space after punctuation if followed by a word. Use `।` (Devanagari Danda) as the primary sentence terminator where appropriate.
    *   The final output string must be clean, with single spaces separating words and tags according to the rules.

**Allowed ENTITY TYPES (Use Only These):**
    [ "PERSON_NAME", "DEITY_NAME", "ORGANIZATION", "LOCATION", "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP_CODE", "CURRENCY", "PRICE", "DATE",
    "TIME", "DURATION", "APPOINTMENT_DATE", "APPOINTMENT_TIME", "DEADLINE", "DELIVERY_DATE", "DELIVERY_TIME", "EVENT", "MEETING", "RITUAL", "FESTIVAL",
    "TASK", "PROJECT_NAME", "ACTION_ITEM", "PRIORITY", "FEEDBACK", "REVIEW", "RATING", "COMPLAINT", "QUESTION", "RESPONSE", "NOTIFICATION_TYPE", "AGENDA",
    "REMINDER", "NOTE", "RECORD", "ANNOUNCEMENT", "UPDATE", "SCHEDULE", "BOOKING_REFERENCE", "APPOINTMENT_NUMBER", "ORDER_NUMBER", "INVOICE_NUMBER", "PAYMENT_METHOD", "PAYMENT_AMOUNT", "BANK_NAME", "ACCOUNT_NUMBER", "CREDIT_CARD_NUMBER", "TAX_ID", "SOCIAL_SECURITY_NUMBER", "DRIVER'S_LICENSE", "PASSPORT_NUMBER", "INSURANCE_PROVIDER", "POLICY_NUMBER", "INSURANCE_PLAN", "CLAIM_NUMBER", "POLICY_HOLDER", "BENEFICIARY",
    "RELATIONSHIP", "EMERGENCY_CONTACT", "PROJECT_PHASE", "VERSION", "DEVELOPMENT_STAGE", "DEVICE_NAME", "OPERATING_SYSTEM", "SOFTWARE_VERSION", "BRAND", "MODEL_NUMBER", "LICENSE_PLATE", "VEHICLE_MAKE", "VEHICLE_MODEL", "VEHICLE_TYPE", "FLIGHT_NUMBER", "HOTEL_NAME", "ROOM_NUMBER", "TRANSACTION_ID", "TICKET_NUMBER", "SEAT_NUMBER", "GATE", "TERMINAL", "TRANSACTION_TYPE", "PAYMENT_STATUS", "PAYMENT_REFERENCE", "INVOICE_STATUS",
    "SYMPTOM", "DIAGNOSIS", "MEDICATION", "DOSAGE", "ALLERGY", "PRESCRIPTION", "TEST_NAME", "TEST_RESULT", "MEDICAL_RECORD", "HEALTH_STATUS", "HEALTH_METRIC", "VITAL_SIGN", "DOCTOR_NAME", "HOSPITAL_NAME", "DEPARTMENT", "WARD", "CLINIC_NAME", "WEBSITE", "URL", "IP_ADDRESS", "MAC_ADDRESS", "USERNAME", "PASSWORD", "LANGUAGE", "CODE_SNIPPET", "DATABASE_NAME", "API_KEY", "WEB_TOKEN", "URL_PARAMETER", "SERVER_NAME", "ENDPOINT", "DOMAIN",
    "SCRIPTURE_NAME", "CHAPTER", "VERSE_NUMBER", "PHILOSOPHICAL_CONCEPT", "SCHOOL_OF_THOUGHT", "TEXT_NAME" ]
    # Added DEITY_NAME, RITUAL, FESTIVAL, SCRIPTURE_NAME, CHAPTER, VERSE_NUMBER, PHILOSOPHICAL_CONCEPT, SCHOOL_OF_THOUGHT, TEXT_NAME

**Examples Demonstrating Correct Formatting:**

*   **Input:** `"अहं श्वः रामं नगरे द्रक्ष्यामि। AGE_30_45 GENDER_MALE EMOTION_NEUTRAL"`
*   **Correct Output:** `"अहं श्वः ENTITY_PERSON_NAME रामम् END ENTITY_CITY नगरे END द्रक्ष्यामि। AGE_30_45 GENDER_MALE EMOTION_NEUTRAL INTENT_INFORM"`

*   **Input:** `"तव नाम किम्? AGE_18_30 GENDER_FEMALE EMOTION_NEUTRAL"`
*   **Correct Output:** `"तव नाम किम्? AGE_18_30 GENDER_FEMALE EMOTION_NEUTRAL INTENT_QUESTION"`

*   **Input:** `"भगवद्गीतायां कर्मयोगः महत्त्वपूर्णः अस्ति। AGE_45_60 GENDER_MALE EMOTION_NEUTRAL"`
*   **Correct Output:** `"ENTITY_SCRIPTURE_NAME भगवद्गीतायाम् END ENTITY_PHILOSOPHICAL_CONCEPT कर्मयोगः END महत्त्वपूर्णः अस्ति। AGE_45_60 GENDER_MALE EMOTION_NEUTRAL INTENT_STATEMENT"`

*   **Input:** `"कृपया उपविशतु। AGE_60PLUS GENDER_FEMALE EMOTION_POLITE"`
*   **Correct Output:** `"कृपया उपविशतु। AGE_60PLUS GENDER_FEMALE EMOTION_POLITE INTENT_REQUEST"`

*   **Input:** `"शिवः सर्वेषां कल्याणं करोतु। AGE_30_45 GENDER_MALE EMOTION_HOPEFUL"`
*   **Correct Output:** `"ENTITY_DEITY_NAME शिवः END सर्वेषां कल्याणं करोतु। AGE_30_45 GENDER_MALE EMOTION_HOPEFUL INTENT_BLESSING"`


**Incorrect Examples (Common Mistakes to Avoid):**
*   `... ENTITY_PERSON_ NAM E रामम् END ...` (Space in TYPE)
*   `... ENTITY_CITY नगरेEND ...` (Missing space before END)
*   `... ENTITY_CITY नगरे ENDद्रक्ष्यामि ...` (Missing space after END)
*   `... द्रक्ष्यामि। END AGE_...` (Unnecessary END before metadata)
*   `... रामम् END नगरे END ...` (Incorrectly adding END after non-entity words)

Provide only the JSON array containing the correctly annotated sentences based precisely on these instructions.

**Sentences to Annotate Now:**
{json.dumps(texts_to_annotate, ensure_ascii=False, indent=2)}
'''
    # *** REFINED SANSKRIT PROMPT ENDS HERE ***

    max_retries = 3
    retry_delay = 5 # seconds
    for attempt in range(max_retries):
        try:
            # Using a potentially more capable model might help adherence
            # model = genai.GenerativeModel("gemini-1.5-pro-latest")
            model = genai.GenerativeModel("gemini-1.5-flash-latest") # Keep flash for speed unless pro is needed

            response = model.generate_content(prompt)
            assistant_reply = response.text.strip()

            # --- Robust JSON Extraction ---
            json_match = re.search(r'\[.*\]', assistant_reply, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # print(f"Debug: Extracted JSON string (length {len(json_str)})") # Debug print
            else:
                if assistant_reply.startswith("```json"):
                    assistant_reply = assistant_reply[len("```json"):].strip()
                elif assistant_reply.startswith("```"):
                     assistant_reply = assistant_reply[len("```"):].strip()
                if assistant_reply.endswith("```"):
                    assistant_reply = assistant_reply[:-len("```")].strip()

                if assistant_reply.startswith('[') and assistant_reply.endswith(']'):
                     json_str = assistant_reply
                     # print("Debug: Extracted JSON string after markdown removal") # Debug print
                else:
                    print(f"Error: Could not extract valid JSON list from response (Attempt {attempt+1}). Response snippet:\n---\n{assistant_reply[:500]}\n---")
                    raise json.JSONDecodeError("Response does not appear to contain a JSON list.", assistant_reply, 0)

            # --- JSON Parsing and Post-Processing ---
            try:
                annotated_sentences_raw = json.loads(json_str)
            except json.JSONDecodeError as json_e:
                print(f"JSON decoding failed specifically on extracted string (Attempt {attempt+1}): {json_e}")
                print("Extracted string snippet:", json_str[:500])
                raise json_e

            if isinstance(annotated_sentences_raw, list) and len(annotated_sentences_raw) == len(texts_to_annotate):
                processed_sentences = []
                for idx, sentence in enumerate(annotated_sentences_raw):
                     if isinstance(sentence, str):
                          final_sentence = fix_end_tags_and_spacing(sentence)
                          processed_sentences.append(final_sentence)
                     else:
                          print(f"Warning: Non-string item received in annotation list at index {idx}: {sentence}")
                          try:
                              original_text = texts_to_annotate[idx]
                              processed_sentences.append(fix_end_tags_and_spacing(original_text) + " ANNOTATION_ERROR_NON_STRING")
                          except IndexError:
                              print(f"Error: Could not map non-string item at index {idx} back to original text.")
                              processed_sentences.append("ANNOTATION_ERROR_UNKNOWN_ORIGINAL")

                # Final length check
                if len(processed_sentences) == len(texts_to_annotate):
                    # print(f"Debug: Batch annotation successful (Attempt {attempt+1}). Lengths match.") # Debug
                    return processed_sentences
                else:
                    print(f"Error: Mismatch after processing non-string elements. Expected {len(texts_to_annotate)}, Got {len(processed_sentences)}")
                    if attempt == max_retries - 1:
                        return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_PROCESSING" for t in texts_to_annotate]
                    else:
                        raise ValueError("Processing error lead to length mismatch.") # Force retry

            else:
                print(f"Error: API returned invalid list or mismatched length (Attempt {attempt+1}). Expected {len(texts_to_annotate)}, Got {len(annotated_sentences_raw) if isinstance(annotated_sentences_raw, list) else type(annotated_sentences_raw)}")
                if attempt == max_retries - 1:
                     return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_LENGTH" for t in texts_to_annotate]
                else:
                    print(f"Retrying annotation (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(retry_delay * (attempt + 1))

        except json.JSONDecodeError as e:
            print(f"JSON decoding failed (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                 return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_JSON_DECODE" for t in texts_to_annotate]
            print(f"Retrying annotation...")
            time.sleep(retry_delay * (attempt + 1))
        except Exception as e:
            print(f"Error calling/processing Generative AI (Attempt {attempt+1}/{max_retries}): {e}")
            import traceback; traceback.print_exc()
            if "rate limit" in str(e).lower():
                 print("Rate limit likely hit.")
                 time.sleep(retry_delay * (attempt + 1) * 5)
            elif "API key not valid" in str(e) or "permission" in str(e).lower():
                 print("FATAL: Invalid Google API Key or permission issue.")
                 return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_KEY" for t in texts_to_annotate]
            else:
                 time.sleep(retry_delay * (attempt + 1))

            if attempt == max_retries - 1:
                 return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_CALL" for t in texts_to_annotate]

    print("Error: Max retries reached for annotation batch.")
    return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_MAX_RETRIES" for t in texts_to_annotate]


# --- Rest of the script remains the same ---

# --- Data Structures (Unchanged) ---
# ... (AudioSegment, ChunkData classes) ...

# --- File Handling (Unchanged) ---
# ... (get_file_pairs, get_transcription functions) ...

# --- Main Processing Function (Unchanged except calls to modified functions and setup) ---
def process_audio_and_annotate(base_dir: str, output_jsonl_path: str, batch_size: int = 10) -> None:
    """
    Processes Sanskrit audio files, extracts metadata, gets transcriptions,
    formats text, annotates with AI (using Sanskrit prompt), and saves final JSONL output.
    """
    output_dir = os.path.dirname(output_jsonl_path)
    os.makedirs(output_dir, exist_ok=True)

    # --- Clear output file at the very beginning ---
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
    # Age/Gender Model
    age_gender_model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    try:
        processor = Wav2Vec2Processor.from_pretrained(age_gender_model_name)
        age_gender_model = AgeGenderModel.from_pretrained(age_gender_model_name).to(device)
        age_gender_model.eval()
        print("Age/Gender model loaded.")
    except Exception as e:
        print(f"FATAL Error loading Age/Gender model: {e}. Exiting.")
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

    print("-" * 30)

    # --- Prepare File Paths ---
    audio_dir = os.path.join(base_dir, "audio") # Standard subfolder names
    text_dir = os.path.join(base_dir, "text")   # Standard subfolder names

    # Check if directories exist
    if not os.path.exists(audio_dir) or not os.path.exists(text_dir):
        print(f"Error: Audio ({audio_dir}) or text ({text_dir}) directory not found in {base_dir}")
        print("Please ensure your Sanskrit data follows the structure:")
        print(f"{base_dir}/")
        print(f"  ├── audio/  (contains .flac/.wav/.mp3 files)")
        print(f"  └── text/   (contains .txt files with matching basenames, UTF-8 encoded)")
        return

    print(f"Processing Sanskrit files from:\n  Audio: {audio_dir}\n  Text:  {text_dir}")
    print(f"Output will be saved to: {output_jsonl_path}")
    print("-" * 30)

    # Get matching audio-text pairs
    file_pairs = get_file_pairs(audio_dir, text_dir)
    if not file_pairs:
        print("No matching audio-text files found. Exiting.")
        return

    total_files = len(file_pairs)
    print(f"Found {total_files} audio-text pairs to process.")

    processed_records_buffer = []
    batch_num = 0
    files_processed_count = 0
    total_records_saved = 0

    # --- Process Files ---
    for i, (audio_path, text_path) in enumerate(file_pairs):
        print(f"\nProcessing file {i+1}/{total_files}: {os.path.basename(audio_path)}")
        try:
            # 1. Get Transcription (Ensures UTF-8 for Sanskrit)
            transcription = get_transcription(text_path)
            if not transcription:
                print(f"  Skipping: Empty or unreadable transcription in {text_path}")
                continue

            # 2. Load Audio
            signal, sr, duration = None, 16000, 0
            try:
                try:
                    signal_raw, sr_orig = sf.read(audio_path, dtype='float32')
                except Exception as sf_err:
                    # print(f"  Soundfile failed ({sf_err}), trying librosa...")
                    signal_raw, sr_orig = librosa.load(audio_path, sr=None, mono=False)

                target_sr = 16000
                if sr_orig != target_sr:
                    signal_float = signal_raw.astype(np.float32) if not np.issubdtype(signal_raw.dtype, np.floating) else signal_raw
                    if signal_float.ndim > 1: signal_mono = np.mean(signal_float, axis=1)
                    else: signal_mono = signal_float
                    signal = librosa.resample(y=signal_mono, orig_sr=sr_orig, target_sr=target_sr)
                    sr = target_sr
                else:
                    signal = signal_raw.astype(np.float32) if not np.issubdtype(signal_raw.dtype, np.floating) else signal_raw
                    sr = sr_orig

                if signal.ndim > 1: signal = np.mean(signal, axis=1) # Ensure mono

                if signal is None or len(signal) == 0:
                    print(f"  Skipping: Failed to load or empty audio in {audio_path}")
                    continue
                duration = round(len(signal) / sr, 2)
                if duration < 0.1:
                    print(f"  Skipping: Audio too short ({duration:.2f}s) in {audio_path}")
                    continue
            except Exception as load_err:
                 print(f"  Skipping: Error loading/processing audio {audio_path}: {load_err}")
                 continue

            # 3. Extract Age/Gender
            age, gender = -1.0, "ERROR"
            try:
                inputs = processor(signal, sampling_rate=sr, return_tensors="pt", padding=True)
                input_values = inputs.input_values
                max_len_inference = 30 * sr # Limit input length
                if input_values.shape[1] > max_len_inference:
                    input_values_truncated = input_values[:, :max_len_inference].to(device)
                else:
                    input_values_truncated = input_values.to(device)

                with torch.no_grad():
                    _, logits_age, logits_gender = age_gender_model(input_values_truncated)
                age = logits_age.cpu().numpy().item()
                gender_idx = torch.argmax(logits_gender, dim=-1).cpu().numpy().item()
                gender_map = {0: "FEMALE", 1: "MALE", 2: "OTHER"}
                gender = gender_map.get(gender_idx, "UNKNOWN")
            except RuntimeError as e:
                 if "CUDA out of memory" in str(e):
                     print(f"  CUDA OOM Error during Age/Gender extraction. Skipping file.")
                     torch.cuda.empty_cache(); gc.collect(); continue
                 else: print(f"  Runtime Error during Age/Gender extraction: {e}"); age = -1.0; gender = "ERROR"
            except Exception as age_gender_err:
                print(f"  Error during Age/Gender extraction: {age_gender_err}"); age = -1.0; gender = "ERROR"

            # 4. Extract Emotion
            emotion = "ERROR"
            try:
                 emotion_signal = signal[:max_len_inference] if 'input_values' in locals() and input_values.shape[1] > max_len_inference else signal
                 emotion = extract_emotion(emotion_signal, sr, emotion_model_info)
            except Exception as emotion_err:
                 print(f"  Error during Emotion extraction: {emotion_err}"); emotion = "ERROR"

            # 5. Create Initial Record
            try: speaker = os.path.splitext(os.path.basename(audio_path))[0].split('_')[0]
            except IndexError: speaker = "UNKNOWN_SPEAKER"
            segment_data = AudioSegment(start_time=0, end_time=duration, speaker=speaker, age=age, gender=gender, transcription=transcription, emotion=emotion, chunk_filename=os.path.basename(audio_path), duration=duration)
            chunk = ChunkData(segments=[segment_data], filepath=os.path.abspath(audio_path))
            initial_formatted_text = chunk.get_formatted_text()
            record = { "audio_filepath": chunk.filepath, "duration": duration, "initial_text": initial_formatted_text, "raw_age_output": age, "raw_gender_prediction": gender, "raw_emotion_prediction": emotion, "speaker_id": speaker }
            processed_records_buffer.append(record)
            files_processed_count += 1

            # 6. Annotate and Save in Batches
            if len(processed_records_buffer) >= batch_size or (i + 1) == total_files:
                batch_num += 1
                current_batch_size = len(processed_records_buffer)
                print(f"\n--- Annotating Batch {batch_num} ({current_batch_size} records) ---")

                texts_to_annotate = [rec["initial_text"] for rec in processed_records_buffer]
                # Use the Sanskrit-specific annotation function
                annotated_texts = annotate_batch_texts(texts_to_annotate)

                if len(annotated_texts) != current_batch_size:
                     print(f"  FATAL BATCH ERROR: Annotation count mismatch! Expected {current_batch_size}, Got {len(annotated_texts)}. Skipping save.")
                else:
                    try:
                        lines_written_in_batch = 0
                        # Ensure UTF-8 for writing Sanskrit text (Devanagari)
                        with open(output_jsonl_path, 'a', encoding='utf-8') as f_out:
                            for record_data, annotated_text in zip(processed_records_buffer, annotated_texts):
                                final_record = {
                                    "audio_filepath": record_data["audio_filepath"],
                                    "duration": record_data["duration"],
                                    "text": annotated_text, # Contains final Sanskrit text with tags
                                }
                                if "ANNOTATION_ERROR" in annotated_text:
                                     print(f"  Warning: Saving record for {os.path.basename(record_data['audio_filepath'])} with annotation error flag: {annotated_text.split()[-1]}")
                                # Ensure JSON dumps handles Unicode correctly (Devanagari)
                                json_str = json.dumps(final_record, ensure_ascii=False)
                                f_out.write(json_str + '\n')
                                lines_written_in_batch += 1
                        total_records_saved += lines_written_in_batch
                        print(f"--- Batch {batch_num} processed and saved ({lines_written_in_batch} records). Total saved: {total_records_saved} ---")
                    except IOError as io_err: print(f"  Error writing batch {batch_num} to {output_jsonl_path}: {io_err}")
                    except Exception as write_err: print(f"  Unexpected error writing batch {batch_num}: {write_err}")

                processed_records_buffer = []
                # Clean up memory
                del texts_to_annotate, annotated_texts
                if 'inputs' in locals(): del inputs
                if 'input_values' in locals(): del input_values
                if 'input_values_truncated' in locals(): del input_values_truncated
                if 'signal' in locals(): del signal
                if 'signal_raw' in locals(): del signal_raw
                if 'signal_float' in locals(): del signal_float
                if 'signal_mono' in locals(): del signal_mono
                if 'logits_age' in locals(): del logits_age, logits_gender
                if 'emotion_signal' in locals(): del emotion_signal
                if 'outputs' in locals(): del outputs
                torch.cuda.empty_cache(); gc.collect()

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")
            break
        except Exception as e:
            print(f"  FATAL ERROR processing file {os.path.basename(audio_path)}: {e}")
            import traceback; traceback.print_exc()
            torch.cuda.empty_cache(); gc.collect()
            continue # Try the next file

    print("\n" + "="*30 + f"\nProcessing Finished.\nTotal files processing attempted: {files_processed_count}/{total_files}\nTotal records saved to {output_jsonl_path}: {total_records_saved}\n" + "="*30)


# --- Main Execution ---
if __name__ == "__main__":
    # --- *** Configuration for Sanskrit Data *** ---
    # >>> IMPORTANT: CHANGE THESE PATHS TO YOUR SANSKRIT DATA LOCATION <<<
    # Example assumes Sanskrit data uses 'san' code or similar structure
    BASE_AUDIO_TEXT_DIR = "/external4/datasets/Shrutilipi/sanskrit/sk"  # Example: /data/sanskrit_corpus/san
    FINAL_OUTPUT_JSONL = "/external4/datasets/Shrutilipi/sanskrit/sk/san_annotated_output.jsonl" # Example: /data/sanskrit_corpus/san/annotated_sanskrit.jsonl
    # <<< END OF PATHS TO CHANGE <<<

    PROCESSING_BATCH_SIZE = 10 # Adjust based on API limits and system memory (start small)

    # --- API Key and Setup ---
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not found.")
        exit(1)
    if genai is None: # Re-check config
        try:
            api_key_recheck = os.getenv("GOOGLE_API_KEY")
            if not api_key_recheck: raise ValueError("GOOGLE_API_KEY still not found.")
            genai.configure(api_key=api_key_recheck)
            print("Google Generative AI re-configured successfully.")
        except Exception as e:
             print(f"ERROR: Failed to configure Google Generative AI: {e}")
             exit(1)

    print("Starting Sanskrit Audio Processing and Annotation Workflow...")
    print(f"Input Base Directory: {BASE_AUDIO_TEXT_DIR}")
    print(f"Final Output File: {FINAL_OUTPUT_JSONL}")
    print(f"Processing Batch Size: {PROCESSING_BATCH_SIZE}")
    print("-" * 30)

    if not os.path.isdir(BASE_AUDIO_TEXT_DIR):
        print(f"ERROR: Base directory not found: {BASE_AUDIO_TEXT_DIR}")
        exit(1)
    output_dir = os.path.dirname(FINAL_OUTPUT_JSONL)
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created/exists: {output_dir}")
    except OSError as e:
        print(f"Error creating output directory {output_dir}: {e}")
        exit(1)

    # Start the main processing function
    process_audio_and_annotate(
        base_dir=BASE_AUDIO_TEXT_DIR,
        output_jsonl_path=FINAL_OUTPUT_JSONL,
        batch_size=PROCESSING_BATCH_SIZE
    )

    print("Workflow complete.")