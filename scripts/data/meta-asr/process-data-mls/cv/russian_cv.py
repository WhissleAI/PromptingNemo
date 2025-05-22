# -*- coding: utf-8 -*- # Ensure editor recognizes UTF-8 for Cyrillic script comments/examples
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
    Wav2Vec2Processor, # Keep for age/gender
    Wav2Vec2Model,     # Keep for age/gender
    Wav2Vec2PreTrainedModel # Keep for age/gender
)
import torch.nn as nn
import google.generativeai as genai
import time
import soundfile as sf
import subprocess # Added for direct ffmpeg call
from tqdm import tqdm # Added for progress tracking

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
    try:
        if 1 < torch.cuda.device_count():
             device = torch.device("cuda:1")
             print(f"Using specified device: {torch.cuda.get_device_name(1)}")
        else:
             device = torch.device("cuda:0")
             print(f"Specified device cuda:1 not available or only one GPU found. Using default cuda:0: {torch.cuda.get_device_name(0)}")
    except Exception as e:
         print(f"Error setting CUDA device: {e}. Using default cuda:0.")
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
    print(f"Error configuring Google Generative AI: {e}. Annotation step will likely fail.")
    genai = None

# --- Audio Conversion Function ---
def convert_opus_to_wav(input_opus_path: str, output_wav_path: str) -> bool:
    """Converts an audio file from OPUS to WAV format (16kHz, mono) using a direct ffmpeg call."""
    global ffmpeg_path # Access the globally defined ffmpeg path
    if not ffmpeg_path or not os.path.exists(ffmpeg_path):
        print(f"  FATAL ERROR: ffmpeg path not set or invalid: {ffmpeg_path}")
        return False
    try:
        output_dir = os.path.dirname(output_wav_path)
        os.makedirs(output_dir, exist_ok=True)

        # Construct the ffmpeg command
        command = [
            ffmpeg_path,
            '-i', input_opus_path,
            '-ar', '16000',
            '-ac', '1',
            '-sample_fmt', 's16',
            '-y',
            output_wav_path
        ]

        # Execute the command
        result = subprocess.run(command, check=False, capture_output=True, text=True, encoding='utf-8')

        # Check if ffmpeg reported an error
        if result.returncode != 0:
            print(f"  Error converting {input_opus_path} to {output_wav_path} using ffmpeg.")
            print(f"  FFmpeg stderr: {result.stderr.strip()}")
            if "Unknown input format: 'opus'" in result.stderr:
                 print("  FATAL: Your ffmpeg build does not seem to support opus decoding.")
            elif "No such file or directory" in result.stderr and input_opus_path in result.stderr:
                 print(f"  FATAL: Input file not found by ffmpeg: {input_opus_path}")
            elif "Permission denied" in result.stderr:
                 print(f"  FATAL: Permission error accessing input/output files or ffmpeg executable.")
            return False
        else:
            if not os.path.exists(output_wav_path) or os.path.getsize(output_wav_path) == 0:
                print(f"  Warning: ffmpeg reported success but output file is missing or empty: {output_wav_path}")
            return True

    except FileNotFoundError:
        print(f"  FATAL ERROR: ffmpeg command not found at '{ffmpeg_path}'. Is the path correct and executable?")
        return False
    except Exception as e:
        print(f"  Unexpected error during subprocess call for {input_opus_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- Age/Gender Model Definition (Predicts Both) ---
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
        self.age_head = ModelHead(config, 1)
        self.gender_head = ModelHead(config, 2)
        self.init_weights()

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        hidden_states_pooled = torch.mean(hidden_states, dim=1)

        logits_age = self.age_head(hidden_states_pooled)
        logits_gender = self.gender_head(hidden_states_pooled)
        return logits_age, logits_gender

# --- Emotion Extraction Function (Unchanged) ---
def extract_emotion(audio_data: np.ndarray, sampling_rate: int = 16000, model_info: dict = None) -> str:
    if model_info is None or 'model' not in model_info or 'feature_extractor' not in model_info:
        print("Error: Emotion model not loaded correctly."); return "ERRORLOADINGMODEL"
    if audio_data is None or len(audio_data) == 0: return "NO_AUDIO"
    try:
        inputs = model_info['feature_extractor'](audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        max_len_inference_emotion = 30 * sampling_rate
        if inputs['input_values'].shape[1] > max_len_inference_emotion:
            print(f"  Warning: Truncating audio from {inputs['input_values'].shape[1]/sampling_rate:.1f}s to {max_len_inference_emotion/sampling_rate:.1f}s for emotion extraction.")
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

# --- Age/Gender Prediction Function (Updated) ---
def predict_age_gender_from_audio(signal: np.ndarray, sampling_rate: int, processor: Wav2Vec2Processor, model: AgeGenderModel) -> Tuple[Optional[float], Optional[torch.Tensor]]:
    if signal is None or len(signal) == 0:
        print("  Age/Gender Prediction Error: No audio signal provided.")
        return None, None
    try:
        max_len_inference_age_gender = 30 * sampling_rate
        inputs = processor(signal, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        input_values = inputs.input_values
        attention_mask = inputs.get("attention_mask")

        input_values_truncated = input_values[:, :max_len_inference_age_gender].to(device)
        attention_mask_truncated = attention_mask[:, :max_len_inference_age_gender].to(device) if attention_mask is not None else None

        if input_values.shape[1] > max_len_inference_age_gender:
             print(f"  Warning: Truncating audio from {input_values.shape[1]/sampling_rate:.1f}s to {max_len_inference_age_gender/sampling_rate:.1f}s for age/gender prediction.")

        with torch.no_grad():
            logits_age, logits_gender = model(input_values_truncated, attention_mask=attention_mask_truncated)

        age_prediction_float = logits_age.cpu().numpy().item()
        gender_logits = logits_gender.cpu()
        return age_prediction_float, gender_logits

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"  CUDA OOM Error during age/gender prediction.");
            torch.cuda.empty_cache(); gc.collect()
            return None, None
        else:
            print(f"  Runtime Error during age/gender prediction: {e}"); return None, None
    except Exception as e:
        print(f"  Error during age/gender prediction: {e}"); return None, None

# --- Age Bucketing Function (Unchanged) ---
def get_age_bucket_from_prediction(age_float: Optional[float]) -> str:
    if age_float is None or age_float < 0: return "UNKNOWN"
    try:
        actual_age = round(age_float * 100)
        if actual_age < 18: return "0_17"
        elif actual_age < 30: return "18_29"
        elif actual_age < 45: return "30_44"
        elif actual_age < 60: return "45_59"
        else: return "60PLUS"
    except Exception as e:
        print(f"  Error bucketing predicted age {age_float}: {e}"); return "UNKNOWN"

# --- Gender Labeling Function (New) ---
def get_gender_label_from_prediction(gender_logits: Optional[torch.Tensor]) -> str:
    if gender_logits is None: return "UNKNOWN"
    try:
        predicted_gender_idx = torch.argmax(gender_logits, dim=-1).item()
        if predicted_gender_idx == 0: return "FEMALE"
        elif predicted_gender_idx == 1: return "MALE"
        else: return "UNKNOWN"
    except Exception as e:
        print(f"  Error determining gender label from logits: {e}"); return "UNKNOWN"

# --- AI Annotation Helper Functions ---
def fix_end_tags_and_spacing(text: str) -> str:
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
    text = re.sub(r'\s+([?!:;,.])', r'\1', text)
    text = re.sub(r'([?!:;,.])([^\s])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- *** MODIFIED PROMPT FOR RUSSIAN *** ---
def annotate_batch_texts(texts_to_annotate: List[str]):
    if not genai:
        print("Error: Google Generative AI not configured. Skipping annotation.")
        return [t + " ANNOTATION_ERROR_NO_GENAI" for t in texts_to_annotate]
    if not texts_to_annotate: return []

    prompt = f'''You are an expert linguistic annotator specifically for **Russian** text written in the **Cyrillic script**. Your task is to process a list of Russian sentences, each already containing `AGE_*`, `GENDER_*`, `EMOTION_*`, and `DOMAIN_*` metadata tags at the end. Follow these instructions with extreme precision:

1.  **Preserve Existing Metadata Tags:**
    *   The `AGE_*`, `GENDER_*`, `EMOTION_*`, and `DOMAIN_*` tags at the end of each sentence **must** remain exactly as they are, including the case (e.g., `DOMAIN_GENERAL`).
    *   **Do not** modify, move, delete, or change these existing metadata tags.

2.  **Entity Annotation (Russian Text Only):**
    *   Identify entities **only** within the main Russian transcription part of the sentence (before the metadata).
    *   Use **only** the entity types provided in the `Allowed ENTITY TYPES` list below. Do not invent new types.

3.  **Strict Entity Tagging Format:**
    *   **Tag Insertion:** Place the entity tag **immediately before** the identified Russian entity text. The tag format is `ENTITY_<TYPE>` (e.g., `ENTITY_CITY`, `ENTITY_PERSON_NAME`).
    *   **NO SPACES IN TYPE:** Ensure the `<TYPE>` part of the tag contains **no spaces** (e.g., `PERSON_NAME` is correct, `PERSON_ NAM E` is **incorrect**).
    *   **`END` Tag Placement:** Place the literal string `END` **immediately after** the *complete* Russian entity text.
    *   **Spacing:** There must be exactly **one space** between the `ENTITY_<TYPE>` tag and the start of the entity text. There must be exactly **one space** between the end of the entity text and the `END` tag. There must be exactly **one space** after the `END` tag before the next word begins (unless followed by punctuation).
    *   **Example:** For the Russian entity "Москва", the correct annotation is `ENTITY_CITY Москва END`.
    *   **Crucial `END` Rule:** Only add **one** `END` tag right after the full entity phrase. Do **not** add `END` tags after individual words within a multi-word entity or after words that simply follow an entity.
    *   **Avoid Extra `END` Before Metadata:** Do **not** place an `END` tag just before the `AGE_`, `GENDER_`, `EMOTION_`, `DOMAIN_`, or `INTENT_` tags unless that `END` tag correctly marks the end of an immediately preceding entity.

4.  **Intent Tag:**
    *   Determine the single primary intent of the Russian sentence (e.g., `INFORM`, `QUESTION`, `REQUEST`, `COMMAND`, `GREETING`, `OTHER`).
    *   Append **one** `INTENT_<INTENT_TYPE>` tag at the **absolute end** of the entire string, after all other tags (`AGE_`, `GENDER_`, `EMOTION_`, `DOMAIN_`).

5.  **Output Format:**
    *   Return a **JSON array** of strings. Each string must be a fully annotated sentence following all the rules above.

6.  **Russian Language Specifics:**
    *   Handle **Cyrillic script** correctly.
    *   Ensure correct spacing around standard punctuation (like `.`, `?`, `!`). Remove space before punctuation, ensure space after punctuation if followed by a word.
    *   The final output string must be clean, with single spaces separating words and tags according to the rules.

**Allowed ENTITY TYPES (Use Only These):**
    [ "PERSON_NAME", "ORGANIZATION", "LOCATION", "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP_CODE", "CURRENCY", "PRICE", "DATE",
    "TIME", "DURATION", "APPOINTMENT_DATE", "APPOINTMENT_TIME", "DEADLINE", "DELIVERY_DATE", "DELIVERY_TIME", "EVENT", "MEETING",
    "TASK", "PROJECT_NAME", "ACTION_ITEM", "PRIORITY", "FEEDBACK", "REVIEW", "RATING", "COMPLAINT", "QUESTION", "RESPONSE", "NOTIFICATION_TYPE", "AGENDA",
    "REMINDER", "NOTE", "RECORD", "ANNOUNCEMENT", "UPDATE", "SCHEDULE", "BOOKING_REFERENCE", "APPOINTMENT_NUMBER", "ORDER_NUMBER", "INVOICE_NUMBER", "PAYMENT_METHOD", "PAYMENT_AMOUNT", "BANK_NAME", "ACCOUNT_NUMBER", "CREDIT_CARD_NUMBER", "TAX_ID", "SOCIAL_SECURITY_NUMBER", "DRIVER'S_LICENSE", "PASSPORT_NUMBER", "INSURANCE_PROVIDER", "POLICY_NUMBER", "INSURANCE_PLAN", "CLAIM_NUMBER", "POLICY_HOLDER", "BENEFICIARY",
    "RELATIONSHIP", "EMERGENCY_CONTACT", "PROJECT_PHASE", "VERSION", "DEVELOPMENT_STAGE", "DEVICE_NAME", "OPERATING_SYSTEM", "SOFTWARE_VERSION", "BRAND", "MODEL_NUMBER", "LICENSE_PLATE", "VEHICLE_MAKE", "VEHICLE_MODEL", "VEHICLE_TYPE", "FLIGHT_NUMBER", "HOTEL_NAME", "ROOM_NUMBER", "TRANSACTION_ID", "TICKET_NUMBER", "SEAT_NUMBER", "GATE", "TERMINAL", "TRANSACTION_TYPE", "PAYMENT_STATUS", "PAYMENT_REFERENCE", "INVOICE_STATUS",
    "SYMPTOM", "DIAGNOSIS", "MEDICATION", "DOSAGE", "ALLERGY", "PRESCRIPTION", "TEST_NAME", "TEST_RESULT", "MEDICAL_RECORD", "HEALTH_STATUS", "HEALTH_METRIC", "VITAL_SIGN", "DOCTOR_NAME", "HOSPITAL_NAME", "DEPARTMENT", "WARD", "CLINIC_NAME", "WEBSITE", "URL", "IP_ADDRESS", "MAC_ADDRESS", "USERNAME", "PASSWORD", "LANGUAGE", "CODE_SNIPPET", "DATABASE_NAME", "API_KEY", "WEB_TOKEN", "URL_PARAMETER", "SERVER_NAME", "ENDPOINT", "DOMAIN" ]

**Examples Demonstrating Correct Formatting (Russian - Using Cyrillic script):**

*   **Input:** `"Меня зовут Иван. AGE_30_44 GENDER_MALE EMOTION_NEUTRAL DOMAIN_GENERAL"`
*   **Correct Output:** `"Меня зовут ENTITY_PERSON_NAME Иван END. AGE_30_44 GENDER_MALE EMOTION_NEUTRAL DOMAIN_GENERAL INTENT_INFORM"`

*   **Input:** `"Он живет в Санкт-Петербурге. AGE_45_59 GENDER_MALE EMOTION_NEUTRAL DOMAIN_LOCATION"`
*   **Correct Output:** `"Он живет в ENTITY_CITY Санкт-Петербурге END. AGE_45_59 GENDER_MALE EMOTION_NEUTRAL DOMAIN_LOCATION INTENT_INFORM"`

*   **Input:** `"Встреча состоится 10 мая. AGE_18_29 GENDER_FEMALE EMOTION_NEUTRAL DOMAIN_MEETING"`
*   **Correct Output:** `"Встреча состоится ENTITY_DATE 10 мая END. AGE_18_29 GENDER_FEMALE EMOTION_NEUTRAL DOMAIN_MEETING INTENT_INFORM"`

*   **Input:** `"Найди в нете мульт стальной гигант AGE_UNKNOWN GENDER_UNKNOWN EMOTION_NEUTRAL DOMAIN_GENERAL"`
*   **Correct Output:** `"Найди в нете мульт ENTITY_TASK стальной гигант END AGE_UNKNOWN GENDER_UNKNOWN EMOTION_NEUTRAL DOMAIN_GENERAL INTENT_REQUEST"`

*   **Input:** `"Пожалуйста, говорите тише? AGE_60PLUS GENDER_MALE EMOTION_ANGRY DOMAIN_INSTRUCTION"`
*   **Correct Output:** `"Пожалуйста, говорите тише? AGE_60PLUS GENDER_MALE EMOTION_ANGRY DOMAIN_INSTRUCTION INTENT_REQUEST"`

Provide only the JSON array containing the correctly annotated sentences based precisely on these instructions. Do not include any explanations or introductory text outside the JSON array itself.

**Sentences to Annotate Now:**
{json.dumps(texts_to_annotate, ensure_ascii=False, indent=2)}
'''
    max_retries = 3; retry_delay = 10
    for attempt in range(max_retries):
        try:
            if not genai:
                 print("Error: genai object not available in annotate_batch_texts. Skipping.");
                 raise ConnectionError("Google Generative AI not configured.")

            model = genai.GenerativeModel("gemini-1.5-flash-latest")
            response = model.generate_content(prompt)

            if not response.parts:
                 print(f"Warning: Gemini response blocked (Attempt {attempt+1}). Block reason: {response.prompt_feedback.block_reason}")
                 if attempt == max_retries - 1:
                      print("Error: Gemini response blocked after max retries.")
                      return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_BLOCKED" for t in texts_to_annotate]
                 else:
                      print(f"Retrying annotation due to block (attempt {attempt + 2}/{max_retries})...");
                      time.sleep(retry_delay * (attempt + 1))
                      continue

            assistant_reply = response.text.strip()

            json_match = re.search(r'\[.*\]', assistant_reply, re.DOTALL)
            if json_match: json_str = json_match.group(0)
            else:
                if assistant_reply.startswith("```json"): assistant_reply = assistant_reply[len("```json"):].strip()
                elif assistant_reply.startswith("```"): assistant_reply = assistant_reply[len("```"):].strip()
                if assistant_reply.endswith("```"): assistant_reply = assistant_reply[:-len("```")].strip()
                if assistant_reply.startswith('[') and assistant_reply.endswith(']'): json_str = assistant_reply
                else:
                    print(f"Error: Could not extract valid JSON list using regex (Attempt {attempt+1}). Response snippet:\n---\n{assistant_reply[:500]}\n---")
                    try:
                        potential_json = json.loads(assistant_reply)
                        if isinstance(potential_json, list):
                            print("Warning: Extracted JSON by parsing the whole response string directly.")
                            json_str = assistant_reply
                        else: raise json.JSONDecodeError("Response parsed but is not a JSON list.", assistant_reply, 0)
                    except json.JSONDecodeError:
                         raise json.JSONDecodeError("Response does not appear to contain a JSON list.", assistant_reply, 0)

            try: annotated_sentences_raw = json.loads(json_str)
            except json.JSONDecodeError as json_e:
                print(f"JSON decoding failed specifically on extracted string (Attempt {attempt+1}): {json_e}"); print("Extracted string snippet:", json_str[:500]); raise json_e

            if isinstance(annotated_sentences_raw, list) and len(annotated_sentences_raw) == len(texts_to_annotate):
                processed_sentences = []
                for idx, sentence in enumerate(annotated_sentences_raw):
                     if isinstance(sentence, str): processed_sentences.append(fix_end_tags_and_spacing(sentence))
                     else:
                          print(f"Warning: Non-string item received in annotation list at index {idx}: {sentence}")
                          try: processed_sentences.append(fix_end_tags_and_spacing(texts_to_annotate[idx]) + " ANNOTATION_ERROR_NON_STRING")
                          except IndexError:
                              print(f"Error: Could not map non-string item at index {idx} back to original text.");
                              processed_sentences.append("ANNOTATION_ERROR_UNKNOWN_ORIGINAL")

                if len(processed_sentences) == len(texts_to_annotate): return processed_sentences
                else:
                    print(f"Error: Mismatch after processing non-string elements. Expected {len(texts_to_annotate)}, Got {len(processed_sentences)}")
                    if attempt == max_retries - 1: return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_PROCESSING" for t in texts_to_annotate]
                    else: raise ValueError("Processing error lead to length mismatch.")
            else:
                print(f"Error: API returned invalid list or mismatched length (Attempt {attempt+1}). Expected {len(texts_to_annotate)}, Got {len(annotated_sentences_raw) if isinstance(annotated_sentences_raw, list) else type(annotated_sentences_raw)}")
                if attempt == max_retries - 1: return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_LENGTH" for t in texts_to_annotate]
                else:
                    print(f"Retrying annotation (attempt {attempt + 2}/{max_retries})...");
                    time.sleep(retry_delay * (attempt + 1))

        except json.JSONDecodeError as e:
            print(f"JSON decoding failed (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1: return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_JSON_DECODE" for t in texts_to_annotate]
            print(f"Retrying annotation..."); time.sleep(retry_delay * (attempt + 1))
        except Exception as e:
            print(f"Error calling/processing Generative AI (Attempt {attempt+1}/{max_retries}): {e}")
            import traceback; traceback.print_exc()
            if "rate limit" in str(e).lower(): print("Rate limit likely hit."); time.sleep(retry_delay * (attempt + 1) * 5)
            elif "API key not valid" in str(e) or "permission" in str(e).lower():
                print("FATAL: Invalid Google API Key or permission issue.");
                return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_KEY" for t in texts_to_annotate]
            elif "ConnectionError" in str(e):
                 print("FATAL: GenAI not configured properly.");
                 return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_NO_GENAI" for t in texts_to_annotate]
            else: time.sleep(retry_delay * (attempt + 1))
            if attempt == max_retries - 1:
                return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_API_CALL" for t in texts_to_annotate]

    print("Error: Max retries reached for annotation batch.")
    return [fix_end_tags_and_spacing(t) + " ANNOTATION_ERROR_MAX_RETRIES" for t in texts_to_annotate]

# --- *** Main Processing Function for Russian Data *** ---
def process_russian_data(
    input_manifest_path: str,
    base_audio_dir: str,
    output_wav_dir: str,
    output_manifest_path: str,
    batch_size: int = 10
) -> None:
    if not os.path.isdir(base_audio_dir):
        print(f"Error: Base audio directory '{base_audio_dir}' not found.")
        return
    if not os.path.isfile(input_manifest_path):
        print(f"Error: Input manifest file '{input_manifest_path}' not found.")
        return

    os.makedirs(output_wav_dir, exist_ok=True)
    output_parent_dir = os.path.dirname(output_manifest_path)
    os.makedirs(output_parent_dir, exist_ok=True)

    try:
        print(f"Attempting to clear output file: {output_manifest_path}")
        with open(output_manifest_path, 'w', encoding='utf-8') as f_clear: f_clear.write("")
        print(f"Output file {output_manifest_path} cleared successfully.")
    except IOError as e: print(f"Error clearing output file {output_manifest_path}: {e}. Please check permissions."); return

    print("Loading models...")
    emotion_model_name = "superb/hubert-large-superb-er"; emotion_model_info = {}
    try:
        emotion_model_info['model'] = AutoModelForAudioClassification.from_pretrained(emotion_model_name).to(device); emotion_model_info['model'].eval()
        emotion_model_info['feature_extractor'] = AutoFeatureExtractor.from_pretrained(emotion_model_name)
        print("Emotion model loaded.")
    except Exception as e:
        print(f"FATAL Error loading Emotion model: {e}. Cannot proceed without emotion model.")
        return

    age_gender_model = None; age_gender_processor = None
    age_gender_model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    try:
        age_gender_processor = Wav2Vec2Processor.from_pretrained(age_gender_model_name)
        age_gender_model = AgeGenderModel.from_pretrained(age_gender_model_name).to(device); age_gender_model.eval()
        print("Age/Gender prediction model loaded.")
    except Exception as e:
        print(f"Warning: Failed to load age/gender prediction model ({e}). Age/Gender prediction will not be available.")
        age_gender_model = None; age_gender_processor = None

    print("-" * 30)

    total_lines = 0
    try:
        with open(input_manifest_path, 'r', encoding='utf-8') as f_count:
            total_lines = sum(1 for _ in f_count)
        print(f"Found {total_lines} records in {input_manifest_path}.")
    except Exception as e:
        print(f"Error reading input manifest for counting: {e}"); return

    if total_lines == 0: print("Input manifest is empty. Exiting."); return

    processed_records_buffer = []
    batch_num = 0
    files_processed_count = 0
    total_records_saved = 0
    skipped_records_count = 0
    conversion_errors = 0
    audio_load_errors = 0

    try:
        with open(input_manifest_path, 'r', encoding='utf-8') as infile:
            print(f"Starting processing of {total_lines} Russian records...")
            print(f"Original Audio Base: {base_audio_dir}")
            print(f"Converted WAV Output Dir: {output_wav_dir}")
            print(f"Final Manifest Output: {output_manifest_path}")
            print("-" * 30)

            for i, line in enumerate(tqdm(infile, total=total_lines, desc="Processing Records")):
                record_num = i + 1
                signal, sr = None, 16000
                converted_wav_path_relative = None
                full_output_wav_path = None

                try:
                    record = json.loads(line.strip())

                    required_keys = ["audio_filepath", "text", "duration"]
                    missing_keys = [key for key in required_keys if key not in record]
                    if missing_keys:
                        skipped_records_count += 1; continue

                    relative_opus_path = record["audio_filepath"]
                    text = record["text"]
                    duration = record["duration"]

                    if not text or not isinstance(text, str) or not text.strip():
                         skipped_records_count += 1; continue
                    if not isinstance(duration, (int, float)) or duration <= 0:
                         skipped_records_count += 1; continue
                    if not isinstance(relative_opus_path, str) or not relative_opus_path.endswith(".opus"):
                         skipped_records_count += 1; continue

                    full_input_opus_path = os.path.normpath(os.path.join(base_audio_dir, relative_opus_path))
                    relative_wav_path = "/external4/datasets/russian_opensr_cv/" + os.path.splitext(relative_opus_path)[0] + ".wav"
                    full_output_wav_path = os.path.normpath(os.path.join(output_wav_dir, relative_wav_path))
                    converted_wav_path_relative = relative_wav_path

                    if not os.path.isfile(full_input_opus_path):
                        skipped_records_count += 1; continue

                    conversion_success = False
                    if os.path.exists(full_output_wav_path):
                        conversion_success = True
                    else:
                        conversion_success = convert_opus_to_wav(full_input_opus_path, full_output_wav_path)

                    if not conversion_success:
                         conversion_errors += 1; skipped_records_count += 1; continue

                    try:
                        signal, sr_orig = sf.read(full_output_wav_path, dtype='float32')
                        sr = 16000
                        if sr_orig != sr:
                             print(f"  Warning: WAV file {full_output_wav_path} has unexpected sample rate {sr_orig}. Resampling.")
                             signal = librosa.resample(y=signal, orig_sr=sr_orig, target_sr=sr)
                        if signal.ndim > 1: signal = np.mean(signal, axis=1)

                        if signal is None or len(signal) == 0:
                            raise ValueError("Loaded audio signal is empty.")

                    except Exception as load_err:
                        print(f"  Skipping record {record_num}: Error loading converted WAV {full_output_wav_path}: {load_err}")
                        audio_load_errors += 1; skipped_records_count += 1; continue

                    emotion = "UNKNOWN"
                    try:
                        emotion_raw = extract_emotion(signal, sr, emotion_model_info)
                        emotion = emotion_raw.upper()
                        if "ERROR" in emotion or emotion == "NO_AUDIO":
                            emotion = "UNKNOWN"
                    except Exception as emotion_err:
                        print(f"  Error during emotion extraction call for record {record_num}: {emotion_err}"); emotion = "UNKNOWN"

                    age_group = "UNKNOWN"; gender = "UNKNOWN"
                    if age_gender_model and age_gender_processor:
                        raw_age_prediction, gender_logits = predict_age_gender_from_audio(signal, sr, age_gender_processor, age_gender_model)

                        if raw_age_prediction is not None:
                            age_group = get_age_bucket_from_prediction(raw_age_prediction)
                        if gender_logits is not None:
                            gender = get_gender_label_from_prediction(gender_logits)

                    cleaned_text = text.strip()
                    domain = "GENERAL"

                    safe_age = re.sub(r'\s+', '_', age_group.strip())
                    safe_gender = re.sub(r'\s+', '_', gender.strip())
                    safe_emotion = re.sub(r'\s+', '_', emotion.strip())
                    safe_domain = re.sub(r'\s+', '_', domain.strip().upper())

                    initial_text_for_gemini = f"{cleaned_text} AGE_{safe_age} GENDER_{safe_gender} EMOTION_{safe_emotion} DOMAIN_{safe_domain}"

                    buffer_item = {
                        "audio_filepath": converted_wav_path_relative,
                        "duration": duration,
                        "text_for_gemini": initial_text_for_gemini,
                        "original_opus_path": relative_opus_path,
                    }
                    processed_records_buffer.append(buffer_item)
                    files_processed_count += 1

                except json.JSONDecodeError:
                    skipped_records_count += 1
                except Exception as e:
                    print(f"  Unexpected error processing record {record_num} (Line: {line.strip()}): {e}")
                    import traceback
                    traceback.print_exc()
                    skipped_records_count += 1
                finally:
                    if 'signal' in locals() and signal is not None : del signal
                    gc.collect()

                if len(processed_records_buffer) >= batch_size or (record_num == total_lines and len(processed_records_buffer) > 0):
                    batch_num += 1; current_batch_size = len(processed_records_buffer)
                    print(f"\n--- Annotating Batch {batch_num}/{-(total_lines // -batch_size)} ({current_batch_size} records) ---")
                    texts_to_annotate = [rec["text_for_gemini"] for rec in processed_records_buffer]
                    annotated_texts = annotate_batch_texts(texts_to_annotate)

                    if len(annotated_texts) != current_batch_size:
                         print(f"  FATAL BATCH ERROR: Annotation count mismatch! Expected {current_batch_size}, Got {len(annotated_texts)}. Skipping save for this batch.")
                         skipped_records_count += current_batch_size
                    else:
                        try:
                            lines_written_in_batch = 0
                            with open(output_manifest_path, 'a', encoding='utf-8') as f_out:
                                for record_data, annotated_text in zip(processed_records_buffer, annotated_texts):
                                    final_record = {
                                        "audio_filepath": record_data["audio_filepath"],
                                        "text": annotated_text,
                                        "duration": record_data["duration"]
                                    }
                                    if "ANNOTATION_ERROR" in annotated_text:
                                        pass
                                    json_str = json.dumps(final_record, ensure_ascii=False)
                                    f_out.write(json_str + '\n'); lines_written_in_batch += 1
                            total_records_saved += lines_written_in_batch
                            print(f"--- Batch {batch_num} processed and saved ({lines_written_in_batch} records). Total saved: {total_records_saved} ---")
                        except IOError as io_err: print(f"  Error writing batch {batch_num} to {output_manifest_path}: {io_err}")
                        except Exception as write_err: print(f"  Unexpected error writing batch {batch_num}: {write_err}")

                    processed_records_buffer = []
                    del texts_to_annotate, annotated_texts
                    torch.cuda.empty_cache(); gc.collect()

    except Exception as E:
         print(f"\n--- A CRITICAL ERROR OCCURRED DURING PROCESSING ---")
         print(f"Error: {E}")
         import traceback
         traceback.print_exc()
         print(f"Processing stopped prematurely.")

    print("\n" + "="*30); print("Processing Finished.")
    print(f"Total records found in input manifest: {total_lines}")
    print(f"Records successfully processed (before annotation): {files_processed_count}")
    print(f"Records skipped (missing data, path error, empty text, etc.): {skipped_records_count}")
    print(f"Audio conversion errors: {conversion_errors}")
    print(f"Audio loading errors (post-conversion): {audio_load_errors}")
    print(f"Total records saved to {output_manifest_path}: {total_records_saved}")
    print("Note: Age/Gender determined via prediction. Domain set to GENERAL.")
    print("Note: Annotation errors within saved records are flagged in the 'text' field.")
    print("="*30)

# --- Main Execution ---
if __name__ == "__main__":
    RUSSIAN_DATASET_BASE_DIR = "/external4/datasets/russian_opensr_train"
    INPUT_MANIFEST = os.path.join(RUSSIAN_DATASET_BASE_DIR, "manifest_raw.jsonl")

    OUTPUT_WAV_DIR = "/external4/datasets/russian_opensr_cv"
    FINAL_OUTPUT_MANIFEST = os.path.join(OUTPUT_WAV_DIR, "manifest_processed_annotated.jsonl")

    PROCESSING_BATCH_SIZE = 10

    local_ffmpeg_dir = "/hydra2-prev/home/compute/workspace_himanshu/ffmpeg-7.0.2-amd64-static"
    ffmpeg_path = os.path.join(local_ffmpeg_dir, "ffmpeg")

    if not os.path.exists(ffmpeg_path):
        print(f"ERROR: Local ffmpeg executable not found at {ffmpeg_path}.")
        system_ffmpeg_path = "/usr/bin/ffmpeg"
        print(f"Attempting to fall back to system ffmpeg at: {system_ffmpeg_path}")
        if os.path.exists(system_ffmpeg_path) and os.access(system_ffmpeg_path, os.X_OK):
            print(f"Using system ffmpeg: {system_ffmpeg_path}")
            ffmpeg_path = system_ffmpeg_path
        else:
            print("FATAL: Neither local nor system ffmpeg found or executable. Cannot proceed.")
            exit(1)
    elif not os.access(ffmpeg_path, os.X_OK):
         print(f"ERROR: Local ffmpeg found at {ffmpeg_path} but is not executable.")
         print(f"Try running: chmod +x {ffmpeg_path}")
         exit(1)
    else:
         print(f"Using local ffmpeg: {ffmpeg_path}")

    print("--- Russian Data Processing & Annotation Script ---")
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"): print("ERROR: GOOGLE_API_KEY not set."); exit(1)
    if genai is None: print("ERROR: Google Generative AI failed to configure."); exit(1)

    try:
        print("Checking ffmpeg version...")
        result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True, check=True, encoding='utf-8')
        print(result.stdout.splitlines()[0])
        print("ffmpeg version check successful.")
    except FileNotFoundError:
        print(f"FATAL ERROR: ffmpeg command failed. Path '{ffmpeg_path}' seems invalid despite earlier checks.")
        exit(1)
    except subprocess.CalledProcessError as e:
        print(f"FATAL ERROR: ffmpeg -version command failed with error code {e.returncode}.")
        print(f"Stderr: {e.stderr}")
        exit(1)
    except Exception as e:
        print(f"Warning: ffmpeg version check produced an unexpected error: {e}")

    if not os.path.isfile(INPUT_MANIFEST):
        print(f"ERROR: Input manifest file not found: {INPUT_MANIFEST}")
        exit(1)
    if not os.path.isdir(RUSSIAN_DATASET_BASE_DIR):
         print(f"ERROR: Base Russian dataset directory not found: {RUSSIAN_DATASET_BASE_DIR}")
         exit(1)

    output_dir_for_manifest = os.path.dirname(FINAL_OUTPUT_MANIFEST)
    try:
        os.makedirs(output_dir_for_manifest, exist_ok=True)
        print(f"Output directory created/exists: {output_dir_for_manifest}")
    except OSError as e:
        print(f"Error creating output directory {output_dir_for_manifest}: {e}")
        exit(1)

    print("\nStarting Russian Data Processing (WAV Conversion, Prediction, Annotation)...")
    start_time = time.time()

    process_russian_data(
        input_manifest_path=INPUT_MANIFEST,
        base_audio_dir=RUSSIAN_DATASET_BASE_DIR,
        output_wav_dir=OUTPUT_WAV_DIR,
        output_manifest_path=FINAL_OUTPUT_MANIFEST,
        batch_size=PROCESSING_BATCH_SIZE
    )

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    print(f"Converted WAV files saved in: {OUTPUT_WAV_DIR}")
    print(f"Final annotated manifest saved to: {FINAL_OUTPUT_MANIFEST}")