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
import time

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
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable not found.")
    print("Google Generative AI configured successfully.")
except Exception as e:
    print(f"Error configuring or testing Google Generative AI: {e}. Annotation step will likely fail.")
    genai = None

# --- Age/Gender Model Definition (Script 1 - unchanged) ---
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

# --- Emotion Extraction Function (Script 1 - unchanged) ---
def extract_emotion(audio_data: np.ndarray, sampling_rate: int = 16000, model_info: dict = None) -> str:
    if model_info is None or 'model' not in model_info or 'feature_extractor' not in model_info:
        print("Error: Emotion model not loaded correctly.")
        return "ErrorLoadingModel"
    if audio_data is None or len(audio_data) == 0:
        return "No_Audio"
    if len(audio_data) < sampling_rate * 0.1:
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


# --- Data Structures ---
@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    speaker: str
    age: float
    gender: str
    transcription: str # This will now store lowercase transcription
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
        transcription = segment.transcription.strip() # Transcription is already lowercase
        metadata = f"AGE_{age_bucket} GENDER_{gender_text} EMOTION_{emotion_text}"
        return f"{transcription.strip()} {metadata.strip()}"

    @staticmethod
    def get_age_bucket(age: float) -> str:
        actual_age = round(age * 100)
        age_brackets = [
            (18, "0_18"), (30, "18_30"), (45, "30_45"),
            (60, "45_60"), (float('inf'), "60PLUS")
        ]
        for threshold, bracket in age_brackets:
            if actual_age < threshold: return bracket
        return "60PLUS"

# --- File Handling (MODIFIED - get_transcription) ---
def get_file_pairs(audio_dir: str, text_dir: str) -> List[Tuple[str, str]]:
    try:
        audio_files_list = [f for f in os.listdir(audio_dir) if f.lower().endswith('.wav')]
        text_files_list = [f for f in os.listdir(text_dir) if f.lower().endswith('.txt')]
        audio_files = {os.path.splitext(f)[0]: os.path.join(audio_dir, f) for f in audio_files_list}
        text_files = {os.path.splitext(f)[0]: os.path.join(text_dir, f) for f in text_files_list}
        pairs = []
        for base_name, audio_path in audio_files.items():
            if base_name in text_files:
                pairs.append((audio_path, text_files[base_name]))
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
        with open(text_file, 'r', encoding='utf-8') as f:
            return f.read().strip().lower() # MODIFIED: Convert transcription to lowercase
    except Exception as e:
        print(f"Error reading transcription file {text_file}: {str(e)}")
        return ""

# --- AI Annotation Functions ---

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
    text = re.sub(r'\s+END\s+END\b', ' END', text)

    pattern_add_end = r'(ENTITY_[A-Z0-9_]+\s+\S.*?)(?<!\sEND)(?=\s+(\bAGE_|\bGENDER_|\bEMOTION_|\bINTENT_|\bENTITY_)|$)'
    text = re.sub(pattern_add_end, r'\1 END', text)

    text = re.sub(r'(ENTITY_[A-Z0-9_]+)(\S)', r'\1 \2', text)
    text = re.sub(r'(\S)(END\b)', r'\1 \2', text)

    text = re.sub(r'\s+([?!:;,.])', r'\1', text)
    text = re.sub(r'([?!:;,.])(\w)', r'\1 \2', text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text

def annotate_batch_texts(texts_to_annotate: List[str]):
    if not genai:
        print("Error: Google Generative AI not configured. Skipping annotation.")
        return texts_to_annotate
    if not texts_to_annotate:
        return []

    # The prompt is kept as is; Gemini should handle lowercase English text for entity recognition.
    # The examples in the prompt use mixed case for entities, but the instruction for entity identification itself is general.
    prompt = f'''You are an expert linguistic annotator for English text.
You will receive a list of English sentences. Each sentence already includes metadata tags (AGE_*, GENDER_*, EMOTION_*) at the end.
The English transcription part will be in lowercase.

Your task is crucial and requires precision:
1.  **PRESERVE EXISTING TAGS:** Keep the `AGE_`, `GENDER_`, and `EMOTION_` tags exactly as they appear at the end of each sentence. DO NOT modify or move them.
2.  **ENTITY ANNOTATION (English Text Only):** Identify entities ONLY within the lowercase English transcription part of the sentence. Use ONLY the entity types from the provided list.
3.  **ENTITY TAG FORMAT (VERY IMPORTANT):**
    *   Insert the tag **BEFORE** the entity: `ENTITY_<TYPE>` (e.g., `ENTITY_CITY`, `ENTITY_PERSON_NAME`).
    *   **NO SPACES** are allowed within the `<TYPE>` part (e.g., use `PERSON_NAME`, NOT `PERSON_ NAM E`).
    *   Immediately **AFTER** the English entity text, add a single space followed by `END`.
    *   Example (assuming input "i saw maria in london yesterday."): `... i saw ENTITY_PERSON_NAME maria END in ENTITY_CITY london END yesterday. ...`
    *   **DO NOT** add an `END` tag just before the `AGE_`, `GENDER_`, `EMOTION_` tags unless it belongs to a preceding entity.
4.  **INTENT TAG:** Determine the single primary intent of the English transcription (e.g., INFORM, QUESTION, REQUEST, COMMAND, GREETING, etc.). Add ONE `INTENT_<INTENT_TYPE>` tag at the absolute end of the entire string, AFTER all other tags.
5.  **OUTPUT FORMAT:** Return a JSON array of strings, where each string is a fully annotated sentence adhering to all rules.
6.  **ENGLISH SPECIFICS:** Handle English script, punctuation (like '.', '?', '!'), and spacing correctly according to standard English rules. Ensure proper spacing around the inserted tags. The entity text you select from the input will be lowercase; preserve this casing in your output within the `ENTITY_TAG ... END` block.

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

**Example Input String (now lowercase):**
"i saw maria in london yesterday. AGE_30_45 GENDER_FEMALE EMOTION_NEUTRAL"

**CORRECT Example Output String (entities are lowercase as in input):**
"i saw ENTITY_PERSON_NAME maria END in ENTITY_CITY london END yesterday. AGE_30_45 GENDER_FEMALE EMOTION_NEUTRAL INTENT_INFORM"

**INCORRECT Example Output String (Spaces in Tag):**
"i saw ENTITY_PERSON_ NAM E maria END in ENTITY_CIT Y london END yesterday. AGE_30_45 GENDER_FEMALE EMOTION_NEUTRAL INTENT_INFORM"

**INCORRECT Example Output String (Extra END before metadata):**
"i saw ENTITY_PERSON_NAME maria END in ENTITY_CITY london END yesterday. END AGE_30_45 GENDER_FEMALE EMOTION_NEUTRAL INTENT_INFORM"


**Sentences to Annotate Now:**
{json.dumps(texts_to_annotate, ensure_ascii=False, indent=2)}
'''

    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash-latest")
            response = model.generate_content(prompt)
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
                          processed_sentences.append(sentence)
                return processed_sentences
            else:
                print(f"Error: API did not return a valid list or mismatched length. Expected {len(texts_to_annotate)}, Got {len(annotated_sentences_raw) if isinstance(annotated_sentences_raw, list) else 'Invalid Type'}")
                if attempt == max_retries - 1:
                    return texts_to_annotate
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
            else:
                 time.sleep(retry_delay * (attempt + 1))
            if attempt == max_retries - 1: return texts_to_annotate

    print("Error: Max retries reached for annotation.")
    return texts_to_annotate


# --- Main Processing Function ---
def process_audio_and_annotate(base_dir: str, output_jsonl_path: str, batch_size: int = 10) -> None:
    output_dir = os.path.dirname(output_jsonl_path)
    os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"Attempting to clear output file: {output_jsonl_path}")
        with open(output_jsonl_path, 'w') as f_clear:
            f_clear.write("")
        print(f"Output file {output_jsonl_path} cleared successfully.")
    except IOError as e:
        print(f"Error clearing output file {output_jsonl_path}: {e}. Please check permissions.")
        return

    print("Loading models...")
    age_gender_model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    try:
        processor = Wav2Vec2Processor.from_pretrained(age_gender_model_name)
        age_gender_model = AgeGenderModel.from_pretrained(age_gender_model_name).to(device)
        age_gender_model.eval()
        print("Age/Gender model loaded.")
    except Exception as e:
        print(f"FATAL Error loading Age/Gender model: {e}. Exiting.")
        return

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

    audio_dir = os.path.join(base_dir, "wav")
    text_dir = os.path.join(base_dir, "txt")

    if not os.path.exists(audio_dir) or not os.path.exists(text_dir):
        print(f"Error: Audio ({audio_dir}) or text ({text_dir}) directory not found in {base_dir}")
        return

    print(f"Processing files from:\n  Audio: {audio_dir}\n  Text:  {text_dir}")
    print(f"Output will be saved to: {output_jsonl_path}")
    print("-" * 30)

    file_pairs = get_file_pairs(audio_dir, text_dir)
    if not file_pairs:
        print("No matching files found. Exiting.")
        return

    total_files = len(file_pairs)
    print(f"Found {total_files} audio-text pairs to process.")

    processed_records_buffer = []
    batch_num = 0
    files_processed_count = 0
    total_records_saved = 0

    for i, (audio_path, text_path) in enumerate(file_pairs):
        print(f"\nProcessing file {i+1}/{total_files}: {os.path.basename(audio_path)}")
        try:
            transcription = get_transcription(text_path) # This will be lowercase
            if not transcription:
                print(f"  Skipping: Empty or unreadable transcription in {text_path}")
                continue

            try:
                signal, sr = librosa.load(audio_path, sr=16000)
                if signal is None or len(signal) == 0:
                    print(f"  Skipping: Failed to load or empty audio in {audio_path}")
                    continue
                duration = round(len(signal) / sr, 2)
                if duration < 0.1:
                     print(f"  Skipping: Audio too short ({duration:.2f}s) in {audio_path}")
                     continue
            except Exception as load_err:
                 print(f"  Skipping: Error loading audio {audio_path}: {load_err}")
                 continue

            try:
                inputs = processor(signal, sampling_rate=sr, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device)
                with torch.no_grad():
                    _, logits_age, logits_gender = age_gender_model(input_values)
                age = logits_age.cpu().numpy().item()
                gender_idx = torch.argmax(logits_gender, dim=-1).cpu().numpy().item()
                gender_map = {0: "FEMALE", 1: "MALE", 2: "OTHER"}
                gender = gender_map.get(gender_idx, "UNKNOWN")
            except Exception as age_gender_err:
                print(f"  Error during Age/Gender extraction: {age_gender_err}")
                age = -1.0
                gender = "ERROR"

            emotion = extract_emotion(signal, sr, emotion_model_info)

            speaker_base = os.path.splitext(os.path.basename(audio_path))[0]
            speaker_parts = speaker_base.split('_')
            speaker = speaker_parts[0] if speaker_parts else speaker_base


            segment_data = AudioSegment(
                start_time=0, end_time=duration, speaker=speaker, age=age,
                gender=gender, transcription=transcription, emotion=emotion,
                chunk_filename=os.path.basename(audio_path), duration=duration
            )
            chunk = ChunkData(segments=[segment_data], filepath=os.path.abspath(audio_path))
            initial_formatted_text = chunk.get_formatted_text()

            record = {
                "audio_filepath": chunk.filepath,
                "duration": duration,
                "initial_text": initial_formatted_text,
                "raw_age_output": age,
                "raw_gender_prediction": gender,
                "raw_emotion_prediction": emotion,
                "speaker_id": speaker,
            }
            processed_records_buffer.append(record)
            files_processed_count += 1

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
                                    "text": annotated_text,
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
                torch.cuda.empty_cache()
                gc.collect()

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")
            break
        except Exception as e:
            print(f"  FATAL ERROR processing file {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*30)
    print(f"Processing Finished.")
    print(f"Total files processed attempt: {files_processed_count}/{total_files}")
    print(f"Total records saved to {output_jsonl_path}: {total_records_saved}")
    print("="*30)


# --- Main Execution ---
if __name__ == "__main__":
    BASE_AUDIO_TEXT_DIR = "/external3/databases/NPTEL2020-Indian-English-Speech-Dataset/nptel-test"
    FINAL_OUTPUT_JSONL = "/external3/databases/NPTEL2020-Indian-English-Speech-Dataset/nptel-test/nptel_english_annotated_v2_lowercase.jsonl" # Changed output file name
    PROCESSING_BATCH_SIZE = 10

    print("Starting English Audio Processing and Annotation Workflow (with lowercase transcriptions)...")
    print(f"Input Base Directory: {BASE_AUDIO_TEXT_DIR}")
    print(f"Final Output File: {FINAL_OUTPUT_JSONL}")
    print(f"Processing Batch Size: {PROCESSING_BATCH_SIZE}")
    print("-" * 30)

    if not genai:
         print("ERROR: Google Generative AI failed to configure. Exiting.")
         exit(1)

    process_audio_and_annotate(
        base_dir=BASE_AUDIO_TEXT_DIR,
        output_jsonl_path=FINAL_OUTPUT_JSONL,
        batch_size=PROCESSING_BATCH_SIZE
    )

    print("Workflow complete.")