import os
import re
import sys
import json
import traceback
import glob
import pandas as pd
from pathlib import Path, PurePath
from pydub import AudioSegment
from tqdm import tqdm
import pprint
import string
import concurrent.futures
import torch
import torch.nn.functional as F
# from transformers import (AutoTokenizer, AutoModelForTokenClassification, 
#                           TokenClassificationPipeline, pipeline, AutoConfig, 
#                           Wav2Vec2FeatureExtractor)
# from transformers import AutoModelForAudioClassification

import torch
import librosa
import soundfile as sf
from datasets import load_dataset
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

#from nemo_text_processing.text_normalization.normalize import Normalizer
#import nemo.collections.nlp as nemo_nlp
#from AudioEmotionClassification.models import (Wav2Vec2ForSpeechClassification, 
#                                              HubertForSpeechClassification)

import librosa
import openai

lang_code_map = {"en": "English", "fr": "French", "de": "German", "it": "Italian", "es": "Spanish"}

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part

from annotate_sentence.gcp_custom_llm import annotate_sentences_custom_vertex
from annotate_sentence.gcp_gemini import annotate_sentences_vertexAI

PROJECT_ID = "stream2action"  # Replace with your actual Google Cloud project ID

# Initialize Vertex AI with the project ID and location
vertexai.init(project=PROJECT_ID, location="us-central1")

# Load the Gemini model
model = GenerativeModel("gemini-1.5-flash-002")

class ModelHead(nn.Module):
    r"""Classification head."""

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
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)

        return hidden_states, logits_age, logits_gender

def age_process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict age and gender or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = age_processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = age_model(y)
        if embeddings:
            y = y[0]
        else:
            y = torch.hstack([y[1], y[2]])

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y

def process_and_interpret(signal, sampling_rate):
    # Get the raw predictions
    predictions = age_process_func(signal, sampling_rate)
    
    # Extract age (predictions[0][0] is between 0 and 1, representing 0-100 years)
    age = predictions[0][0] * 100  # Scale to actual age
    
    # Extract gender probabilities
    female_prob = predictions[0][1]
    male_prob = predictions[0][2]
    child_prob = predictions[0][3]
    
    # Get the most likely gender
    gender_probs = {
        'female': female_prob,
        'male': male_prob,
        'child': child_prob
    }
    predicted_gender = max(gender_probs, key=gender_probs.get)
    
    return {
        'age': round(age, 1),
        'predicted_gender': predicted_gender,
        'gender_probabilities': {
            'female': round(female_prob, 3),
            'male': round(male_prob, 3),
            'child': round(child_prob, 3)
        }
    }

def map_to_array(example):
    speech, _ = librosa.load(example["file"], sr=16000, mono=True)
    example["speech"] = speech
    return example

print("HELLO HELLO")

'''
TODO:
1. all labels used are written properly to the taglist file
2. Multiprocessing for the tsv files
'''



# def convert_mp3_to_wav(mp3_file_path, wav_file_path, sample_rate=16000):
#     # Load the MP3 file
#     audio = AudioSegment.from_mp3(mp3_file_path)
#     duration_ms = len(audio)
#     duration_seconds = duration_ms / 1000.0

#     # Export as WAV
#     audio = audio.set_frame_rate(sample_rate)
    
#     audio.export(wav_file_path, format="wav")

#     return duration_seconds


def normalize(text):

    text = text.lower()
    normalized = normalizer.normalize(text, verbose=True, punct_post_process=True)
    normalized = [normalized]
    norm_punctuated = punctuator.add_punctuation_capitalization(normalized)[0]
    return norm_punctuated


def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Emotion": emotion_config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in
               enumerate(scores)]
    return outputs

def speech_file_to_array_fn(path, sampling_rate):
    """Convert audio file to array using librosa instead of torchaudio"""
    speech_array, _ = librosa.load(path, sr=sampling_rate)
    return speech_array

def convert_mp3_to_wav(mp3_file_path, wav_file_path, sample_rate=16000):
    """Convert MP3 to WAV using librosa instead of pydub"""
    try:
        # Load the audio file with librosa
        y, sr = librosa.load(mp3_file_path, sr=sample_rate)
        
        # Get duration in seconds
        duration_seconds = librosa.get_duration(y=y, sr=sr)
        
        # Save as WAV
        import soundfile as sf
        sf.write(wav_file_path, y, sr, format='WAV')
        
        return duration_seconds
    except Exception as e:
        print(f"Error converting {mp3_file_path}: {e}")
        return None

# Update the detect_emotion function to use GPU
def detect_emotion(audio_path):
    """Detect emotion in audio file using GPU"""
    try:
        speech, _ = librosa.load(audio_path, sr=16000, mono=True)
        
        inputs = feature_extractor(speech, sampling_rate=16000, padding=True, return_tensors="pt")
        # Move input tensors to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():  # Add this for inference
            logits = emotion_classifier(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            labels = [emotion_classifier.config.id2label[_id] for _id in predicted_ids.tolist()]

        print("Emotion", labels)
        return labels[0]
    except Exception as e:
        print(f"Error detecting emotion for {audio_path}: {e}")
        return "neutral"  # default fallback

def get_classification_labels_hf(model, raw_wav, sampling_rate=16000, score=50.0):

    #get mean/std
    mean = model.config.mean
    std = model.config.std

    #normalize the audio by mean/std
    raw_wav_tensor = torch.tensor(raw_wav).float().to(model.device)
    norm_wav = (raw_wav_tensor - mean) / (std+0.000001)

    #generate the mask
    mask = torch.ones(1, len(norm_wav))
    mask.to(model.device)

    #batch it (add dim)
    wavs = torch.tensor(norm_wav).unsqueeze(0)

    #predict
    with torch.no_grad():
        pred = model(wavs, mask)
    
    id2label = model.config.id2label
      
    #print(pred)
    #{0: 'Angry', 1: 'Sad', 2: 'Happy', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Contempt', 7: 'Neutral'}
    #tensor([[0.0015, 0.3651, 0.0593, 0.0315, 0.0600, 0.0125, 0.0319, 0.4382]])

    #convert logits to probability
    probabilities = torch.nn.functional.softmax(pred, dim=1)
    
    return id2label, probabilities
    
    
def write_taglist(taglist,filename):

    taglist = "\n".join(taglist)

    with open(filename, 'w') as f:
        f.write(taglist)
        f.write("\n")
    f.close()

def process_tsv(tsvfile, audioclips, audioclipswav, manifestfile, taglistfile, checkpoint_file, lang_code):
    
    tsvfile = pd.read_csv(tsvfile, sep="\t")


    # Check if a checkpoint file exists
    if os.path.exists(checkpoint_file):
        # If it exists, read the last processed row index from the checkpoint file
        with open(checkpoint_file, 'r') as checkpoint:
            last_processed_row = int(checkpoint.read())
    else:
        # If it doesn't exist, start from the beginning
        last_processed_row = 0


    print("Output Manifest File Path:", manifestfile)
    manifest = open(str(manifestfile),'a')
    #data_top = tsvfile.columns.values

    #print(data_top)
    taglist = []
    batch_text = []
    batch_emotion = []
    batch_langid = []
    batch_meta = []
    
    for index, row in tqdm(enumerate(tsvfile.iterrows()), total=len(tsvfile), initial=last_processed_row):
        
        # Skip rows that have already been processed
        if index < last_processed_row:
            continue
        
        row = row[1]

        
        print("yeah row with meta")
        audiofile = audioclips / Path(row['path'])
        wavfilepath = audioclipswav / PurePath(row['path'].split(".")[0]+".wav")
        
        duration = convert_mp3_to_wav(audiofile, wavfilepath)
        
        os.system("rm "+str(audiofile))
        
        
        text = row['sentence']
        #text = normalize(text)
        batch_text.append(text)

        emotion_label = detect_emotion(str(wavfilepath))
        print("Emotion Label:", emotion_label)
        batch_emotion.append(emotion_label)
        #batch_meta.append(str(row['gender']) + " " + str(row['age']) + " " + str(row['variant']))
    
    
        if len(batch_text) == 10:
            
            batch_text_annotated = annotate_sentences(batch_text)
            
            if len(batch_text_annotated) == len(batch_text):
                print("Batch Text Annotated:", batch_text_annotated)
                for i in range(len(batch_text)):
                    
            
                    wavfilepath = str(wavfilepath)

                    sample_dict = {}
                    sample_dict['duration'] = duration
                    sample_dict['audio_filepath'] = wavfilepath
                    sample_dict['text'] = batch_text_annotated[i]
                    sample_dict['emotion'] = batch_emotion[i]
                    sample_dict['langid'] = lang_code
                    #sample_dict['meta'] = batch_meta[i]
                    sample_dict['tasks'] = ["transcription", "keyphrases","speaker-meta"]
                    sample_dict['instruction'] = "Transcribe and mark keyphrases and speaker metadata"
                    json.dump(sample_dict, manifest, ensure_ascii=False)
                    manifest.write("\n") 
                
            batch_text = []
            batch_emotion = []
            batch_meta = []
            batch_langid = []
            

        # Update the checkpoint with the current row index
        with open(checkpoint_file, 'w') as checkpoint:
            checkpoint.write(str(index))

    manifest.close()
    
def process_manifest(input_manifest_file, manifestfile, taglistfile, checkpoint_file, lang_code):
    print("Processing manifest file:", input_manifest_file)
    input_manifest_file = open(input_manifest_file, 'r')
    
    batch_text = []
    batch_emotion = []
    batch_langid = []
    batch_meta = []
    
    output_manifest_file = open(manifestfile, 'w')
    
    print("Output Manifest File Path:", manifestfile)
        
    for sample in input_manifest_file:
        sample = json.loads(sample)
        text = sample['text']
        audiofile = sample['audio_filepath']
        duration = sample['duration']
        
        emotion_label = detect_emotion(audiofile).upper()
        #emotion_label = "NEUTRAL"
        
        batch_text.append(text)
        batch_emotion.append(emotion_label)
        batch_langid.append(lang_code)
        
        signal, sample_rate = sf.read(audiofile)

        #print("Audio File:", audiofile)
        #print("detecting age")
        
        result = process_and_interpret(signal, sample_rate)
        age = str(int(result['age']))
        gender = result['predicted_gender'].upper()
        #age = "30"
        #gender = "MALE"
        batch_meta.append("GENDER_"+gender+" AGE_"+age)
        
        if len(batch_text) == 20:
            print("\n--- Processing Batch ---")
            print("Input Batch Text:", json.dumps(batch_text, indent=2))
            
            try:
                
                # batch_text_annotated = annotate_sentences_custom_vertex(
                #     sentences=batch_text,
                #     project_id="495570340582",
                #     endpoint_id="3407238100408074240"
                # )

                batch_text_annotated = annotate_sentences_vertexAI(batch_text)
                print("Successfully received annotations")
                print("Annotated Batch Text:", json.dumps(batch_text_annotated, indent=2))
                
                if not isinstance(batch_text_annotated, list):
                    print("Warning: annotate_sentences did not return a list")
                    print(f"Received type: {type(batch_text_annotated)}")
                    continue
                
                if len(batch_text_annotated) != len(batch_text):
                    print(f"Warning: Length mismatch - Input: {len(batch_text)}, Output: {len(batch_text_annotated)}")
                    continue
                
                for i in range(len(batch_text)):
                    sample_dict = {}
                    sample_dict['duration'] = duration
                    sample_dict['audio_filepath'] = audiofile
                    sample_dict['text'] = batch_text_annotated[i]
                    sample_dict['emotion'] = batch_emotion[i]
                    sample_dict['meta'] = batch_meta[i]
                    sample_dict['langid'] = lang_code
                    sample_dict['tasks'] = ["transcription", "keyphrases","speaker-meta"]
                    sample_dict['instruction'] = "Transcribe and mark keyphrases and speaker metadata"
                    
                    print(f"\nWriting sample {i+1}/{len(batch_text)}:")
                    print(json.dumps(sample_dict, indent=2))
                    
                    json.dump(sample_dict, output_manifest_file, ensure_ascii=False)
                    output_manifest_file.write("\n")
                
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                print("Traceback:", traceback.format_exc())
            
            print("--- Batch Processing Complete ---\n")
            batch_text = []
            batch_emotion = []
            batch_meta = []
            batch_langid = []    
        
    output_manifest_file.close()

def annotate_sentences_vertexAI(sentences):
    """
    Annotates sentences using Google's Vertex AI Gemini model.
    """
    print("\nStarting sentence annotation with Vertex AI")
    
    try:
        # Initialize Vertex AI
        vertexai.init(project="stream2action", location="us-central1")
        model = GenerativeModel("gemini-1.5-flash-002")
        
        prompt = f'''
        Given a list of sentences in English, annotate each sentence individually with the appropriate entity tags from the provided list. The sentences may relate to various actions such as managing tasks, controlling devices, sending notifications, scheduling events, updating information, making purchases, or offering assistance.

        **Instructions:**

        - Annotate each sentence separately with entity tags based on the entities in the provided list.
        - Use the format `ENTITY_<type>` to start each entity and `END` to close it.
        - For each sentence, add a relevant intent label at the end in the format `INTENT_<type>`, where `<type>` represents the sentence's main action or goal.
        - Only use the entity types provided in the list below.
        - Do not add any additional text, explanations, or comments.
        - Ensure the output is a JSON array containing only the annotated sentences in the same order as the input sentences, without any markdown or formatting.

        **Entities**:

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
            "API_KEY", "WEB_TOKEN", "URL_PARAMETER", "SERVER_NAME", "ENDPOINT", "DOMAIN", 
            
            "PRODUCT", "SERVICE", "CATEGORY", "BRAND", "ORDER_STATUS", "DELIVERY_METHOD", "RETURN_STATUS", "WARRANTY_PERIOD", 
            "CANCELLATION_REASON", "REFUND_AMOUNT", "EXCHANGE_ITEM", "GIFT_OPTION", "GIFT_MESSAGE", 
            
            "FOOD_ITEM", "DRINK_ITEM", "CUISINE", "MENU_ITEM", "ORDER_NUMBER", "DELIVERY_ESTIMATE", "RECIPE", "INGREDIENT", 
            "DISH_NAME", "PORTION_SIZE", "COOKING_TIME", "PREPARATION_METHOD", 
            
            "AGE", "GENDER", "NATIONALITY", "RELIGION", "MARITAL_STATUS", "OCCUPATION", "EDUCATION_LEVEL", "DEGREE", 
            "SKILL", "EXPERIENCE", "YEARS_OF_EXPERIENCE", "CERTIFICATION", 
            
            "MEASUREMENT", "DISTANCE", "WEIGHT", "HEIGHT", "VOLUME", "TEMPERATURE", "SPEED", "CAPACITY", "DIMENSION", "AREA", 
            "SHAPE", "COLOR", "MATERIAL", "TEXTURE", "PATTERN", "STYLE", 
            
            "WEATHER_CONDITION", "TEMPERATURE_SETTING", "HUMIDITY_LEVEL", "WIND_SPEED", "RAIN_INTENSITY", "AIR_QUALITY", 
            "POLLUTION_LEVEL", "UV_INDEX", 
            
            "QUESTION_TYPE", "REQUEST_TYPE", "SUGGESTION_TYPE", "ALERT_TYPE", "REMINDER_TYPE", "STATUS", "ACTION", "COMMAND"
        ]

        **Example**:

        Input Sentences:
        [
            "Can you set up a meeting with John at 3 PM tomorrow?",
            "Play some jazz music in the living room.",
            "Schedule a delivery for October 15th at 10 AM.",
            "Please book a table at the Italian restaurant for two at 7 PM.",
            "Remind me to take my medication at 9 AM."
        ]

        Expected Output:
        [
            "Can you ENTITY_ACTION set up END a ENTITY_MEETING meeting END with ENTITY_PERSON_NAME John END at ENTITY_TIME 3 PM END on ENTITY_DATE tomorrow END? INTENT_SCHEDULE_MEETING",
            
            "ENTITY_ACTION Play END some ENTITY_CATEGORY jazz music END in the ENTITY_LOCATION living room END. INTENT_MEDIA_CONTROL",
            
            "ENTITY_ACTION Schedule END a ENTITY_DELIVERY delivery END for ENTITY_DATE October 15th END at ENTITY_TIME 10 AM END. INTENT_SCHEDULE_DELIVERY",
            
            "Please ENTITY_ACTION book END a ENTITY_TABLE table END at the ENTITY_CUISINE Italian restaurant END for ENTITY_PARTY_SIZE two END at ENTITY_TIME 7 PM END. INTENT_BOOK_RESERVATION",
            
            "ENTITY_ACTION Remind END me to ENTITY_ACTION take END my ENTITY_MEDICATION medication END at ENTITY_TIME 9 AM END. INTENT_SET_REMINDER"
        ]

        **Sentences to Annotate:**
        {json.dumps(sentences, ensure_ascii=False)}
        '''

        print("Sending request to Vertex AI...")
        response = model.generate_content(prompt)
        print("Received response from Vertex AI")
        
        try:
            # Extract the text content and parse as JSON
            response_text = response.text.strip()
            print("Raw response:", response_text)
            
            # Try to find and extract JSON array if response contains additional text
            import re
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                response_text = json_match.group()
            
            annotated_sentences = json.loads(response_text)
            
            if isinstance(annotated_sentences, list):
                print(f"Successfully parsed {len(annotated_sentences)} annotated sentences")
                
                # Validate the number of sentences matches
                if len(annotated_sentences) != len(sentences):
                    print(f"Warning: Length mismatch - Input: {len(sentences)}, Output: {len(annotated_sentences)}")
                    print("Falling back to original sentences")
                    return sentences
                    
                return annotated_sentences
            else:
                print("Warning: Response was not a list after JSON parsing")
                return sentences
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print("Falling back to original sentences")
            return sentences
            
    except Exception as e:
        print(f"Error during Vertex AI processing: {str(e)}")
        print("Traceback:", traceback.format_exc())
        return sentences  # Return original sentences as fallback




def annotate_sentences(sentences):
    print("\nStarting sentence annotation")
    lang_code = "English"
    
    prompt = f'''
    Given a list of sentences in English, annotate each sentence individually with the appropriate entity tags from the provided list. The sentences may relate to various actions such as managing tasks, controlling devices, sending notifications, scheduling events, updating information, making purchases, or offering assistance.

    **Instructions:**

    - Annotate each sentence separately with entity tags based on the entities in the provided list.
    - Use the format `ENTITY_<type>` to start each entity and `END` to close it.
    - For each sentence, add a relevant intent label at the end in the format `INTENT_<type>`, where `<type>` represents the sentence's main action or goal.
    - Only use the entity types provided in the list below.
    - Do not add any additional text, explanations, or comments.
    - Ensure the output is a JSON array containing only the annotated sentences in the same order as the input sentences, without any markdown or formatting.

    **Entities**:

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
        "API_KEY", "WEB_TOKEN", "URL_PARAMETER", "SERVER_NAME", "ENDPOINT", "DOMAIN", 
        
        "PRODUCT", "SERVICE", "CATEGORY", "BRAND", "ORDER_STATUS", "DELIVERY_METHOD", "RETURN_STATUS", "WARRANTY_PERIOD", 
        "CANCELLATION_REASON", "REFUND_AMOUNT", "EXCHANGE_ITEM", "GIFT_OPTION", "GIFT_MESSAGE", 
        
        "FOOD_ITEM", "DRINK_ITEM", "CUISINE", "MENU_ITEM", "ORDER_NUMBER", "DELIVERY_ESTIMATE", "RECIPE", "INGREDIENT", 
        "DISH_NAME", "PORTION_SIZE", "COOKING_TIME", "PREPARATION_METHOD", 
        
        "AGE", "GENDER", "NATIONALITY", "RELIGION", "MARITAL_STATUS", "OCCUPATION", "EDUCATION_LEVEL", "DEGREE", 
        "SKILL", "EXPERIENCE", "YEARS_OF_EXPERIENCE", "CERTIFICATION", 
        
        "MEASUREMENT", "DISTANCE", "WEIGHT", "HEIGHT", "VOLUME", "TEMPERATURE", "SPEED", "CAPACITY", "DIMENSION", "AREA", 
        "SHAPE", "COLOR", "MATERIAL", "TEXTURE", "PATTERN", "STYLE", 
        
        "WEATHER_CONDITION", "TEMPERATURE_SETTING", "HUMIDITY_LEVEL", "WIND_SPEED", "RAIN_INTENSITY", "AIR_QUALITY", 
        "POLLUTION_LEVEL", "UV_INDEX", 
        
        "QUESTION_TYPE", "REQUEST_TYPE", "SUGGESTION_TYPE", "ALERT_TYPE", "REMINDER_TYPE", "STATUS", "ACTION", "COMMAND"
    ]

    **Example**:

    Input Sentences:
    [
        "Can you set up a meeting with John at 3 PM tomorrow?",
        "Play some jazz music in the living room.",
        "Schedule a delivery for October 15th at 10 AM.",
        "Please book a table at the Italian restaurant for two at 7 PM.",
        "Remind me to take my medication at 9 AM."
    ]

    Expected Output:
    [
        "Can you ENTITY_ACTION set up END a ENTITY_MEETING meeting END with ENTITY_PERSON_NAME John END at ENTITY_TIME 3 PM END on ENTITY_DATE tomorrow END? INTENT_SCHEDULE_MEETING",
        
        "ENTITY_ACTION Play END some ENTITY_CATEGORY jazz music END in the ENTITY_LOCATION living room END. INTENT_MEDIA_CONTROL",
        
        "ENTITY_ACTION Schedule END a ENTITY_DELIVERY delivery END for ENTITY_DATE October 15th END at ENTITY_TIME 10 AM END. INTENT_SCHEDULE_DELIVERY",
        
        "Please ENTITY_ACTION book END a ENTITY_TABLE table END at the ENTITY_CUISINE Italian restaurant END for ENTITY_PARTY_SIZE two END at ENTITY_TIME 7 PM END. INTENT_BOOK_RESERVATION",
        
        "ENTITY_ACTION Remind END me to ENTITY_ACTION take END my ENTITY_MEDICATION medication END at ENTITY_TIME 9 AM END. INTENT_SET_REMINDER"
    ]

    **Sentences to Annotate:**
    {json.dumps(sentences, ensure_ascii=False)}
    '''
    
    openai.api_key = "openai_api_key"
    
    try:
        print("Making OpenAI API call...")
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            temperature=0.5
        )
        
        assistant_reply = response.choices[0].message.content
        print("Received API response")
        print("Raw response:", assistant_reply)
        
        try:
            annotated_sentences = json.loads(assistant_reply)
            if isinstance(annotated_sentences, list):
                print(f"Successfully parsed {len(annotated_sentences)} annotated sentences")
                return annotated_sentences
            else:
                print("Warning: API response was not a list after JSON parsing")
                return [assistant_reply] * len(sentences)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print("Falling back to raw reply")
            return [assistant_reply] * len(sentences)
            
    except Exception as e:
        print(f"Error during API call: {str(e)}")
        print("Traceback:", traceback.format_exc())
        return sentences  # Return original sentences as fallback
    
if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process audio files and create annotated manifests')
    parser.add_argument('lang_code', help='Language code (e.g., en, fr, de)')
    parser.add_argument('--mode', choices=['tsv', 'manifest'], required=True, 
                        help='Processing mode: tsv or manifest')
    parser.add_argument('--input-manifest', 
                        help='Path to input manifest file or directory containing manifest files')
    parser.add_argument('--max-workers', type=int, default=1,
                        help='Maximum number of parallel processes (default: 4)')
    
    args = parser.parse_args()
    
    # Define base paths
    data_path = PurePath("/projects/whissle/datasets/cv/cv-corpus-15.0-2023-09-08/")
    data_path_lang = data_path / args.lang_code
    manifestfolder = "/external1/datasets/peoples_speech/manifests_processed"
    
    # Set up device and models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load emotion recognition model
    print("Loading emotion recognition model...")    
    emotion_classifier = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er")
    emotion_classifier.to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
    
    # load model from hub
    device = 'cpu'
    model_name = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
    age_processor = Wav2Vec2Processor.from_pretrained(model_name)
    age_model = AgeGenderModel.from_pretrained(model_name)

    # dummy signal
    #sampling_rate = 16000
    #signal = np.zeros((1, sampling_rate), dtype=np.float32)
    
    if args.mode == 'tsv':
        # Process TSV files
        train_annotations = data_path_lang / PurePath("train.tsv")
        audioclips = data_path_lang / PurePath("clips")
        audioclipswav = PurePath(str(audioclips) + "-wav")
        os.system("mkdir -p " + str(audioclipswav))
        print(f"Processing TSV files for {args.lang_code}")
        
        # Define dataset configurations
        datasets = {
            "train": {
                "tsvfile": train_annotations,
                "audioclips": audioclips,
                "audioclipswav": audioclipswav,
                "manifestfile": os.path.join(manifestfolder, f"train_cv_{args.lang_code}.json"),
                "taglistfile": os.path.join(manifestfolder, f"taglist_train_{args.lang_code}.txt"),
                "checkpoint_file": os.path.join(manifestfolder, f"checkpoint_cv_train_{args.lang_code}.txt"),
                "lang_code": args.lang_code
            },
        }
        
        # Process TSV files with limited parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for dataset in datasets.values():
                future = executor.submit(process_tsv, **dataset)
                futures.append(future)
                
            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f'Generated an exception: {exc}')
                    
    elif args.mode == 'manifest':
        if not args.input_manifest:
            parser.error("--input-manifest is required when using manifest mode")
            
        input_path = Path(args.input_manifest)
        
        if input_path.is_file():
            # Process single manifest file
            print(f"Processing single manifest file: {input_path}")
            manifest_files = [input_path]
        elif input_path.is_dir():
            # Process all manifest files in directory
            print(f"Processing manifest files from directory: {input_path}")
            manifest_files = list(input_path.glob("train*.json"))
            if not manifest_files:
                print(f"No JSON manifest files found in {input_path}")
                sys.exit(1)
        else:
            parser.error("Input manifest path does not exist")
        
        # Process each manifest file
        manifests_to_process = []
        for input_manifest in manifest_files:
            # Generate output paths for each manifest
            output_manifest = os.path.join(manifestfolder, f"annotated_{input_manifest.name}")
            print(f"Output manifest file: {output_manifest}")
            taglist_file = os.path.join(manifestfolder, f"taglist_{input_manifest.stem}_{args.lang_code}.txt")
            checkpoint_file = os.path.join(manifestfolder, f"checkpoint_{input_manifest.stem}_{args.lang_code}.txt")
            
            manifests_to_process.append({
                "input_manifest_file": str(input_manifest),
                "manifestfile": output_manifest,
                "taglistfile": taglist_file,
                "checkpoint_file": checkpoint_file,
                "lang_code": args.lang_code
            })
        
        print(f"Will process {len(manifests_to_process)} manifest files using up to {args.max_workers} workers")
        
        # Process manifests with limited parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for manifest_config in manifests_to_process:
                future = executor.submit(process_manifest, **manifest_config)
                futures.append(future)
            
            # Track progress
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                print(f"Completed {completed}/{len(manifests_to_process)} manifests")
                future.result()  # This will raise any exceptions that occurred
                    
        print(f"Finished processing {len(manifests_to_process)} manifest files")