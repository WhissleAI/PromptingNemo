import os
import re
import sys
import json
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
from datasets import load_dataset
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

from nemo_text_processing.text_normalization.normalize import Normalizer
import nemo.collections.nlp as nemo_nlp
#from AudioEmotionClassification.models import (Wav2Vec2ForSpeechClassification, 
#                                              HubertForSpeechClassification)

import librosa
import openai

lang_code_map = {"en": "English", "fr": "French", "de": "German", "it": "Italian", "es": "Spanish"}

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
    
    input_manifest_file = open(input_manifest_file, 'r')
    
    batch_text = []
    batch_emotion = []
    batch_langid = []
    batch_meta = []
    
    output_manifest_file = open(manifestfile, 'w')
        
    for sample in input_manifest_file:
        
        text = sample['text']
        audiofile = sample['audio_filepath']
        
        emotion_label = detect_emotion(audiofile)
        
        batch_text.append(text)
        batch_emotion.append(emotion_label)
        batch_langid.append(lang_code)
        
        if len(batch_text) == 10:
            
            batch_text_annotated = annotate_sentences(batch_text)
            
            if len(batch_text_annotated) == len(batch_text):
                print("Batch Text Annotated:", batch_text_annotated)
                for i in range(len(batch_text)):
                    
                    sample_dict = {}
                    sample_dict['duration'] = duration
                    sample_dict['audio_filepath'] = audiofile
                    sample_dict['text'] = batch_text_annotated[i]
                    sample_dict['emotion'] = batch_emotion[i]
                    sample_dict['langid'] = lang_code
                    sample_dict['meta'] = batch_meta[i]
                    sample_dict['tasks'] = ["transcription", "keyphrases","speaker-meta"]
                    sample_dict['instruction'] = "Transcribe and mark keyphrases and speaker metadata"
                    json.dump(sample_dict, manifest, ensure_ascii=False)
                    output_manifest_file.write("\n") 
                
            batch_text = []
            batch_emotion = []
            batch_meta = []
            batch_langid = []    
        
        
    
    

#def process_tsv(tsvfile, audioclips, audioclipswav, manifestfile, taglistfile, checkpoint_file):
    
def annotate_sentences(sentences):
    # Prepare the prompt with multiple sentences
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

    try:
        prompt = prompt.replace("English", lang_code_map[lang_code])
        client = openai.OpenAI(api_key='<openai-api-key>')
        response = client.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
                n=1,
                stop=None,
                temperature=0.5,
            )
            # Get the assistant's reply
        assistant_reply = dict(response).get('choices')[0].message.content
        try:
            annotated_sentences = json.loads(assistant_reply)
            if isinstance(annotated_sentences, list):
                return annotated_sentences
            else:
                print("Assistant did not return a list. Fallback to raw reply.")
                return [assistant_reply] * len(sentences)  # Fallback
        except json.JSONDecodeError:
            print("JSON decoding failed. Fallback to raw replies.")
            return [assistant_reply] * len(sentences)  # Fallback
    except Exception as e:
        print(f"Error processing sentences: {e}")
        return ["sentence"]

    
if __name__ == "__main__":
    
    
    ### Intitiate text normalizer and puctuator
    #normalizer = Normalizer(input_case='lower_cased', lang="en")
    #punctuator = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained("punctuation_en_distilbert")

    ### Define all data path (SLURP here)
    data_path = PurePath("/projects/whissle/datasets/cv/cv-corpus-15.0-2023-09-08/")
    lang_code = sys.argv[1]
    data_path_lang = data_path / lang_code
    train_annotations = data_path_lang / PurePath("train.tsv")
    dev_annotations = data_path_lang / PurePath("dev.tsv")
    test_annotations = data_path_lang / PurePath("test.tsv")

    audioclips = data_path_lang / PurePath("clips")
    audioclipswav = PurePath(str(audioclips) + "-wav")
    os.system("mkdir -p " + str(audioclipswav))
    print(audioclipswav)

    ### Named entity tagger
    #entity_tagger = PretrainedPipeline("onto_recognize_entities_sm")


    # Audio Emotion Classification
    #emotion_model = HubertForSpeechClassification.from_pretrained("Rajaram1996/Hubert_emotion")
    #feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    #sampling_rate=16000 # defined by the model; must convert mp3 to this rate.
    #emotion_config = AutoConfig.from_pretrained("Rajaram1996/Hubert_emotion")

    # At the top of the file, add device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # When loading the emotion classifier, move it to GPU
    print("Loading emotion recognition model...")    
    emotion_classifier = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er")
    emotion_classifier.to(device)  # Move model to GPU
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")


    manifestfolder = "/projects/whissle/datasets/cv/manifests"


    # Define the file paths and parameters for each dataset
    datasets = {
        "train": {
            "tsvfile": train_annotations,
            "audioclips": audioclips,
            "audioclipswav": audioclipswav,
            "manifestfile": os.path.join(manifestfolder, "train_cv_"+lang_code+".json"),
            "taglistfile": os.path.join(manifestfolder, "taglist_train_en.txt"),
            "checkpoint_file": os.path.join(manifestfolder,"checkpoint_cv_train"+lang_code+".txt"),
            "lang_code": lang_code
        },
    }

    # Run process_tsv in parallel for each dataset
    with concurrent.futures.ThreadPoolExecutor() as executor:
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


# # Define the file paths and parameters
# tsvfile_dev = dev_annotations
# audioclips_dev = audioclips
# audioclipswav_dev = audioclipswav
# manifestfile_dev = os.path.join(manifestfolder, "dev_cv_en.json")
# taglistfile_dev = os.path.join(manifestfolder, "taglist_dev_en.txt")
# checkpoint_file_dev = "checkpoint_dev.txt"

# # Call the process_tsv function with the defined parameters
# process_tsv(
#     tsvfile=tsvfile_dev,
#     audioclips=audioclips_dev,
#     audioclipswav=audioclipswav_dev,
#     manifestfile=manifestfile_dev,
#     taglistfile=taglistfile_dev,
#     checkpoint_file=checkpoint_file_dev
# )


# # Define the file paths and parameters for the second call
# tsvfile_train = train_annotations
# audioclips_train = Path(audioclips)
# audioclipswav_train = audioclipswav
# manifestfile_train = os.path.join(manifestfolder, "train_cv_en.json")
# taglistfile_train = os.path.join(manifestfolder, "taglist_train_en.txt")
# checkpoint_file_train = "checkpoint_train.txt"  # Define the checkpoint file

# # Call the process_tsv function for the train data with the defined parameters
# process_tsv(
#     tsvfile=tsvfile_train,
#     audioclips=audioclips_train,
#     audioclipswav=audioclipswav_train,
#     manifestfile=manifestfile_train,
#     taglistfile=taglistfile_train,
#     checkpoint_file=checkpoint_file_train  # Use the defined checkpoint file
# )

# # Define the file paths and parameters for the third call
# tsvfile_test = test_annotations
# audioclips_test = audioclips
# audioclipswav_test = audioclipswav
# manifestfile_test = os.path.join(manifestfolder, "test_cv_en.json")
# taglistfile_test = os.path.join(manifestfolder, "taglist_test_en.txt")
# checkpoint_file_test = "checkpoint_test.txt"  # Define the checkpoint file

# # Call the process_tsv function for the test data with the defined parameters
# process_tsv(
#     tsvfile=tsvfile_test,
#     audioclips=audioclips_test,
#     audioclipswav=audioclipswav_test,
#     manifestfile=manifestfile_test,
#     taglistfile=taglistfile_test,
#     checkpoint_file=checkpoint_file_test  # Use the defined checkpoint file
# )