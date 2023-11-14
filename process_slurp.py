# Process SLURP dataset for 1SSI: OneStep speech intructor

import os
import json
import re
from pathlib import Path

from pathlib import PurePath
from pydub import AudioSegment

from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo.collections import nlp as nemo_nlp

### Intitiate text normalizer and puctuator
normalizer = Normalizer(input_case='lower_cased', lang="en")
punctuator = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained("punctuation_en_distilbert")

def normalize(text):

    text = text.lower()
    normalized = normalizer.normalize(text, verbose=True, punct_post_process=True)
    normalized = [normalized]
    norm_punctuated = punctuator.add_punctuation_capitalization(normalized)[0]
    return norm_punctuated


### Define all data path (SLURP here)
slurp_annotations = Path("/n/disk1/audio_datasets/slurp/dataset/slurp/")
train_annotations = slurp_annotations / Path("train.jsonl")
dev_annotations = slurp_annotations / Path("devel.jsonl")
test_annotations = slurp_annotations / Path("test.jsonl")

audio_real = Path("/n/disk1/audio_datasets/slurp/audio/slurp_real")
audio_synth = Path("/n/disk1/audio_datasets/slurp/audio/slurp_synth/")


### SLURP tag aligner
def convert_entity_format(text):
    # Regular expression to find any entity type pattern
    pattern = r'\[([a-zA-Z_]+) : ([^\]]+)\]'

    # Function to replace the found pattern
    def replace_pattern(match):
        entity_type = match.group(1).strip().upper()  # Convert entity type to uppercase
        entity_value = match.group(2).strip()

        return f"B-{entity_type} {entity_value} E-{entity_type}"

    # Replace all occurrences of the pattern in the text
    converted_text = re.sub(pattern, replace_pattern, text)

    return converted_text

def add_entity_tags(input1, input2):
    # Find all entities in input2
    entities = re.findall(r'B-([A-Z_]+) (.*?) E-\1', input2)

    # Function to handle punctuation and casing
    def replace_entity(match):
        before, entity, after = match.groups()
        # Use the original entity text from Input1 for replacement
        original_entity = input1[match.start(2):match.end(2)]
        return f"{before}B-{entity_type} {original_entity} E-{entity_type}{after}"

    # Replace the text in input1 with tagged text from input2
    for entity in entities:
        entity_type, entity_value = entity
        # Pattern to include possible punctuation around the entity
        pattern = r'(\W?)(\b' + re.escape(entity_value) + r'\b)(\W?)'
        input1 = re.sub(pattern, replace_entity, input1, flags=re.IGNORECASE)

    return input1


### Get emotion labels using HUBERT audio classification
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
from AudioEmotionClassification.models import Wav2Vec2ForSpeechClassification, HubertForSpeechClassification

emotion_model = HubertForSpeechClassification.from_pretrained("Rajaram1996/Hubert_emotion")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
sampling_rate=16000 # defined by the model; must convert mp3 to this rate.
config = AutoConfig.from_pretrained("Rajaram1996/Hubert_emotion")

def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate, sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in
               enumerate(scores)]
    return outputs

def get_emotion_labels(audio_file, sampling_rate=16000, score=50.0):
    sound_array = speech_file_to_array_fn(audio_file, sampling_rate)
    
    inputs = feature_extractor(sound_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to("cpu").float() for key in inputs}

    with torch.no_grad():
        logits = emotion_model(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

    outputs = [{
        "emo": config.id2label[i],
        "score": round(score * 100, 1)}
        for i, score in enumerate(scores)
    ]

    #[{'emo': 'female_neutral', 'score': 73.9}, {'emo': 'female_happy', 'score': 24.8}]
    emotion_labels = [row for row in sorted(outputs, key=lambda x:x["score"], reverse=True) if row['score'] != '0.0%'][:2]

    all_labels = []
    for emotion_dict in emotion_labels:
        label = emotion_dict['emo'].split("_")[1].upper()
        score = emotion_dict['score']

        if score > 50.0:
            all_labels.append(label)

    return all_labels


### parse jsonl file and create manifest
ALL_ENTITIES = {}


def jsonl_process(jsonlfile, audiofolder, manifestfolder):

    print(jsonlfile)

    wavfolder = str(audiofolder) + "-wav"
    os.system("mkdir -p "+wavfolder)
    wavfolder = Path(wavfolder)

    jsonlfileread = open(str(jsonlfile),'r').readlines()

    manifest = open(manifestfolder + "/" + jsonlfile.name.replace(jsonlfile.suffix, "") + "-slurp.json",'w')


    for line in jsonlfileread:

        line = json.loads(line)
        #print(line)
        annotation = line['sentence_annotation']
        text = line['sentence']
        text_clean = normalize(text)
        text_tagged = convert_entity_format(line['sentence_annotation'])
        text_clean_tagged = add_entity_tags(text_clean, text_tagged)
        
        intent = line['intent'].upper()

        recordings = line['recordings']

        #print("Final text:", text_clean_tagged)

        for recording in recordings:
            audiofile = recording['file']
            audiofilepath = audiofolder / Path(audiofile)

            audiofile = PurePath(audiofile)
            filekey = audiofile.name.replace(audiofile.suffix, "")
            wavfilepath = str(wavfolder) + "/" + filekey + ".wav"
            flac_tmp_audio_data = AudioSegment.from_file(audiofilepath, audiofilepath.suffix[1:])
            flac_tmp_audio_data.export(wavfilepath, format="wav")
            
            
            print(audiofilepath)

            sample_dict = {}
            sample_dict['audiofilepath'] = wavfilepath
            sample_dict['text'] = text_clean
            sample_dict['tagged_text'] = text_clean

            flac_tmp_audio_data = AudioSegment.from_file(audiofilepath, audiofilepath.suffix[1:])
            flac_tmp_audio_data.export(wavfilepath, format="wav")
            sample_dict['instruction'] = "transcribe speech"

            json.dump(sample_dict, manifest)
            manifest.write("\n")

            sample_dict['tagged_text'] = text_clean_tagged
            sample_dict['instruction'] = "transcribe and mark entities"
            json.dump(sample_dict, manifest)
            manifest.write("\n")


            emotion_labels = get_emotion_labels(audio_file=wavfilepath, sampling_rate=16000)
            emotion_labels = ' '.join(emotion_labels)

            final_transcription = text_clean_tagged + " " + emotion_labels

            sample_dict['tagged_text'] = final_transcription
            sample_dict['instruction'] = "transcribe, mark entitites and track speaker emotion"
            json.dump(sample_dict, manifest)
            manifest.write("\n")

            sample_dict['tagged_text'] = text_clean_tagged + " " + intent
            sample_dict['instruction'] = "transcribe, mark entitites, get speaker intent"
            json.dump(sample_dict, manifest)
            manifest.write("\n")

            sample_dict['tagged_text'] = final_transcription + " " + intent
            sample_dict['instruction'] = "transcribe, mark entitites, get emotion and intent labels"
            json.dump(sample_dict, manifest)
            manifest.write("\n")        
    
    manifest.close() 


manifestfolder = "/n/disk1/audio_datasets/manifests"

jsonl_process(jsonlfile=train_annotations, audiofolder=audio_real, manifestfolder=manifestfolder)
jsonl_process(jsonlfile=dev_annotations, audiofolder=audio_real, manifestfolder=manifestfolder)
jsonl_process(jsonlfile=test_annotations, audiofolder=audio_real, manifestfolder=manifestfolder)