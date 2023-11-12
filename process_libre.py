# Organizing English LibreSpeech Corpus into Manfests for Speech

import os
import re
import json
import glob
from pathlib import Path

from pathlib import PurePath
from pydub import AudioSegment

from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo.collections import nlp as nemo_nlp


### Intitiate text normalizer and puctuator
normalizer = Normalizer(input_case='lower_cased', lang="en")
punctuator = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained("punctuation_en_distilbert")


### Start Hugging Face NLP systems
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

entity_tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
entity_model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")

hf_nlp = pipeline("ner", model=entity_model, tokenizer=entity_tokenizer, grouped_entities=True)

def tag_entities(text):

    ner_results = hf_nlp(text)

    # example: [{'entity_group': 'PER', 'score': 0.8913538, 'word': 'Min', 'start': 0, 'end': 3}, {'entity_group': 'LOC', 'score': 0.9983326, 'word': 'West Van Buren Street', 'start': 93, 'end': 114}]
    for ner_dict in ner_results:

        entity_group = ner_dict['entity_group']
        start = ner_dict['start']
        end = ner_dict['end']
        word = ner_dict['word']

        text = text.replace(word, "B-"+entity_group+" "+word+" E-"+entity_group)

    return text


### Start pretrained Emotion Classification system
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


### Librespeech: Get data, un-compress it and then set paths
#define paths to folders created afte unzipping
LIBRE = '/n/disk1/audio_datasets/EN_libre/'
TRAIN_DATA = Path(LIBRE+'/LibriSpeech/train-clean-360/')
TRAIN_DATA_WAV = str(TRAIN_DATA) + '-wav/'
os.system('mkdir -p ' + TRAIN_DATA_WAV)
TRAIN_DATA_WAV = Path(TRAIN_DATA_WAV)

DEV_DATA = Path(LIBRE+'/LibriSpeech/dev-clean/')
DEV_DATA_WAV = str(DEV_DATA) + '-wav/'
os.system('mkdir -p ' + DEV_DATA_WAV)
DEV_DATA_WAV = Path(DEV_DATA_WAV)

TEST_DATA = Path(LIBRE+'/LibriSpeech/test-clean/')
TEST_DATA_WAV = str(TEST_DATA) + '-wav/'
os.system('mkdir -p ' + TEST_DATA_WAV)
TEST_DATA_WAV = Path(TEST_DATA_WAV)


allpath = [TRAIN_DATA, DEV_DATA, TEST_DATA]

def normalize(text):

    text = text.lower()
    normalized = normalizer.normalize(text, verbose=True, punct_post_process=True)
    normalized = [normalized]
    norm_punctuated = punctuator.add_punctuation_capitalization(normalized)[0]
    return norm_punctuated

def read_transcription(filepath):

    trans = open(filepath,'r').readlines()
    trans_dict = {}
    for line in trans:
        line = line.strip().split()
        text = ' '.join(line[1:])
        text = normalize(text)

        trans_dict[line[0]] = text
    
    return trans_dict

def process_librispeech(datakey):

    datafolders = glob.glob(str(datakey)+'/*')

    datakey_wav = str(datakey) + '-wav/'
    os.system('mkdir -p ' + datakey_wav)
    datakey_wav = Path(datakey_wav)

    manifest = open(datakey_wav.name+'.json','w')

    for folder in datafolders:
        sessdirs = glob.glob(folder + '/*')

        for sessdir in sessdirs:
            segments = glob.glob(sessdir + '/*')

            transcription = [x for x in segments if re.search(".txt", x)][0]
            trans_dict = read_transcription(transcription)


            for segment in segments:
                #allfiles = glob.glob(segment + '/*')
                
                filepath = PurePath(segment)

                if ".flac" in str(filepath):
                    
                    sample_dict = {}

                    filekey = filepath.name.replace(filepath.suffix, "")
                    transcription = trans_dict[filekey]
                    wav_filepath = str(datakey_wav) + "/" + filekey + ".wav"
                    sample_dict['audiofilepath'] = wav_filepath
                    sample_dict['text'] = transcription
                    sample_dict['tagged_text'] = transcription

                    flac_tmp_audio_data = AudioSegment.from_file(filepath, filepath.suffix[1:])
                    flac_tmp_audio_data.export(wav_filepath, format="wav")
                    sample_dict['instruction'] = "transcribe speech"

                    json.dump(sample_dict, manifest)
                    manifest.write("\n")


                    tagged_transcription = tag_entities(transcription)
                    sample_dict['text'] = transcription
                    sample_dict['tagged_text'] = tagged_transcription
                    sample_dict['instruction'] = "transcribe and mark named entities"
                    json.dump(sample_dict, manifest)
                    manifest.write("\n")


                    emotion_labels = get_emotion_labels(audio_file=wav_filepath, sampling_rate=16000)
                    emotion_labels = ' '.join(emotion_labels)

                    final_transcription = tagged_transcription + " " + emotion_labels

                    sample_dict['text'] = transcription
                    sample_dict['tagged_text'] = final_transcription
                    sample_dict['instruction'] = "transcribe, mark named entitites and track speaker emotion"
                    json.dump(sample_dict, manifest)
                    manifest.write("\n")

                    sample_dict['prompt'] = final_transcription
    
    manifest.close()

for datakey in allpath:
    print(datakey)
    process_librispeech(datakey)