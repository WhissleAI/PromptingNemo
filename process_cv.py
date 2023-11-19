import os
import re
import json
import glob
import pandas as pd


from pathlib import Path
from pathlib import PurePath
from pydub import AudioSegment

from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo.collections import nlp as nemo_nlp

from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
from transformers import pipeline

from flair.data import Sentence
from flair.models import SequenceTagger
import pprint
import string


def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    # Load the MP3 file
    audio = AudioSegment.from_mp3(mp3_file_path)

    # Export as WAV
    audio.export(wav_file_path, format="wav")

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
cv_english = PurePath("/audio_datasets/CommonVoice/datasets/cv-corpus-15.0-2023-09-08/en/")
train_annotations = cv_english / PurePath("train.tsv")
dev_annotations = cv_english / PurePath("dev.tsv")
test_annotations = cv_english / PurePath("test.tsv")

audioclips = PurePath("/audio_datasets/CommonVoice/datasets/cv-corpus-15.0-2023-09-08/en/clips")
audioclipswav = PurePath(str(audioclips) + "-wav")
os.system("mkdir -p " + str(audioclipswav))
print(audioclipswav)


### Named entity tagger
entity_tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
entity_model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
hf_nlp = pipeline("ner", model=entity_model, tokenizer=entity_tokenizer, grouped_entities=True)

def tag_entities(text, lowercase=True):

    ner_results = hf_nlp(text)
    print(ner_results)

    # example: [{'entity_group': 'PER', 'score': 0.8913538, 'word': 'Min', 'start': 0, 'end': 3}, {'entity_group': 'LOC', 'score': 0.9983326, 'word': 'West Van Buren Street', 'start': 93, 'end': 114}]
    for ner_dict in ner_results:

        entity_group = ner_dict['entity_group'].upper()
        start = ner_dict['start']
        end = ner_dict['end']
        word = ner_dict['word']

        if lowercase:
            word = word.lower()

        text = text.replace(word, "ENTITY-"+entity_group+" "+word+" END")

    print("ner tagged text", text)
    return text

### Part of Speech Tagger
# load tagger
pos_tagger = SequenceTagger.load("flair/pos-english")


def tag_pos(text):
    sentence = Sentence(text)
    pos_tagger.predict(sentence)

    tagged = []
    for entity in sentence.labels:

        word = entity.data_point.text
        tag = entity.value

        if tag not in string.punctuation:
            tagged.append(tag)
            tagged.append(word)
            tagged.append("END")
        else:
            tagged.append(word)

    text = " ".join(tagged)
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

def process_tsv(tsvfile, audioclips, audioclipswav, manifestfile):
    
    tsvfile = pd.read_csv(tsvfile, sep="\t")
    print(manifestfile)
    manifest = open(str(manifestfile),'w')
    #data_top = tsvfile.columns.values

    #print(data_top)
    for index, row in tsvfile.iterrows():
        audiofile = audioclips / row['path']
        wavfilepath = audioclipswav / PurePath(row['path'].split(".")[0]+".wav")
        
        convert_mp3_to_wav(audiofile, wavfilepath)
        
        text = row['sentence']
        print(text)
        text_pos = tag_pos(text)
        print(text_pos)
        text_tagged = tag_entities(text)
        emotion_labels = get_emotion_labels(audio_file=wavfilepath, sampling_rate=16000)
        text_tagged_emotion = text_tagged + " " + " ".join(emotion_labels)

        wavfilepath = str(wavfilepath)


        sample_dict = {}
        sample_dict['audio_filepath'] = wavfilepath
        sample_dict['text'] = text
        sample_dict['tasks'] = ["transcription"]
        sample_dict['instruction'] = "Transcribe what is begin spoken"
        pprint.pprint(sample_dict)
        json.dump(sample_dict, manifest)
        manifest.write("\n")

        emotion_labels = get_emotion_labels(audio_file=wavfilepath, sampling_rate=16000)
        for label in emotion_labels:
            if label not in taglist:
                taglist.append(label)
        emotion_labels = ' '.join(emotion_labels)
        sample_dict['text'] = text + " " + emotion_labels
        sample_dict['tasks'] = ["transcription", "emotion"]
        sample_dict['instruction'] = "Transcribe and track speaker emotion"
        json.dump(sample_dict, manifest)
        manifest.write("\n")

        sample_dict['text'] = text_tagged
        sample_dict['tasks'] = ["transcription", "entity-named"]
        sample_dict['instruction'] = "Transcribe, mark named entities"
        json.dump(sample_dict, manifest)
        manifest.write("\n")

        sample_dict['text'] = text_clean_tagged + " " + intent + " " + emotion_labels
        sample_dict['tasks'] = ["transcription", "entity-named", "emotion"]
        sample_dict['instruction'] = "Transcribe and mark named entities and emotion"
        json.dump(sample_dict, manifest)

        sample_dict['text'] = text_clean_tagged + " " + intent + " " + emotion_labels
        sample_dict['tasks'] = ["transcription", "emotion", "entity-named"]
        sample_dict['instruction'] = "Transcribe, track speaker emotion and mark named entities"
        json.dump(sample_dict, manifest)

        manifest.write("\n")
        print(text_tagged, audiofilewav)
    manifest.close()

manifestfolder = "/audio_datasets/manifests"
process_tsv(tsvfile=dev_annotations, audioclips=audioclips, audioclipswav=audioclipswav, manifestfile=manifestfolder+"/dev_cv_en.json")
process_tsv(tsvfile=train_annotations, audioclips=audioclips, audioclipswav=audioclipswav, manifestfile=manifestfolder+"/train_cv_en.json")
process_tsv(tsvfile=test_annotations, audioclips=audioclips, audioclipswav=audioclipswav, manifestfile=manifestfolder+"/test_cv_en.json")
