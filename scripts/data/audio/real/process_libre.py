# Organizing English LibreSpeech Corpus into Manfests for Speech

import os
import re
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
import torchaudio
from transformers import (AutoTokenizer, AutoModelForTokenClassification, 
                          TokenClassificationPipeline, pipeline, AutoConfig, 
                          Wav2Vec2FeatureExtractor)
from nemo_text_processing.text_normalization.normalize import Normalizer
import nemo.collections.nlp as nemo_nlp
from flair.data import Sentence
from flair.models import SequenceTagger
from AudioEmotionClassification.models import (Wav2Vec2ForSpeechClassification, 
                                               HubertForSpeechClassification)

from johnsnowlabs import nlp
dependency_parser = nlp.load('ner')
from sparknlp.pretrained import PretrainedPipeline

'''
TODO:
1. all labels used are written properly to the taglist file
2. Multiprocessing for the tsv files
'''


### Intitiate text normalizer and puctuator
normalizer = Normalizer(input_case='lower_cased', lang="en")
punctuator = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained("punctuation_en_distilbert")

### Named entity tagger
dependency_parser = PretrainedPipeline("dependency_parse")
entity_tagger = PretrainedPipeline("onto_recognize_entities_sm")


# Audio Emotion Classification
emotion_model = HubertForSpeechClassification.from_pretrained("Rajaram1996/Hubert_emotion")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
sampling_rate=16000 # defined by the model; must convert mp3 to this rate.
emotion_config = AutoConfig.from_pretrained("Rajaram1996/Hubert_emotion")



def normalize(text):

    text = text.lower()
    normalized = normalizer.normalize(text, verbose=True, punct_post_process=True)
    normalized = [normalized]
    norm_punctuated = punctuator.add_punctuation_capitalization(normalized)[0]
    return norm_punctuated




def tag_ner(text):
    """
    Wraps each token with its corresponding BIO tag and returns the wrapped text and a list of unique tags used.

    Parameters:
    text (str): The original text to be annotated.

    Returns:
    tuple: A tuple containing the wrapped text and a list of unique tags used.
    """
    entities = entity_tagger.annotate(text)

    tokens = entities['token']
    tags = entities['ner']

    wrapped_text = ""
    current_entity = ""
    current_tag = ""
    used_tags = set()

    for token, tag in zip(tokens, tags):
        # Standardize tag format
        formatted_tag = f"NER_{tag[2:]}" if tag != 'O' else tag

        if tag.startswith('B-'):
            # Close previous entity if any
            if current_entity:
                wrapped_text += f"{current_tag} {current_entity} END "
                used_tags.add(current_tag)
                used_tags.add("END")
                current_entity = ""

            # Start a new entity
            current_entity = token
            current_tag = formatted_tag
        elif tag.startswith('I-') and "NER_" + tag[2:] == current_tag:
            # Continue current entity
            current_entity += " " + token
        else:
            # Close previous entity if any
            if current_entity:
                wrapped_text += f"{current_tag} {current_entity} END "
                used_tags.add(current_tag)
                used_tags.add("END")
                current_entity = ""

            # Add non-entity tokens as is
            wrapped_text += token + " "
    
    # Close the last entity if any
    if current_entity:
        wrapped_text += f"{current_tag} {current_entity} END"
        used_tags.add(current_tag)
        used_tags.add("END")

    return wrapped_text.strip(), list(used_tags)

### Part of Speech Tagger
# load tagger

def tag_pos(text):
    """
    Wraps each token with its corresponding tag. Assigns unique tags to different punctuation marks.

    Parameters:
    tokens (list of str): Tokens to be wrapped.
    tags (list of str): Corresponding tags for each token.

    Returns:
    list of str: List of tokens wrapped with their tags.
    """
    dp = dependency_parser.annotate(text)

    tags = dp['pos']
    tokens = dp['token']

    punctuation_tags = {
        '.': 'PUNCT_DOT',
        ',': 'PUNCT_COMMA',
        ';': 'PUNCT_SEMICOLON',
        ':': 'PUNCT_COLON',
        '!': 'PUNCT_EXCLAMATION',
        '?': 'PUNCT_QUESTION',
        '-': 'PUNCT_HYPHEN',
        '(': 'PUNCT_LPAREN',
        ')': 'PUNCT_RPAREN',
        '[': 'PUNCT_LBRACKET',
        ']': 'PUNCT_RBRACKET',
        '{': 'PUNCT_LCURLY',
        '}': 'PUNCT_RCURLY',
        '\'\'': 'PUNCT_DOUBLEQUOTE',
        '``': 'PUNCT_BACKTICK'
    }

    wrapped_tokens = []
    used_tags = set()
    for token, tag in zip(tokens, tags):
        # Use a specific tag for each type of punctuation
        if token in punctuation_tags:
            tag = punctuation_tags[token]
            used_tags.add(tag)
        else:
            used_tags.add("POS_"+tag)
       
        used_tags.add("END")
        tag = f"POS_{tag}"
        wrapped_token = f"{tag} {token} END"
        wrapped_tokens.append(wrapped_token)
    
    return " ".join(wrapped_tokens), list(used_tags)


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
    outputs = [{"Emotion": emotion_config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in
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
        "emo": emotion_config.id2label[i],
        "score": round(score * 100, 1)}
        for i, score in enumerate(scores)
    ]

    #[{'emo': 'female_neutral', 'score': 73.9}, {'emo': 'female_happy', 'score': 24.8}]
    emotion_labels = [row for row in sorted(outputs, key=lambda x:x["score"], reverse=True) if row['score'] != '0.0%'][:2]

    final_label  = "EMOTION_DONTKNOW"
    for emotion_dict in emotion_labels:

        label = "EMOTION_"+emotion_dict['emo'].split("_")[1].upper()
        score = emotion_dict['score']

        if score > 50.0:
            final_label = label
        
    return final_label

def write_taglist(taglist,filename):

    taglist = "\n".join(taglist)

    with open(filename, 'w') as f:
        f.write(taglist)
        f.write("\n")
    f.close()





def read_transcription(filepath):

    trans = open(filepath,'r').readlines()
    trans_dict = {}
    for line in trans:
        line = line.strip().split()
        text = ' '.join(line[1:])
        text = normalize(text)

        trans_dict[line[0]] = text
    
    return trans_dict

def process_librispeech(datakey, taglistfile):

    datafolders = glob.glob(str(datakey)+'/*')

    datakey_wav = str(datakey) + '-wav/'
    os.system('mkdir -p ' + datakey_wav)
    datakey_wav = Path(datakey_wav)

    taglist = []
    for folder in datafolders:
        sessdirs = glob.glob(folder + '/*')

        for sessdir in sessdirs:
            segments = glob.glob(sessdir + '/*')

            manifest = open(datakey_wav.name+'.json','a')

            transcription = [x for x in segments if re.search(".txt", x)][0]
            trans_dict = read_transcription(transcription)


            for segment in segments:
                #allfiles = glob.glob(segment + '/*')
                
                filepath = PurePath(segment)

                if ".flac" in str(filepath):
                    
                    sample_dict = {}

                    filekey = filepath.name.replace(filepath.suffix, "")
                    text = trans_dict[filekey]

                    text = normalize(text)

                    text_pos, used_pos = tag_pos(text)
                    taglist = taglist + used_pos

                    text_ner, used_ner = tag_ner(text)
                    taglist = taglist + used_ner

                    wavfilepath = str(datakey_wav) + "/" + filekey + ".wav"
                    flacaudio = AudioSegment.from_file(filepath, filepath.suffix[1:])
                    duration_ms = len(flacaudio)
                    duration_seconds = duration_ms / 1000.0
                    flacaudio.export(wavfilepath, format="wav")

                    emotion_label = get_emotion_labels(audio_file=wavfilepath, sampling_rate=16000)
                    emotion_labels = [emotion_label]
                    taglist = taglist + emotion_labels
                    taglist = list(set(taglist)) # remove duplicates




                    sample_dict = {}
                    sample_dict['duration'] = duration_seconds
                    sample_dict['audio_filepath'] = wavfilepath
                    sample_dict['text'] = text
                    sample_dict['tasks'] = ["transcription"]
                    sample_dict['instruction'] = "Transcribe what is begin spoken"
                    json.dump(sample_dict, manifest)
                    manifest.write("\n")

                    emotion_labels = ' '.join(emotion_labels)
                    sample_dict['text'] = text + " " + emotion_label
                    sample_dict['tasks'] = ["transcription", "emotion"]
                    sample_dict['instruction'] = "Transcribe and track speaker emotion"
                    json.dump(sample_dict, manifest)
                    manifest.write("\n")

                    sample_dict['text'] = text_ner
                    sample_dict['tasks'] = ["transcription", "ner"]
                    sample_dict['instruction'] = "Transcribe and mark named entities"
                    json.dump(sample_dict, manifest)
                    manifest.write("\n")

                    sample_dict['text'] = text_ner + " " + emotion_label
                    sample_dict['tasks'] = ["transcription", "ner", "emotion"]
                    sample_dict['instruction'] = "Transcribe, mark named entities and track speaker emotion"
                    json.dump(sample_dict, manifest)
                    manifest.write("\n")

                    sample_dict['text'] = text_pos
                    sample_dict['tasks'] = ["transcription", "pos"]
                    sample_dict['instruction'] = "Transcribe and tag parts of speech of each word"
                    json.dump(sample_dict, manifest)
                    manifest.write("\n")

                    sample_dict['text'] = text_pos + " " + emotion_labels
                    sample_dict['tasks'] = ["transcription", "pos", "emotion"]
                    sample_dict['instruction'] = "Transcribe, tag parts of speech of each word and track speaker information"
                    json.dump(sample_dict, manifest)
                    manifest.write("\n")
    
            manifest.close()
    write_taglist(taglist,taglistfile)

### Librespeech: Get data, un-compress it and then set paths
#define paths to folders created afte unzipping
LIBRE = '/audio_datasets/EN_libre/'
TRAIN_DATA = Path(LIBRE+'/LibriSpeech/train-clean-360/')

DEV_DATA = Path(LIBRE+'/LibriSpeech/dev-clean/')


TEST_DATA = Path(LIBRE+'/LibriSpeech/test-clean/')

allpath = [DEV_DATA, TEST_DATA, TRAIN_DATA]

for datakey in allpath:
    print(datakey)
    taglistfile = LIBRE + datakey.name + '.taglist.txt'
    process_librispeech(datakey,taglistfile)
