# Process SLURP dataset for 1SSI: OneStep speech intructor

import os
import json
import re
import itertools
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
slurp_annotations = Path("/external2/datasets/slurp/dataset/slurp/")
train_annotations = slurp_annotations / Path("train.jsonl")
dev_annotations = slurp_annotations / Path("devel.jsonl")
test_annotations = slurp_annotations / Path("test.jsonl")

audio_real = Path("/external2/datasets/slurp/audio/slurp_real")
audio_synth = Path("/external2/datasets/slurp/audio/slurp_synth/")


### SLURP tag aligner
def convert_entity_format(text, taglist):
    # Regular expression to find any entity type pattern
    pattern = r'\[([a-zA-Z_]+) : ([^\]]+)\]'

    # Function to replace the found pattern
    def replace_pattern(match):
        entity_type = match.group(1).strip().upper()  # Convert entity type to uppercase
        entity_value = match.group(2).strip()

        begin_tag = "ENTITY-"+f"{entity_type}".upper()
        end_tag =  f"END"

        if begin_tag not in taglist:

            taglist.append(begin_tag)

        if end_tag not in taglist:
            taglist.append(end_tag)

        return f"{begin_tag} {entity_value} {end_tag}"

    # Replace all occurrences of the pattern in the text
    converted_text = re.sub(pattern, replace_pattern, text)

    return converted_text, taglist

# def merge_text_and_tags(text_clean, text_tagged):
#     # Regular expression pattern to match words (including contractions), tags, and punctuation
#     pattern = r"ENTITY-\d+|END|\w+(?:'\w+)?|[.,!?;]"

#     # Split the tagged text and clean text into their respective components
#     tagged_parts = re.findall(pattern, text_tagged)
#     clean_parts = re.findall(r"\w+(?:'\w+)?|[.,!?;]", text_clean)

#     merged_text = []
#     clean_iter = iter(clean_parts)

#     for part in tagged_parts:
#         if 'ENTITY-' in part or part == 'END':
#             merged_text.append(part)
#             # Check if next clean part is punctuation and append it if so
#             next_clean = next(clean_iter, None)
#             if next_clean and re.match(r"[.,!?;]", next_clean):
#                 merged_text.append(next_clean)
#         elif re.match(r"\w+(?:'\w+)?", part):
#             # Append the corresponding word from the clean text
#             word = next(clean_iter, '')
#             merged_text.append(word)
#             # Check if next clean part is punctuation and append it if so
#             next_clean = next(clean_iter, None)
#             if next_clean and re.match(r"[.,!?;]", next_clean):
#                 merged_text.append(next_clean)

#     return ' '.join(merged_text)

def merge_text_and_tags(text_clean, text_tagged):
    # Regular expression pattern to match words (including contractions), tags, and punctuation
    pattern = r"ENTITY-\d+|END|\w+(?:'\w+)?|[.,!?;]"

    # Split the tagged text and clean text into their respective components
    tagged_parts = re.findall(pattern, text_tagged)
    clean_parts = re.findall(r"\w+(?:'\w+)?|[.,!?;]", text_clean)

    merged_text = []
    clean_iter = iter(clean_parts)
    last_tag_was_end = False

    for part in tagged_parts:
        if part.startswith('ENTITY-'):
            # Append entity tags
            merged_text.append(part)
            last_tag_was_end = False
        elif part == 'END':
            # Append 'END' tag and peek next for punctuation
            merged_text.append(part)
            next_clean = next(clean_iter, None)
            if next_clean and re.match(r"[.,!?;]", next_clean):
                merged_text.append(next_clean)
            last_tag_was_end = True
        elif part.isalpha():
            if not last_tag_was_end:
                # Append the corresponding word from the clean text
                word = next(clean_iter, '')
                merged_text.append(word)
            last_tag_was_end = False
        else:
            # Append punctuation
            if not last_tag_was_end:
                merged_text.append(part)

    # Check if any unprocessed clean parts are left and append them
    while True:
        next_clean = next(clean_iter, None)
        if next_clean is None:
            break
        merged_text.append(next_clean)

    # Join the parts with a space but avoid double spacing
    return ' '.join(merged_text).replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?')



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



def convert_single_flac_to_wav(audiofile, wavfile):

    # Load the .flac file
    audio = AudioSegment.from_file(audiofile, format="flac")
    
    # Set frame rate to 16kHz
    audio = audio.set_frame_rate(16000)
    
    # Export as .wav file
    audio.export(wavfile, format="wav")


def jsonl_process(jsonlfile, audiofolder, manifestfolder):

    wavfolder = str(audiofolder) + "-wav"
    os.system("mkdir -p "+wavfolder)
    wavfolder = Path(wavfolder)

    jsonlfileread = open(str(jsonlfile),'r').readlines()

    manifest = open(manifestfolder + "/" + jsonlfile.name.replace(jsonlfile.suffix, "") + "-slurp-tagged.json",'w')


    taglist = []

    for line in jsonlfileread:

        line = json.loads(line)
        #print(line)
        annotation = line['sentence_annotation']
        text = line['sentence']
        #text_clean = normalize(text)
        text_clean = text + "."

        text_tagged, taglist = convert_entity_format(line['sentence_annotation'], taglist)
        text_tagged = text_tagged + "."

        #text_clean_tagged = merge_text_and_tags(text_clean, text_tagged)
        text_clean_tagged = text_tagged

        intent = line['intent'].upper()
        intent = "INTENT-"+intent

        if intent not in taglist:
            
            taglist.append(intent) 
        

        recordings = line['recordings']

        #print("Final text:", text_clean_tagged)

        for recording in recordings:
            audiofile = recording['file']
            audiofilepath = audiofolder / Path(audiofile)

            audiofile = PurePath(audiofile)
            filekey = audiofile.name.replace(audiofile.suffix, "")
            wavfilepath = str(wavfolder) + "/" + filekey + ".wav"
            
            convert_single_flac_to_wav(audiofilepath, wavfilepath)
            
            #flac_tmp_audio_data = AudioSegment.from_file(audiofilepath, audiofilepath.suffix[1:])
            #flac_tmp_audio_data.export(wavfilepath, format="wav")        

            sample_dict = {}
            sample_dict['audio_filepath'] = wavfilepath

            emotion_labels = get_emotion_labels(audio_file=wavfilepath, sampling_rate=16000)
            for label in emotion_labels:
                if label not in taglist:
                    taglist.append(label)
            emotion_labels = ' '.join(emotion_labels)

            sample_dict['text'] = text_clean_tagged + " " + intent + " " + emotion_labels
            sample_dict['tasks'] = ["transcription", "entity", "intent","emotion"]
            sample_dict['instruction'] = "transcribe, track entities, get intent and emotion"
            json.dump(sample_dict, manifest)
            manifest.write("\n")        
    
    manifest.close()
    taglistfile = open(manifestfolder+"/taglistfile.json",'w')
    json.dump(taglist,taglistfile)
    taglistfile.close()



manifestfolder = "/external2/datasets/slurp"

jsonl_process(jsonlfile=train_annotations, audiofolder=audio_real, manifestfolder=manifestfolder)
jsonl_process(jsonlfile=dev_annotations, audiofolder=audio_real, manifestfolder=manifestfolder)
jsonl_process(jsonlfile=test_annotations, audiofolder=audio_real, manifestfolder=manifestfolder)
