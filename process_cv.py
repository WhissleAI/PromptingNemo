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
import nemo.collections as nemo_nlp
from flair.data import Sentence
from flair.models import SequenceTagger
from AudioEmotionClassification.models import (Wav2Vec2ForSpeechClassification, 
                                               HubertForSpeechClassification)

### Intitiate text normalizer and puctuator
#normalizer = Normalizer(input_case='lower_cased', lang="en")
#punctuator = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained("punctuation_en_distilbert")

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

# Audio Emotion Classification
emotion_model = HubertForSpeechClassification.from_pretrained("Rajaram1996/Hubert_emotion")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
sampling_rate=16000 # defined by the model; must convert mp3 to this rate.
emotion_config = AutoConfig.from_pretrained("Rajaram1996/Hubert_emotion")



def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    # Load the MP3 file
    audio = AudioSegment.from_mp3(mp3_file_path)

    # Export as WAV
    audio.export(wav_file_path, format="wav")



def normalize(text):

    text = text.lower()
    normalized = normalizer.normalize(text, verbose=True, punct_post_process=True)
    normalized = [normalized]
    norm_punctuated = punctuator.add_punctuation_capitalization(normalized)[0]
    return norm_punctuated




def tag_entities(text, lowercase=True):

    ner_results = hf_nlp(text)

    taglist = []
    # example: [{'entity_group': 'PER', 'score': 0.8913538, 'word': 'Min', 'start': 0, 'end': 3}, {'entity_group': 'LOC', 'score': 0.9983326, 'word': 'West Van Buren Street', 'start': 93, 'end': 114}]
    for ner_dict in ner_results:

        entity_group = ner_dict['entity_group'].upper()
        start = ner_dict['start']
        end = ner_dict['end']
        word = ner_dict['word']

        if lowercase:
            word = word.lower()
        
        begin_tag = "ENTITY-"+entity_group
        end_tag = "END"

        text = text.replace(word, begin_tag+" "+word+" "+end_tag)

    return text, taglist

### Part of Speech Tagger
# load tagger
pos_tagger = SequenceTagger.load("flair/pos-english")


def tag_pos(text, taglist):
    sentence = Sentence(text)
    pos_tagger.predict(sentence)

    tagged = []
    for entity in sentence.labels:

        word = entity.data_point.text
        tag = entity.value


        if tag not in string.punctuation:

            if tag not in taglist:
                taglist.append(tag)
            if "END" not in taglist:
                taglist.append("END")
            
            tagged.append(tag)
            tagged.append(word)
            tagged.append("END")
        else:
            tagged.append(word)

    text = " ".join(tagged)
    return text, taglist



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

def get_emotion_labels(audio_file, taglist, sampling_rate=16000, score=50.0):
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

    all_labels = []
    for emotion_dict in emotion_labels:

        label = "EMOTION-"+emotion_dict['emo'].split("_")[1].upper()
        score = emotion_dict['score']

        if score > 50.0:
            all_labels.append(label)

        if label not in taglist:
            taglist.append(label)
        
    return all_labels, taglist

def write_taglist(taglist,filename):

    taglist = "\n".join(taglist)

    with open(filename, 'w') as f:
        f.write(taglist)
        f.write("\n")
    f.close()

def process_tsv(tsvfile, audioclips, audioclipswav, manifestfile, taglistfile, checkpoint_file):
    
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
    for index, row in tqdm(enumerate(tsvfile.iterrows()), total=len(tsvfile), initial=last_processed_row):
        
        # Skip rows that have already been processed
        if index < last_processed_row:
            continue
        
        row = row[1]
        audiofile = audioclips / Path(row['path'])
        wavfilepath = audioclipswav / PurePath(row['path'].split(".")[0]+".wav")
        
        convert_mp3_to_wav(audiofile, wavfilepath)
        
        text = row['sentence']
        text_pos, taglist = tag_pos(text, taglist)
        text_tagged, taglist = tag_entities(text, taglist)
        emotion_labels, taglist = get_emotion_labels(audio_file=wavfilepath, taglist=taglist, sampling_rate=16000)
        text_tagged_emotion = text_tagged + " " + " ".join(emotion_labels)

        wavfilepath = str(wavfilepath)

        sample_dict = {}
        sample_dict['audio_filepath'] = wavfilepath
        sample_dict['text'] = text
        sample_dict['tasks'] = ["transcription"]
        sample_dict['instruction'] = "Transcribe what is begin spoken"
        json.dump(sample_dict, manifest)
        manifest.write("\n")

        emotion_labels = ' '.join(emotion_labels)
        sample_dict['text'] = text_tagged_emotion
        sample_dict['tasks'] = ["transcription", "emotion"]
        sample_dict['instruction'] = "Transcribe and track speaker emotion"
        json.dump(sample_dict, manifest)
        manifest.write("\n")

        sample_dict['text'] = text_tagged
        sample_dict['tasks'] = ["transcription", "entity-named"]
        sample_dict['instruction'] = "Transcribe and mark named entities"
        json.dump(sample_dict, manifest)
        manifest.write("\n")

        sample_dict['text'] = text_tagged + " " + emotion_labels
        sample_dict['tasks'] = ["transcription", "entity-named", "emotion"]
        sample_dict['instruction'] = "Transcribe, mark named entities and speaker emotion"
        json.dump(sample_dict, manifest)
        manifest.write("\n")

        sample_dict['text'] = text_tagged + " " + emotion_labels
        sample_dict['tasks'] = ["transcription", "entity-named", "emotion"]
        sample_dict['instruction'] = "Transcribe, track speaker emotion and mark named entities"
        json.dump(sample_dict, manifest)
        manifest.write("\n")

        sample_dict['text'] = text_tagged + " " + emotion_labels
        sample_dict['tasks'] = ["transcription", "entity-named", "emotion"]
        sample_dict['instruction'] = "Transcribe, track speaker emotion and mark named entities"
        json.dump(sample_dict, manifest)
        manifest.write("\n")

        sample_dict['text'] = text_tagged + " " + emotion_labels
        sample_dict['tasks'] = ["transcription", "entity-named", "emotion"]
        sample_dict['instruction'] = "Transcribe, track speaker emotion and mark named entities"
        json.dump(sample_dict, manifest)
        manifest.write("\n")

        sample_dict['text'] = text_pos + " " + emotion_labels
        sample_dict['tasks'] = ["transcription", "parts-of-speech", "emotion"]
        sample_dict['instruction'] = "Transcribe, tag parts-of-speech and track speaker emotion"
        json.dump(sample_dict, manifest)
        manifest.write("\n")

        sample_dict['text'] = text_pos + " " + emotion_labels
        sample_dict['tasks'] = ["transcription", "parts-of-speech"]
        sample_dict['instruction'] = "Transcribe and tag parts-of-speech"
        json.dump(sample_dict, manifest)
        manifest.write("\n")

        # Update the checkpoint with the current row index
        with open(checkpoint_file, 'w') as checkpoint:
            checkpoint.write(str(index))

    manifest.close()
    write_taglist(taglist,taglistfile)


manifestfolder = "/audio_datasets/manifests"


# Define the file paths and parameters for each dataset
datasets = {
    "dev": {
        "tsvfile": dev_annotations,
        "audioclips": audioclips,
        "audioclipswav": audioclipswav,
        "manifestfile": os.path.join(manifestfolder, "dev_cv_en.json"),
        "taglistfile": os.path.join(manifestfolder, "taglist_dev_en.txt"),
        "checkpoint_file": os.path.join(manifestfolder,"checkpoint_cv_dev.txt")
    },
    "train": {
        "tsvfile": train_annotations,
        "audioclips": Path(audioclips),
        "audioclipswav": audioclipswav,
        "manifestfile": os.path.join(manifestfolder, "train_cv_en.json"),
        "taglistfile": os.path.join(manifestfolder, "taglist_train_en.txt"),
        "checkpoint_file": os.path.join(manifestfolder,"checkpoint_cv_train.txt")
    },
    "test": {
        "tsvfile": test_annotations,
        "audioclips": audioclips,
        "audioclipswav": audioclipswav,
        "manifestfile": os.path.join(manifestfolder, "test_cv_en.json"),
        "taglistfile": os.path.join(manifestfolder, "taglist_test_en.txt"),
        "checkpoint_file": os.path.join(manifestfolder,"checkpoint_cv_test.txt")
    }
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