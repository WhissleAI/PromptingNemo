import os
import json
import pandas as pd
from pathlib import Path, PurePath
from unidecode import unidecode
from pydub import AudioSegment
from tqdm import tqdm
import concurrent.futures
import torch
from transformers import (AutoTokenizer, AutoModelForTokenClassification, 
                          XLMRobertaForSequenceClassification, pipeline)

import concurrent.futures
import threading

'''
TODO:
1. all labels used are written properly to the taglist file
2. Multiprocessing for the tsv files
'''

### Define all data path (SLURP here)
common_voice = PurePath("~/working_dir/audio_datasets/CommonVoice/datasets/cv-corpus-15.0-2023-09-08/")
cv_italian = common_voice / PurePath("it/")
it_train_annotations = cv_italian / PurePath("train.tsv")
it_dev_annotations = cv_italian / PurePath("dev.tsv")
it_test_annotations = cv_italian / PurePath("test.tsv")

cv_german = common_voice / PurePath("de/")
de_train_annotations = cv_german / PurePath("train.tsv")
de_dev_annotations = cv_german / PurePath("dev.tsv")
de_test_annotations = cv_german / PurePath("test.tsv")

cv_spanish = common_voice / PurePath("es/")
es_train_annotations = cv_spanish / PurePath("train.tsv")
es_dev_annotations = cv_spanish / PurePath("dev.tsv")
es_test_annotations = cv_spanish / PurePath("test.tsv")

cv_french = common_voice / PurePath("fr/")
fr_train_annotations = cv_french / PurePath("train.tsv")
fr_dev_annotations = cv_french / PurePath("dev.tsv")
fr_test_annotations = cv_french / PurePath("test.tsv")



### Named entity tagger
ner_tokenizer = AutoTokenizer.from_pretrained("Davlan/xlm-roberta-large-ner-hrl")
ner_model = AutoModelForTokenClassification.from_pretrained("Davlan/xlm-roberta-large-ner-hrl")
entity_tagger = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy='simple')

# Audio Emotion Classification
emotion_tokenizer = AutoTokenizer.from_pretrained("MilaNLProc/xlm-emo-t")
emotion_model = XLMRobertaForSequenceClassification.from_pretrained("MilaNLProc/xlm-emo-t", problem_type="multi_label_classification")

def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    # Load the MP3 file
    audio = AudioSegment.from_mp3(mp3_file_path)
    duration_ms = len(audio)
    duration_seconds = duration_ms / 1000.0

    # Export as WAV
    audio.export(wav_file_path, format="wav")

    return duration_seconds

def tag_ner(text):
    """
    Wraps each token with its corresponding BIO tag and returns the wrapped text and a list of unique tags used.

    Parameters:
    text (str): The original text to be annotated.

    Returns:
    tuple: A tuple containing the wrapped text and a list of unique tags used.
    """
    entities = entity_tagger(text)
    used_tags = set()
    prev_end = 0
    wrap_text = []

    for entity in entities:
        start = entity['start']
        end = entity['end']
        entity_class = "NER_" + entity['entity_group']

        wrap_text.append(text[prev_end:start])
        wrap_text.append(entity_class)
        wrap_text.append(text[start:end])
        wrap_text.append('END')
        used_tags.add("END")
        used_tags.add(entity_class)
        prev_end = end

    wrap_text.append(text[prev_end+1:])
    wrap_text = [i.strip() for i in wrap_text]

    return ' '.join(wrap_text), list(used_tags)

def get_emotion_labels(text):

    inputs = emotion_tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        logits = emotion_model(**inputs).logits

    predicted_class_id = torch.argmax(torch.sigmoid(logits).squeeze(dim=0))

    final_emotion_label = emotion_model.config.id2label[predicted_class_id.item()]
    final_emotion_label = "EMOTION_"+ final_emotion_label.upper()
    return final_emotion_label

def write_taglist(taglist,filename):

    taglist = "\n".join(taglist)

    with open(filename, 'w') as f:
        f.write(taglist)
        f.write("\n")
    f.close()

def process_tsv(langid, tsvfile, audioclips, audioclipswav, manifestfile, taglistfile, checkpoint_file):
    
    langid_orig = langid
    langid = "LANGUAGEID_"+langid.upper()
    
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
    if langid not in taglist:
        taglist.append(langid) 
    
    for index, row in tqdm(enumerate(tsvfile.iterrows()), total=len(tsvfile), initial=last_processed_row):
        
        # Skip rows that have already been processed
        if index < last_processed_row:
            continue
        
        row = row[1]
        audiofile = audioclips / Path(row['path'])
        wavfilepath = audioclipswav / PurePath(row['path'].split(".")[0]+".wav")
        
        duration = convert_mp3_to_wav(audiofile, wavfilepath)
        
        wavfilepath =  PurePath("/working_dir/audio_datasets/CommonVoice/datasets/cv-corpus-15.0-2023-09-08/" + langid_orig + "/clips-wav/") / PurePath(row['path'].split(".")[0]+".wav")
        
        text = row['sentence']
        text = unidecode(text)

        text_ner, used_ner = tag_ner(text)
        taglist = taglist + used_ner

        emotion_label = get_emotion_labels(text)
        emotion_labels = [emotion_label]
        taglist = taglist + emotion_labels
        # print(text, taglist)
        taglist = list(set(taglist)) # remove duplicates
        
        wavfilepath = str(wavfilepath)

        sample_dict = {}
        sample_dict['duration'] = duration
        sample_dict['audio_filepath'] = wavfilepath
        sample_dict['text'] = langid + " " + text
        sample_dict['tasks'] = ["transcription"]
        sample_dict['instruction'] = "Transcribe what is begin spoken"
        json.dump(sample_dict, manifest)
        manifest.write("\n")

        emotion_labels = ' '.join(emotion_labels)
        sample_dict['text'] = langid + " " + text + " " + emotion_label
        sample_dict['tasks'] = ["transcription", "emotion"]
        sample_dict['instruction'] = "Transcribe and track speaker emotion"
        json.dump(sample_dict, manifest)
        manifest.write("\n")

        sample_dict['text'] = langid + " " + text_ner
        sample_dict['tasks'] = ["transcription", "ner"]
        sample_dict['instruction'] = "Transcribe and mark named entities"
        json.dump(sample_dict, manifest)
        manifest.write("\n")

        sample_dict['text'] = langid + " " + text_ner + " " + emotion_label
        sample_dict['tasks'] = ["transcription", "ner", "emotion"]
        sample_dict['instruction'] = "Transcribe, mark named entities and track speaker emotion"
        json.dump(sample_dict, manifest)
        manifest.write("\n")

        # Update the checkpoint with the current row index
        with open(checkpoint_file, 'w') as checkpoint:
            checkpoint.write(str(index))

    manifest.close()
    write_taglist(taglist,taglistfile)


manifestfolder = "/home/ubuntu/working_dir/audio_datasets/manifests_euro"

audioclips_base = "/home/ubuntu/working_dir/audio_datasets/CommonVoice/datasets/cv-corpus-15.0-2023-09-08/"

language_paths = {
    "it": f"{audioclips_base}it/clips",
    "de": f"{audioclips_base}de/clips",
    "es": f"{audioclips_base}es/clips",
    "fr": f"{audioclips_base}fr/clips"
}

language_paths_wav = {
    "it": f"{audioclips_base}it/clips-wav",
    "de": f"{audioclips_base}de/clips-wav",
    "es": f"{audioclips_base}es/clips-wav",
    "fr": f"{audioclips_base}fr/clips-wav"
}

# Create directories if they don't exist
for path in language_paths_wav.values():
    os.makedirs(path, exist_ok=True)


# Define the file paths and parameters for each dataset
datasets = {
    "it_dev": {
        "langid": "it",
        "tsvfile": it_dev_annotations,
        "audioclips": language_paths['it'],
        "audioclipswav": language_paths_wav['it'],
        "manifestfile": os.path.join(manifestfolder, "dev_cv_it.json"),
        "taglistfile": os.path.join(manifestfolder, "taglist_dev_it.txt"),
        "checkpoint_file": os.path.join(manifestfolder,"checkpoint_cv_it_dev.txt")
    },
    "it_test": {
        "langid": "it",
        "tsvfile": it_test_annotations,
        "audioclips": language_paths['it'],
        "audioclipswav": language_paths_wav['it'],
        "manifestfile": os.path.join(manifestfolder, "test_cv_it.json"),
        "taglistfile": os.path.join(manifestfolder, "taglist_test_it.txt"),
        "checkpoint_file": os.path.join(manifestfolder,"checkpoint_cv_it_test.txt")
    },
    "it_train": {
        "langid": "it",
        "tsvfile": it_train_annotations,
        "audioclips": language_paths['it'],
        "audioclipswav": language_paths_wav['it'],
        "manifestfile": os.path.join(manifestfolder, "train_cv_it.json"),
        "taglistfile": os.path.join(manifestfolder, "taglist_train_it.txt"),
        "checkpoint_file": os.path.join(manifestfolder,"checkpoint_cv_it_train.txt")
    },
    "de_dev": {
        "langid": "de",
        "tsvfile": de_dev_annotations,
        "audioclips": language_paths['de'],
        "audioclipswav": language_paths_wav['de'],
        "manifestfile": os.path.join(manifestfolder, "dev_cv_de.json"),
        "taglistfile": os.path.join(manifestfolder, "taglist_dev_de.txt"),
        "checkpoint_file": os.path.join(manifestfolder,"checkpoint_cv_de_dev.txt")
    },
    "de_test": {
        "langid": "de",
        "tsvfile": de_test_annotations,
        "audioclips": language_paths['de'],
        "audioclipswav": language_paths_wav['de'],
        "manifestfile": os.path.join(manifestfolder, "test_cv_de.json"),
        "taglistfile": os.path.join(manifestfolder, "taglist_test_de.txt"),
        "checkpoint_file": os.path.join(manifestfolder,"checkpoint_cv_de_test.txt")
    },
    "de_train": {
        "langid": "de",
        "tsvfile": de_train_annotations,
        "audioclips": language_paths['de'],
        "audioclipswav": language_paths_wav['de'],
        "manifestfile": os.path.join(manifestfolder, "train_cv_de.json"),
        "taglistfile": os.path.join(manifestfolder, "taglist_train_de.txt"),
        "checkpoint_file": os.path.join(manifestfolder,"checkpoint_cv_de_train.txt")
    },
    "es_dev": {
        "langid": "es",
        "tsvfile": es_dev_annotations,
        "audioclips": language_paths['es'],
        "audioclipswav": language_paths_wav['es'],
        "manifestfile": os.path.join(manifestfolder, "dev_cv_es.json"),
        "taglistfile": os.path.join(manifestfolder, "taglist_dev_es.txt"),
        "checkpoint_file": os.path.join(manifestfolder,"checkpoint_cv_es_dev.txt")
    },
    "es_test": {
        "langid": "es",
        "tsvfile": es_test_annotations,
        "audioclips": language_paths['es'],
        "audioclipswav": language_paths_wav['es'],
        "manifestfile": os.path.join(manifestfolder, "test_cv_es.json"),
        "taglistfile": os.path.join(manifestfolder, "taglist_test_es.txt"),
        "checkpoint_file": os.path.join(manifestfolder,"checkpoint_cv_es_test.txt")
    },
    "es_train": {
        "langid": "es",
        "tsvfile": es_train_annotations,
        "audioclips": language_paths['es'],
        "audioclipswav": language_paths_wav['es'],
        "manifestfile": os.path.join(manifestfolder, "train_cv_es.json"),
        "taglistfile": os.path.join(manifestfolder, "taglist_train_es.txt"),
        "checkpoint_file": os.path.join(manifestfolder,"checkpoint_cv_es_train.txt")
    },
    "fr_dev": {
        "langid": "fr",
        "tsvfile": fr_dev_annotations,
        "audioclips": language_paths['fr'],
        "audioclipswav": language_paths_wav['fr'],
        "manifestfile": os.path.join(manifestfolder, "dev_cv_fr.json"),
        "taglistfile": os.path.join(manifestfolder, "taglist_dev_fr.txt"),
        "checkpoint_file": os.path.join(manifestfolder,"checkpoint_cv_fr_dev.txt")
    },
    "fr_test": {
        "langid": "fr",
        "tsvfile": fr_test_annotations,
        "audioclips": language_paths['fr'],
        "audioclipswav": language_paths_wav['fr'],
        "manifestfile": os.path.join(manifestfolder, "test_cv_fr.json"),
        "taglistfile": os.path.join(manifestfolder, "taglist_test_fr.txt"),
        "checkpoint_file": os.path.join(manifestfolder,"checkpoint_cv_fr_test.txt")
    },
    "fr_train": {
        "langid": "fr",
        "tsvfile": fr_train_annotations,
        "audioclips": language_paths['fr'],
        "audioclipswav": language_paths_wav['fr'],
        "manifestfile": os.path.join(manifestfolder, "train_cv_fr.json"),
        "taglistfile": os.path.join(manifestfolder, "taglist_train_fr.txt"),
        "checkpoint_file": os.path.join(manifestfolder,"checkpoint_cv_fr_train.txt")
    },
}


max_concurrent_jobs = 8

def run_jobs(semaphore, dataset):
    with semaphore:
        process_tsv(**dataset)

# Use threading.Semaphore instead of concurrent.futures.Semaphore
with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_jobs) as executor:
    semaphore = threading.Semaphore(max_concurrent_jobs)
    futures = [executor.submit(run_jobs, semaphore, dataset) for dataset in list(datasets.values())[:max_concurrent_jobs]]

    # Wait for all futures to complete
    for future in concurrent.futures.as_completed(futures):
        future.result()