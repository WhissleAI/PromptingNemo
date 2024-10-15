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
from ai4bharat.transliteration import XlitEngine

import concurrent.futures
import threading

'''
TODO:
1. all labels used are written properly to the taglist file
2. Multiprocessing for the tsv files
'''

### Define all data path (SLURP here)
indicSuperb = PurePath("~/Downloads/indicSuperb/kb_data_clean_m4a")
ktb_bn = indicSuperb / PurePath("bengali/")
bn_train_annotations = ktb_bn / PurePath("train/transcription_n2w.txt")
bn_val_annotations = ktb_bn / PurePath("valid/transcription_n2w.txt")
bn_test_annotations = ktb_bn / PurePath("test_known/transcription_n2w.txt")
bn_testunk_annotations = ktb_bn / PurePath("test/transcription_n2w.txt")


### Named entity tagger
ner_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicNER")
ner_model = AutoModelForTokenClassification.from_pretrained("ai4bharat/IndicNER")

# Audio Emotion Classification
# emotion_tokenizer = AutoTokenizer.from_pretrained("MilaNLProc/xlm-emo-t")
# emotion_model = XLMRobertaForSequenceClassification.from_pretrained("MilaNLProc/xlm-emo-t", problem_type="multi_label_classification")

#transliteration
transliterator = XlitEngine( beam_width=4, rescore=False, src_script_type = "indic")


def convert_m4a_to_wav(m4a_file_path, wav_file_path, sample_rate=16000):
    # Load the MP3 file
    audio = AudioSegment.from_file(m4a_file_path)
    duration_ms = len(audio)
    duration_seconds = duration_ms / 1000.0

    # Export as WAV
    audio = audio.set_frame_rate(sample_rate)
    
    audio.export(wav_file_path, format="wav")

    return duration_seconds

def get_ner_predictions( sentence, tokenizer, model ):
  # Let us first tokenize the sentence - split words into subwords
  tok_sentence = tokenizer(sentence, return_tensors='pt')

  with torch.no_grad():
    # we will send the tokenized sentence to the model to get predictions
    logits = model(**tok_sentence).logits.argmax(-1)
    
    # We will map the maximum predicted class id with the class label
    predicted_tokens_classes = [model.config.id2label[t.item()] for t in logits[0]]
    
    predicted_labels = []
    
    previous_token_id = 0
    # we need to assign the named entity label to the head word and not the following sub-words
    word_ids = tok_sentence.word_ids()
    for word_index in range(len(word_ids)):
        if word_ids[word_index] == None:
            previous_token_id = word_ids[word_index]
        elif word_ids[word_index] == previous_token_id:
            previous_token_id = word_ids[word_index]
        else:
            predicted_labels.append( predicted_tokens_classes[ word_index ] )
            previous_token_id = word_ids[word_index]
    
    return predicted_labels

def transliterate(text, lang_id):
    return transliterator.translit_sentence(text, lang_id)

def tag_ner(text, lang_id):
    """
    Wraps each token with its corresponding BIO tag and returns the wrapped text and a list of unique tags used.

    Parameters:
    text (str): The original text to be annotated.

    Returns:
    tuple: A tuple containing the wrapped text and a list of unique tags used.
    """
    tags = get_ner_predictions(sentence=text, 
                                   tokenizer=ner_tokenizer,
                                   model=ner_model
                                   )
    text = transliterate(text, lang_id)
    
    tokens = text.split(' ')

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

# def get_emotion_labels(text):

#     inputs = emotion_tokenizer(text, return_tensors="pt")

#     with torch.no_grad():
#         logits = emotion_model(**inputs).logits

#     predicted_class_id = torch.argmax(torch.sigmoid(logits).squeeze(dim=0))

#     final_emotion_label = emotion_model.config.id2label[predicted_class_id.item()]
#     final_emotion_label = "EMOTION_"+ final_emotion_label.upper()
#     return final_emotion_label

def write_taglist(taglist,filename):

    taglist = "\n".join(taglist)

    with open(filename, 'w') as f:
        f.write(taglist)
        f.write("\n")
    f.close()      

def process_tsv(langid, tsvfile, audioclips, audioclipswav, manifestfile, taglistfile, checkpoint_file):
    
    langid_orig = langid
    langid = "LANGUAGEID_"+langid.upper()
    
    tsvfile = pd.read_csv(tsvfile, sep="\t", names=['path', 'sentence'])


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
        
        duration = convert_m4a_to_wav(audiofile, wavfilepath)
        
        # wavfilepath =  PurePath("/working_dir/audio_datasets/CommonVoice/datasets/cv-corpus-15.0-2023-09-08/" + langid_orig + "/clips-wav/") / PurePath(row['path'].split(".")[0]+".wav")
        
        text = row['sentence']
        text = unidecode(text)

        text_ner, used_ner = tag_ner(text, langid_orig)
        taglist = taglist + used_ner
        try:
            # emotion_label = get_emotion_labels(text)
            # emotion_labels = [emotion_label]
            # taglist = taglist + emotion_labels
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

            # emotion_labels = ' '.join(emotion_labels)
            # sample_dict['text'] = langid + " " + text + " " + emotion_label
            # sample_dict['tasks'] = ["transcription", "emotion"]
            # sample_dict['instruction'] = "Transcribe and track speaker emotion"
            # json.dump(sample_dict, manifest)
            # manifest.write("\n")

            sample_dict['text'] = langid + " " + text_ner
            sample_dict['tasks'] = ["transcription", "ner"]
            sample_dict['instruction'] = "Transcribe and mark named entities"
            json.dump(sample_dict, manifest)
            manifest.write("\n")

            # sample_dict['text'] = langid + " " + text_ner + " " + emotion_label
            # sample_dict['tasks'] = ["transcription", "ner", "emotion"]
            # sample_dict['instruction'] = "Transcribe, mark named entities and track speaker emotion"
            # json.dump(sample_dict, manifest)
            # manifest.write("\n")

            # Update the checkpoint with the current row index
            with open(checkpoint_file, 'w') as checkpoint:
                checkpoint.write(str(index))
        except:
            continue

    manifest.close()
    write_taglist(taglist,taglistfile)


manifestfolder = "/home/ubuntu/working_dir/audio_datasets/manifests_euro"

audioclips_base = "/home/ubuntu/working_dir/audio_datasets/CommonVoice/datasets/cv-corpus-15.0-2023-09-08/"

# language_paths = {
#     "bn": f"{audioclips_base}it/clips",
#     "de": f"{audioclips_base}de/clips",
#     "es": f"{audioclips_base}es/clips",
#     "fr": f"{audioclips_base}fr/clips"
# }

# language_paths_wav = {
#     "it": f"{audioclips_base}it/clips-wav",
#     "de": f"{audioclips_base}de/clips-wav",
#     "es": f"{audioclips_base}es/clips-wav",
#     "fr": f"{audioclips_base}fr/clips-wav"
# }

# Define the file paths and parameters for each dataset
datasets = {

    "bn_val": {
        "langid": "bn",
        "tsvfile": bn_val_annotations,
        "audioclips": os.path.join(bn_val_annotations.parent, "audio/"),
        "audioclipswav": os.path.join(bn_val_annotations.parent, "audio_wav/"),
        "manifestfile": os.path.join(manifestfolder, "valid_ktb_bn.json"),
        "taglistfile": os.path.join(manifestfolder, "taglist_valid_bn.txt"),
        "checkpoint_file": os.path.join(manifestfolder,"checkpoint_ktb_bn_valid.txt")
    }
}

# Create directories if they don't exist
for lang_split in datasets:
    os.makedirs(datasets[lang_split]["audioclipswav"], exist_ok=True)

max_concurrent_jobs = 1

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
