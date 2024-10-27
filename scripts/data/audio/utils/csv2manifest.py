import json
import os
import librosa
import re
from pathlib import Path

language = "marathi"
DATA_FOLDER = Path("/projects/whissle/datasets/") / language

AUDIO_FOLDER = DATA_FOLDER / "wavs_train"
annotation_file = DATA_FOLDER / "marathi_annotated_data_train.json"

output_file_path = DATA_FOLDER / "train_manifest.json"

keyword_file = open(DATA_FOLDER / "keywords.txt", "w", encoding="utf-8")


#AUDIO_FOLDER = os.path.join(DATA_FOLDER, "wavs_train/")
#annotation_file = "annotated_data.json"
#output_file_path = os.path.join(DATA_FOLDER, "train_manifest.json")

#keyword_file = open(os.path.join(DATA_FOLDER, "keywords.txt"), "w", encoding="utf-8")

# Open files with UTF-8 encoding
print("Loading data from", os.path.join(DATA_FOLDER, annotation_file))
with open(os.path.join(DATA_FOLDER, annotation_file), "r", encoding="utf-8") as f:
    all_lines = json.load(f)

# Open output file with UTF-8 encoding
output_file = open(output_file_path, "w", encoding="utf-8")

def get_capitalized_words(text):
    # Regex to capture uppercase words with underscores, digits, and symbols like + or -
    capitalized_words = re.findall(r'\b[A-Z0-9_+\-]+\b', text)
    keywords = []

    for word in capitalized_words:
        if word == "AGE_60":
            word = "AGE_60+"
        if "_" in word:
            keywords.extend(split_words(word))
        else:
            keywords.append(word)
    
    return keywords

def split_words(text):
    # Split by one or more underscores and filter out empty strings
    return [word for word in re.split(r'_+', text) if word]

print("Total lines:", len(all_lines))
all_keywords = []

for line in all_lines:
    sample = {}
    fileid = os.path.basename(line['path'])
    text = line['Final Output']

    audio_file = os.path.join(AUDIO_FOLDER, fileid)
    
    if not os.path.exists(audio_file):
        print("Missing file:", audio_file)
    else:
        sample['audio_filepath'] = audio_file
        sample['duration'] = librosa.get_duration(filename=audio_file)
        sample['text'] = text

        # Extract keywords from text
        keywords = get_capitalized_words(text)

        for keyword in keywords:
            if keyword not in all_keywords:
                all_keywords.append(keyword)

        # Write sample to output file with UTF-8 encoding
        json.dump(sample, output_file, ensure_ascii=False)
        output_file.write("\n")

print("Total keywords:", len(all_keywords))
print(all_keywords[:50])
for keyword in all_keywords:
    keyword_file.write(keyword + "\n")
keyword_file.close()

print(all_lines[0])

# Close the output file
output_file.close()
