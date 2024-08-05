import re
import os
import time
from pathlib import Path
import random
from langdetect import detect, detect_langs
import unicodedata
import sys
import glob

language_codes = [
    "EN",  # English
    "ES",  # Spanish
    "FR",  # French
    "DE",  # German
    "ZH",  # Chinese
    "JA",  # Japanese
    "RU",  # Russian
    "PT",  # Portuguese
    "AR",  # Arabic
    "HI",  # Hindi
    "BN",  # Bengali
    "PA",  # Punjabi
    "ID",  # Indonesian
    "KO",  # Korean
    "VI",  # Vietnamese
    "IT",  # Italian
    "TR",  # Turkish
    "FA",  # Persian
    "PL",  # Polish
    "NL",  # Dutch
    "MR",  # Marathi
    "SV"   # Swedish
]

# Create a translation table for full-width to half-width punctuation
translation_table = str.maketrans(
    {
        "。": ".",
        "，": ",",
        "！": "!",
        "？": "?",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "「": "{",
        "」": "}",
        "『": "<",
        "』": ">",
        "《": "<",
        "》": ">",
        "〈": "<",
        "〉": ">",
        "．": ".",
        "、": ",",
        "·": "."
    }
)

def clean_extra(sentence):
    # Extract ENTITY and INTENT
    entities = re.findall(r'ENTITY_[A-Z_]+ .*? END', sentence)
    intent = re.search(r'INTENT_[A-Z_]+', sentence)
    
    # Join the extracted parts
    cleaned_sentence = ' '.join(entities)
    if intent:
        cleaned_sentence += ' ' + intent.group()
    
    return cleaned_sentence

def load_tag_file(tag_file):
    tag_dict = {}
    with open(tag_file, 'r') as file:
        for line in file:
            tag, tag_num = line.strip().split()
            tag_dict[tag] = tag_num
    return tag_dict


def clean_data(text):
    # Extract words that start with ENTITY_, INTENT_, DOMAIN-, or END
    
    tags = re.findall(r'\b(ENTITY-\w+|INTENT-\w+|LANG_\w+|DOMAIN-[\w-]+|END)', text)
    
    # Split tags into separate components
    split_tags = []
    
    for tag in tags:
        if 'LANG_' in tag:
            split_tags.append('LANG')
            split_tags.extend(tag.split('_')[1:])
        if 'ENTITY-' in tag:
            split_tags.append('ENTITY')
            split_tags.extend(tag.split('_')[1:])
        elif 'INTENT-' in tag:
            split_tags.append('INTENT')
            split_tags.extend(tag.split('_')[1:])
        elif 'DOMAIN-' in tag:
            split_tags.append('DOMAIN')
            split_tags.extend(tag.split('-')[1:])
        else:
            split_tags.append(tag)
    
    # Remove the word "END"
    text = text.replace("END", "")
    
    
    # Remove words that start with ENTITY_, INTENT_, and DOMAIN-
    text = re.sub(r'\bENTITY-\w+', '', text)
    text = re.sub(r'\bINTENT-\w+', '', text)
    text = re.sub(r'\bDOMAIN-[\w-]+', '', text)
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text, split_tags

# Helper function to replace tags in a compound word
def replace_tags(compound_word, tag_to_number):
    # Split by '-' or '_'
    components = re.split(r'[-_]', compound_word)
    
    # Replace each component with its corresponding T<num>
    replaced_components = [f"{tag_to_number.get(component, component)}" for component in components]
    # Join them back with the original separator
    if '-' in compound_word:
        return '-'.join(replaced_components)
    else:
        return '_'.join(replaced_components)
    
def capitalize_intents(text):
    # Capitalize words that start with INTENT_
    return re.sub(r'\b(INTENT_\w+)', lambda match: match.group(1).upper(), text)


def clean_file(input_file, text_file, text_tagged_file, tags_file):
    all_tags = set()
    valid_sents = []
    
    infile = open(input_file, 'r').readlines()
    outfile = open(text_file, 'w')
    
    
    for line in infile:
        if len(line.strip().split()) < 100:
            langid, line = clean_noisy_line(line)
            #print(line)
            valid = check_noisy(line)
            #print(valid)
            if valid != "NOISY_LINE":
                    cleaned_line, tags = clean_data(line)
                    #print("Cleaned line:", cleaned_line)
                    #print("Tags:", tags)
                    #print(line)
                    all_tags.update(tags)
                    outfile.write("LANG_"+langid + " " + cleaned_line + "\n")
                    #line = clean_extra(line)
                    valid_sents.append(line)
            #else:
            #    print("Noisy line: ", line)
                    
    outfile.close()

    print("Valid sentences: ", langid, len(valid_sents))
    
    print("example valid sentence: ", valid_sents[:10])

    print("total tags: ", len(all_tags))
    
    if not tags_file.exists():
        tagfile = open(tags_file, "w")
        tag_dict = {}
        special_tags = ["EOS", "TASK", "TAGGER", "TRANSCRIBE", "UNK"]
        for number, tag in enumerate(special_tags, start=0):
            tag_dict[tag] = "T" + str(number)
            tagfile.write(f"{tag} T{number}\n")
    else:
        tag_dict = load_tag_file(tags_file)
        tagfile = open(tags_file, "a")
    
    start = len(tag_dict.keys())
    all_tags = all_tags - set(tag_dict.keys())
    all_tags = list(all_tags)
    for number, tag in enumerate(all_tags, start=start):
            tag_dict[tag] = "T" + str(number)
            tagfile.write(f"{tag} T{number}\n")
    tagfile.close()
    
    
    
    with open(text_tagged_file, 'w') as taggedfile:
        for sent in valid_sents:
            sent = "TASK TAGGER " + sent + " EOS"
            #print(sent)
            words = sent.split()
            tagged_words = []
            for word in words:

                if '-' in word or '_' in word:
                    tagged_words.append(replace_tags(word, tag_dict))
                elif word in tag_dict:
                    tagged_words.append(tag_dict[word])
                else:
                    tagged_words.append(word)
            
            tagged_sent = " ".join(tagged_words)
        
            #tagged_sent = " ".join(replace_tags(word, tag_dict) for word in words)
            taggedfile.write(tagged_sent+"\n")


def create_cleaned_input_file(input_file, cleaned_file):
    with open(input_file, 'r') as infile, open(cleaned_file, 'w') as outfile:
        for line in infile:
            capitalized_line = capitalize_intents(line)
            outfile.write(capitalized_line)


def check_noisy(line):
    # Define regex patterns to match valid lines with the correct spacing and tags
    entity_pattern = re.compile(r"ENTITY-[A-Z_]+ [^ ]+ .*? END")
    intent_pattern = re.compile(r"INTENT-[A-Z_]+")
    
    # Find all entities and intents in the line
    entities = entity_pattern.findall(line)
    intents = intent_pattern.findall(line)
    
    
    # Check if each ENTITY has a corresponding END
    entity_start_pattern = re.compile(r"ENTITY-[A-Z_]+ [^ ]+")
    entity_end_pattern = re.compile(r"END")
    
    entity_starts = entity_start_pattern.findall(line)
    entity_ends = entity_end_pattern.findall(line)
    

    # Ensure the number of entity starts matches the number of entity ends
    if len(entity_starts) == len(entity_ends) and len(intents) > 0:
        output = "CLEAN_LINE"
    else:
        output = "NOISY_LINE"
    
    return output


def clean_noisy_line(line):
    # Remove all characters that are not alphabets, digits, or spaces
    

    line = line.strip().split(" ")
    
    langid = line[0].split("_")[1]
    line = " ".join(line[1:])
    
    line = line.replace("END.", "END .")
    
    return langid, line




            
def clean_files(input_folder):
    
    
    input_folder = Path(input_folder)
    output_folder = Path(input_folder / "processed")
    os.system(f"mkdir -p {output_folder}")
    
    input_files = list(input_folder.glob("*.txt"))
    tags_file = output_folder / f"alltags.txt"

    for input_file in input_files:
        text_tagged_file = str(output_folder / f"{input_file.stem}_clean.txt")  # replace with your desired output file path
        text_file = str(output_folder / f"{input_file.stem}_notag.txt")  # replace with your desired output file path
        clean_file(input_file, text_file, text_tagged_file, tags_file)
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <data_path>")
        sys.exit(1)

    clean_files(sys.argv[1])

#input_folder = Path("/home/ksingla/workspace/PromptingNemo/data_v2/synthetic/")

#clean_files(input_folder)

# input_file = str(output_folder / "text_tagged_train11.txt")  # replace with your input file path
# text_tagged_file = str(output_folder / "text_tagged_train_cleaned_gpt3.txt")  # replace with your desired output file path
# text_file = str(output_folder / "text_train_gpt3.txt")  # replace with your desired output file path
# tags_file = str(output_folder / "alltags_gpt3.txt")  # replace with your desired output file path
# clean_file_valid(input_file, text_file, text_tagged_file, tags_file)


# input_file = str(output_folder / "text_tagged_valid_v2.txt")  # replace with your input file path
# text_tagged_file = str(output_folder / "text_tagged_valid_cleaned_v4.txt")  # replace with your desired output file path
# text_file = str(output_folder / "text_valid_v4.txt")  # replace with your desired output file path
# tags_file = str(output_folder / "alltags_v4.txt")  # replace with your desired output file path
# clean_file_valid(input_file, text_file, text_tagged_file, tags_file)
