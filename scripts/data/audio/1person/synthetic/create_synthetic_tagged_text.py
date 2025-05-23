import openai
import re
import os
import time
from pathlib import Path
import random
import json

openai.api_key = 'sk-proj-wBhxVeSmc5c9wq0MccFNT3BlbkFJPnPgz351rUnyoyLziIRu'


MAP_PROMPT_TO_DOMAIN = {
    "smarthome-assistant": "DOMAIN-SMARTHOME-ASSISTANT",
    "medical-lab": "DOMAIN-MEDICAL-LAB-ASSISTANT",
    "finance-assistant": "DOMAIN-FINANCE-ASSISTANT",
    "personal-fitness-assistant": "DOMAIN-PERSONAL-FITNESS-ASSISTANT",
    "soccer-highlights": "DOMAIN-SOCCER-HIGHLIGHTS",
    "psychtherapist-assistant": "DOMAIN-PSYCHOTHERAPY-ASSISTANT",
    "travel-assistant": "DOMAIN-TRAVEL-ASSISTANT",
    "medical-patient": "DOMAIN-MEDICAL-PATIENT",
    "entertainment-assistant": "DOMAIN-ENTERTAINMENT-ASSISTANT",
    "grocery-assistant": "DOMAIN-GROCERY-ASSISTANT",
    "recipe-assistant": "DOMAIN-RECIPE-ASSISTANT",
}

def validate_and_correct_annotations(conll_output):
    lines = conll_output.split('\n')
    formatted_output = []
    temp_sentence = []

    for line in lines:
        if line.strip() == "":  # Check for empty line to separate sentences
            if temp_sentence:
                formatted_output.append("\n".join(temp_sentence))
                temp_sentence = []
        else:
            if len(temp_sentence) % 2 == 0:
                token = line.strip()
                temp_sentence.append(token)
            else:
                tag = line.strip()
                temp_sentence[-1] = f"{temp_sentence[-1]} {tag}"
    
    if temp_sentence:  # Append the last sentence if not empty
        formatted_output.append("\n".join(temp_sentence))
    
    # Join all sentences with an extra newline
    #print(formatted_output)
    return formatted_output


def create_data(prompt_files, output_folder, lang_map, example_samples):
    
    for prompt_file in prompt_files:
        print("Processing file:", prompt_file)
        filename = os.path.basename(prompt_file)  # Get the file name with extension
        name, ext = os.path.splitext(filename)  # Split the file name and extension
            
        for lang in lang_map.keys():
            
            output_file = open(output_folder / f"tagged_{lang_map[lang]}.txt", 'a')
            prompt = open(prompt_file, 'r').read()
            prompt = prompt.replace("{lang}", lang)
            prompt = prompt.replace("{examples}", example_samples)
            
            print("Prompt:", prompt)
            
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4096,
                    n=1,
                    stop=None,
                    temperature=0.7,  # Higher value to increase creativity and diversity
                )
                            
                tagged_output = response.choices[0].message['content'].strip()
                tagged_output = tagged_output.replace("```json\n", "")
                tagged_output = tagged_output.replace("```", "")
                #print(tagged_output)
                tagged_output = json.loads(tagged_output)
                #tagged_output = validate_and_correct_annotations(tagged_output)
                for line in tagged_output:
                    line = "LANG_" + lang_map[lang] + " " + line
                    output_file.write(line)
                    output_file.write("\n")
                
                output_file.close()
            except:
                print("Error in processing prompt")
                continue

import requests

def create_data_anthropic(prompt_files, output_folder, lang_map, example_samples):
    
    api_key = "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"
    api_url = "https://api.anthropic.com/v1/complete"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    for prompt_file in prompt_files:
        print("Processing file:", prompt_file)
        filename = os.path.basename(prompt_file)  # Get the file name with extension
        name, ext = os.path.splitext(filename)  # Split the file name and extension
            
        for lang in lang_map.keys():
            
            output_file_path = output_folder / f"tagged_{lang_map[lang]}.txt"
            with open(output_file_path, 'a') as output_file:
                with open(prompt_file, 'r') as file:
                    prompt = file.read()
                
                # Ensure prompt starts with "Human:" and ends with "Assistant:"
                prompt = prompt.replace("{lang}", lang)
                prompt = prompt.replace("{examples}", example_samples)
                
                if not prompt.startswith("\n\nHuman:"):
                    prompt = "\n\nHuman:" + prompt
                
                if not prompt.endswith("\n\nAssistant:"):
                    prompt = prompt.strip() + "\n\nAssistant:"
                
                #print("Prompt:", prompt)
                
                payload = {
                    "model": "claude-2.1",
                    "max_tokens_to_sample": 4096,
                    "prompt": prompt
                }
                
                response = requests.post(api_url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    response_json = response.json()
                    tagged_output = response_json.get('completion', '').strip()
                    print(tagged_output)
                    tagged_output = validate_and_correct_annotations(tagged_output)
                    
                    for line in tagged_output.splitlines():
                        line = "LANG_" + lang_map[lang] + " " + line
                        output_file.write(line + "\n")
                else:
                    print("Error:", response.status_code, response.text)

def create_data_n_times(prompt_files, output_folder, lang_map, examples, n=40):
    
    
    for i in range(n):
        example_samples = random.sample(examples, 5)
        example_samples = "\n".join(example_samples)
        create_data(prompt_files, output_folder, lang_map, example_samples)



            
'''
Collect samples from GPT-4
'''

if __name__ == "__main__":

    output_folder = Path("/external2/datasets/slurp/synthetic6/")
    prompt_folder = Path("/root/workspace/PromptingNemo/datasets/prompts/data_extension/")
    prompt_files = list(prompt_folder.glob("*.txt"))
    random.shuffle(prompt_files)
    
    
    extension_dataset = Path("/external2/datasets/slurp/train-slurp-tagged.txt")
    examples = []
    with open(extension_dataset, 'r') as f:
        examples = f.readlines()
    

    EURO = {"English": "EN", "Spanish": "ES", "French": "FR", "German": "DE", "Italian" : "IT", "Dutch": "NL", "Portuguese": "PT"}
    INDIAN = {"Hindi": "HI", "Punjabi": "PA", "Bengali": "BN", "Marathi": "MR", "Gujrati": "GU", "Kannada": "KN", "Telugu": "TE"}

    # lang_map = EURO.copy()  # Copy EURO dictionary to avoid modifying the original
    # lang_map.update(INDIAN)  # Update with INDIAN dictionary
    #INDIAN.update(EURO)

    os.system(f"mkdir -p {output_folder}")
    create_data_n_times(prompt_files, output_folder, EURO, examples, n=500)


# input_file = str(output_folder / "text_tagged_train_v2.txt")  # replace with your input file path
# text_tagged_file = str(output_folder / "text_tagged_train_cleaned_v2.txt")  # replace with your desired output file path
# text_file = str(output_folder / "text_train_v2.txt")  # replace with your desired output file path
# tags_file = str(output_folder / "alltags_v2.txt")  # replace with your desired output file path
# clean_file(input_file, text_file, text_tagged_file, tags_file)

# input_file = str(output_folder / "text_tagged_valid.txt")  # replace with your input file path
# text_tagged_file = str(output_folder / "text_tagged_valid_cleaned.txt")  # replace with your desired output file path
# text_file = str(output_folder / "text_valid.txt")  # replace with your desired output file path
# tags_file = str(output_folder / "alltags.txt")  # replace with your desired output file path
# clean_file_valid(input_file, text_file, text_tagged_file, tags_file)


# DATA_PATH = Path("/home/ksingla/workspace/medical-ner/data/iot-"+RUN_NAME)
# os.system(f"mkdir -p {DATA_PATH}")
# output_file = str(Path(DATA_PATH, 'text_tagged_train.txt'))
# collect_samples(output_file, num_batches=1)

# output_file = str(Path(DATA_PATH, 'text_tagged_valid.txt'))
# collect_samples(output_file, num_batches=1)

# print("Dataset saved to conll_dataset.txt")



# # Clean data
# input_file = str(DATA_PATH / "text_tagged_train.txt")  # replace with your input file path
# input_file_cleaned = str(DATA_PATH / "text_tagged_train_cleaned.txt")  # replace with your desired output file path
# output_file = str(DATA_PATH / "text_train.txt")  # replace with your desired output file path
# tags_file = str(DATA_PATH / "alltags.txt")  # replace with your desired output file path
# clean_file(input_file, output_file, tags_file)
# create_cleaned_input_file(input_file, input_file_cleaned)


# print(f"Cleaned data has been written to {output_file}")


# # Clean data
# input_file = str(DATA_PATH / "text_tagged_valid.txt")  # replace with your input file path
# input_file_cleaned = str(DATA_PATH / "text_tagged_valid_cleaned.txt")  # replace with your desired output file path
# output_file = str(DATA_PATH / "text_valid.txt")  # replace with your desired output file path
# tags_file = str(DATA_PATH / "alltags_valid.txt")  # replace with your desired output file path
# clean_file(input_file, output_file, tags_file)
# create_cleaned_input_file(input_file, input_file_cleaned)

# print(f"Cleaned data has been written to {output_file}")
