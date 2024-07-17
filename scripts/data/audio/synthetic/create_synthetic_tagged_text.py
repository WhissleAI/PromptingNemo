import openai
import re
import os
import time
from pathlib import Path
import random

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
    print(formatted_output)
    return formatted_output


def create_data(prompt_files, output_file):
    
    output_file = open(output_file, 'a')

    for prompt_file in prompt_files:
        print("Processing file:", prompt_file)
        filename = os.path.basename(prompt_file)  # Get the file name with extension
        name, ext = os.path.splitext(filename)  # Split the file name and extension
        domain = MAP_PROMPT_TO_DOMAIN[name]  # Get the domain from the file name
        with open(prompt_file, 'r') as file:
            prompt = file.read()
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
                n=1,
                stop=None,
                temperature=0.4
            )
            
            tagged_output = response.choices[0].message['content'].strip()
            tagged_output = validate_and_correct_annotations(tagged_output)
            for line in tagged_output:
                line = domain + " " + line
                output_file.write(line)
                output_file.write("\n")
    
    output_file.close()


def create_data_n_times(prompt_files, output_file, n=40):
    for i in range(n):
        create_data(prompt_files, output_file)



            
'''
Collect samples from GPT-4
'''
output_folder = Path("/home/ksingla/workspace/PromptingNemo/data/synthetic/ASIAN/")

prompt_folder = Path("/home/ksingla/workspace/PromptingNemo/data/prompts/ASIAN/")
prompt_files = list(prompt_folder.glob("*.txt"))
random.shuffle(prompt_files)

output_file = output_folder / "text_tagged_train_noisy_ASIAN.txt"
os.system(f"mkdir -p {output_folder}")
#os.system(f"touch {output_file}")
create_data_n_times(prompt_files, output_file, n=50)


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
