import json

from multi_wer import multi_word_error_rate

# Path to the JSON file
json_file_path = '/projects/whissle/datasets/hindi/valid_pred.json'

# Load JSON data from the file
with open(json_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Initialize an empty list to store the converted lists
# data_list = []
hypotheses = []
references = []
tags = {
    'NER': [],
    "INTENT": [],
    'AGE': [],
    'GENDER': [],
    'DIALECT': [],
    "END": "END"
}

# Convert each JSON object to a list and add it to data_list
for line in lines:
    data = json.loads(line)
    entry = [data["audio_filepath"], data["text"], data["predicted_text"]]
    hypotheses.append(data["predicted_text"])
    references.append(data["text"])
    text = data["text"]

    for word in text.split(' '):
        if word.startswith('ENTITY_'):
            tags['NER'].append(word)
        elif word.startswith('INTENT_'):
            tags['INTENT'].append(word)
        elif word.startswith('AGE_'):
            tags['AGE'].append(word)
        elif word.startswith('GENDER_'):
            tags['GENDER'].append(word)
        elif word.startswith('DIALECT_'):
            tags['DIALECT'].append(word)

tags['NER'] = list(set(tags['NER']))
tags['INTENT'] = list(set(tags['INTENT']))
tags['AGE'] = list(set(tags['AGE']))
tags['GENDER'] = list(set(tags['GENDER']))
tags['DIALECT'] = list(set(tags['DIALECT']))


# print(tags)
with open('tags.json', 'a') as tags_f:
    json.dump(tags, tags_f, ensure_ascii=False)


print(multi_word_error_rate(hypotheses, references, tags, 'standard'))

# Calculate WER and detailed metrics
wer, metrics = multi_word_error_rate(hypotheses, references, tags, 'standard')

# Prepare output dictionary
output_data = {
    "WER": wer,
    "metrics": metrics
}

# Save WER and metrics to a JSON file
output_file_path = 'wer_metrics.json'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(output_data, output_file, ensure_ascii=False, indent=4)

print(f"Results saved to {output_file_path}")
