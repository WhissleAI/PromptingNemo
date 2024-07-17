import json

file1 = 'text_train_cleaned_manifest_v4_0.json'
file2 = 'text_train_cleaned_manifest_gpt3.json'
output_file = 'tem.json'

def read_json_lines(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

data1 = read_json_lines(file1)
data2 = read_json_lines(file2)

# Assuming both JSON files contain lists of records, you can concatenate them
merged_data = data1 + data2

with open(output_file, 'w') as f:
    for entry in merged_data:
        json.dump(entry, f)
        f.write('\n')

print(f"Merged {file1} and {file2} into {output_file}")

