import os
import sys
from deepmultilingualpunctuation import PunctuationModel

# Load the punctuation model
punc_model = PunctuationModel()

def normalize_text(text):
    # Add missing punctuation and capitalize sentences
    corrected_text = punc_model.restore_punctuation(text)
    return corrected_text

# Define the input and output directories
input_folder = sys.argv[1]  # Update this path to where your .ctm files are located
output_folder = sys.argv[2]  # Update to a valid directory where you have write permissions

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over each .ctm file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.ctm'):
        input_path = os.path.join(input_folder, filename)
        input_lines = open(input_path, 'r').readlines()
        
        output_path = os.path.join(output_folder, filename.replace('.ctm', '.tsv'))
        outfile = open(output_path, 'w')
        segment = []
        for line in input_lines:
            line = line.strip()
            line = line.split()
            if "D:" in line[4]:
                
                if segment != []:
                    segment = " ".join(segment)
                    normalized_text = normalize_text(segment)
                    outfile.write(f'{start_time}\t{end_time}\t{role}\t{normalized_text}\n')
                
                role = "Doctor"
                start_time = float(line[2])
                segment = []
            elif "P:" in line:
                if segment != []:
                    segment = " ".join(segment)
                    normalized_text = normalize_text(segment)
                    outfile.write(f'{start_time}\t{end_time}\t{role}\t{normalized_text}\n')
                
                role = "Patient"
                start_time = float(line[2])
                segment = []
            else:
                segment.append(line[4])
                end_time = float(line[2]) + float(line[3])
                
        outfile.close()
            
        print(f'Processed {filename} and saved to {output_path}')

print('Processing completed.')
