import json
import os
from nemo.collections.nlp.models import PunctuationCapitalizationModel

# Load the pre-trained model
model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")

input_jsonl_path = "/external3/databases/NPTEL2020-Indian-English-Speech-Dataset/english_190/english_190_processed.jsonl"  # Replace with your input file path
output_jsonl_path = "/external3/databases/NPTEL2020-Indian-English-Speech-Dataset/english_190/english_190_itn.jsonl"  # Replace with your desired output file path

def process_jsonl():
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_jsonl_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process the JSONL file
    with open(input_jsonl_path, 'r') as infile, open(output_jsonl_path, 'w') as outfile:
        for line in infile:
            # Parse the JSON line
            data = json.loads(line.strip())
            
            # Extract the text field
            original_text = data["text"]
            
            # Process the text with NeMo's model
            # We'll need to extract just the transcription part before metadata tags
            text_parts = original_text.split("AGE_")
            if len(text_parts) > 1:
                # Extract the text part before metadata tags
                text_for_processing = text_parts[0].strip()
            else:
                # If no metadata tags, use the whole text
                text_for_processing = original_text
            
            # Apply punctuation and capitalization
            processed_result = model.add_punctuation_capitalization([text_for_processing])
            processed_text = processed_result[0]
            
            # Replace the original text part with the processed text
            if len(text_parts) > 1:
                metadata = "AGE_" + text_parts[1]
                data["text"] = processed_text + " " + metadata
            else:
                data["text"] = processed_text
            
            # Write the modified JSON line to the output file
            outfile.write(json.dumps(data) + "\n")
    
    print(f"Processing complete. Output saved to {output_jsonl_path}")

if __name__ == "__main__":
    process_jsonl()