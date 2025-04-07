import json
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
import re
import os

def process_jsonl(input_file, output_file):

    inverse_normalizer = InverseNormalizer(lang='en')

    ner_pattern = re.compile(r'NER_(\w+)\s+(.*?)\s+END')
    
    def process_line(line):
        try:
            data = json.loads(line.strip())
            text = data['text']
            
   
            def replace_ner(match):
                tag_type = match.group(1)
                content = match.group(2).strip()
                
    
                if tag_type == 'CARDINAL':
                    try:
                        normalized = inverse_normalizer.inverse_normalize(content, verbose=False)
                        return f"NER_CARDINAL {normalized} END"
                    except Exception as e:
                        print(f"Warning: Could not normalize cardinal value '{content}': {str(e)}")
                        return f"NER_CARDINAL {content} END"

                elif tag_type in ['PERSON', 'ORG', 'GPE', 'LOC', 'EVENT', 'PRODUCT', 'WORK_OF_ART',
                                'FAC', 'LANGUAGE', 'LAW', 'NORP']:
                    capitalized = ' '.join(word.capitalize() for word in content.split())
                    return f"NER_{tag_type} {capitalized} END"
                
       
                else:
                    return f"NER_{tag_type} {content} END"
       
            processed_text = ner_pattern.sub(replace_ner, text)
            data['text'] = processed_text
            return json.dumps(data)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse line as JSON: {str(e)}")
            return line.strip()
        except Exception as e:
            print(f"Warning: Error processing line: {str(e)}")
            return line.strip()


        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            for line_num, line in enumerate(infile, 1):
                try:
                    processed_line = process_line(line)
                    outfile.write(processed_line + '\n')
                except Exception as e:
                    print(f"Error processing line {line_num}: {str(e)}")
                    outfile.write(line)  

        print(f"Processing complete. Output written to: {output_file}")



if __name__ == "__main__":

    input_file = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/model_ready_manifest/toy.jsonl"  # Update this path
    output_file = "processed_manifest.jsonl"  
    
    process_jsonl(input_file, output_file)