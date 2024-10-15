import sys
import os
import openai
import json


openai.api_key = "sk-nDKRYoQG7GGBAl_CHSjtNyG2plOIrUY9CDIHKZWfvzT3BlbkFJ-pgkybJi1A_NRgKygxMyGfVhS2HDpnDZocvWGGsjwA"

import anthropic

def generate_annotations_anthropic(transcripts, prompt, model="claude-2"):
    """
    Function to generate annotations for patient-doctor conversations using Anthropic Claude model.

    Args:
    transcripts (list of str): List of conversation turns to annotate.
    prompt (str): The prompt template to provide to the model.
    model (str): The Claude model to use, default is "claude-2".

    Returns:
    list of str: Annotated sentences.
    """

    # Present the transcripts in sequential format to guide the model to use context across turns
    transcripts_text = "\n".join([f'{i+1}. "{transcript}"' for i, transcript in enumerate(transcripts)])
    
    # Replace placeholder in the prompt with the transcripts
    prompt = prompt.replace("{transcripts_text}", transcripts_text)

    # Call the Anthropic Claude API
    client = anthropic.Client(api_key="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")
    
    response = client.completions.create(
        prompt=f"{anthropic.HUMAN_PROMPT} You are an expert medical text annotator.\n\n{prompt}{anthropic.AI_PROMPT}",
        model=model,  # Use an Anthropic model like "claude-2"
        max_tokens_to_sample=1000,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        temperature=0.7
    )

    # Access the 'completion' content correctly
    annotations = response.completion.strip()
    
    try:
        annotations = json.loads(annotations)
    except json.JSONDecodeError:
        print("Error: Unable to parse response as JSON.")
        annotations = []

    return annotations

def generate_annotations(transcripts, prompt, model="gpt-4o"):
    """
    Function to generate annotations for patient-doctor conversations using provided examples.

    Args:
    transcripts (list of str): List of conversation turns to annotate.
    examples (list of str): List of example annotations to provide to GPT for guidance.
    model (str): The OpenAI GPT model to use, default is "gpt-4".

    Returns:
    list of str: Annotated sentences.
    """

    # Present the transcripts in sequential format to guide the model to use context across turns
    transcripts_text = "\n".join([f'{i+1}. "{transcript}"' for i, transcript in enumerate(transcripts)])

    prompt = prompt.replace("{transcripts_text}", transcripts_text)

    #print(prompt)
    # Call the OpenAI API
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert medical text annotator."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract and return the annotated sentences
    annotations = response.choices[0].message['content']
    annotations = annotations.replace("```json", "")
    annotations = annotations.replace("```", "")
    #print(annotations)
    annotations = json.loads(annotations)
    # Attempt to parse the string as JSON
    return annotations



def annotate_tsv(input_file, output_file, prompt, batch_size=10):
    
    print(f'Annotating {input_file} and saving to {output_file}')
    with open(input_file, 'r') as f:
        lines = f.readlines()
    output_file = open(output_file, 'w')
    
    
    for num in range(0, len(lines), batch_size):
        print("Batch: ", num)
        
        try:
            batch = lines[num: num + batch_size]
            
            sentences = []
            for line in batch:
                line = line.strip()
                start_time, end_time, role, text = line.split('\t')
                sentences.append(role+': '+text)
            
            #print("#Sentences: ", len(sentences))
            
            #annotations = generate_annotations_anthropic(sentences, prompt, model="claude-2")
            annotations = generate_annotations(sentences, prompt, model="gpt-4o-mini")
            #print("\n\nANNOTATIONS\n\n")
            #print(annotations)
            #print("#Annotations: ", len(annotations))
            
            print("\n\nAnnotations DONE\n\n")
            
            for num in range(len(batch)):
                line = batch[num]
                start_time, end_time, role, text = line.split('\t')
                annotated_sentence = annotations[num]
                annotated_sentence = annotated_sentence.split(": ")[1]
                output_line = f'{start_time}\t{end_time}\t{role}\t{annotated_sentence}\n'
                
                output_file.write(output_line)
                output_file.write('\n')
        except:
            print("Error")
            continue
    output_file.close()

    print(f'Annotated file saved to {output_file}')



if __name__ == '__main__':
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    prompt_file = sys.argv[3] #/root/workspace/2person_data_creation/datasets/prompts/psychotherapy/patient-doctor.txt
    
    prompt = open(prompt_file, 'r').read()
    
    os.makedirs(output_folder, exist_ok=True)
    
    #input_files = glob.glob(os.path.join(input_folder, '*.tsv'))
    
    for filename in os.listdir(input_folder):
        
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename.replace('.tsv', '_annotated.tsv'))
        
        annotate_tsv(input_file, output_file, prompt)
                    
        
    
    
    