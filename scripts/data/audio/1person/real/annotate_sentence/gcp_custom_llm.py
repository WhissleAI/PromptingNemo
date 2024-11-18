from typing import List
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import json
import traceback
import re

def annotate_sentences_custom_vertex(
    sentences: List[str],
    project_id: str,
    endpoint_id: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
) -> List[str]:
    """
    Annotate sentences using a custom language model deployed on Vertex AI.
    
    Args:
        sentences: List of sentences to annotate
        project_id: Google Cloud project ID
        endpoint_id: The endpoint ID to make prediction request to
        location: Region where the endpoint is deployed
        api_endpoint: API endpoint URL
    
    Returns:
        List of annotated sentences
    """
    print("\nStarting sentence annotation using Vertex AI")
    
    # Modified prompt template with more explicit instructions and forced output format
    prompt_template = f'''You are an expert entity and intent annotator. Your task is to add entity tags and intent labels to the following sentences.

Required entity types: ENTITY_PERSON, ENTITY_ACTION, ENTITY_LOCATION
Required format: Each line must contain at least one entity tag and one intent label.

Format rules:
1. Wrap each entity with its type and END marker
2. Add exactly one INTENT type at the end of each sentence
3. Return one sentence per line
4. Do not include any other text in your response

Example input: "John ran to the store"
Example output: "ENTITY_PERSON John END ENTITY_ACTION ran END to the ENTITY_LOCATION store END INTENT_ACTION"

Input sentences to annotate:
{json.dumps(sentences)}

Annotated output (one per line):'''

    try:
        # Initialize Vertex AI client
        client = aiplatform.gapic.PredictionServiceClient(
            client_options={"api_endpoint": api_endpoint}
        )
        
        # Prepare the instance with adjusted parameters
        instance = {
            "prompt": prompt_template,
            "max_tokens": 6800,  # Increased for longer responses
            "temperature": 0.5,   # Very low temperature for consistency
            "top_p": 0.95,
            "top_k": 40,
            "stop": ["\n\n", "System:", "Human:", "Assistant:", "Output:", "Example:"]
        }
        
        # Get endpoint path and make prediction
        endpoint = client.endpoint_path(
            project=project_id,
            location=location,
            endpoint=endpoint_id
        )
        
        response = client.predict(
            endpoint=endpoint,
            instances=[json_format.ParseDict(instance, Value())],
            parameters=json_format.ParseDict({}, Value())
        )
        
        print("Received response from Vertex AI")
        print("Raw response:", response)
        
        # Extract generated text
        generated_text = (
            response.predictions[0].string_value 
            if hasattr(response.predictions[0], 'string_value')
            else str(response.predictions[0])
        )
        
        def extract_annotations(text: str) -> List[str]:
            # First attempt: Try to find and parse JSON array
            try:
                # Look for the last occurrence of a JSON array in the text
                json_pattern = r'\[(?:[^][]|\[(?:[^][]|\[(?:[^][]|\[[^]]*\])*\])*\])*\]'
                json_matches = list(re.finditer(json_pattern, text))
                
                if json_matches:
                    last_match = json_matches[-1].group()
                    parsed = json.loads(last_match)
                    if isinstance(parsed, list):
                        # Validate that annotations are present
                        valid_annotations = []
                        for sent in parsed:
                            if ('ENTITY_' in sent and 'INTENT_' in sent):
                                valid_annotations.append(sent)
                            else:
                                print(f"Warning: Invalid annotation format: {sent[:100]}...")
                        
                        if len(valid_annotations) == len(sentences):
                            return valid_annotations
            except json.JSONDecodeError:
                pass
            
            # Second attempt: Try to extract line by line
            try:
                lines = [line.strip() for line in text.split('\n')]
                # Filter for properly annotated lines
                annotated = [
                    line.strip('"\'')
                    for line in lines
                    if 'ENTITY_' in line and 'INTENT_' in line and not line.startswith('Example')
                ]
                
                if len(annotated) == len(sentences):
                    return annotated
            except Exception as e:
                print(f"Line extraction failed: {str(e)}")
            
            print("Warning: Could not extract proper annotations, returning original sentences")
            return sentences
        
        annotated_sentences = extract_annotations(generated_text)
        
        # Validation
        if len(annotated_sentences) != len(sentences):
            print(f"Warning: Got {len(annotated_sentences)} annotations for {len(sentences)} sentences")
            return sentences
            
        # Verify each sentence has proper annotations
        for i, sent in enumerate(annotated_sentences):
            if 'ENTITY_' not in sent or 'INTENT_' not in sent:
                print(f"Warning: Sentence {i} missing required annotations")
                annotated_sentences[i] = sentences[i]
        
        print(f"Successfully processed {len(annotated_sentences)} annotations")
        return annotated_sentences
        
    except Exception as e:
        print(f"Error during annotation: {str(e)}")
        print("Traceback:", traceback.format_exc())
        return sentences