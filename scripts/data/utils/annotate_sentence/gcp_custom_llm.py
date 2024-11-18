from typing import Dict, List, Union

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: List[Dict],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    """
    Make predictions using a custom trained model deployed on Vertex AI.
    
    Args:
        project: Google Cloud project ID
        endpoint_id: The endpoint ID to make prediction request to
        instances: List of prediction requests containing prompt and generation parameters
        location: Region where the endpoint is deployed
        api_endpoint: API endpoint URL
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    
    # Initialize client that will be used to create and send requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    
    # Convert instances to protobuf Value format
    instances_value = [
        json_format.ParseDict(instance, Value()) for instance in instances
    ]
    
    # Empty parameters since they're included in instances
    parameters = json_format.ParseDict({}, Value())
    
    # Get the full endpoint path
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    
    # Make prediction request
    response = client.predict(
        endpoint=endpoint, 
        instances=instances_value, 
        parameters=parameters
    )
    
    print("Response:")
    print(f"Deployed model ID: {response.deployed_model_id}")
    
    # Process predictions
    for i, prediction in enumerate(response.predictions):
        print(f"\nPrediction {i + 1}:")
        print(f"Input prompt: {instances[i]['prompt']}")
        
        # Handle MapComposite object
        try:
            # Convert to dictionary and extract relevant fields
            prediction_dict = dict(prediction)
            
            # Print all available keys in the prediction
            print("Available keys in prediction:", prediction_dict.keys())
            
            # Try to get the generated text from common response fields
            generated_text = prediction_dict.get('content', 
                           prediction_dict.get('text',
                           prediction_dict.get('generated_text',
                           prediction_dict.get('response', str(prediction_dict)))))
            
            print(f"Generated response: {generated_text}")
            
            # Optionally print the full prediction dictionary for debugging
            print("Full prediction dictionary:", prediction_dict)
            
        except Exception as e:
            print(f"Error processing prediction: {e}")
            print(f"Raw prediction object: {prediction}")
            print(f"Prediction type: {type(prediction)}")


# Example usage
instances = {
    "instances": [
        {
            "prompt": "System: Follow these rules:\n1. Provide only the direct answer\n2. Don't generate additional questions\n3. Keep responses brief and factual\n\nUser: What is the capital of France?\nAssistant: ",
            "max_tokens": 10,
            "temperature": 0.1,
            "stop": ["\n", "[End]"]
        },
        {
            "prompt": "System: Follow these rules:\n1. Provide only the direct answer\n2. Don't generate additional questions\n3. Keep responses brief and factual\n\nUser: What is the capital of Germany?\nAssistant: ",
            "max_tokens": 10,
            "temperature": 0.1,
            "stop": ["\n", "[End]"]
        }
    ]
}

predict_custom_trained_model_sample(
    project="<your-project-id>",
    endpoint_id="<your-endpoint-id>",
    location="us-central1",
    instances=instances["instances"]  # Pass just the instances array
)