import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

text_pipe = load_model()

# Load model and tokenizer
def load_model():
    # Check if GPU (CUDA) is available and set the device accordingly
    device = 0 if torch.cuda.is_available() else -1

    tokenizer = AutoTokenizer.from_pretrained("RedHenLabs/news-reporter-euro-3b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("RedHenLabs/news-reporter-euro-3b", trust_remote_code=True)

    # Create pipeline with the loaded model and tokenizer, using the appropriate device
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, trust_remote_code=True)
    return pipe

# Function to generate response from user query
def generate_response(user_query):
    # Load model and pipeline

    # Prepare message format for the pipeline
    messages = [
        {"role": "user", "content": user_query},
    ]

    # Generate response using the pipeline
    response = text_pipe(messages)
    return response[0]['generated_text']

# Example usage
# user_input = "Who are you?"
# print(generate_response(user_input))