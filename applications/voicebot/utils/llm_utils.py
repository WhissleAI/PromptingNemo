from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch
import requests


class HFLanguageModel:
    def __init__(self, model_name_or_path):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.llm_model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                                          trust_remote_code=False, safetensors=True).to(device)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)

    def generate_response(self, input_text, emotion, conversation_history):
        prompt_template = f'''system In a heartfelt phone conversation, the user just revealed: "{input_text}" system It's clear that you are feeling {emotion}. I'm here to provide comfort and give an answer in less than 30 words. Let's continue our heartfelt conversation. system The recent conversation history: {conversation_history[-5:]} assistant'''

        tokens = self.llm_tokenizer(
            prompt_template,
            return_tensors='pt'
        ).input_ids.cuda()

        generation_output = self.llm_model.generate(
            tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_new_tokens=512
        )

        response_text = self.llm_tokenizer.decode(generation_output[0])
        return response_text

import time
import requests
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def _generate(self, prompt, max_length):
        pass

    @abstractmethod
    def _llm_type(self):
        pass

class HuggingFaceAPI(BaseLLM):
    def __init__(self, model_id, api_token):
        self.model_id = model_id
        self.api_token = api_token
        self.headers = {
            "Authorization": f"Bearer {api_token}"
        }
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"

    def _generate(self, user_input, max_length=100, retries=5, wait_time=30):
        # Define the prompt template with roles
        prompt_template = f"""
        User: {user_input}
        Assistant: 
        """
        
        payload = {
            "inputs": prompt_template.strip(),
            "parameters": {
                "max_length": max_length,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        for attempt in range(retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload)
                response.raise_for_status()  # Raises an HTTPError for bad responses
                
                # Print the response for debugging
                print(response.text)
                
                response_json = response.json()
                if isinstance(response_json, list):
                    generated_text = response_json[0].get('generated_text', '').strip()
                else:
                    generated_text = response_json.get('generated_text', '').strip()
                
                # Extract the Assistant's reply
                assistant_reply_start = generated_text.find("Assistant:")
                if assistant_reply_start != -1:
                    generated_text = generated_text[assistant_reply_start + len("Assistant:"):].strip()
                
                return generated_text

            except requests.exceptions.HTTPError as errh:
                if response.status_code == 503 and 'loading' in response.text.lower():
                    print(f"Model is loading, retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retries})")
                    time.sleep(wait_time)
                else:
                    print(f"HTTP Error: {errh}\nResponse Content: {response.text}")
                    break
            except requests.exceptions.ConnectionError as errc:
                print(f"Error Connecting: {errc}")
            except requests.exceptions.Timeout as errt:
                print(f"Timeout Error: {errt}")
            except requests.exceptions.RequestException as err:
                print(f"Error: {err}")
                break
        return None

    def _llm_type(self):
        return "Hugging Face API"


# # Your Hugging Face API token and model ID
# api_token = "your_hf_api_token"
# model_id = "gpt-2"  # Replace with the model you intend to use

# # Initialize the Hugging Face API model
# hf_api_model = HuggingFaceAPI(model_id, api_token)

# # Use LangChain with the Hugging Face model
# from langchain.chains import LangChain

# lang_chain = LangChain(llm=hf_api_model)
# response = lang_chain.complete("What is the capital of France?")
# print(response)