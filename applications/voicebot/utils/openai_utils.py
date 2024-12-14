from openai import OpenAI
import openai
import os

def get_openai_response(text: str, instruction: str, token:str, history: list = []):
    temperature = 0.1
    client = openai.OpenAI(api_key=token) 


    messages = [{"role": "system", "content": instruction}]
    
    # Append conversation history
    for entry in history:
        messages.append({"role": "user", "content": entry["query"]})
        messages.append({"role": "assistant", "content": entry["answer"]})
    
    # Append the latest user query
    messages.append({"role": "user", "content": text})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature,
        max_tokens=300,
    )
    response = dict(response)

    print("response", response)
    response_text = response.get('choices')[0].message.content
    input_tokens = response.get('usage').prompt_tokens
    output_tokens = response.get('usage').completion_tokens

    return response_text, input_tokens, output_tokens
