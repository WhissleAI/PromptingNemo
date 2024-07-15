import openai
from openai import OpenAI
import os

'''
TODO
input: conversation_history = [], text, instruction="Provide meaningful reply", max_tokens=150

messages = conversation_history + messages
'''

def get_openai_response(text: str, instruction: str, history: list = []):
    api_key = os.environ.get("OPENAI_API_KEY", "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")
    temperature = 0.1

    client = openai.OpenAI(
        api_key=api_key,
    )

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

    print("response", response)
    response_text = response.choices[0].message.content

    return response_text


