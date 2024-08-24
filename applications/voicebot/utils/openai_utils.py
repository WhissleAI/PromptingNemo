import openai
import os

def get_openai_response(text: str, instruction: str, history: list = []):
    api_key = os.environ.get("OPENAI_API_KEY", "sk-proj-wBhxVeSmc5c9wq0MccFNT3BlbkFJPnPgz351rUnyoyLziIRu")
    temperature = 0.1

    openai.api_key = api_key

    messages = [{"role": "system", "content": instruction}]
    
    # Append conversation history
    for entry in history:
        messages.append({"role": "user", "content": entry["query"]})
        messages.append({"role": "assistant", "content": entry["answer"]})
    
    # Append the latest user query
    messages.append({"role": "user", "content": text})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature,
        max_tokens=300,
    )

    print("response", response)
    response_text = response.choices[0].message['content']
    input_tokens = response['usage']['prompt_tokens']
    output_tokens = response['usage']['completion_tokens']

    return response_text, input_tokens, output_tokens
