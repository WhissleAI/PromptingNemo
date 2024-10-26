import requests
import yaml
import openai
# Load configuration from YAML file
with open("config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# Set up API key from the config
openai.api_key = config['openai']['api_key']

alpha_vantage_api_key = config['alpha_vantage']['api_key']

NVAPI_URL = config['nvidia']['api_url']
NVAPI_KEY = config['nvidia']['api_key']

def get_stock_price(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={alpha_vantage_api_key}'
    response = requests.get(url)
    data = response.json()
    
    if 'Time Series (1min)' in data:
        latest_time = list(data['Time Series (1min)'].keys())[0]
        latest_close = data['Time Series (1min)'][latest_time]['4. close']
        return f"The latest price for {symbol} is {latest_close}."
    else:
        return f"Could not retrieve data for {symbol}."

def stock_agent(prompt):
    # Extract stock symbol from the prompt
    # completion = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     temperature=0,
    #     messages=[
    #         {"role": "system", "content": "You are a virtual assistant that helps users get stock prices. When given a prompt, you should respond with the format 'stock SYMBOL'."},
    #         {"role": "user", "content": prompt},
    #     ]
    # )
    # print(prompt)
    # reply_content = completion.choices[0].message.content

    headers = {
        "Authorization": f"Bearer {NVAPI_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [
            {
                "role": "system", 
                "content": "You are a virtual assistant that helps users get stock prices. When given a prompt, you should respond with the format 'stock SYMBOL'."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 100  # Adjust as needed
    }

    reply_content = requests.post(NVAPI_URL, headers=headers, json=data)
    reply_content = reply_content.json()['choices'][0]['message']['content'].strip()
    
    if reply_content.startswith("stock "):
        symbol = reply_content[len("stock "):].strip()
        return get_stock_price(symbol)
    else:
        return "Sorry, I don't understand the request."

# Example usage:
if __name__ == "__main__":
    prompt = "What is the latest price of google?"
    print(stock_agent(prompt))
