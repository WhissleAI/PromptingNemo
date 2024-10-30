import requests
from bs4 import BeautifulSoup
import openai
from document_embedding import query_agent, query_agent_stream
import yaml

# Load configuration from YAML file
with open("config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# Assign variables from the config
openai.api_key = config['openai']['api_key']

def get_page_text(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            return ' '.join([p.text for p in soup.find_all('p')])
        else:
            print(f"Failed to fetch page: {url}, status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching page: {url}, error: {e}")
        return None

def search_duckduckgo(query, num_results=2):
    search_results = []
    try:
        url = f"https://duckduckgo.com/html/?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.find_all('a', {'class': 'result__a'}, limit=num_results)
            for result in results:
                link = result['href']
                page_text = get_page_text(link)
                if page_text:
                    search_results.append({"url": link, "text": page_text})
        else:
            print(f"Failed to fetch search results for query: {query}, status code: {response.status_code}")
    except Exception as e:
        print(f"Error fetching search results for query: {query}, error: {e}")
    return search_results

def trim_text(text, start_index=450, length=1500):
    return text[start_index:start_index + length]

def duckduckgo_agent(prompt, cutoff=6):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {"role": "system", "content": "You analyze a user's input to a large language model with \
                training data that cuts off at September 2021. The current year is 2023. You decide how \
                likely it is that a user's request will benefit from an internet search to help address the \
                question. Respond with a number in the range 1-10, where 1 is very unlikely that a \
                search would be beneficial, and 10 meaning a search is highly necessary."},
            {"role": "user", "content": prompt},
        ]
    )
    print(prompt)
    print("Cut-off probability: ", cutoff)
    print("Google probability: ", completion.choices[0].message.content)
    google_probability = int(completion.choices[0].message.content)
    if google_probability >= cutoff:
        search_results = search_duckduckgo(prompt)
        trimmed_results = [trim_text(result['text']) for result in search_results]
        query_with_context = prompt + str(trimmed_results)
        response = query_agent_stream(query_with_context)
        return response
    else:
        return False

# # Example usage
# prompt = "Medals won by india at the current olympics"
# response = google_agent(prompt)
# print("Response: ", response)
