import openai
import requests
import yaml
import json
from fuzzywuzzy import process
import re
# Load configuration from YAML file
with open("config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# Set up OpenAI API key from the config
openai.api_key = config['openai']['api_key']

# Set up Notion API token and database ID from the config
notion_token = config['notion']['token']
database_id = config['notion']['database_id']

# Initialize the Nvidia API details
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY = "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"


headers = {
    "Authorization": f"Bearer {notion_token}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

def create_notion_page(page_title, page_content):
    create_url = 'https://api.notion.com/v1/pages'
    
    new_page_data = {
        "parent": {"database_id": database_id},
        "properties": {
            "title": {
                "title": [
                    {
                        "text": {
                            "content": page_title
                        }
                    }
                ]
            }
        },
        "children": [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": page_content
                            }
                        }
                    ]
                }
            }
        ]
    }

    response = requests.post(create_url, headers=headers, data=json.dumps(new_page_data))
    if response.status_code == 200:
        return "Page created successfully!"
    else:
        return f"Failed to create page: {response.status_code}, {response.text}"

def get_all_notion_page_titles():
    query_url = f"https://api.notion.com/v1/databases/{database_id}/query"
    response = requests.post(query_url, headers=headers)
    
    if response.status_code != 200:
        return []
    
    data = response.json()
    all_titles = []
    for page in data['results']:
        try:
            title = page['properties']['Page']['title'][0]['text']['content']
            all_titles.append(title)
        except:
            continue
    #titles = [page['properties']['Page']['title'][0]['text']['content'] for page in data['results']]
    return all_titles

def search_notion_page_id(page_title):
    query_url = f"https://api.notion.com/v1/databases/{database_id}/query"
    response = requests.post(query_url, headers=headers)
    
    if response.status_code != 200:
        return None
    
    data = response.json()
    for page in data['results']:
        title = page['properties']['title']['title'][0]['text']['content']
        if title == page_title:
            return page['id']
    
    return None

def extract_clean_text(data):
    # Regular expression to find the 'content' values
    content_regex = r"'content': '([^']*)'"
    
    # Find all matches for 'content'
    matches = re.findall(content_regex, data)
    
    # Join all the found content into a single clean string
    clean_text = "\n".join(matches)
    
    return clean_text

def summarize_content_with_nvidia(content):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "nv-mistralai/mistral-nemo-12b-instruct",
        "messages": [
            {"role": "system", "content": "Please summarize the following text into a concise paragraph less than 100 words."},
            {"role": "user", "content": content}
        ],
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 300
    }

    response = requests.post(API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        summarized_content = response.json()['choices'][0]['message']['content'].strip()
        return summarized_content
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return "Failed to summarize content"
    
    
def summarize_content_with_gpt(content):
    inference_prompt = f"Please summarize the following text into a concise paragraph:\n\n{content}"
    
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that infers details from user input for note management."},
            {"role": "user", "content": inference_prompt},
        ]
    )
    
    summarized_content = completion['choices'][0]['message']['content']
    return summarized_content

def search_notion_page_and_get_content(page_title):
    all_titles = get_all_notion_page_titles()
    closest_match, _ = process.extractOne(page_title, all_titles)
    
    query_url = f"https://api.notion.com/v1/databases/{database_id}/query"
    response = requests.post(query_url, headers=headers)
    
    if response.status_code != 200:
        return closest_match, "Failed to retrieve content"
    
    data = response.json()
    for page in data['results']:
        try:
            title = page['properties']['Page']['title'][0]['text']['content']
            if title == closest_match:
                page_id = page['id']
                retrieve_url = f"https://api.notion.com/v1/blocks/{page_id}/children"
                content_response = requests.get(retrieve_url, headers=headers)
                if content_response.status_code == 200:
                    content_data = content_response.json()
                    clean_text = extract_clean_text(str(content_data))
                    
                    # Summarize the content using GPT-4
                    summarized_content = summarize_content_with_nvidia(clean_text)
                    
                    return closest_match, summarized_content
                else:
                    return closest_match, "Failed to retrieve content"
        except:
            continue

    return closest_match, "Content not found"

def append_to_notion_page(page_title, new_content):
    page_id = search_notion_page_id(page_title)
    
    if not page_id:
        return f"No page found with title '{page_title}'"

    append_url = f'https://api.notion.com/v1/blocks/{page_id}/children'
    
    new_block = {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [
                {
                    "type": "text",
                    "text": {
                        "content": new_content
                    }
                }
            ]
        }
    }

    response = requests.patch(append_url, headers=headers, data=json.dumps({"children": [new_block]}))
    
    if response.status_code == 200:
        return "Content appended successfully!"
    else:
        return f"Failed to append content: {response.status_code}, {response.text}"

def infer_note_details(refined_input):
    inference_prompt = f'''
    Analyze the following refined input and infer the intent (such as "create_note", "search_note", or "append_note"), 
    the note title if available, and the note content if available. Clean the input to remove any noise or unnecessary information.
    Output the result in the following JSON format:
    {{
        "intent": "<inferred_intent>",
        "note_title": "<inferred_note_title>",
        "note_content": "<inferred_note_content>"
    }}
    Refined Input: {refined_input}
    '''
    
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that infers details from user input for note management."},
            {"role": "user", "content": inference_prompt},
        ]
    )
    
    inferred_details = completion.choices[0].message.content.strip()
    return yaml.safe_load(inferred_details)

def notion_agent(refined_input):
    # Use GPT-4 to infer the intent, note title, and content from the refined input
    inferred_details = infer_note_details(refined_input)
    print(inferred_details)
    intent = inferred_details.get("intent", "").lower()
    
    if intent == "create_note":
        note_title = inferred_details.get("note_title", "Untitled Note")
        note_content = inferred_details.get("note_content", "")
        
        if isinstance(note_content, list):
            note_content = ", ".join(note_content)
        
        # If no title was extracted, create a title based on the content
        if note_title == "Untitled Note" and note_content:
            note_title = note_content.split('.')[0]  # Use the first sentence as a title
        
        response = create_notion_page(note_title, note_content)
        return response
    
    elif intent == "search_note":
        note_title = inferred_details.get("note_title", "")
        
        closest_match, content = search_notion_page_and_get_content(note_title)
        return f"Note found with closest matching title '{closest_match}'. Here is a summary: {content}" if content else f"No note found with a close match to title '{note_title}'"
    
    elif intent == "append_note":
        note_title = inferred_details.get("note_title", "")
        note_content = inferred_details.get("note_content", "")
        
        if isinstance(note_content, list):
            note_content = ", ".join(note_content)
        
        response = append_to_notion_page(note_title, note_content)
        return response
    
    else:
        return "Command not recognized"

# # Example usage:
# example = """
# {
#   "intent": "search_document",
#   "entities": {
#     "document_type": "business plan"
#   }
# }
# """
# print(notion_agent(example))
