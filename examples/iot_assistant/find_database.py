import requests

# Replace with your Notion API token
notion_token = "secret_M1dy5X7ZXYtIp1neOOUXB5Juf7jRnhiBJUmdkCE6VIJ"

headers = {
    "Authorization": f"Bearer {notion_token}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

def list_databases():
    url = "https://api.notion.com/v1/search"
    data = {
        "filter": {
            "value": "database",
            "property": "object"
        }
    }

    response = requests.post(url, headers=headers, json=data)
    print(f"Response: {response.status_code}")
    if response.status_code == 200:
        results = response.json().get('results', [])
        print(f"Found {len(results)} databases:")
        for result in results:
            print(f"Database Title: {result['properties']['title']['title'][0]['text']['content']}")
            print(f"Database ID: {result['id']}\n")
    else:
        print(f"Failed to retrieve databases: {response.status_code}, {response.text}")

# List all databases
list_databases()
