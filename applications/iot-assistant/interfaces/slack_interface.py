from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import yaml
import openai

# Load configuration from YAML file
with open("config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# Set up API key from the config
slack_token = config['slack']['token']
openai.api_key = config['openai']['api_key']

client = WebClient(token=slack_token)

def send_slack_message(channel, message):
    #try:
        response = client.chat_postMessage(channel=channel, text=message)
        return response
    #except SlackApiError as e:
    #    print(f"Error sending message: {e.response['error']}")

def slack_agent(prompt):
    # Use OpenAI to process the prompt and determine the channel and message
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a virtual assistant that helps users send Slack messages. When given a prompt, you should respond with the format 'slack message <channel> <message>'."},
            {"role": "user", "content": prompt},
        ]
    )
    reply_content = completion.choices[0].message.content
    print(f"Reply content: {reply_content}")
    if reply_content.startswith("slack message"):
        parts = reply_content.split(maxsplit=3)
        if len(parts) < 4:
            return "Please provide a channel and a message. Usage: slack message <channel> <message>"
        channel = parts[2]
        message = parts[3]
        print(f"Channel: {channel}, Message: {message}")
        send_slack_message(channel, message)
        return f"Message sent to Slack channel {channel}: {message}"
    return "Invalid Slack message format. Usage: slack message <channel> <message>"

# Example usage
if __name__ == "__main__":
    response = slack_agent("Send a message to #general saying 'Hello, this is a test message.'")
    print(response)
