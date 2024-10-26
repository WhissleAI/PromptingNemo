import subprocess
import yaml
import openai
import json

# Load configuration from YAML file
with open("config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# Set up API key from the config
openai.api_key = config['openai']['api_key']

def send_message(recipient, message):
    apple_script = f'''
    tell application "Messages"
        set targetService to 1st service whose service type is iMessage
        set targetBuddy to buddy "{recipient}" of targetService
        send "{message}" to targetBuddy
    end tell
    '''
    try:
        subprocess.run(['osascript', '-e', apple_script], check=True)
        print(f"Message sent to {recipient} with content: {message}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to send message: {e}")

def read_last_n_messages(n):
    apple_script = f'''
    tell application "Messages"
        set recentMessages to ""
        set messageCount to {n}
        set chatCount to 0
        repeat with chat in (chats whose service type is iMessage)
            set chatMessages to texts of chat
            repeat with msg in chatMessages
                set recentMessages to recentMessages & msg & return
                set chatCount to chatCount + 1
                if chatCount ≥ messageCount then exit repeat
            end repeat
            if chatCount ≥ messageCount then exit repeat
        end repeat
        if (count of paragraphs of recentMessages) > messageCount then
            set recentMessages to paragraphs -(messageCount) thru -1 of recentMessages
        end if
        return recentMessages
    end tell
    '''
    try:
        result = subprocess.run(['osascript', '-e', apple_script], capture_output=True, text=True, check=True)
        print(f"AppleScript output: {result.stdout}")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Failed to read messages: {e}")
        print(f"AppleScript error output: {e.stderr}")
        return None

def message_agent(prompt):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are an assistant that helps manage messages. Respond only with structured JSON commands. You can send messages and read the last received messages. You can also read the last N messages."},
                {"role": "user", "content": prompt},
            ]
        )
        print("Prompt:", prompt)
        reply_content = completion.choices[0]['message']['content']
        print("Reply content:", reply_content)
        
        response = json.loads(reply_content)

        if response.get("command") in ["send", "sendMessage"]:
            recipient = response["params"].get("recipient")
            message = response["params"].get("message")
            send_message(recipient, message)
            return f"Message sent to {recipient} with content: {message}"
        elif response.get("command") == "read":
            n = response["parameters"].get("count")
            last_messages = read_last_n_messages(n)
            return f"Last {n} messages: {last_messages}" if last_messages else "Failed to read messages."
        else:
            return "Command not recognized"
    except openai.error.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return "There was an error processing your request."
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return "Failed to decode the response."

# # Example usage
# prompt = "Please send message to John Doe saying Hello, how are you?"
# response = message_agent(prompt)
# print(response)

# prompt = "Please read last 5 messages"
# response = message_agent(prompt)
# print(response)
