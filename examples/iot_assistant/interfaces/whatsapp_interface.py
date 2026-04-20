import subprocess
import openai
import yaml

# Load configuration from YAML file
with open("config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# Set up API key from the config
openai.api_key = config['openai']['api_key']

def send_whatsapp_message(contact_name, message):
    # AppleScript command to send a message via WhatsApp
    applescript_command = f'''
    tell application "WhatsApp"
        activate
        delay 1
    end tell

    tell application "System Events"
        tell process "WhatsApp"
            delay 1
            keystroke "f" using command down -- Open search
            delay 1
            keystroke "{contact_name}" -- Type contact name
            delay 2
            -- Use arrow keys to navigate to the correct contact
            key code 125 -- Down arrow to move to the contact in the search results
            delay 1
            keystroke return -- Select the contact
            delay 2
            -- Type and send the message
            keystroke "{message}"
            delay 1
            keystroke return
        end tell
    end tell
    '''
    subprocess.call(['osascript', '-e', applescript_command])

def make_whatsapp_call(contact_name):
    # AppleScript command to make a phone call using WhatsApp
    applescript_command = f'''
    tell application "WhatsApp"
        activate
        delay 1
    end tell

    tell application "System Events"
        tell process "WhatsApp"
            delay 1
            keystroke "f" using command down -- Open search
            delay 1
            keystroke "{contact_name}" -- Type contact name
            delay 2
            -- Use arrow keys to navigate to the correct contact
            key code 125 -- Down arrow to move to the contact in the search results
            delay 1
            keystroke return -- Select the contact
            delay 2
            -- Navigate to the call button and initiate the call
            keystroke tab -- Focus on call button
            delay 1
            keystroke tab -- Move to call button
            delay 1
            keystroke return -- Initiate the call
        end tell
    end tell
    '''
    subprocess.call(['osascript', '-e', applescript_command])

def whatsapp_agent(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a virtual assistant that helps users send WhatsApp messages and make phone calls. When given a prompt, you should respond with either 'message Contact Name: Message Content' or 'call Contact Name'."},
            {"role": "user", "content": prompt},
        ]
    )
    
    reply_content = completion.choices[0].message.content
    if reply_content.lower().startswith("message "):
        contact_name, message = reply_content[len("message "):].split(":", 1)
        send_whatsapp_message(contact_name.strip(), message.strip())
    elif reply_content.lower().startswith("call "):
        contact_name = reply_content[len("call "):].strip()
        make_whatsapp_call(contact_name)

# Example usage:
# whatsapp_agent("Send a message to Midhun Home saying 'Hello, this is a test message.'")
# #whatsapp_agent("Call Midhun Home")
