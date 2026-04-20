import openai
import psutil
import yaml
import subprocess
import time
import os
import re

# Load configuration from YAML file
with open("config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# Set up API key from the config
openai.api_key = config['openai']['api_key']

def take_screenshot(save_path):
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        subprocess.run(['screencapture', save_path])
        return f"Screenshot saved at {save_path}"
    except Exception as e:
        return str(e)

def extract_path_from_prompt(prompt):
    match = re.search(r"save (?:it |the screenshot )?to (.+)", prompt, re.IGNORECASE)
    if match:
        return match.group(1).strip().strip("'\"")
    return None

def screenshot_agent(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {"role": "system", "content": "You control screenshot or capture image functionality on a Mac based on a user request."},
            {"role": "user", "content": prompt},
        ]
    )
    reply_content = completion.choices[0].message.content
    default_save_path = os.path.expanduser("~/Desktop/screenshot.png")

    if "take a screenshot" or "capture screen" in prompt.lower():
        save_path = extract_path_from_prompt(prompt) or default_save_path
        return take_screenshot(save_path)
    else:
        return "Command not recognized"

# Example usage:
#print(screenshot_agent("take a screenshot and save it to '/Users/yourusername/Desktop/my_screenshot.png'"))
#print(screenshot_agent("take a screenshot"))

