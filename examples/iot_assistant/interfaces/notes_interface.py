import openai
import psutil
import yaml
import subprocess
import time
from fuzzywuzzy import process

# Load configuration from YAML file
with open("config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# Set up API key from the config
openai.api_key = config['openai']['api_key']

def is_notes_running():
    for process in psutil.process_iter():
        try:
            if 'notes' in process.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

def open_notes():
    if not is_notes_running():
        subprocess.call(['open', '-a', 'Notes'])
        time.sleep(1)

def create_note_with_title(note_title, note_content):
    apple_script = f'''
    tell application "Notes"
        tell folder "Notes"
            make new note with properties {{name:"{note_title}", body:"{note_content}"}}
        end tell
    end tell
    '''
    subprocess.run(['osascript', '-e', apple_script])

def get_all_note_titles():
    apple_script = '''
    tell application "Notes"
        set noteTitles to ""
        tell folder "Notes"
            set allNotes to every note
            repeat with aNote in allNotes
                set noteTitles to noteTitles & (name of aNote) & "|"
            end repeat
        end tell
        return noteTitles
    end tell
    '''
    result = subprocess.run(['osascript', '-e', apple_script], capture_output=True, text=True)
    return result.stdout.strip().split('|')

def search_note_and_get_content(note_title):
    all_titles = get_all_note_titles()
    closest_match, _ = process.extractOne(note_title, all_titles)
    
    apple_script = f'''
    tell application "Notes"
        set noteContent to ""
        tell folder "Notes"
            set theNote to first note whose name is "{closest_match}"
            set noteContent to body of theNote
        end tell
        return noteContent
    end tell
    '''
    result = subprocess.run(['osascript', '-e', apple_script], capture_output=True, text=True)
    return closest_match, result.stdout.strip()

def infer_note_details(refined_input):
    inference_prompt = f'''
    Analyze the following refined input and infer the intent (such as "create_note" or "search_note"), 
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

def notes_agent(refined_input):
    # Use GPT-4 to infer the intent, note title, and content from the refined input
    inferred_details = infer_note_details(refined_input)
    print("INFER1",inferred_details)
    intent = inferred_details.get("intent", "").lower()
    print("LABEL", intent)
    if intent == "create_note":
        note_title = inferred_details.get("note_title", "Untitled Note")
        note_content = inferred_details.get("note_content", "")
        
        if isinstance(note_content, list):
            note_content = ", ".join(note_content)
        
        # If no title was extracted, create a title based on the content
        if note_title == "Untitled Note" and note_content:
            note_title = note_content.split('.')[0]  # Use the first sentence as a title
        
        open_notes()
        create_note_with_title(note_title, note_content)
        return f"Note created with title '{note_title}' and content: {note_content}"
    
    elif intent == "search_note":
        note_title = inferred_details.get("note_title", "")
        
        closest_match, content = search_note_and_get_content(note_title)
        return f"Note found with closest matching title '{closest_match}' and content: {content}" if content else f"No note found with a close match to title '{note_title}'"
    
    else:
        return "Command not recognized"

# Example usage:
example = """
{
  "intent": "LISTS_CREATEORADD",
  "entities": {
    "list_name": "grocery",
    "items": ["apples", "mangos", "bananas"]
  }
}

"""
#print(notes_agent(example))
#print(notes_agent("Can you find the note about the groceries list?"))
