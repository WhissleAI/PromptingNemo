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

def is_reminders_running():
    for process in psutil.process_iter():
        try:
            if 'reminders' in process.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

def open_reminders():
    if not is_reminders_running():
        subprocess.call(['open', '-a', 'Reminders'])
        time.sleep(1)

def create_reminder_with_details(reminder_title, reminder_content, reminder_date=None, reminder_time=None, reminder_location=None):
    apple_script = f'''
    tell application "Reminders"
        tell list "Reminders"
            set newReminder to make new reminder with properties {{name:"{reminder_title}", body:"{reminder_content}"}}
    '''
    
    if reminder_date and reminder_time:
        # Combine date and time into a single AppleScript date object
        apple_script += f'''
            set due date of newReminder to date "{reminder_date} at {reminder_time}"
        '''
    elif reminder_date:
        apple_script += f'''
            set due date of newReminder to date "{reminder_date}"
        '''
    
    if reminder_location:
        # Setting location requires location services to be enabled and a valid location to be set
        apple_script += f'''
            set location of newReminder to "{reminder_location}"
        '''
    
    apple_script += '''
        end tell
    end tell
    '''
    
    subprocess.run(['osascript', '-e', apple_script])


def get_all_reminder_titles():
    apple_script = '''
    tell application "Reminders"
        set reminderTitles to ""
        tell list "Reminders"
            set allReminders to every reminder
            repeat with aReminder in allReminders
                set reminderTitles to reminderTitles & (name of aReminder) & "|"
            end repeat
        end tell
        return reminderTitles
    end tell
    '''
    result = subprocess.run(['osascript', '-e', apple_script], capture_output=True, text=True)
    return result.stdout.strip().split('|')

def search_reminder_and_get_content(reminder_title):
    all_titles = get_all_reminder_titles()
    closest_match, _ = process.extractOne(reminder_title, all_titles)
    
    apple_script = f'''
    tell application "Reminders"
        set reminderContent to ""
        tell list "Reminders"
            set theReminder to first reminder whose name is "{closest_match}"
            set reminderContent to body of theReminder
        end tell
        return reminderContent
    end tell
    '''
    result = subprocess.run(['osascript', '-e', apple_script], capture_output=True, text=True)
    return closest_match, result.stdout.strip()

def infer_reminder_details(refined_input):
    inference_prompt = f'''
    Analyze the following refined input and infer the intent (such as "create_reminder" or "search_reminder"), 
    the reminder title if available, the reminder content, date, time, and location if available.
    Clean the input to remove any noise or unnecessary information.
    Output the result in the following JSON format:
    {{
        "intent": "<inferred_intent>",
        "reminder_title": "<inferred_reminder_title>",
        "reminder_content": "<inferred_reminder_content>",
        "reminder_date": "<inferred_reminder_date>",
        "reminder_time": "<inferred_reminder_time>",
        "reminder_location": "<inferred_reminder_location>"
    }}
    Refined Input: {refined_input}
    '''
    
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that infers details from user input for reminder management."},
            {"role": "user", "content": inference_prompt},
        ]
    )
    
    inferred_details = completion.choices[0].message.content.strip()
    return yaml.safe_load(inferred_details)

def reminder_agent(refined_input):
    # Use GPT-4 to infer the intent, reminder title, and content from the refined input
    inferred_details = infer_reminder_details(refined_input)
    
    intent = inferred_details.get("intent", "").lower()
    
    if intent == "create_reminder":
        reminder_title = inferred_details.get("reminder_title", "Untitled Reminder")
        reminder_content = inferred_details.get("reminder_content", "")
        reminder_date = inferred_details.get("reminder_date", None)
        reminder_time = inferred_details.get("reminder_time", None)
        reminder_location = inferred_details.get("reminder_location", None)
        
        if isinstance(reminder_content, list):
            reminder_content = ", ".join(reminder_content)
        
        # If no title was extracted, create a title based on the content
        if reminder_title == "Untitled Reminder" and reminder_content:
            reminder_title = reminder_content.split('.')[0]  # Use the first sentence as a title
        
        open_reminders()
        create_reminder_with_details(reminder_title, reminder_content, reminder_date, reminder_time, reminder_location)
        return f"Reminder created with title '{reminder_title}', content: {reminder_content}, date: {reminder_date}, time: {reminder_time}, location: {reminder_location}"
    
    elif intent == "search_reminder":
        reminder_title = inferred_details.get("reminder_title", "")
        
        closest_match, content = search_reminder_and_get_content(reminder_title)
        return f"Reminder found with closest matching title '{closest_match}' and content: {content}" if content else f"No reminder found with a close match to title '{reminder_title}'"
    
    else:
        return "Command not recognized"

# Example usage:
# example = """
# {
#   "intent": "REMINDERS_CREATEORADD",
#   "entities": {
#     "reminder_name": "meeting with team",
#     "details": ["prepare agenda", "send invites"],
#     "date": "2024-08-10",
#     "time": "10:00 AM"
#   }
# }
# """
# print(reminder_agent(example))
#print(reminder_agent("Set a reminder for tomorrow morning to have fun at 10am tomorrow at the park."))
