from twilio.rest import Client
import csv
from difflib import SequenceMatcher
import openai
import yaml

# Load configuration from YAML file
with open("config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# Assign variables from the config
openai.api_key = config['openai']['api_key']
twilio_config = config['twilio']

def load_contacts(filename):
    contacts = {}
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            contacts[row['Name'].strip()] = row['Phone 1 - Value'].strip()
    return contacts

def find_best_matching_contact(name, contacts):
    best_match = None
    best_similarity = 0.0
    for contact_name in contacts.keys():
        similarity = SequenceMatcher(None, name.lower(), contact_name.lower()).ratio()
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = contact_name
    return best_match if best_similarity > 0.5 else None

def send_sms(to, body):    
    client = Client(twilio_config['account_sid'], twilio_config['auth_token'])
    message = client.messages.create(
                        body=body,
                        from_=twilio_config['phone_number'],
                        to=to
                    )

def sms_agent(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {"role":"system", "content": "You identify the recipient and content of an SMS based on a user request. Format the output as 'recipient | sms text'."},
            {"role":"user", "content": prompt}
        ]
    )
    reply_content = completion.choices[0].message.content
    sms_data = reply_content.strip().split('|')
    recipient, body = sms_data[0].strip(), sms_data[1].strip()

    if not any(char.isdigit() for char in recipient):  # Check if recipient is not a phone number
        contacts = load_contacts("utils/contacts.csv")
        best_match = find_best_matching_contact(recipient, contacts)
        if best_match:
            recipient = contacts[best_match]
        else:
            print(f"No matching contact found for {recipient}")
            return

    send_sms(recipient, body)

# Example usage:
# sms_agent("Send an SMS to John saying Hello, how are you?")
