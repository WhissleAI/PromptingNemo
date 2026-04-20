import openai
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.mime.text import MIMEText
import base64
import csv
from difflib import SequenceMatcher
import os
import yaml

# Load configuration from YAML file
with open("config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# Assign OpenAI API key from the config
openai.api_key = config['openai']['api_key']

def load_contacts():
    filename = config['contacts']['file']
    contacts = {}
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')  # Specify the tab delimiter here
        for row in reader:
            print("Row", row)
            contacts[row['Name']] = row['Email']
    return contacts

def find_best_matching_contact(name, contacts):
    best_match = None
    best_similarity = 0.0
    for contact_name in contacts:
        similarity = SequenceMatcher(None, name.lower(), contact_name.lower()).ratio()
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = contact_name
    return best_match if best_similarity > 0.5 else None

def send_email(prompt):
    SCOPES = config['google']['scopes']
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {"role": "system", "content": "You identify the recipient, title, and body of an email to be sent based on a user request. The output must be in the format: 'recipient | email title | email body'"},
            {"role": "user", "content": prompt},
        ]
    )
    reply_content = completion.choices[0].message.content
    email_data = reply_content.strip().split('|')
    
    print(reply_content)
    to, subject, body = map(str.strip, email_data)
    print(to, "::",subject, "::",body)
    
    if '@' not in to:
        contacts = load_contacts()
        print("Contacts:", contacts)
        best_match = find_best_matching_contact(to, contacts)
        print("Best match:", best_match)
        to = contacts.get(best_match, None)
        if to is None:
            print(f"No valid email address found for {best_match}")
            return f"No matching contact found for {to}"

    try:
        creds = None
        if os.path.exists(config['google']['token_file']):
            creds = Credentials.from_authorized_user_file(config['google']['token_file'], SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(config['google']['client_secrets_file'], SCOPES)
                creds = flow.run_local_server(port=0)
            with open(config['google']['token_file'], 'w') as token:
                token.write(creds.to_json())

        service = build('gmail', 'v1', credentials=creds)
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject
        create_message = {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}
        service.users().messages().send(userId="me", body=create_message).execute()
        status = "Email has been sent."
    except HttpError as error:
        print(f'An error occurred: {error}')
        status = "An error occurred, message not sent."
    return status

# # Example usage
# print(send_email("Send an email to karan | Meeting reminder | Remember our meeting tomorrow at 10 AM."))
