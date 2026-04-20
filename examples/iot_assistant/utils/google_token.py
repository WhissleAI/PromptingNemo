from google_auth_oauthlib.flow import InstalledAppFlow

# Replace 'google_client_secret.json' with the path to the downloaded client secrets file
flow = InstalledAppFlow.from_client_secrets_file(
    'google_client_secret.json',
    scopes=['https://www.googleapis.com/auth/gmail.send']
)

# This will open a new browser window for authentication using the specified port
creds = flow.run_local_server(port=3000)

# Save the credentials for the next run
with open('token.json', 'w') as token:
    token.write(creds.to_json())